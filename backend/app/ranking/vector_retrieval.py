"""Stage-1 job retrieval in the database (``STAGE1_RETRIEVAL_BACKEND=db``): Atlas ``$vectorSearch``
over the pre-whitened job vectors, in place of ``Stage1Retriever``'s in-process matrix.

Only the top-K hits and the fields later stages read come back; no job vector is loaded or whitened
in-process. Occupations are not searched here: they keep the in-process stage 1. Per user:

1. **Query vector.** The user's concat embedding, whitened with the service's artifact. Whitening
   re-normalises each vector, so it cannot be applied to the query alone: the documents must already
   be whitened, and only documents whitened with *this* artifact are searched (the whitening
   SHA-256 is a pre-filter). Other versions are never mixed in; they are counted at startup.
2. **Jobs.** One search per location tier of the user's ``UserLocationKeys`` (``app.location.keys``):
   consecutive keys sharing a tier weight are searched together (remote + local, then each hub).
   Each search is pre-filtered on ``is_active``, the artifact, the tier's ``location_keys`` and the
   education gate. The hits are merged and ranked by cosine x tier (``UserLocationKeys.tier_for``,
   as in memory); hits outside the user's keys are dropped and the top K kept. A separate search per
   tier keeps the weighted top K complete: every item of it is in its own tier's top K.
   ``concat_cosine_similarity`` stays the unweighted search cosine.

The education gate and location tiers are re-checked on the hits (K rows, cheap). A hit dropped
there means a document's pre-filter field disagrees with its content, so it is counted in
``Stage1SearchStats``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import ConfigDict, Field

from app.domain.base import DomainModel
from app.domain.user import UserProfile
from app.jobs.job_document import (
    JOB_REQUIRES_POST_SECONDARY_FIELD,
    JOB_WHITENING_SHA256_FIELD,
    LOCATION_KEYS_FIELD,
)
from app.jobs.repository import IJobsRepository, JobSearchHit, JobsSearchResult
from app.location.keys import UserLocationKeys
from app.ranking.retrieval import (
    job_requires_post_secondary,
    user_lacks_post_secondary,
)
from app.ranking.types import Corpus, MatchItem
from app.server_dependencies.vector_search import (
    VectorSearchQuery,
    num_candidates_for,
)

logger = logging.getLogger(__name__)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


# --------------------------------------------------------------------------------------------------
# pre-filters
# --------------------------------------------------------------------------------------------------


class LocationTier(DomainModel):
    """One job search of a user: the location keys it is restricted to."""

    keys: tuple[str, ...] | None = Field(
        description="``location_keys`` values to match (any); None = no location restriction."
    )


def job_location_tiers(location_keys: UserLocationKeys | None) -> list[LocationTier]:
    """The job searches to run for one user, best tier first.

    Consecutive keys of ``location_keys`` with the same tier weight share a search (without hub
    chains every key weighs 1.0, so that is one search); None is one unrestricted search.
    """
    if location_keys is None:
        return [LocationTier(keys=None)]
    groups: list[tuple[float, list[str]]] = []
    for key, tier in location_keys.tiers:
        if groups and groups[-1][0] == tier:
            groups[-1][1].append(key)
        else:
            groups.append((tier, [key]))
    return [LocationTier(keys=tuple(sorted(keys))) for _, keys in groups]


def job_prefilter(
    user: UserProfile, tier: LocationTier, *, artifact_sha256: str
) -> dict[str, Any]:
    """The ``$vectorSearch`` filter of one job search: active, same whitening, tier, education."""
    clauses: list[dict[str, Any]] = [
        {"is_active": {"$eq": True}},
        {JOB_WHITENING_SHA256_FIELD: {"$eq": artifact_sha256}},
    ]
    if tier.keys is not None:
        clauses.append({LOCATION_KEYS_FIELD: {"$in": list(tier.keys)}})
    if user_lacks_post_secondary(user):
        clauses.append({JOB_REQUIRES_POST_SECONDARY_FIELD: {"$ne": True}})
    return {"$and": clauses}


# --------------------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------------------


class Stage1Hit(DomainModel):
    """One shortlisted job of one user, before the per-skill detail is attached."""

    item: MatchItem = Field(description="The job (no embedding).")
    concat_cosine_similarity: float = Field(
        description="Whitened concat cosine from the search (not tier-weighted)."
    )


class Stage1SearchStats(DomainModel):
    """Counts and timings of one corpus's searches for one request (logged and traced)."""

    corpus: Corpus = Field(description="Which corpus was searched.")
    n_searches: int = Field(default=0, description="``$vectorSearch`` calls made.")
    n_returned: int = Field(default=0, description="Documents the searches returned.")
    n_skipped_invalid: int = Field(
        default=0, description="Job documents that failed validation."
    )
    n_education_dropped: int = Field(
        default=0,
        description="Hits the in-process education check dropped (pre-filter field out of date).",
    )
    n_off_location: int = Field(
        default=0,
        description="Job hits outside the user's location keys (tier 0).",
    )
    search_ms_max: float = Field(default=0.0, description="Slowest single search.")
    wall_ms: float = Field(default=0.0, description="Wall time of all the searches.")


class Stage1SearchResult(DomainModel):
    """One corpus's shortlist per user, in user order, and how it was found."""

    shortlists: list[list[Stage1Hit]] = Field(
        description="Per user: hits, best first (tier-weighted), at most K."
    )
    stats: Stage1SearchStats = Field(description="Counts and timings.")

    @property
    def n_distinct_items(self) -> int:
        return len({h.item.uuid for hits in self.shortlists for h in hits})


class DbStage1(DomainModel):
    """Stage-1 job results from the database for one request, handed to ``MatchingService.rank``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_unit_vectors: np.ndarray = Field(
        description="The users' L2-normalised (unwhitened) embeddings, ``[n_users, dim]``."
    )
    jobs: Stage1SearchResult = Field(description="Job shortlists.")


# --------------------------------------------------------------------------------------------------
# retriever
# --------------------------------------------------------------------------------------------------


class IDbStage1Retriever(ABC):
    """Interface for job stage 1 in the database (one search per user and tier)."""

    @abstractmethod
    async def shortlist_jobs(
        self,
        users: Sequence[UserProfile],
        user_white_vectors: np.ndarray,
        *,
        artifact_sha256: str,
        retrieve_top_k: int,
        location_keys: Sequence[UserLocationKeys] | None,
    ) -> Stage1SearchResult:
        """
        Searches jobs for every user, concurrently.

        :param users: The users, aligned with ``user_white_vectors``
        :param user_white_vectors: Whitened, unit-length user embeddings ``[n_users, dim]``
        :param artifact_sha256: SHA-256 of the whitening artifact that made the query vectors
        :param retrieve_top_k: Shortlist size per user
        :param location_keys: Per-user eligible location keys and tiers (aligned with ``users``) to
            search, weight by and drop ineligible jobs with, or None for no location step
        :return: One shortlist per user
        :raises pymongo.errors.PyMongoError: If a search fails
        """
        raise NotImplementedError()


class VectorSearchStage1Retriever(IDbStage1Retriever):
    """``IDbStage1Retriever`` over the jobs repository's ``search_stage1``."""

    def __init__(
        self,
        *,
        jobs_repository: IJobsRepository,
        num_candidates_factor: int,
        exact: bool,
        concurrency: int,
    ):
        self._jobs_repository = jobs_repository
        self._num_candidates_factor = num_candidates_factor
        self._exact = exact
        self._concurrency = max(1, concurrency)

    def _query(
        self, vector: np.ndarray, query_filter: dict[str, Any], limit: int
    ) -> VectorSearchQuery:
        return VectorSearchQuery(
            vector=[float(x) for x in vector],
            filter=query_filter,
            limit=limit,
            num_candidates=num_candidates_for(limit, self._num_candidates_factor),
            exact=self._exact,
        )

    async def shortlist_jobs(
        self,
        users: Sequence[UserProfile],
        user_white_vectors: np.ndarray,
        *,
        artifact_sha256: str,
        retrieve_top_k: int,
        location_keys: Sequence[UserLocationKeys] | None,
    ) -> Stage1SearchResult:
        t0 = time.perf_counter()
        rt = max(1, int(retrieve_top_k))
        gate = asyncio.Semaphore(self._concurrency)

        async def search(query: VectorSearchQuery) -> JobsSearchResult:
            async with gate:
                return await self._jobs_repository.search_stage1(query)

        user_keys: list[UserLocationKeys | None] = (
            list(location_keys) if location_keys is not None else [None] * len(users)
        )
        tiers = [job_location_tiers(keys) for keys in user_keys]
        flat = [
            self._query(
                user_white_vectors[i],
                job_prefilter(user, tier, artifact_sha256=artifact_sha256),
                rt,
            )
            for i, user in enumerate(users)
            for tier in tiers[i]
        ]
        results = await asyncio.gather(*(search(q) for q in flat))

        stats = Stage1SearchStats(
            corpus="jobs",
            n_searches=len(results),
            n_returned=sum(r.n_returned for r in results),
            n_skipped_invalid=sum(r.n_skipped_invalid for r in results),
            search_ms_max=max((r.search_ms for r in results), default=0.0),
        )
        shortlists: list[list[Stage1Hit]] = []
        n_education = n_off_location = 0
        start = 0
        for i, user in enumerate(users):
            user_results = results[start : start + len(tiers[i])]
            start += len(tiers[i])
            hits, dropped_education, off_location = self._rank_job_hits(
                user,
                [h for r in user_results for h in r.hits],
                rt=rt,
                location_keys=user_keys[i],
            )
            shortlists.append(hits)
            n_education += dropped_education
            n_off_location += off_location
        stats = stats.model_copy(
            update={
                "n_education_dropped": n_education,
                "n_off_location": n_off_location,
                "wall_ms": _ms(t0),
            }
        )
        if n_education:
            logger.warning(
                "stage-1 vector search: %d job hits require post-secondary education but passed "
                "the pre-filter; their %s field is out of date.",
                n_education,
                JOB_REQUIRES_POST_SECONDARY_FIELD,
            )
        return Stage1SearchResult(shortlists=shortlists, stats=stats)

    @staticmethod
    def _rank_job_hits(
        user: UserProfile,
        hits: Sequence[JobSearchHit],
        *,
        rt: int,
        location_keys: UserLocationKeys | None,
    ) -> tuple[list[Stage1Hit], int, int]:
        """Merge one user's tier hits: dedupe, gate, rank by cosine x tier, keep ``rt``.

        :return: ``(hits, n_dropped_by_education, n_off_location)``
        """
        user_no_ps = user_lacks_post_secondary(user)
        seen: set[str] = set()
        ranked: list[tuple[float, Stage1Hit]] = []
        n_education = n_off_location = 0
        for hit in hits:
            job = hit.job
            if job.uuid in seen:
                continue
            seen.add(job.uuid)
            if user_no_ps and job_requires_post_secondary(job):
                n_education += 1
                continue
            tier = 1.0
            if location_keys is not None:
                tier = location_keys.tier_for(job.location_keys)
                if tier <= 0.0:
                    n_off_location += 1
                    continue
            ranked.append(
                (
                    hit.cosine * tier,
                    Stage1Hit(item=job, concat_cosine_similarity=hit.cosine),
                )
            )
        ranked.sort(key=lambda pair: -pair[0])
        return [h for _, h in ranked[:rt]], n_education, n_off_location
