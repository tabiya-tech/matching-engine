"""Stage-1 candidate retrieval: concat-embedding text and vectors, whitening, the post-secondary education gate, location tiers, and the whitened-cosine shortlist."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from app.artifacts.types import ConcatWhitening
from app.domain.base import DomainModel
from app.domain.user import UserProfile
from app.ranking.skills import CosineSkillMatcher
from app.ranking.types import Candidate, MatchItem

logger = logging.getLogger(__name__)

ItemT = TypeVar("ItemT", bound=MatchItem)

# --------------------------------------------------------------------------------------------------
# vectors
#
# Stage-1 concat embedding vectors on jobs / occupations, and row normalisation.
# --------------------------------------------------------------------------------------------------


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Rows scaled to unit L2 norm (zero rows unchanged), as float32."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms > 0, norms, 1.0).astype(np.float32)
    return (mat / norms).astype(np.float32)


def stage1_vector(item: MatchItem, *, dim: int) -> np.ndarray | None:
    """The item's concat embedding as float32, or None when absent or not ``dim`` long."""
    vec = item.embedding
    if vec is None or vec.ndim != 1 or vec.size != dim:
        return None
    return vec.astype(np.float32, copy=False)


def is_prewhitened(item: MatchItem) -> bool:
    """True iff the item's stage-1 embedding is ALREADY whitened (DB-side for jobs, once at load
    for occupations); raw vectors are whitened in-process."""
    return item.embedding_whitened


def index_by_uuid(items: Sequence[ItemT]) -> dict[str, ItemT]:
    """Items keyed by ``uuid`` (items without one are dropped; the last duplicate wins)."""
    return {it.uuid: it for it in items if it.uuid}


# --------------------------------------------------------------------------------------------------
# whitening
#
# Concat-embedding whitening ((x - mu) @ W, re-normalised) and its p99 rescale target.
# --------------------------------------------------------------------------------------------------


class ConcatWhitener:
    """Whitening for the user/job concat embedding space.

    ``artifact`` comes from ``ArtifactsRepository.load_concat_whitening``; None when the artifact is
    unavailable — then rows are only L2-normalised and the rescale target is 0.0.
    """

    def __init__(self, artifact: ConcatWhitening | None):
        self._artifact = artifact

    def whiten_rows(self, vecs: np.ndarray) -> np.ndarray:
        """L2-normalise rows, apply the concat whitening ((.-mu)@W), re-normalise -> unit whitened rows.
        If the artifact is unavailable, returns the L2-normalised rows unchanged."""
        v = l2_normalize_rows(np.asarray(vecs, dtype=np.float64))
        cw = self._artifact
        if cw is None:
            return v
        return l2_normalize_rows((v - cw.mu) @ cw.W)

    def rescale_target(self) -> float:
        """p99 rescale target for the whitened concat cosine (0.0 if the artifact is unavailable)."""
        cw = self._artifact
        return cw.target if cw is not None else 0.0


# --------------------------------------------------------------------------------------------------
# concat_embedding_text
#
# Single string per side for embedding: concatenate skill labels only.
# --------------------------------------------------------------------------------------------------

DEFAULT_SEPARATOR = " | "


def _dedupe_stable(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in labels:
        s = str(raw).strip()
        if not s:
            continue
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def build_concat_embedding_text(
    labels: Sequence[str],
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    """Join skill labels into one line for embedding models."""

    return separator.join(_dedupe_stable(list(labels)))


def user_skill_labels_for_concat(user: UserProfile) -> list[str]:
    """The user's skill labels (``skills_vector.top_skills[].preferredLabel``), de-duplicated."""
    return _dedupe_stable(
        [s.preferredLabel for s in user.skills_vector.top_skills if s.preferredLabel]
    )


def job_skill_labels_for_concat(item: MatchItem) -> list[str]:
    """The item's essential ∪ optional skill labels, de-duplicated."""
    return _dedupe_stable(
        [s.label for s in [*item.essential_skills, *item.optional_skills] if s.label]
    )


def user_concat_embedding_text(
    user: UserProfile,
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    """The user's embedding text: skill labels joined by ``separator``."""
    return build_concat_embedding_text(
        user_skill_labels_for_concat(user), separator=separator
    )


def job_concat_embedding_text(
    item: MatchItem,
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    """The item's embedding text: skill labels joined by ``separator``."""
    return build_concat_embedding_text(
        job_skill_labels_for_concat(item), separator=separator
    )


# --------------------------------------------------------------------------------------------------
# education
#
# Post-secondary education eligibility gate (shared by all matching endpoints).
#
# A job may declare ``requires_post_secondary`` (boolean) under
# ``llm_job_attributes.attributes`` in Mongo; ``build_job_from_ranked`` normalises it onto
# ``Job.requires_post_secondary`` (occupations carry the same field).
#
# The user side comes from ``MatchRequest.any_post_secondary_educ`` (0/1, optional).
#
# Gate (applied uniformly across /match, /match_v2, /match_v3, /match_v4):
# a user is **ineligible** for a job only when the job requires post-secondary
# education *and* the user explicitly reported having none.
#
# Fail-open by design — only positive evidence excludes:
# * Job has no / false ``requires_post_secondary``  -> eligible (never hide a job
#   that simply lacks the field).
# * User did not supply ``any_post_secondary_educ`` (None) -> eligible; preserves
#   behaviour for clients that omit the optional field. Only an explicit ``0`` excludes.
# --------------------------------------------------------------------------------------------------


def job_requires_post_secondary(item: MatchItem) -> bool:
    """True only when the item explicitly requires post-secondary education.

    Absent / unknown (None) is "not required", so an item is never hidden for lacking the field.
    The flag is normalised from bool / 0/1 / strings when the document is read (``parse_flag``).
    """
    return item.requires_post_secondary is True


def user_lacks_post_secondary(user: UserProfile) -> bool:
    """True only when the user explicitly reported no post-secondary education (``0``).

    ``1``, a stray value, or an omitted field (None) is not treated as lacking.
    """
    return user.any_post_secondary_educ == 0


def is_education_eligible(user: UserProfile, item: MatchItem) -> bool:
    """Education gate: ineligible iff the item requires post-secondary and the user has none."""
    return not (job_requires_post_secondary(item) and user_lacks_post_secondary(user))


def filter_jobs_by_education(user: UserProfile, items: Sequence[ItemT]) -> list[ItemT]:
    """Drop items the user is education-ineligible for.

    No-op (returns a shallow copy) when the user has — or did not report lacking —
    post-secondary education, so the common path stays cheap.
    """
    if not user_lacks_post_secondary(user):
        return list(items)
    return [j for j in items if not job_requires_post_secondary(j)]


def zqf_annotation(
    user_zqf: int | None, job_zqf_min: int | None
) -> tuple[bool | None, int | None]:
    """(zqf_eligible, zqf_gap) or (None, None) when either side is missing."""
    if user_zqf is not None and job_zqf_min is not None:
        return (user_zqf >= job_zqf_min, abs(user_zqf - job_zqf_min))
    return (None, None)


# --------------------------------------------------------------------------------------------------
# location
#
# Tiered urban-pull location matching.
#
# Maps a user's county to an ordered fallback **hub chain** (local -> regional hub -> national hub) and
# scores each job by which tier it falls in, so non-hub users (whose own county has few jobs) still get a
# full list while local jobs stay preferred. Hub counties do not pull outward.
#
# Two consumers:
#   * the Mongo prefilter (app.services.location_query.location_or_clauses_for_one_user) uses ``chain_for`` to widen a
#     user's candidate pool to include their hub regions;
#   * the v4 opportunity ranker uses ``tier_factor_for_job`` as a per-uuid [0,1] multiplier on final_score
#     (local=1.0, regional=W_REGIONAL, national=W_NATIONAL, off-chain=0.0).
#
# Region matching uses ``job_matches_user_location`` — the same lenient casefold-substring + always-remote
# rule every matching path uses.
#
# The hub data is a small JSON of exceptions (see resources/location/location_hub_chains.json, loaded by
# ``ArtifactsRepository.load_hub_chains``); every county not named there defaults to
# ``[self, national_hub]``. If the file is missing/malformed the loader returns None and callers fall back to
# today's strict behaviour (the feature becomes a no-op).
# --------------------------------------------------------------------------------------------------


def _cf(s: str | None) -> str:
    """Casefold + strip, aligned with _norm / app.services.location_query._norm_loc_value."""
    if s is None:
        return ""
    t = s.strip()
    return t.casefold() if t else ""


def _norm(v: str | None) -> str:
    return str(v).strip().casefold() if v is not None else ""


class Place(DomainModel):
    """A location as the matching rules see it: city, province and a free-text location."""

    city: str = Field(default="", description="City/town; empty when unknown.")
    province: str = Field(
        default="", description="Province/county; empty when unknown."
    )
    location: str = Field(
        default="",
        description="Free-text location (jobs: ``city province``); empty when unknown.",
    )

    @classmethod
    def of_item(cls, item: MatchItem) -> Place:
        """Where a job is, or the county an occupation row applies to."""
        return cls(city=item.city, province=item.province, location=item.location)

    @classmethod
    def of_user(cls, user: UserProfile) -> Place:
        """Where the user is (no free-text location)."""
        return cls(city=user.city, province=user.province)

    @classmethod
    def region(cls, name: str) -> Place:
        """A single region name used for every field (counties, hubs)."""
        return cls(city=name, province=name, location=name)


def job_matches_user_location(job: Place, user: Place) -> bool:
    """Lenient location match.
    - Always matches 'Remote' jobs
    - Matches if city or province match (case-insensitive, substring)
    """
    user_city = _norm(user.city)
    user_province = _norm(user.province)

    job_city = _norm(job.city)
    job_province = _norm(job.province)
    job_loc = _norm(job.location)

    #  Always include Remote jobs
    if "remote" in job_city or "remote" in job_province or "remote" in job_loc:
        return True

    if not user_city or not user_province:
        return False

    # Check City Match (Lenient)
    if job_city and (user_city in job_city or job_city in user_city):
        return True

    # Check Province Match (Lenient)
    if job_province and (
        user_province in job_province or job_province in user_province
    ):
        return True

    # Fallback to location string match
    if job_loc:
        return user_city in job_loc or user_province in job_loc

    return False


def user_matches_any_county(user: UserProfile, counties: Sequence[str]) -> bool:
    """True if the user's location matches at least one of the given occupation counties."""
    where = Place.of_user(user)
    return any(job_matches_user_location(Place.region(c), where) for c in counties)


class HubChains:
    """Parsed county -> hub-chain map. Built from the exceptions JSON; chains derived lazily."""

    def __init__(
        self,
        national_hub: str,
        regional_hubs: dict[str, list[str]] | None,
        hub_self_only: list[str] | None,
    ) -> None:
        self.national: str = _cf(national_hub)
        self.regional_of: dict[str, str] = {}  # county_cf -> regional hub_cf
        for hub, counties in (regional_hubs or {}).items():
            h = _cf(hub)
            if not h:
                continue
            for c in counties or []:
                cc = _cf(c)
                if cc:
                    self.regional_of[cc] = h
        self.self_only: set = {_cf(h) for h in (hub_self_only or []) if _cf(h)}

    def chain_for(self, county_cf: str) -> list[str]:
        """Ordered fallback regions for a user county: [local, regional hub?, national hub?].

        Hubs in ``hub_self_only`` -> ``[self]`` (no outward pull). Unknown/empty counties -> the
        national hub only (safe default). Order encodes tier: index 0 = local.
        """
        c = _cf(county_cf)
        if not c:
            return [self.national] if self.national else []
        if c in self.self_only:
            return [c]
        chain = [c]
        reg = self.regional_of.get(c)
        if reg and reg not in chain:
            chain.append(reg)
        if self.national and self.national not in chain:
            chain.append(self.national)
        return chain

    def tier_factor_for_job(
        self,
        job: Place,
        county_cf: str,
        *,
        w_regional: float,
        w_national: float,
    ) -> float:
        """Location multiplier for one job given the user's county.

        local (chain[0]) -> 1.0; national hub -> ``w_national``; any other in-chain (regional) hub ->
        ``w_regional``; remote jobs -> 1.0 (always allowed); anything off-chain -> 0.0. Roles are keyed
        by identity (local / national), NOT chain index, so a county whose chain skips the regional tier
        (e.g. Kitui -> [kitui, nairobi]) still scores Nairobi at the national weight.
        """
        chain = self.chain_for(county_cf)
        if not chain:
            return 1.0  # no chain (unknown national hub) -> don't penalise; degrade to neutral
        local = chain[0]
        for region in chain:
            if job_matches_user_location(job, Place.region(region)):
                if region == local:
                    return 1.0
                if region == self.national:
                    return float(w_national)
                return float(w_regional)
        return 0.0


# --------------------------------------------------------------------------------------------------
# retrieval
#
# Stage-1 retrieval: whitened concat-cosine shortlist per user, with the education gate and the
# optional location-tier (urban-pull) weighting applied before the ``retrieve_top_k`` cutoff.
# --------------------------------------------------------------------------------------------------


class Stage1Corpus(BaseModel):
    """The items of one corpus that carry a stage-1 vector, aligned row-for-row."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rows: SkipValidation[list[MatchItem]] = Field(
        description="Items with a stage-1 vector, in input order (the caller's own objects)."
    )
    n_loaded: int = Field(description="Items passed in, with or without a vector.")
    matrix: np.ndarray = Field(
        default_factory=lambda: np.zeros((0, 0)),
        description="Raw stacked vectors (float64).",
    )
    normalized: np.ndarray = Field(
        default_factory=lambda: np.zeros((0, 0)),
        description="L2-normalised ``matrix`` (float64).",
    )
    ids: list[str] = Field(
        default_factory=list, description="Item uuids, aligned with ``rows``."
    )
    requires_post_secondary: list[bool] = Field(
        default_factory=list, description="Education-gate flags, aligned with ``rows``."
    )


def _sorted_indices_desc(sim_row: np.ndarray) -> np.ndarray:
    return np.argsort(-sim_row, kind="stable")


class IStage1Retriever(ABC):
    """Interface for the stage-1 shortlist."""

    @abstractmethod
    def prepare(self, items: Sequence[MatchItem]) -> Stage1Corpus:
        """
        Collects the items that carry a stage-1 vector.

        :param items: Jobs or occupation rows
        :return: The aligned corpus (``rows`` empty when no item has a vector)
        """
        raise NotImplementedError()

    @abstractmethod
    def shortlist(
        self,
        users: Sequence[UserProfile],
        corpus: Stage1Corpus,
        user_unit_vectors: np.ndarray,
        *,
        matcher: CosineSkillMatcher,
        whitener: ConcatWhitener,
        retrieve_top_k: int,
        hub_chains: HubChains | None = None,
        w_regional: float = 1.0,
        w_national: float = 1.0,
    ) -> list[list[Candidate]]:
        """
        Ranks the corpus per user and keeps the top ``retrieve_top_k`` eligible rows.

        :param users: The users, aligned with ``user_unit_vectors``
        :param corpus: A non-empty corpus from ``prepare``
        :param user_unit_vectors: L2-normalised user embeddings ``[n_users, dim]``
        :param matcher: Per-skill matcher whose ``score_pair`` detail is attached to each candidate
        :param whitener: Concat whitening (a no-op when its artifact is unavailable)
        :param retrieve_top_k: Shortlist size per user
        :param hub_chains: Location tiers to weight the ranking by, or None to skip tiering
        :param w_regional: Tier weight of a regional hub
        :param w_national: Tier weight of the national hub
        :return: One cosine-ranked shortlist per user
        """
        raise NotImplementedError()


class Stage1Retriever(IStage1Retriever):
    """Whitened concat-cosine shortlist with the education gate and optional location tiers."""

    def __init__(self, *, embedding_dim: int):
        self._embedding_dim = embedding_dim

    def prepare(self, items: Sequence[MatchItem]) -> Stage1Corpus:
        rows: list[MatchItem] = []
        vectors: list[np.ndarray] = []
        for item in items:
            v = stage1_vector(item, dim=self._embedding_dim)
            if v is None:
                continue
            rows.append(item)
            vectors.append(v)

        corpus = Stage1Corpus(
            rows=rows,
            n_loaded=len(items),
            # Post-secondary education gate: aligned with rows, used to skip candidates per user.
            requires_post_secondary=[job_requires_post_secondary(j) for j in rows],
        )
        if not rows:
            return corpus

        corpus.matrix = np.stack(vectors, axis=0).astype(np.float64)
        corpus.normalized = l2_normalize_rows(corpus.matrix.astype(np.float32)).astype(
            np.float64
        )
        corpus.ids = [j.uuid for j in rows]
        return corpus

    def shortlist(
        self,
        users: Sequence[UserProfile],
        corpus: Stage1Corpus,
        user_unit_vectors: np.ndarray,
        *,
        matcher: CosineSkillMatcher,
        whitener: ConcatWhitener,
        retrieve_top_k: int,
        hub_chains: HubChains | None = None,
        w_regional: float = 1.0,
        w_national: float = 1.0,
    ) -> list[list[Candidate]]:
        rt = max(1, int(retrieve_top_k))
        rows = corpus.rows
        j_mat = corpus.matrix
        j_norm = corpus.normalized
        job_requires_ps = corpus.requires_post_secondary
        u_norm = user_unit_vectors

        # Whitened-space stage-1 retrieval. The concat artifact (same one the DB used to whiten
        # job_embedding) is present in practice, so we rank in the de-anisotropised whitened space (the
        # meaningful signal; raw concat cosine sd ~0.02). Jobs already whitened on the DB side are used
        # as-is; RAW vectors (occupations, offline, not-yet-whitened jobs) are whitened in-process once
        # (numerically identical to the DB result — same artifact). When the artifact is unavailable
        # (target==0) we fall back to the legacy raw cosine and log loudly (DB-whitened jobs degrade).
        if whitener.rescale_target() > 0:
            j_used = j_norm.copy()
            raw_idx = [k for k, jr in enumerate(rows) if not is_prewhitened(jr)]
            if raw_idx:
                j_used[raw_idx] = whitener.whiten_rows(j_mat[raw_idx])
            u_used = whitener.whiten_rows(u_norm)
        else:
            if any(is_prewhitened(jr) for jr in rows):
                logger.error(
                    "concat whitening artifact unavailable but DB job_embedding is whitened; stage-1 "
                    "cosine will be raw-user vs whitened-job (degraded). Ship concat_whitening_gemini.npz."
                )
            j_used, u_used = j_norm, u_norm

        places = [Place.of_item(jr) for jr in rows] if hub_chains is not None else []
        shortlists: list[list[Candidate]] = []
        for i, user in enumerate(users):
            sim_row = (u_used[i : i + 1] @ j_used.T).reshape(-1)
            # Location-tier weighting of the stage-1 ranking (urban-pull). Rank by cosine * tier so local
            # jobs are favoured for the shortlist; keep the RAW cosine for the stored similarity downstream.
            loc_tier_vec = None
            if hub_chains is not None:
                county = user.province or user.city or ""
                loc_tier_vec = np.array(
                    [
                        hub_chains.tier_factor_for_job(
                            p, county, w_regional=w_regional, w_national=w_national
                        )
                        for p in places
                    ],
                    dtype=float,
                )
                rank_row = sim_row * loc_tier_vec
            else:
                rank_row = sim_row
            order = _sorted_indices_desc(rank_row)
            user_no_ps = user_lacks_post_secondary(user)

            candidates: list[Candidate] = []
            for ji in order:
                ji = int(ji)
                if user_no_ps and job_requires_ps[ji]:
                    continue  # job requires post-secondary education the user does not have
                if loc_tier_vec is not None and loc_tier_vec[ji] <= 0.0:
                    continue  # off-chain location for this user: excluded at retrieval
                item = rows[ji]
                candidates.append(
                    Candidate(
                        item=item,
                        rank=len(candidates) + 1,
                        concat_cosine_similarity=round(float(sim_row[ji]), 6),
                        per_job_skill=matcher.score_pair(user, item).per_job_skill,
                    )
                )
                if len(candidates) >= rt:
                    break
            shortlists.append(candidates)
        return shortlists
