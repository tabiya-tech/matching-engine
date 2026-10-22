"""Stage-1 job retrieval in the database (``app.ranking.vector_retrieval``), against fake repositories.

The fakes run an exact search over in-memory vectors and honour the ``$vectorSearch`` pre-filter,
so the retriever can be checked against the in-process ``Stage1Retriever`` on the same data: same
shortlist, same order, same cosines. No Atlas, no Mongo.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.artifacts.types import ConcatWhitening
from app.domain.job import Job
from app.domain.user import UserProfile
from app.jobs.repository import JobSearchHit, JobsSearchResult
from app.location.keys import HubChains, UserLocationKeys, user_location_keys
from app.ranking.retrieval import ConcatWhitener, Stage1Retriever
from app.ranking.types import SkillPairScore
from app.ranking.vector_retrieval import (
    LocationTier,
    VectorSearchStage1Retriever,
    job_location_tiers,
    job_prefilter,
)
from app.server_dependencies.vector_search import VectorSearchQuery
from tests.domain_factories import make_job, make_user

DIM = 8
SHA = "sha-deployed"
HUBS = HubChains("Nairobi", {"Mombasa": ["Kilifi"]}, ["Nairobi"])


def _get(doc: Mapping[str, Any], path: str) -> object:
    value: object = doc
    for part in path.split("."):
        value = value.get(part) if isinstance(value, Mapping) else None
    return value


def _holds(op: str, arg: object, values: list[object]) -> bool:
    if op == "$eq":
        return arg in values
    if op == "$ne":
        return arg not in values
    if op == "$in" and isinstance(arg, list):
        return any(v in arg for v in values)
    if op == "$nin" and isinstance(arg, list):
        return not any(v in arg for v in values)
    raise ValueError(f"unsupported operator {op}")


def matches(doc: Mapping[str, Any], flt: Mapping[str, Any]) -> bool:
    """The MQL subset the pre-filters use: $and, $eq, $ne, $in, $nin (arrays match any element)."""
    if "$and" in flt:
        return all(matches(doc, c) for c in flt["$and"])
    for path, cond in flt.items():
        value = _get(doc, path)
        values = value if isinstance(value, list) else [value]
        conditions = cond if isinstance(cond, Mapping) else {"$eq": cond}
        if not all(_holds(op, arg, values) for op, arg in conditions.items()):
            return False
    return True


def _unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeJobsSearch:
    """``search_stage1`` as an exact search over stored job documents."""

    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs
        self.queries: list[VectorSearchQuery] = []

    async def search_stage1(self, query: VectorSearchQuery) -> JobsSearchResult:
        self.queries.append(query)
        q = np.asarray(query.vector)
        found = sorted(
            (
                (float(q @ d["job_embedding"]), d)
                for d in self.docs
                if matches(d, query.filter)
            ),
            key=lambda pair: -pair[0],
        )[: query.limit]
        return JobsSearchResult(
            hits=[
                JobSearchHit(
                    job=d["job"].model_copy(update={"embedding": None}), cosine=c
                )
                for c, d in found
            ],
            n_returned=len(found),
            n_skipped_inactive=0,
            n_skipped_invalid=0,
            search_ms=1.0,
        )


def _retriever(jobs: FakeJobsSearch) -> VectorSearchStage1Retriever:
    return VectorSearchStage1Retriever(
        jobs_repository=jobs,  # type: ignore[arg-type]
        num_candidates_factor=15,
        exact=False,
        concurrency=4,
    )


def _matcher() -> MagicMock:
    matcher = MagicMock()
    matcher.score_pair.return_value = SkillPairScore(
        mean_best_cosine=0.0,
        min_best_cosine=0.0,
        n_user_skills_embedded=0,
        n_job_skills_embedded=0,
        per_job_skill=[],
    )
    return matcher


# Identity whitening: the stored (pre-whitened) vectors and the user vectors are already unit length.
_WHITENER = ConcatWhitener(ConcatWhitening(mu=np.zeros(DIM), W=np.eye(DIM), target=0.5))

_COUNTIES = ["Kilifi", "Mombasa", "Nairobi", "Kitui", "Remote"]


def _job_corpus(n: int = 80, seed: int = 7) -> tuple[list[Job], list[dict[str, Any]]]:
    """Jobs spread over counties (one key each) with random vectors; a third need post-secondary."""
    rng = np.random.default_rng(seed)
    jobs: list[Job] = []
    docs: list[dict[str, Any]] = []
    for i in range(n):
        county = _COUNTIES[i % len(_COUNTIES)]
        vec = _unit(rng)
        job = make_job(
            uuid=f"j{i:03d}",
            city=county,
            province=county,
            location=f"{county} {county}",
            requires_post_secondary=(i % 3 == 0),
            embedding=vec,
            embedding_whitened=True,
        )
        jobs.append(job)
        docs.append(
            {
                "job": job,
                "job_embedding": vec,
                "is_active": True,
                "llm_reranker_meta": {
                    "embedding": {"whitening": {"artifact_sha256": SHA}}
                },
                "location_keys": [county.casefold()],
                "requires_post_secondary": i % 3 == 0,
            }
        )
    return jobs, docs


_USERS = [
    make_user(
        user_id="kilifi-no-ps",
        city="Kilifi",
        province="Kilifi",
        any_post_secondary_educ=0,
    ),
    make_user(user_id="kitui", city="Kitui", province="Kitui"),
    make_user(
        user_id="nairobi", city="Nairobi", province="Nairobi", any_post_secondary_educ=1
    ),
]


def _user_vectors(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([_unit(rng) for _ in _USERS]).astype(np.float64)


def _keys(
    users: list[UserProfile], hub_chains: HubChains | None
) -> list[UserLocationKeys]:
    return [
        user_location_keys(u, hub_chains, w_regional=0.7, w_national=0.5) for u in users
    ]


# --------------------------------------------------------------------------------------------------
# pre-filters
# --------------------------------------------------------------------------------------------------


class TestJobLocationTiers:
    def test_tiers_follow_the_hub_chain_local_first_with_remote(self):
        (keys,) = _keys([make_user(city="Kilifi", province="Kilifi")], HUBS)

        assert job_location_tiers(keys) == [
            LocationTier(keys=("kilifi", "remote")),
            LocationTier(keys=("mombasa",)),
            LocationTier(keys=("nairobi",)),
        ]

    def test_a_hub_that_does_not_pull_outward_is_one_tier(self):
        (keys,) = _keys([make_user(city="Nairobi", province="Nairobi")], HUBS)

        assert job_location_tiers(keys) == [LocationTier(keys=("nairobi", "remote"))]

    def test_hubs_with_the_same_weight_share_a_search(self):
        keys = UserLocationKeys(
            tiers=(("remote", 1.0), ("kilifi", 1.0), ("mombasa", 0.5), ("nairobi", 0.5))
        )

        assert job_location_tiers(keys) == [
            LocationTier(keys=("kilifi", "remote")),
            LocationTier(keys=("mombasa", "nairobi")),
        ]

    def test_without_tiers_one_search_over_city_province_and_remote(self):
        (keys,) = _keys([make_user(city="Thika", province="Kiambu")], None)

        assert job_location_tiers(keys) == [
            LocationTier(keys=("kiambu", "remote", "thika"))
        ]

    def test_unlocated_user_gets_remote_jobs_only(self):
        user = make_user(city="", province="")

        for hub_chains in (HUBS, None):
            (keys,) = _keys([user], hub_chains)
            assert job_location_tiers(keys) == [LocationTier(keys=("remote",))]

    def test_no_location_restriction_without_keys(self):
        assert job_location_tiers(None) == [LocationTier(keys=None)]


class TestPrefilters:
    def test_job_prefilter_always_pins_active_and_the_whitening_artifact(self):
        flt = job_prefilter(
            make_user(any_post_secondary_educ=1),
            LocationTier(keys=None),
            artifact_sha256=SHA,
        )

        assert flt == {
            "$and": [
                {"is_active": {"$eq": True}},
                {"llm_reranker_meta.embedding.whitening.artifact_sha256": {"$eq": SHA}},
            ]
        }

    def test_job_prefilter_adds_location_keys_and_the_education_gate(self):
        flt = job_prefilter(
            make_user(any_post_secondary_educ=0),
            LocationTier(keys=("kilifi", "remote")),
            artifact_sha256=SHA,
        )

        assert {"location_keys": {"$in": ["kilifi", "remote"]}} in flt["$and"]
        assert {"requires_post_secondary": {"$ne": True}} in flt["$and"]

    @pytest.mark.parametrize("educ", [1, None])
    def test_education_gate_only_for_users_who_reported_none(self, educ):
        flt = job_prefilter(
            make_user(any_post_secondary_educ=educ),
            LocationTier(keys=None),
            artifact_sha256=SHA,
        )

        assert not any("requires_post_secondary" in c for c in flt["$and"])

    def test_documents_whitened_with_another_artifact_never_match(self):
        doc = {
            "is_active": True,
            "llm_reranker_meta": {
                "embedding": {"whitening": {"artifact_sha256": "sha-old"}}
            },
        }
        flt = job_prefilter(make_user(), LocationTier(keys=None), artifact_sha256=SHA)

        assert not matches(doc, flt)


# --------------------------------------------------------------------------------------------------
# jobs: parity with the in-process retriever
# --------------------------------------------------------------------------------------------------


def _memory_job_shortlists(
    jobs: list[Job], *, k: int, hub_chains: HubChains | None
) -> list[list[tuple[str, float]]]:
    retriever = Stage1Retriever(embedding_dim=DIM)
    corpus = retriever.prepare(jobs)
    shortlists = retriever.shortlist(
        _USERS,
        corpus,
        _user_vectors(),
        matcher=_matcher(),
        whitener=_WHITENER,
        retrieve_top_k=k,
        location_keys=_keys(_USERS, hub_chains) if hub_chains is not None else None,
    )
    return [[(c.uuid, c.concat_cosine_similarity) for c in s] for s in shortlists]


def _db_job_shortlists(
    docs: list[dict[str, Any]], *, k: int, hub_chains: HubChains | None
) -> list[list[tuple[str, float]]]:
    result = asyncio.run(
        _retriever(FakeJobsSearch(docs)).shortlist_jobs(
            _USERS,
            _user_vectors(),
            artifact_sha256=SHA,
            retrieve_top_k=k,
            location_keys=_keys(_USERS, hub_chains),
        )
    )
    return [
        [(h.item.uuid, round(h.concat_cosine_similarity, 6)) for h in s]
        for s in result.shortlists
    ]


@pytest.mark.parametrize("k", [3, 10, 40])
def test_tiered_job_shortlist_matches_the_in_process_retriever(k):
    """Per-tier searches merged by cosine x tier give the in-process top K exactly."""
    jobs, docs = _job_corpus()

    assert _db_job_shortlists(docs, k=k, hub_chains=HUBS) == _memory_job_shortlists(
        jobs, k=k, hub_chains=HUBS
    )


def test_untiered_job_shortlist_matches_the_in_process_retriever_per_user_pool():
    jobs, docs = _job_corpus()
    memory: list[list[tuple[str, float]]] = []
    for i, user in enumerate(_USERS):
        # Without tiers, the in-process pool is what the location prefilter loaded for the user.
        pool = [j for j in jobs if j.city in (user.city, "Remote")]
        retriever = Stage1Retriever(embedding_dim=DIM)
        memory.append(
            [
                (c.uuid, c.concat_cosine_similarity)
                for c in retriever.shortlist(
                    [user],
                    retriever.prepare(pool),
                    _user_vectors()[i : i + 1],
                    matcher=_matcher(),
                    whitener=_WHITENER,
                    retrieve_top_k=10,
                )[0]
            ]
        )

    assert _db_job_shortlists(docs, k=10, hub_chains=None) == memory


def test_job_hits_keep_the_unweighted_cosine_and_rank_by_tier():
    jobs, docs = _job_corpus()
    result = asyncio.run(
        _retriever(FakeJobsSearch(docs)).shortlist_jobs(
            _USERS[:1],
            _user_vectors()[:1],
            artifact_sha256=SHA,
            retrieve_top_k=20,
            location_keys=_keys(_USERS[:1], HUBS),
        )
    )
    weight = {"Kilifi": 1.0, "Remote": 1.0, "Mombasa": 0.7, "Nairobi": 0.5}
    hits = result.shortlists[0]
    weighted = [h.concat_cosine_similarity * weight[h.item.city] for h in hits]

    assert weighted == sorted(weighted, reverse=True)
    assert {h.item.city for h in hits} <= set(weight), "Kitui is off the Kilifi chain"
    q = _user_vectors()[0]
    by_uuid = {j.uuid: j for j in jobs}
    for h in hits:
        assert h.concat_cosine_similarity == pytest.approx(
            float(q @ by_uuid[h.item.uuid].embedding)
        )
    assert result.stats.n_searches == 3


def test_education_gate_is_rechecked_and_counted_when_the_prefilter_field_is_stale():
    jobs, docs = _job_corpus()
    for d in docs:
        d["requires_post_secondary"] = False  # stale: the job itself still requires it
    result = asyncio.run(
        _retriever(FakeJobsSearch(docs)).shortlist_jobs(
            _USERS[:1],
            _user_vectors()[:1],
            artifact_sha256=SHA,
            retrieve_top_k=50,
            location_keys=_keys(_USERS[:1], HUBS),
        )
    )

    assert not any(h.item.requires_post_secondary for h in result.shortlists[0])
    assert result.stats.n_education_dropped > 0


def test_searches_carry_the_whitened_vector_limit_and_candidates():
    _, docs = _job_corpus()
    fake = FakeJobsSearch(docs)
    asyncio.run(
        _retriever(fake).shortlist_jobs(
            _USERS[1:2],
            _user_vectors()[1:2],
            artifact_sha256=SHA,
            retrieve_top_k=12,
            location_keys=_keys(_USERS[1:2], None),
        )
    )

    (query,) = fake.queries
    assert query.limit == 12 and query.num_candidates == 180 and not query.exact
    np.testing.assert_allclose(query.vector, _user_vectors()[1])
