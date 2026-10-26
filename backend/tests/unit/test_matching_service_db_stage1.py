"""``MatchingService.match`` with job stage 1 in the database (``STAGE1_RETRIEVAL_BACKEND=db``).

The database retriever is a fake returning fixed job hits; the cross-encoder is a pass-through.
Checks that the request never loads the job pool, that the whitened user vectors, the artifact's
SHA-256 and the users' location keys reach the search, that Phase 2, skill gaps and formatting run on
the hits, and that occupations keep the in-process stage 1.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.artifacts.repository import get_artifacts_repository
from app.artifacts.types import ConcatWhitening
from app.jobs.repository import IJobsRepository
from app.location.keys import HubChains, UserLocationKeys
from app.matching import service as svc
from app.matching.service import MatchingService
from app.matching.types import MatchOptions
from app.ranking.preference import UnifiedPreferenceScorer
from app.ranking.retrieval import ConcatWhitener, Stage1Retriever
from app.ranking.types import SkillPairScore
from app.ranking.vector_retrieval import (
    IDbStage1Retriever,
    Stage1Hit,
    Stage1SearchResult,
    Stage1SearchStats,
)
from app.schemas import MatchRequest
from app.services import location_query
from tests.domain_factories import make_job, make_occupation, make_skills
from tests.unit.test_matching_service_rank import _gate_matcher

DIM = 4
SHA = "sha-deployed"
_SKILLS = make_skills(("E1", "welding"), ("E2", "blueprint reading"))
_NEAR = make_job(
    uuid="near", city="Nairobi", province="Nairobi", essential_skills=_SKILLS
)
_FAR = make_job(
    uuid="far", city="Nairobi", province="Nairobi", essential_skills=_SKILLS
)
_OCC_ROWS = [
    make_occupation(
        uuid=f"occ_{c}",
        originUuid="occ",
        county=c,
        essential_skills=_SKILLS,
        embedding=np.array([1.0, 0.0, 0.0, 0.0]),
        embedding_whitened=True,
    )
    for c in ("Nairobi", "Kitui")
]


class FakeDbRetriever(IDbStage1Retriever):
    def __init__(self) -> None:
        self.job_calls: list[dict] = []

    async def shortlist_jobs(self, users, user_white_vectors, **kwargs):
        self.job_calls.append({"vectors": user_white_vectors, **kwargs})
        return Stage1SearchResult(
            shortlists=[
                [
                    Stage1Hit(item=_NEAR, concat_cosine_similarity=0.4),
                    Stage1Hit(item=_FAR, concat_cosine_similarity=0.1),
                ]
                for _ in users
            ],
            stats=Stage1SearchStats(corpus="jobs", n_searches=len(users)),
        )


def _retrieval_matcher() -> MagicMock:
    matcher = MagicMock()
    matcher.score_pair.return_value = SkillPairScore(
        mean_best_cosine=0.5,
        min_best_cosine=0.5,
        n_user_skills_embedded=1,
        n_job_skills_embedded=2,
        per_job_skill=[],
    )
    return matcher


def _rerank() -> MagicMock:
    rerank = MagicMock()
    rerank.build_pairs.return_value = []
    rerank.apply.side_effect = lambda recs, _scores, **_kw: list(recs)
    return rerank


def _service(retriever: IDbStage1Retriever, jobs_repository: IJobsRepository):
    return MatchingService(
        jobs_repository=jobs_repository,
        artifacts_repository=get_artifacts_repository(),
        embedding_client=MagicMock(model_name="m", embedding_dim=DIM),
        cross_encoder_provider=lambda: MagicMock(model_name="ce"),
        retrieval_matcher_provider=_retrieval_matcher,
        gate_matcher_provider=_gate_matcher,
        whitener_provider=lambda: ConcatWhitener(
            ConcatWhitening(mu=np.zeros(DIM), W=np.diag([2.0, 1, 1, 1]), target=0.5)
        ),
        skill_scorer_provider=MagicMock,
        preference_scorer_provider=lambda: UnifiedPreferenceScorer(
            get_artifacts_repository().load_attribute_schema()
        ),
        retriever=Stage1Retriever(embedding_dim=DIM),
        rerank=_rerank(),
        db_retriever=retriever,
    )


_USER_VECTOR = np.array([[0.6, 0.8, 0.0, 0.0]])


def _match(
    retriever,
    *,
    sha: str | None = SHA,
    gaps: list | None = None,
    hub_chains: HubChains | None = None,
    retrieval_filter: bool = True,
    disable_location: bool = False,
):
    jobs_repository = AsyncMock(spec=IJobsRepository)
    service = _service(retriever, jobs_repository)
    if hub_chains is not None:
        service._artifacts_repository = MagicMock()
        service._artifacts_repository.load_hub_chains.return_value = hub_chains

    def skill_gaps(user, items, *_a, **_kw):
        if gaps is not None:
            gaps.append([i.uuid for i in items])
        return []

    with (
        patch.object(svc, "MATCH_V4_DISABLE_OCCUPATIONS", False),
        patch.object(svc, "MATCH_V4_DISABLE_LOCATION", disable_location),
        patch.object(location_query, "MATCH_V4_DISABLE_LOCATION", disable_location),
        patch.object(svc, "JOBS_RETRIEVAL_FILTER", retrieval_filter),
        patch.object(location_query, "LOCATION_TIER_ENABLED", hub_chains is not None),
        patch.object(svc, "skill_gaps_for", side_effect=skill_gaps),
        patch(
            "app.server_dependencies.model_dependencies.get_concat_whitening_sha256",
            return_value=sha,
        ),
        patch.object(service, "embed_users", return_value=_USER_VECTOR),
        patch.object(
            service, "_load_occupations", AsyncMock(return_value=(_OCC_ROWS, None))
        ),
    ):
        out = asyncio.run(
            service.match(
                [
                    MatchRequest.model_validate(
                        {
                            "user_id": "u1",
                            "city": "Nairobi",
                            "province": "Nairobi",
                            "preference_vector": {"earnings_per_month": 0.9},
                        }
                    )
                ],
                MatchOptions(retrieve_top_k=10, final_top_k=10),
            )
        )
    return out, jobs_repository


def test_db_stage1_never_loads_the_job_pool():
    retriever = FakeDbRetriever()

    (response,), jobs_repository = _match(retriever)

    jobs_repository.find_jobs_with_timing.assert_not_awaited()
    assert [o.uuid for o in response.opportunity_recommendations] == ["near", "far"]
    assert [o.uuid for o in response.occupation_recommendations] == ["occ_Nairobi"]


def test_search_gets_the_whitened_user_vector_and_the_artifact_sha():
    retriever = FakeDbRetriever()

    _match(retriever)

    (call,) = retriever.job_calls
    whitened = np.array([1.2, 0.8, 0.0, 0.0]) / np.linalg.norm([1.2, 0.8])
    np.testing.assert_allclose(call["vectors"][0], whitened)
    assert call["artifact_sha256"] == SHA
    assert call["retrieve_top_k"] == 10
    assert call["location_keys"] == [
        UserLocationKeys(tiers=(("remote", 1.0), ("nairobi", 1.0)))
    ]


def test_disable_location_passes_no_location_keys_to_the_db_search():
    """MATCH_V4_DISABLE_LOCATION=True: the db stage-1 search gets no location restriction at
    all, and the response carries no location_tier_factor."""
    retriever = FakeDbRetriever()

    (response,), jobs_repository = _match(
        retriever, hub_chains=HubChains("nairobi", None, None), disable_location=True
    )

    jobs_repository.find_jobs_with_timing.assert_not_awaited()
    (call,) = retriever.job_calls
    assert call["location_keys"] is None
    for rec in response.opportunity_recommendations:
        assert rec.score_breakdown.location_tier_factor is None
    assert [o.uuid for o in response.occupation_recommendations] == ["occ_Nairobi"]


def test_phase2_p_hat_is_the_search_cosine_rescaled():
    (response,), _ = _match(FakeDbRetriever())
    near, far = (o.score_breakdown for o in response.opportunity_recommendations)

    # whitened cosine / target 0.5
    assert near.p_hat == pytest.approx(0.8)
    assert far.p_hat == pytest.approx(0.2)
    assert near.p_hat_source == "concat_cosine_whitened"


def test_skill_gaps_come_from_the_users_job_shortlist():
    gaps: list[list[str]] = []

    _match(FakeDbRetriever(), gaps=gaps)

    assert gaps == [["near", "far"]]


def test_without_the_whitening_artifact_the_request_fails_loudly():
    with pytest.raises(RuntimeError, match="whitening artifact"):
        _match(FakeDbRetriever(), sha=None)


def test_tiers_are_passed_to_the_job_search_when_enabled():
    retriever = FakeDbRetriever()

    _match(retriever, hub_chains=HubChains("Mombasa", {}, []))

    (call,) = retriever.job_calls
    assert call["location_keys"] == [
        UserLocationKeys(
            tiers=(
                ("remote", 1.0),
                ("nairobi", 1.0),
                ("mombasa", location_query.LOCATION_TIER_W_NATIONAL),
            )
        )
    ]


def test_no_location_step_when_both_the_filter_and_tiers_are_off():
    retriever = FakeDbRetriever()

    _match(retriever, retrieval_filter=False)

    (call,) = retriever.job_calls
    assert call["location_keys"] is None
