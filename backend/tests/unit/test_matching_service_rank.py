"""``MatchingService.rank`` end to end on typed values: slate -> Phase-2 -> preferences -> response.

Retrieval and the cross-encoder are stubbed (``shortlist_and_rerank`` returns a fixed slate); the
Phase-2 overrides, preference scoring, location tiers and formatting are the real ones.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.artifacts.repository import get_artifacts_repository
from app.artifacts.types import ConcatWhitening
from app.matching import service as svc
from app.matching.service import MatchingService
from app.ranking.preference import UnifiedPreferenceScorer
from app.ranking.retrieval import ConcatWhitener
from app.ranking.types import RetrievalSummary, SkillMatchDetail, UserShortlist
from app.schemas import MatchResponse
from tests.domain_factories import (
    make_attributes,
    make_candidate,
    make_job,
    make_occupation,
    make_skills,
    make_user,
)

DIM = 4


def _vec(index: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[index] = 1.0
    return v


def _gate_matcher() -> MagicMock:
    """Every item: essential E1 met exactly, E2 not met."""
    matcher = MagicMock()
    matcher.score_pair_v4.return_value = [
        SkillMatchDetail(
            job_skill_id="E1",
            job_skill_label="welding",
            best_user_skill_id="E1",
            best_user_skill_label="welding",
            cosine_similarity=1.0,
            cosine_similarity_raw=1.0,
            exact=True,
        ),
        SkillMatchDetail(
            job_skill_id="E2",
            job_skill_label="blueprint reading",
            best_user_skill_id=None,
            best_user_skill_label=None,
            cosine_similarity=0.0,
            cosine_similarity_raw=0.0,
        ),
    ]
    matcher.essential_ids.return_value = frozenset({"E1", "E2"})
    return matcher


def _summary() -> RetrievalSummary:
    return RetrievalSummary(
        stage1="s1",
        stage2="s2",
        gemini_user_embed_model="m",
        embedding_dim=DIM,
        n_jobs_with_stage1_embedding=2,
        n_jobs_active_loaded=2,
    )


def _service(whitening: ConcatWhitening | None) -> MatchingService:
    return MatchingService(
        jobs_repository=MagicMock(),
        artifacts_repository=get_artifacts_repository(),
        embedding_client=MagicMock(model_name="m", embedding_dim=DIM),
        cross_encoder_provider=MagicMock,
        retrieval_matcher_provider=MagicMock,
        gate_matcher_provider=_gate_matcher,
        whitener_provider=lambda: ConcatWhitener(whitening),
        skill_scorer_provider=MagicMock,
        preference_scorer_provider=lambda: UnifiedPreferenceScorer(
            get_artifacts_repository().load_attribute_schema()
        ),
        retriever=MagicMock(),
        rerank=MagicMock(),
    )


_SKILLS = make_skills(("E1", "welding"), ("E2", "blueprint reading"))
_JOBS = [
    make_job(
        uuid="near",
        city="Nairobi",
        province="Nairobi",
        location="Nairobi Nairobi",
        essential_skills=_SKILLS,
        attributes=make_attributes(earnings_per_month="earn_70k"),
        embedding=_vec(0),
    ),
    make_job(
        uuid="far",
        city="Nairobi",
        province="Nairobi",
        location="Nairobi Nairobi",
        essential_skills=_SKILLS,
        embedding=_vec(1),
    ),
]
_OCCUPATIONS = [
    make_occupation(
        uuid="occ_Nairobi",
        originUuid="occ",
        county="Nairobi",
        essential_skills=_SKILLS,
        attributes=make_attributes(expected_demand="High Expected Demand"),
        embedding=_vec(0),
    ),
    make_occupation(
        uuid="occ_Kitui",
        originUuid="occ",
        county="Kitui",
        essential_skills=_SKILLS,
        embedding=_vec(0),
    ),
]


def _slates(users, items, **_kwargs):
    """``shortlist_and_rerank`` stub: every item, in input order, cosine 0.9, 0.8, ..."""
    return [
        UserShortlist(
            user_id=u.user_id or "",
            n_jobs_scored=len(items),
            n_jobs_active_loaded=len(items),
            candidates=[
                make_candidate(item, rank=i + 1, cosine=0.9 - 0.1 * i)
                for i, item in enumerate(items)
            ],
            config_summary=_summary(),
        )
        for u in users
    ]


def _rank(whitening: ConcatWhitening | None) -> list[MatchResponse]:
    service = _service(whitening)
    user = make_user(
        user_id="u1",
        city="Nairobi",
        province="Nairobi",
        preference_vector={"earnings_per_month": 0.9},
    )
    with (
        patch.object(svc, "MATCH_V4_DISABLE_OCCUPATIONS", False),
        patch.object(svc, "location_hub_chains", return_value=None),
        patch.object(svc, "skill_gaps_for", return_value=[]),
        patch.object(
            service, "embed_users", return_value=_vec(0)[None, :].astype(np.float64)
        ),
        patch.object(service, "shortlist_and_rerank", side_effect=_slates),
    ):
        return service.rank(
            [user], _JOBS, _OCCUPATIONS, retrieve_top_k=10, final_top_k=10
        )


@pytest.mark.parametrize(
    "whitening",
    [
        None,
        ConcatWhitening(mu=np.zeros(DIM), W=np.eye(DIM), target=0.5),
    ],
    ids=["phase1", "phase2"],
)
def test_rank_returns_typed_responses(whitening):
    (response,) = _rank(whitening)

    assert response.user_id == "u1"
    assert [o.uuid for o in response.opportunity_recommendations] == ["near", "far"]
    assert [o.rank for o in response.opportunity_recommendations] == [1, 2]
    top = response.opportunity_recommendations[0]
    assert top.score_breakdown.essential_coverage == pytest.approx(0.5)
    assert top.matched_skills.essential_skill_matches[0].match_tier == "exact"
    assert top.matched_preferences, "preference contributions reach the response"
    # Occupations: the user's own county row only, deduplicated by code.
    assert [o.uuid for o in response.occupation_recommendations] == ["occ_Nairobi"]
    assert response.occupation_recommendations[0].score_breakdown.demand_score == 0.75


def test_phase2_uses_whitened_p_hat_and_coverage_demotion():
    (response,) = _rank(ConcatWhitening(mu=np.zeros(DIM), W=np.eye(DIM), target=0.5))
    near = response.opportunity_recommendations[0].score_breakdown

    # user == near's vector: whitened cosine 1.0 / target 0.5 -> clamped to 1.0
    assert near.p_hat == pytest.approx(1.0)
    assert near.p_hat_source == "concat_cosine_whitened"
    assert near.coverage_factor is not None and near.coverage_factor < 1.0


def test_phase1_keeps_stage1_cosine_without_demotion():
    (response,) = _rank(None)
    near = response.opportunity_recommendations[0].score_breakdown

    assert near.p_hat == pytest.approx(0.9)
    assert near.p_hat_source == "concat_cosine_similarity"
    assert near.coverage_factor is None
