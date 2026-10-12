"""Deterministic v3-stage ranking with mocked embeddings (no Gemini)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from app.artifacts.repository import get_artifacts_repository
from app.clients.gemini_embedding_client import EMBEDDING_DIM, MODEL_NAME
from app.matching.service import MatchingService
from app.ranking.retrieval import Stage1Retriever
from app.ranking.types import SkillPairScore
from app.server_dependencies.model_dependencies import get_concat_whitener
from tests.domain_factories import make_job, make_user


def _service(matcher=None) -> MatchingService:
    """Real stage-1 retrieval; mocked matcher, Gemini and cross-encoder; pass-through rerank."""
    rerank = MagicMock()
    rerank.build_pairs.return_value = []
    rerank.apply.side_effect = lambda recs, _scores, **_kw: list(recs)
    return MatchingService(
        jobs_repository=MagicMock(),
        artifacts_repository=get_artifacts_repository(),
        embedding_client=MagicMock(model_name=MODEL_NAME, embedding_dim=EMBEDDING_DIM),
        cross_encoder_provider=lambda: MagicMock(model_name="test-cross-encoder"),
        retrieval_matcher_provider=lambda: matcher or MagicMock(),
        gate_matcher_provider=MagicMock,
        whitener_provider=get_concat_whitener,
        skill_scorer_provider=MagicMock,
        preference_scorer_provider=MagicMock,
        retriever=Stage1Retriever(embedding_dim=EMBEDDING_DIM),
        rerank=rerank,
    )


def _unit_vector(index: int) -> np.ndarray:
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v[index] = 1.0
    return v


def _job(uuid: str, vec_index: int | None, **overrides):
    fields = {
        "uuid": uuid,
        "opportunity_title": f"Job {uuid}",
        "employer": "TestCo",
        "location": "Remote",
        "embedding": _unit_vector(vec_index) if vec_index is not None else None,
    }
    fields.update(overrides)
    return make_job(**fields)


def _matcher() -> MagicMock:
    matcher = MagicMock()
    matcher.score_pair.return_value = SkillPairScore(
        mean_best_cosine=0.5,
        min_best_cosine=0.5,
        n_user_skills_embedded=1,
        n_job_skills_embedded=1,
        per_job_skill=[],
    )
    return matcher


class TestMatchConcatMockedEmbeddings:
    def test_identical_user_job_vector_ranks_first(self):
        """User embedding == Job A vector → Job A must be rank 1 in stage-1 cosine."""
        # Pass-through rerank: preserve cosine order
        service = _service(_matcher())
        jobs = [_job("job-a", 0), _job("job-b", 1)]
        user = make_user(
            user_id="u1",
            skills_vector={
                "top_skills": [
                    {
                        "originUUID": "00000000-0000-4000-8000-000000000001",
                        "preferredLabel": "manage staff",
                    }
                ]
            },
        )
        u_vecs = np.stack([_unit_vector(0).astype(np.float64)], axis=0)
        out = service.shortlist_and_rerank(
            [user],
            jobs,
            retrieve_top_k=5,
            final_top_k=5,
            user_unit_vectors=u_vecs,
        )
        candidates = out[0].candidates
        assert len(candidates) >= 1
        assert candidates[0].uuid == "job-a"
        assert candidates[0].rank == 1

    def test_education_gate_applied_in_stage1(self):
        service = _service(_matcher())
        jobs = [_job("job-ps", 0, requires_post_secondary=True), _job("job-ok", 1)]
        user = make_user(user_id="u1", any_post_secondary_educ=0)
        # User aligned with job-ps vector — would rank first without gate
        u_vecs = np.stack([_unit_vector(0).astype(np.float64)], axis=0)
        out = service.shortlist_and_rerank(
            [user],
            jobs,
            retrieve_top_k=5,
            final_top_k=5,
            user_unit_vectors=u_vecs,
        )
        uuids = [c.uuid for c in out[0].candidates]
        assert "job-ps" not in uuids
        assert "job-ok" in uuids

    def test_no_embeddings_returns_empty_recommendations(self):
        user = make_user(user_id="u1")
        jobs = [_job("j1", None, opportunity_title="No embed")]
        out = _service().shortlist_and_rerank(
            [user], jobs, retrieve_top_k=5, final_top_k=5
        )
        assert out[0].candidates == []
        assert out[0].n_jobs_scored == 0
