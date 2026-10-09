"""Deterministic v3-stage ranking with mocked embeddings (no Gemini)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from app.artifacts.repository import get_artifacts_repository
from app.clients.gemini_embedding_client import EMBEDDING_DIM, MODEL_NAME
from app.matching.service import MatchingService
from app.ranking.retrieval import Stage1Retriever
from app.server_dependencies.model_dependencies import get_concat_whitener


def _service(matcher=None) -> MatchingService:
    """Real stage-1 retrieval; mocked matcher, Gemini and cross-encoder; pass-through rerank."""
    rerank = MagicMock()
    rerank.build_pairs.return_value = []
    rerank.apply.side_effect = lambda recs, _scores, **_kw: list(recs)
    return MatchingService(
        jobs_repository=MagicMock(),
        occupations_repository=MagicMock(),
        artifacts_repository=get_artifacts_repository(),
        embedding_client=MagicMock(model_name=MODEL_NAME, embedding_dim=EMBEDDING_DIM),
        cross_encoder_provider=MagicMock,
        retrieval_matcher_provider=lambda: matcher or MagicMock(),
        gate_matcher_provider=MagicMock,
        whitener_provider=get_concat_whitener,
        skill_scorer_provider=MagicMock,
        preference_scorer_provider=MagicMock,
        retriever=Stage1Retriever(embedding_dim=EMBEDDING_DIM),
        rerank=rerank,
    )


def _unit_vector(index: int) -> list[float]:
    v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v[index] = 1.0
    return v.tolist()


def _job(uuid: str, vec_index: int):
    return {
        "uuid": uuid,
        "opportunity_title": f"Job {uuid}",
        "employer": "TestCo",
        "location": "Remote",
        "essential_skills": [],
        "optional_skills": [],
        "job_embedding": _unit_vector(vec_index),
    }


class TestMatchConcatMockedEmbeddings:
    def test_identical_user_job_vector_ranks_first(self):
        """User embedding == Job A vector → Job A must be rank 1 in stage-1 cosine."""
        matcher = MagicMock()
        matcher.score_pair.return_value = {
            "mean_best_cosine": 0.5,
            "per_job_skill": [],
        }
        # Pass-through rerank: preserve cosine order
        service = _service(matcher)
        jobs = [_job("job-a", 0), _job("job-b", 1)]
        user = {
            "user_id": "u1",
            "skills_vector": {
                "top_skills": [
                    {
                        "originUUID": "00000000-0000-4000-8000-000000000001",
                        "preferredLabel": "manage staff",
                    }
                ]
            },
        }
        u_vecs = np.stack([np.asarray(_unit_vector(0), dtype=np.float64)], axis=0)
        out = service.shortlist_and_rerank(
            [user],
            jobs,
            retrieve_top_k=5,
            final_top_k=5,
            user_unit_vectors=u_vecs,
        )
        recs = out[0]["concat_gemini_ce_recommendations"]
        assert len(recs) >= 1
        assert recs[0]["job_uuid"] == "job-a"
        assert recs[0]["rank"] == 1

    def test_education_gate_applied_in_stage1(self):
        matcher = MagicMock()
        matcher.score_pair.return_value = {
            "mean_best_cosine": 0.5,
            "per_job_skill": [],
        }
        service = _service(matcher)
        jobs = [
            {
                **_job("job-ps", 0),
                "requires_post_secondary": True,
                "llm_job_attributes": {"attributes": {"requires_post_secondary": True}},
            },
            _job("job-ok", 1),
        ]
        user = {
            "user_id": "u1",
            "any_post_secondary_educ": 0,
            "skills_vector": {"top_skills": []},
        }
        # User aligned with job-ps vector — would rank first without gate
        u_vecs = np.stack([np.asarray(_unit_vector(0), dtype=np.float64)], axis=0)
        out = service.shortlist_and_rerank(
            [user],
            jobs,
            retrieve_top_k=5,
            final_top_k=5,
            user_unit_vectors=u_vecs,
        )
        uuids = [r["job_uuid"] for r in out[0]["concat_gemini_ce_recommendations"]]
        assert "job-ps" not in uuids
        assert "job-ok" in uuids

    def test_no_embeddings_returns_empty_recommendations(self):
        user = {"user_id": "u1", "skills_vector": {"top_skills": []}}
        jobs = [{"uuid": "j1", "opportunity_title": "No embed"}]
        out = _service().shortlist_and_rerank(
            [user], jobs, retrieve_top_k=5, final_top_k=5
        )
        assert out[0]["concat_gemini_ce_recommendations"] == []
        assert out[0]["n_jobs_scored"] == 0
