"""Deterministic v3-stage ranking with mocked embeddings (no Gemini)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from app.services.cross_encoder.gemini_embeddings import EMBEDDING_DIM
from app.services.match_concat_gemini_ce_service import run_match_concat_gemini_ce


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
    @patch("app.services.match_concat_gemini_ce_service._get_matcher")
    @patch("app.services.match_concat_gemini_ce_service._get_reranker")
    def test_identical_user_job_vector_ranks_first(self, mock_reranker, mock_matcher):
        """User embedding == Job A vector → Job A must be rank 1 in stage-1 cosine."""
        mock_matcher.return_value.score_pair.return_value = {
            "mean_best_cosine": 0.5,
            "per_job_skill": [],
        }
        # Pass-through rerank: preserve cosine order
        mock_reranker.return_value = MagicMock()
        with patch(
            "app.services.match_concat_gemini_ce_service.rerank_cosine_recommendations",
            side_effect=lambda _labels, recs, **_kw: recs,
        ):
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
            out = run_match_concat_gemini_ce(
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

    @patch("app.services.match_concat_gemini_ce_service._get_matcher")
    @patch("app.services.match_concat_gemini_ce_service._get_reranker")
    def test_education_gate_applied_in_stage1(self, mock_reranker, mock_matcher):
        mock_matcher.return_value.score_pair.return_value = {
            "mean_best_cosine": 0.5,
            "per_job_skill": [],
        }
        with patch(
            "app.services.match_concat_gemini_ce_service.rerank_cosine_recommendations",
            side_effect=lambda _labels, recs, **_kw: recs,
        ):
            jobs = [
                {
                    **_job("job-ps", 0),
                    "requires_post_secondary": True,
                    "llm_job_attributes": {
                        "attributes": {"requires_post_secondary": True}
                    },
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
            out = run_match_concat_gemini_ce(
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
        out = run_match_concat_gemini_ce([user], jobs, retrieve_top_k=5, final_top_k=5)
        assert out[0]["concat_gemini_ce_recommendations"] == []
        assert out[0]["n_jobs_scored"] == 0
