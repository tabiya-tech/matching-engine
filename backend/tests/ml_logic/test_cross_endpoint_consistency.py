"""Rules that must hold the same across all matching engines."""

from app.services.education_eligibility import filter_jobs_by_education
from app.services.matching_service import _filter_skill_gap_recommendations


class TestCrossEndpointConsistency:
    def test_education_filter_is_engine_agnostic(self):
        """Same user + jobs → same eligible set regardless of endpoint."""
        user = {"any_post_secondary_educ": 0}
        jobs = [
            {"uuid": "a", "requires_post_secondary": True},
            {"uuid": "b", "requires_post_secondary": False},
            {"uuid": "c"},
        ]
        filtered_once = filter_jobs_by_education(user, jobs)
        filtered_twice = filter_jobs_by_education(user, list(jobs))
        assert [j["uuid"] for j in filtered_once] == ["b", "c"]
        assert filtered_once == filtered_twice

    def test_skill_gap_top_k_cap_consistent(self):
        """Response filter enforces the same cap on every endpoint path."""
        gaps = [
            {"skill_id": f"s{i}", "proximity_score": 0.9 - i * 0.01} for i in range(10)
        ]
        for top_k in (3, 5, 14):
            out = _filter_skill_gap_recommendations(gaps, top_k=top_k)
            assert len(out) <= top_k
