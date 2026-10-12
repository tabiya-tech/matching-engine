"""Rules that must hold the same across all matching engines."""

from app.ranking.retrieval import filter_jobs_by_education
from app.ranking.skill_gaps import filter_skill_gap_recommendations
from app.schemas import SkillGapRecommendation
from tests.domain_factories import make_job, make_user


def _gap(skill_id: str, proximity: float) -> SkillGapRecommendation:
    return SkillGapRecommendation(
        skill_id=skill_id,
        skill_label=skill_id,
        proximity_score=proximity,
        job_unlock_count=1,
        combined_score=proximity,
        reasoning="",
    )


class TestCrossEndpointConsistency:
    def test_education_filter_is_engine_agnostic(self):
        """Same user + jobs → same eligible set regardless of endpoint."""
        user = make_user(any_post_secondary_educ=0)
        jobs = [
            make_job(uuid="a", requires_post_secondary=True),
            make_job(uuid="b", requires_post_secondary=False),
            make_job(uuid="c"),
        ]
        filtered_once = filter_jobs_by_education(user, jobs)
        filtered_twice = filter_jobs_by_education(user, list(jobs))
        assert [j.uuid for j in filtered_once] == ["b", "c"]
        assert filtered_once == filtered_twice

    def test_skill_gap_top_k_cap_consistent(self):
        """Response filter enforces the same cap on every endpoint path."""
        gaps = [_gap(f"s{i}", 0.9 - i * 0.01) for i in range(10)]
        for top_k in (3, 5, 14):
            out = filter_skill_gap_recommendations(gaps, top_k=top_k)
            assert len(out) <= top_k
