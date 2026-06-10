"""CosineSkillMatcher component tests."""

from tests.components.conftest import _job_with_essential, _user_with_skills

_SKILL = "manage staff"


class TestSkillScorer:
    def test_score_pair_returns_expected_keys(self, cosine_matcher):
        user = _user_with_skills([_SKILL])
        job = _job_with_essential([_SKILL])
        out = cosine_matcher.score_pair(user, job)
        assert "mean_best_cosine" in out
        assert "per_job_skill" in out
        assert 0.0 <= out["mean_best_cosine"] <= 1.0

    def test_empty_user_skills_yield_zero_score(self, cosine_matcher):
        user = {"skills_vector": {"top_skills": []}}
        job = _job_with_essential([_SKILL])
        out = cosine_matcher.score_pair(user, job)
        assert out["mean_best_cosine"] == 0.0
        assert out["per_job_skill"] == []

    def test_empty_job_skills_yield_zero_score(self, cosine_matcher):
        user = _user_with_skills([_SKILL])
        job = {"essential_skills": [], "optional_skills": []}
        out = cosine_matcher.score_pair(user, job)
        assert out["mean_best_cosine"] == 0.0
