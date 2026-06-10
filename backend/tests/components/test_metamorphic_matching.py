"""Metamorphic tests — relationships that must hold after algo changes (4b).

No golden rankings: only score/order invariants under input transforms.
"""

from __future__ import annotations

import copy

import pytest

from tests.components.conftest import _job_with_essential, _user_with_skills

# Labels known to resolve in the project taxonomy (skills.csv).
_SKILL_A = "manage staff"
_SKILL_B = "supervise correctional procedures"


@pytest.fixture(scope="module")
def _base_pair(cosine_matcher):
    """User with one skill vs job requiring that skill — baseline score."""
    user = _user_with_skills([_SKILL_A])
    job = _job_with_essential([_SKILL_A, _SKILL_B])
    score = cosine_matcher.score_pair(user, job)["mean_best_cosine"]
    if score <= 0:
        pytest.skip(f"Taxonomy miss: {_SKILL_A!r} did not embed")
    return user, job, score


class TestMetamorphicMatching:
    def test_reordering_user_skills_does_not_change_score(self, cosine_matcher):
        user_ab = _user_with_skills([_SKILL_A, _SKILL_B])
        user_ba = _user_with_skills([_SKILL_B, _SKILL_A])
        job = _job_with_essential([_SKILL_A, _SKILL_B])
        s_ab = cosine_matcher.score_pair(user_ab, job)["mean_best_cosine"]
        s_ba = cosine_matcher.score_pair(user_ba, job)["mean_best_cosine"]
        if s_ab <= 0:
            pytest.skip("Skills did not resolve in taxonomy")
        assert s_ba == pytest.approx(s_ab, abs=1e-6)

    def test_duplicating_user_skill_does_not_change_score(
        self, cosine_matcher, _base_pair
    ):
        user, job, baseline = _base_pair
        dup = copy.deepcopy(user)
        dup["skills_vector"]["top_skills"] = dup["skills_vector"]["top_skills"] * 2
        score = cosine_matcher.score_pair(dup, job)["mean_best_cosine"]
        assert score == pytest.approx(baseline, abs=1e-6)

    def test_adding_matching_skill_does_not_lower_score(self, cosine_matcher):
        """Superset of user skills vs same job — score must not drop."""
        minimal = _user_with_skills([_SKILL_A])
        extended = _user_with_skills([_SKILL_A, _SKILL_B])
        job = _job_with_essential([_SKILL_A, _SKILL_B])
        s_min = cosine_matcher.score_pair(minimal, job)["mean_best_cosine"]
        s_ext = cosine_matcher.score_pair(extended, job)["mean_best_cosine"]
        if s_min <= 0:
            pytest.skip("Baseline skill did not resolve in taxonomy")
        assert s_ext >= s_min - 1e-6

    def test_identical_user_and_job_skill_yields_high_similarity(self, cosine_matcher):
        user = _user_with_skills([_SKILL_A])
        job = _job_with_essential([_SKILL_A])
        score = cosine_matcher.score_pair(user, job)["mean_best_cosine"]
        if score <= 0:
            pytest.skip(f"Taxonomy miss: {_SKILL_A!r}")
        assert score >= 0.5

    def test_unrelated_skills_score_lower_than_overlap(self, cosine_matcher):
        user = _user_with_skills([_SKILL_A])
        job_match = _job_with_essential([_SKILL_A])
        job_unrelated = _job_with_essential([_SKILL_B])
        s_match = cosine_matcher.score_pair(user, job_match)["mean_best_cosine"]
        s_unrelated = cosine_matcher.score_pair(user, job_unrelated)["mean_best_cosine"]
        if s_match <= 0:
            pytest.skip("Overlap baseline did not resolve")
        assert s_match >= s_unrelated
