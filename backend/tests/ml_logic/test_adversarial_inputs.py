"""Graceful handling of empty / invalid matching inputs."""

import pytest

from app.services.matching_service import match_user_with_data
from tests.domain_factories import make_job, make_skills, make_user


class TestAdversarialInputs:
    def test_missing_user_id_raises(self):
        with pytest.raises(ValueError, match="user_id"):
            match_user_with_data(make_user(city="Lusaka"), [], [])

    def test_empty_jobs_and_occupations_returns_empty_lists(self, base_user):
        out = match_user_with_data(base_user, [], [])
        assert out.user_id == base_user.user_id
        assert out.opportunity_recommendations == []
        assert out.occupation_recommendations == []
        assert out.skill_gap_recommendations == []

    def test_user_with_no_skills_does_not_crash(self):
        user = make_user(user_id="no-skills", city="Lusaka", province="Lusaka")
        jobs = [
            make_job(
                uuid="j1",
                city="Remote",
                essential_skills=make_skills(("s1", "manage staff")),
            )
        ]
        out = match_user_with_data(user, jobs, [])
        assert out.user_id == "no-skills"
        assert isinstance(out.skill_gap_recommendations, list)
