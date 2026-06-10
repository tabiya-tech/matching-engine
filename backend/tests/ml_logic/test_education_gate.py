"""Post-secondary education gate invariants."""

from app.services.education_eligibility import (
    filter_jobs_by_education,
    is_education_eligible,
    job_requires_post_secondary,
    user_lacks_post_secondary,
)


class TestEducationGate:
    def test_job_requires_post_secondary_from_top_level(self):
        assert job_requires_post_secondary({"requires_post_secondary": True}) is True
        assert job_requires_post_secondary({"requires_post_secondary": 1}) is True
        assert job_requires_post_secondary({"requires_post_secondary": False}) is False

    def test_job_requires_post_secondary_from_attributes(self):
        job = {"attributes": {"requires_post_secondary": True}}
        assert job_requires_post_secondary(job) is True

    def test_user_lacks_post_secondary_only_when_explicit_zero(self):
        assert user_lacks_post_secondary({"any_post_secondary_educ": 0}) is True
        assert user_lacks_post_secondary({"any_post_secondary_educ": 1}) is False
        assert user_lacks_post_secondary({}) is False
        assert user_lacks_post_secondary({"any_post_secondary_educ": None}) is False

    def test_filter_drops_ps_jobs_for_user_without_postsec(self):
        user = {"any_post_secondary_educ": 0}
        jobs = [
            {"uuid": "j1", "requires_post_secondary": True},
            {"uuid": "j2", "requires_post_secondary": False},
            {"uuid": "j3"},
        ]
        out = filter_jobs_by_education(user, jobs)
        assert [j["uuid"] for j in out] == ["j2", "j3"]

    def test_filter_keeps_all_when_user_has_postsec(self):
        user = {"any_post_secondary_educ": 1}
        jobs = [
            {"uuid": "j1", "requires_post_secondary": True},
            {"uuid": "j2"},
        ]
        assert len(filter_jobs_by_education(user, jobs)) == 2

    def test_is_education_eligible_fail_open_when_user_omits_field(self):
        user = {}
        job = {"requires_post_secondary": True}
        assert is_education_eligible(user, job) is True
