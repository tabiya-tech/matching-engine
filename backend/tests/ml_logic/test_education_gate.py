"""Post-secondary education gate invariants."""

from app.ranking.retrieval import (
    filter_jobs_by_education,
    is_education_eligible,
    job_requires_post_secondary,
    user_lacks_post_secondary,
)
from tests.domain_factories import make_job, make_user


class TestEducationGate:
    def test_job_requires_post_secondary_only_when_true(self):
        assert (
            job_requires_post_secondary(make_job(requires_post_secondary=True)) is True
        )
        assert (
            job_requires_post_secondary(make_job(requires_post_secondary=False))
            is False
        )
        assert job_requires_post_secondary(make_job()) is False

    def test_user_lacks_post_secondary_only_when_explicit_zero(self):
        assert user_lacks_post_secondary(make_user(any_post_secondary_educ=0)) is True
        assert user_lacks_post_secondary(make_user(any_post_secondary_educ=1)) is False
        assert user_lacks_post_secondary(make_user()) is False
        assert (
            user_lacks_post_secondary(make_user(any_post_secondary_educ=None)) is False
        )

    def test_filter_drops_ps_jobs_for_user_without_postsec(self):
        user = make_user(any_post_secondary_educ=0)
        jobs = [
            make_job(uuid="j1", requires_post_secondary=True),
            make_job(uuid="j2", requires_post_secondary=False),
            make_job(uuid="j3"),
        ]
        out = filter_jobs_by_education(user, jobs)
        assert [j.uuid for j in out] == ["j2", "j3"]

    def test_filter_keeps_all_when_user_has_postsec(self):
        user = make_user(any_post_secondary_educ=1)
        jobs = [make_job(uuid="j1", requires_post_secondary=True), make_job(uuid="j2")]
        assert len(filter_jobs_by_education(user, jobs)) == 2

    def test_is_education_eligible_fail_open_when_user_omits_field(self):
        assert (
            is_education_eligible(make_user(), make_job(requires_post_secondary=True))
            is True
        )
