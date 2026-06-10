"""Location / remote matching invariants."""

from app.services.matching_service import _job_matches_user_location


class TestLocationMatching:
    def test_remote_job_matches_any_user(self, base_user, remote_job):
        assert _job_matches_user_location(remote_job, base_user) is True

    def test_local_city_matches(self, base_user, local_job):
        assert _job_matches_user_location(local_job, base_user) is True

    def test_foreign_city_does_not_match(self, base_user, foreign_job):
        assert _job_matches_user_location(foreign_job, base_user) is False

    def test_remote_in_location_field_matches(self, base_user):
        job = {"city": "", "province": "", "location": "Fully Remote role"}
        assert _job_matches_user_location(job, base_user) is True

    def test_empty_user_location_does_not_match_non_remote(self, local_job):
        user = {"city": "", "province": ""}
        assert _job_matches_user_location(local_job, user) is False
