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

    def test_empty_user_location_matches_any_job(self, local_job, foreign_job):
        # MatchRequest contract: omitting location relaxes the filter.
        for user in ({"city": "", "province": ""}, {"city": None, "province": None}, {}):
            assert _job_matches_user_location(local_job, user) is True
            assert _job_matches_user_location(foreign_job, user) is True

    def test_province_only_user_matches_on_province(self, local_job, foreign_job):
        user = {"city": "", "province": "Lusaka"}
        assert _job_matches_user_location(local_job, user) is True
        assert _job_matches_user_location(foreign_job, user) is False

    def test_city_only_user_matches_on_city(self, local_job, foreign_job):
        user = {"city": "Lusaka", "province": None}
        assert _job_matches_user_location(local_job, user) is True
        assert _job_matches_user_location(foreign_job, user) is False

    def test_province_only_user_does_not_match_via_empty_city_in_location(self):
        # "" is a substring of every string: an empty user city must not match job_loc.
        job = {"city": "", "province": "", "location": "Ndola, Copperbelt"}
        assert _job_matches_user_location(job, {"city": "", "province": "Lusaka"}) is False
        assert _job_matches_user_location(job, {"city": "", "province": "Copperbelt"}) is True

    def test_cross_field_matches(self):
        # A user value matches the job's city or province, whichever field the source used.
        assert _job_matches_user_location({"city": "Lusaka", "province": ""}, {"city": "", "province": "Lusaka"}) is True
        assert _job_matches_user_location({"city": "", "province": "Lusaka"}, {"city": "Lusaka", "province": ""}) is True
        assert _job_matches_user_location({"city": "Ndola", "province": ""}, {"city": "", "province": "Lusaka"}) is False
