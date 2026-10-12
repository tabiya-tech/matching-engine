"""Location / remote matching invariants."""

from app.ranking.retrieval import Place, job_matches_user_location


def _matches(job, user) -> bool:
    return job_matches_user_location(Place.of_item(job), Place.of_user(user))


class TestLocationMatching:
    def test_remote_job_matches_any_user(self, base_user, remote_job):
        assert _matches(remote_job, base_user) is True

    def test_local_city_matches(self, base_user, local_job):
        assert _matches(local_job, base_user) is True

    def test_foreign_city_does_not_match(self, base_user, foreign_job):
        assert _matches(foreign_job, base_user) is False

    def test_remote_in_location_field_matches(self, base_user):
        job = Place(city="", province="", location="Fully Remote role")
        assert job_matches_user_location(job, Place.of_user(base_user)) is True

    def test_empty_user_location_matches_any_job(self, local_job, foreign_job):
        # MatchRequest contract: omitting location relaxes the filter.
        for user in (Place(city="", province=""), Place()):
            assert job_matches_user_location(Place.of_item(local_job), user) is True
            assert job_matches_user_location(Place.of_item(foreign_job), user) is True

    def test_province_only_user_matches_on_province(self, local_job, foreign_job):
        user = Place(city="", province="Lusaka")
        assert job_matches_user_location(Place.of_item(local_job), user) is True
        assert job_matches_user_location(Place.of_item(foreign_job), user) is False

    def test_city_only_user_matches_on_city(self, local_job, foreign_job):
        user = Place(city="Lusaka")
        assert job_matches_user_location(Place.of_item(local_job), user) is True
        assert job_matches_user_location(Place.of_item(foreign_job), user) is False

    def test_province_only_user_does_not_match_via_empty_city_in_location(self):
        # "" is a substring of every string: an empty user city must not match job_loc.
        job = Place(city="", province="", location="Ndola, Copperbelt")
        assert job_matches_user_location(job, Place(city="", province="Lusaka")) is False
        assert job_matches_user_location(job, Place(city="", province="Copperbelt")) is True

    def test_cross_field_matches(self):
        # A user value matches the job's city or province, whichever field the source used.
        assert job_matches_user_location(Place(city="Lusaka"), Place(province="Lusaka")) is True
        assert job_matches_user_location(Place(province="Lusaka"), Place(city="Lusaka")) is True
        assert job_matches_user_location(Place(city="Ndola"), Place(province="Lusaka")) is False
