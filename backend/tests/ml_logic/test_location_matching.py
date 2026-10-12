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

    def test_empty_user_location_does_not_match_non_remote(self, local_job):
        user = Place(city="", province="")
        assert job_matches_user_location(Place.of_item(local_job), user) is False
