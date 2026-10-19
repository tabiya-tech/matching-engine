"""Location / remote matching invariants (location keys, without hub chains)."""

from app.location.keys import user_location_keys
from tests.domain_factories import make_job, make_user


def _matches(job, user) -> bool:
    return user_location_keys(user).tier_for(job.location_keys) > 0.0


class TestLocationMatching:
    def test_remote_job_matches_any_user(self, base_user, remote_job):
        assert _matches(remote_job, base_user) is True

    def test_local_city_matches(self, base_user, local_job):
        assert _matches(local_job, base_user) is True

    def test_foreign_city_does_not_match(self, base_user, foreign_job):
        assert _matches(foreign_job, base_user) is False

    def test_remote_in_province_matches(self, base_user):
        job = make_job(city="", province="National/Remote")
        assert _matches(job, base_user) is True

    def test_one_of_several_listed_regions_matches(self, base_user):
        job = make_job(city="Ndola, Lusaka", province="Copperbelt, Lusaka")
        assert _matches(job, base_user) is True

    def test_region_contained_in_another_name_does_not_match(self):
        # "western" is part of "north western" but a different province.
        user = make_user(user_id="u", city="Mongu", province="Western")
        job = make_job(city="Solwezi", province="North Western")
        assert _matches(job, user) is False

    def test_empty_user_location_does_not_match_non_remote(self, local_job):
        user = make_user(user_id="u", city="", province="")
        assert _matches(local_job, user) is False
