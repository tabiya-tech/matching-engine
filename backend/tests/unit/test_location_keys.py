"""Tests for the location-key rule (app.location.keys): canonical keys and users' eligible keys."""

import json
from pathlib import Path

import pytest

from app.location.keys import (
    REMOTE_KEY,
    HubChains,
    canonical_location,
    location_keys_for,
    user_home_keys,
    user_location_keys,
)
from tests.domain_factories import make_user

_VECTORS = (
    Path(__file__).resolve().parents[1] / "fixtures" / "location_key_vectors.json"
)

# national=nairobi; coastal counties fall back to mombasa; nairobi & mombasa are self-only hubs.
_HC = HubChains("Nairobi", {"Mombasa": ["Kilifi", "Kwale"]}, ["nairobi", "mombasa"])


@pytest.mark.parametrize(
    "case",
    json.loads(_VECTORS.read_text(encoding="utf-8"))["cases"],
    ids=lambda c: " | ".join(str(v) for v in c["values"]),
)
def test_shared_key_vectors(case):
    # These cases are shared with the llm-reranker, which writes the stored keys.
    assert list(location_keys_for(*case["values"])) == case["keys"]


class TestCanonicalLocation:
    def test_trims_collapses_and_casefolds(self):
        assert canonical_location("  Taita   Taveta. ") == "taita taveta"

    def test_blank_is_empty(self):
        assert canonical_location(None) == ""
        assert canonical_location(" () ") == ""


class TestUserLocationKeys:
    def test_missing_city_or_province_is_remote_only(self):
        for user in (
            make_user(user_id="u", city="", province=""),
            make_user(user_id="u", city="Nairobi", province=""),
            make_user(user_id="u", city="", province="Nairobi"),
        ):
            assert user_home_keys(user) == ()
            assert user_location_keys(user, _HC).tiers == ((REMOTE_KEY, 1.0),)
            assert user_location_keys(user).tiers == ((REMOTE_KEY, 1.0),)

    def test_without_hub_chains_city_and_province_at_full_tier(self):
        user = make_user(user_id="u", city="Thika", province="Kiambu")

        keys = user_location_keys(user)

        assert keys.keys == {"remote", "thika", "kiambu"}
        assert keys.tier_for(("kiambu",)) == 1.0
        assert keys.tier_for(("nairobi",)) == 0.0

    def test_local_user_gets_local_regional_and_national_tiers(self):
        user = make_user(user_id="u", city="Kilifi", province="Kilifi")

        keys = user_location_keys(user, _HC, w_regional=0.7, w_national=0.5)

        assert keys.tiers == (
            ("remote", 1.0),
            ("kilifi", 1.0),
            ("mombasa", 0.7),
            ("nairobi", 0.5),
        )

    def test_county_without_regional_hub_scores_national_at_national_weight(self):
        user = make_user(user_id="u", city="Kitui", province="Kitui")

        keys = user_location_keys(user, _HC, w_regional=0.7, w_national=0.5)

        assert keys.tier_for(("nairobi",)) == 0.5
        assert keys.tier_for(("mombasa",)) == 0.0

    def test_hub_self_only_users_do_not_pull_outward(self):
        nairobi = make_user(user_id="u", city="Nairobi", province="Nairobi")
        mombasa = make_user(user_id="u", city="Mombasa", province="Mombasa")

        assert user_location_keys(nairobi, _HC).keys == {"remote", "nairobi"}
        assert user_location_keys(mombasa, _HC).keys == {"remote", "mombasa"}

    def test_chain_is_keyed_on_the_province(self):
        user = make_user(user_id="u", city="Malindi", province="Kilifi")

        assert user_location_keys(user, _HC).keys == {
            "remote",
            "kilifi",
            "mombasa",
            "nairobi",
        }

    def test_remote_items_are_eligible_at_full_tier(self):
        user = make_user(user_id="u", city="Kitui", province="Kitui")

        keys = user_location_keys(user, _HC, w_regional=0.7, w_national=0.5)

        assert keys.tier_for(("nairobi", "remote")) == 1.0

    def test_item_tier_is_the_users_best_matching_key(self):
        user = make_user(user_id="u", city="Kilifi", province="Kilifi")

        keys = user_location_keys(user, _HC, w_regional=0.7, w_national=0.5)

        assert keys.tier_for(("nairobi", "kilifi")) == 1.0
        assert keys.tier_for(("nairobi", "mombasa")) == 0.7
