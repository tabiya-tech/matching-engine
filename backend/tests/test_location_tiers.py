"""Unit tests for tiered urban-pull location matching (services/location_tiers.py).

`tier_factor_for_job` matches regions with the real (pure) `job_matches_user_location`: lenient
casefold-substring + always-remote.
"""

import pytest

from app.ranking import retrieval as location_tiers
from app.ranking.retrieval import HubChains, Place


@pytest.fixture
def fake_matching_service():
    # Tiers are matched with the real (pure) job_matches_user_location.
    return location_tiers.job_matches_user_location


# national=nairobi; coastal counties fall back to mombasa; nairobi & mombasa are self-only hubs.
HC = HubChains(
    "nairobi", {"mombasa": ["kilifi", "kwale", "lamu"]}, ["nairobi", "mombasa"]
)


def test_hub_chain_for_tiers():
    assert HC.chain_for("nairobi") == ["nairobi"]  # national hub: no outward pull
    assert HC.chain_for("mombasa") == ["mombasa"]  # regional hub: self only
    assert HC.chain_for("kilifi") == [
        "kilifi",
        "mombasa",
        "nairobi",
    ]  # coastal: local->reg->nat
    assert HC.chain_for("kitui") == [
        "kitui",
        "nairobi",
    ]  # inland non-hub: local->national (skips regional)
    assert HC.chain_for("Garissa") == [
        "garissa",
        "nairobi",
    ]  # unknown county -> default; casefolded
    assert HC.chain_for("") == ["nairobi"]  # empty -> national fallback


def _job(city):
    return Place.region(city)


def test_tier_factor_kilifi_user(fake_matching_service):
    def f(city):
        return HC.tier_factor_for_job(
            _job(city), "kilifi", w_regional=0.85, w_national=0.70
        )

    assert f("Kilifi") == 1.0  # local
    assert f("Mombasa") == 0.85  # regional hub
    assert f("Nairobi") == 0.70  # national hub
    assert f("Nakuru") == 0.0  # off-chain -> excluded
    assert (
        HC.tier_factor_for_job(
            Place(location="Remote"), "kilifi", w_regional=0.85, w_national=0.70
        )
        == 1.0
    )  # remote always allowed


def test_tier_factor_hub_users_dont_pull(fake_matching_service):
    # Nairobi user: only Nairobi at 1.0, everything else off-chain.
    assert (
        HC.tier_factor_for_job(
            _job("Nairobi"), "nairobi", w_regional=0.85, w_national=0.70
        )
        == 1.0
    )
    assert (
        HC.tier_factor_for_job(
            _job("Mombasa"), "nairobi", w_regional=0.85, w_national=0.70
        )
        == 0.0
    )
    # Mombasa user: only Mombasa; not even Nairobi.
    assert (
        HC.tier_factor_for_job(
            _job("Mombasa"), "mombasa", w_regional=0.85, w_national=0.70
        )
        == 1.0
    )
    assert (
        HC.tier_factor_for_job(
            _job("Nairobi"), "mombasa", w_regional=0.85, w_national=0.70
        )
        == 0.0
    )


def test_tier_factor_kitui_national_not_regional(fake_matching_service):
    # Kitui chain skips the regional tier: Nairobi must score at the NATIONAL weight, not regional.
    assert (
        HC.tier_factor_for_job(_job("Kitui"), "kitui", w_regional=0.85, w_national=0.70)
        == 1.0
    )
    assert (
        HC.tier_factor_for_job(
            _job("Nairobi"), "kitui", w_regional=0.85, w_national=0.70
        )
        == 0.70
    )
    assert (
        HC.tier_factor_for_job(
            _job("Mombasa"), "kitui", w_regional=0.85, w_national=0.70
        )
        == 0.0
    )
