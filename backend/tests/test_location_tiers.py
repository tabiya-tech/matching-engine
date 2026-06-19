"""Unit tests for tiered urban-pull location matching (services/location_tiers.py).

Hermetic + torch-free: `tier_factor_for_job` lazily imports
`matching_service._job_matches_user_location`, so we inject a fake module that mirrors that
function's lenient casefold-substring + always-remote semantics.
"""

import pytest

from app.services import location_tiers
from app.services.location_tiers import HubChains


def _fake_matches(job, user):
    """Mirror of matching_service._job_matches_user_location (lenient substring + remote)."""
    jc = str(job.get("city") or "").casefold()
    jp = str(job.get("province") or "").casefold()
    jl = str(job.get("location") or "").casefold()
    uc = str(user.get("city") or "").casefold()
    up = str(user.get("province") or "").casefold()
    if "remote" in jc or "remote" in jp or "remote" in jl:
        return True
    if not uc or not up:
        return False
    if jc and (uc in jc or jc in uc):
        return True
    if jp and (up in jp or up in jp):
        return True
    if jl:
        return uc in jl or up in jl
    return False


@pytest.fixture
def fake_matching_service(monkeypatch):
    # Inject the fake straight into the cached matcher slot (avoids importing the torch-heavy
    # matching_service module). Reset afterwards so the cache doesn't leak across tests.
    monkeypatch.setattr(location_tiers, "_JMUL", _fake_matches)
    return _fake_matches


# national=nairobi; coastal counties fall back to mombasa; nairobi & mombasa are self-only hubs.
HC = HubChains("nairobi", {"mombasa": ["kilifi", "kwale", "lamu"]}, ["nairobi", "mombasa"])


def test_hub_chain_for_tiers():
    assert HC.chain_for("nairobi") == ["nairobi"]          # national hub: no outward pull
    assert HC.chain_for("mombasa") == ["mombasa"]          # regional hub: self only
    assert HC.chain_for("kilifi") == ["kilifi", "mombasa", "nairobi"]  # coastal: local->reg->nat
    assert HC.chain_for("kitui") == ["kitui", "nairobi"]   # inland non-hub: local->national (skips regional)
    assert HC.chain_for("Garissa") == ["garissa", "nairobi"]  # unknown county -> default; casefolded
    assert HC.chain_for("") == ["nairobi"]                 # empty -> national fallback


def _job(city):
    return {"city": city, "province": city, "location": city}


def test_tier_factor_kilifi_user(fake_matching_service):
    f = lambda city: HC.tier_factor_for_job(_job(city), "kilifi", w_regional=0.85, w_national=0.70)
    assert f("Kilifi") == 1.0          # local
    assert f("Mombasa") == 0.85        # regional hub
    assert f("Nairobi") == 0.70        # national hub
    assert f("Nakuru") == 0.0          # off-chain -> excluded
    assert HC.tier_factor_for_job({"city": "", "province": "", "location": "Remote"}, "kilifi",
                                  w_regional=0.85, w_national=0.70) == 1.0  # remote always allowed


def test_tier_factor_hub_users_dont_pull(fake_matching_service):
    # Nairobi user: only Nairobi at 1.0, everything else off-chain.
    assert HC.tier_factor_for_job(_job("Nairobi"), "nairobi", w_regional=0.85, w_national=0.70) == 1.0
    assert HC.tier_factor_for_job(_job("Mombasa"), "nairobi", w_regional=0.85, w_national=0.70) == 0.0
    # Mombasa user: only Mombasa; not even Nairobi.
    assert HC.tier_factor_for_job(_job("Mombasa"), "mombasa", w_regional=0.85, w_national=0.70) == 1.0
    assert HC.tier_factor_for_job(_job("Nairobi"), "mombasa", w_regional=0.85, w_national=0.70) == 0.0


def test_tier_factor_kitui_national_not_regional(fake_matching_service):
    # Kitui chain skips the regional tier: Nairobi must score at the NATIONAL weight, not regional.
    assert HC.tier_factor_for_job(_job("Kitui"), "kitui", w_regional=0.85, w_national=0.70) == 1.0
    assert HC.tier_factor_for_job(_job("Nairobi"), "kitui", w_regional=0.85, w_national=0.70) == 0.70
    assert HC.tier_factor_for_job(_job("Mombasa"), "kitui", w_regional=0.85, w_national=0.70) == 0.0
