"""Mongo location prefilter (``build_mongo_filter_active_and_location``) when user location is partial.

Regression: a user with an empty city (Compass never sends one) used to get a remote-only filter,
so almost no jobs were retrieved.
"""

from __future__ import annotations

import pytest

from app import database
from app.database import (
    RANKED_JOBS_ACTIVE_FILTER,
    _location_or_clauses_for_one_user,
    build_mongo_filter_active_and_location,
)


@pytest.fixture(autouse=True)
def _no_location_tiers(monkeypatch):
    # Keep needles to the user's own {city, province}; hub-chain widening is tested elsewhere.
    monkeypatch.setattr(database, "LOCATION_TIER_ENABLED", False)


def _regex_needles(clauses):
    return {
        cond["$regex"]
        for c in clauses
        for cond in c.values()
        if isinstance(cond, dict) and "$regex" in cond
    }


@pytest.mark.parametrize("user", [{"city": "", "province": ""}, {"city": None, "province": None}, {}])
def test_no_location_yields_no_clauses(user):
    assert _location_or_clauses_for_one_user(user) == []


def test_no_location_user_gets_active_only_filter():
    assert build_mongo_filter_active_and_location([{"city": "", "province": ""}]) == RANKED_JOBS_ACTIVE_FILTER


def test_province_only_user_is_filtered_on_province():
    clauses = _location_or_clauses_for_one_user({"city": "", "province": "Lusaka"})
    assert _regex_needles(clauses) == {"remote", "lusaka"}


def test_city_only_user_is_filtered_on_city():
    clauses = _location_or_clauses_for_one_user({"city": "Ndola", "province": None})
    assert _regex_needles(clauses) == {"remote", "ndola"}


def test_province_only_user_filter_is_not_remote_only():
    filt = build_mongo_filter_active_and_location([{"city": "", "province": "Lusaka"}])
    assert filt != RANKED_JOBS_ACTIVE_FILTER
    assert _regex_needles(filt["$and"][1]["$or"]) == {"remote", "lusaka"}


def test_batch_with_a_no_location_user_is_not_narrowed():
    users = [{"city": "Lusaka", "province": "Lusaka"}, {"city": "", "province": ""}]
    assert build_mongo_filter_active_and_location(users) == RANKED_JOBS_ACTIVE_FILTER
