"""Mongo location filter (``build_location_filter``) when user location is partial or missing.

Regression (CORE-847): a user with an empty city (Compass never sends one) used to get a
remote-only filter, so almost no jobs were retrieved.
"""

from __future__ import annotations

from app.jobs.job_document import LOCATION_KEYS_FIELD, RANKED_JOBS_ACTIVE_FILTER
from app.location.keys import user_location_keys
from app.services.location_query import build_location_filter
from tests.domain_factories import make_user


def _filter(*users):
    # Without hub chains: users are matched to their own city/province keys only.
    return build_location_filter([user_location_keys(u) for u in users])


def test_no_location_user_gets_active_only_filter():
    assert (
        _filter(make_user(user_id="u", city="", province=""))
        == RANKED_JOBS_ACTIVE_FILTER
    )


def test_province_only_user_is_filtered_on_province():
    filt = _filter(make_user(user_id="u", city="", province="Lusaka"))
    assert filt[LOCATION_KEYS_FIELD] == {"$in": ["lusaka", "remote"]}


def test_city_only_user_is_filtered_on_city():
    filt = _filter(make_user(user_id="u", city="Ndola", province=""))
    assert filt[LOCATION_KEYS_FIELD] == {"$in": ["ndola", "remote"]}


def test_batch_with_a_no_location_user_is_not_narrowed():
    users = (
        make_user(user_id="a", city="Lusaka", province="Lusaka"),
        make_user(user_id="b", city="", province=""),
    )
    assert _filter(*users) == RANKED_JOBS_ACTIVE_FILTER
