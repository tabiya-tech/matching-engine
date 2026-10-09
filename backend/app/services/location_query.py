"""Mongo prefilter that narrows the /match job pool to jobs near the requesting users.

Pure query building (no I/O): the result is handed to ``IJobsRepository.find_jobs_with_timing``.
It is a superset of ``matching_service._job_matches_user_location`` on ``classifier_metadata``
fields; the in-process location scoring stays the source of truth for ranking.
"""

import re
from collections.abc import Sequence
from typing import Any

from app.config import LOCATION_HUB_CHAINS_PATH, LOCATION_TIER_ENABLED
from app.jobs.job_document import (
    CITY_FIELD,
    COUNTY_FIELD,
    PROVINCE_FIELD,
    RANKED_JOBS_ACTIVE_FILTER,
)


def _norm_loc_value(v: Any) -> str:
    """Casefold + strip, aligned with matching_service._norm for city/province."""
    if v is None:
        return ""
    s = str(v).strip()
    return s.casefold() if s else ""


def _remote_substring_ors() -> list[dict[str, Any]]:
    r = "remote"
    return [
        {CITY_FIELD: {"$regex": r, "$options": "i"}},
        {COUNTY_FIELD: {"$regex": r, "$options": "i"}},
    ]


def _field_contains_substr_regex(field: str, needle_cf: str) -> dict[str, Any] | None:
    if not needle_cf:
        return None
    return {field: {"$regex": re.escape(needle_cf), "$options": "i"}}


def _expr_haystack_contains_mongo_subfield(
    haystack_casefold: str, dollar_field: str
) -> dict[str, Any] | None:
    """True when haystack (user string) contains the job’s city/county (Python: job in user).

    Requires a non-empty job field: MongoDB matches an empty substring at index 0 for
    ``$indexOfCP``, which would incorrectly match every document if city/county were missing.
    """
    if not haystack_casefold:
        return None
    needle = {"$ifNull": [{"$toLower": dollar_field}, ""]}
    return {
        "$expr": {
            "$and": [
                {"$gt": [{"$strLenCP": needle}, 0]},
                {"$gte": [{"$indexOfCP": [haystack_casefold, needle]}, 0]},
            ]
        }
    }


def location_or_clauses_for_one_user(user: dict) -> list[dict[str, Any]]:
    """Superset of matching_service._job_matches_user_location, on classifier_metadata fields.

    By default the needle set is the user's {city, province}. When LOCATION_TIER_ENABLED (urban-pull
    Part A), it is widened to also include the user's hub-chain regions (regional + national hub) so
    hub jobs become candidates in the pool; the per-user soft tier re-rank (match_v4_full_service)
    then keeps local jobs preferred. Strictly additive — remote clauses always come first.
    """
    uc = _norm_loc_value(user.get("city"))
    up = _norm_loc_value(user.get("province"))
    ors: list[dict[str, Any]] = list(_remote_substring_ors())
    if not uc or not up:
        return ors
    needles = {uc, up}
    if LOCATION_TIER_ENABLED:
        from app.services.location_tiers import load_hub_chains

        hc = load_hub_chains(LOCATION_HUB_CHAINS_PATH)
        if hc is not None:
            needles.update(hc.chain_for(up))  # local + regional + national hub regions
    for needle in sorted(n for n in needles if n):
        for field in (CITY_FIELD, COUNTY_FIELD, PROVINCE_FIELD):
            f = _field_contains_substr_regex(field, needle)
            if f is not None:
                ors.append(f)
        for fpath in (
            "$classifier_metadata.city",
            "$classifier_metadata.county",
            "$classifier_metadata.province",
        ):
            ex = _expr_haystack_contains_mongo_subfield(needle, fpath)
            if ex is not None:
                ors.append(ex)
    return ors


def build_mongo_filter_active_and_location(
    users: Sequence[dict],
) -> dict[str, Any] | None:
    """
    is_active and (OR of all per-user location clauses). None if the caller should
    use active-only (no user context or empty list).
    """
    if not users:
        return None
    parts: list[dict[str, Any]] = []
    for u in users:
        parts.extend(location_or_clauses_for_one_user(u))
    if not parts:
        return RANKED_JOBS_ACTIVE_FILTER
    return {"$and": [RANKED_JOBS_ACTIVE_FILTER, {"$or": parts}]}
