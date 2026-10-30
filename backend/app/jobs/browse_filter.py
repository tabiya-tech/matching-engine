"""Mongo filter for the /jobs browse endpoint."""

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app.jobs.job_document import (
    CITY_FIELD,
    COUNTY_FIELD,
    DEFAULT_EMPLOYMENT_TYPE,
    EMPLOYMENT_TYPE_FIELD,
    LOCATION_KEYS_FIELD,
    PROVINCE_FIELD,
    RANKED_JOBS_ACTIVE_FILTER,
)
from app.location.keys import location_keys_for


def _employment_type_pattern(value: str) -> str:
    """Whole-string regex matching ``value`` regardless of case or ``_``/``-``/space separators.

    Employment type is stored inconsistently (``Full_time``, ``Full-time``, ``full time``); an
    exact match would otherwise require every caller to know every spelling variant on file.
    """
    parts = [re.escape(part) for part in re.split(r"[_\-\s]+", value.strip()) if part]
    return "^" + r"[_\-\s]+".join(parts) + "$"


def build_jobs_browse_filter(
    *,
    search: str | None = None,
    category: list[str] | None = None,
    employment_type: list[str] | None = None,
    location: list[str] | None = None,
    skills: str | None = None,
    days: int | None = None,
) -> dict[str, Any]:
    """Mongo filter for the /jobs browse endpoint, composed with ``is_active``.

    Every clause is optional; all supplied clauses are AND-ed together (a job must match
    all of them); a multi-valued clause (``category``, ``employment_type``, ``location``)
    matches a job against any of its supplied values (OR). Field paths target the raw
    ``classifier_metadata`` / ``llm_classified_skills`` document so the filter is applied by
    Mongo before shaping. ``category`` spans several candidate field names because the stored
    data is not uniform. ``location`` matches via ``location_keys`` ``$in`` (same mechanism as
    ``/match``), OR-ed with a raw-field regex so documents a backfill hasn't reached yet (no
    stored ``location_keys``) still match.
    """
    clauses: list[dict[str, Any]] = [RANKED_JOBS_ACTIVE_FILTER]

    if search and search.strip():
        clauses.append(
            {
                "classifier_metadata.title": {
                    "$regex": re.escape(search.strip()),
                    "$options": "i",
                }
            }
        )
    if category:
        rx = [
            {"$regex": re.escape(value.strip()), "$options": "i"}
            for value in category
            if value and value.strip()
        ]
        if rx:
            clauses.append(
                {
                    "$or": [
                        *[{"classifier_metadata.category": r} for r in rx],
                        *[{"classifier_metadata.sector": r} for r in rx],
                        *[{"classifier_metadata.isco_occupation_group": r} for r in rx],
                    ]
                }
            )
    if employment_type:
        values = [value.strip() for value in employment_type if value and value.strip()]
        rx = [
            {"$regex": _employment_type_pattern(value), "$options": "i"}
            for value in values
        ]
        if rx:
            or_clauses: list[dict[str, Any]] = [{EMPLOYMENT_TYPE_FIELD: r} for r in rx]
            # The jobs-board display falls back to DEFAULT_EMPLOYMENT_TYPE when a job has no
            # employment_type of its own (see build_job_from_ranked); mirror that here so
            # selecting that displayed value also surfaces the documents it was derived from.
            if any(
                re.match(
                    _employment_type_pattern(value),
                    DEFAULT_EMPLOYMENT_TYPE,
                    re.IGNORECASE,
                )
                for value in values
            ):
                or_clauses.append({EMPLOYMENT_TYPE_FIELD: {"$in": [None, ""]}})
                or_clauses.append({EMPLOYMENT_TYPE_FIELD: {"$exists": False}})
            clauses.append({"$or": or_clauses})
    if location:
        keys = location_keys_for(*location)
        if keys:
            rx = [
                {"$regex": re.escape(value.strip()), "$options": "i"}
                for value in location
                if value and value.strip()
            ]
            clauses.append(
                {
                    "$or": [
                        {LOCATION_KEYS_FIELD: {"$in": list(keys)}},
                        *[{CITY_FIELD: r} for r in rx],
                        *[{COUNTY_FIELD: r} for r in rx],
                        *[{PROVINCE_FIELD: r} for r in rx],
                    ]
                }
            )
    if skills and skills.strip():
        rx = {"$regex": re.escape(skills.strip()), "$options": "i"}
        clauses.append(
            {
                "$or": [
                    {"llm_classified_skills.essential.label": rx},
                    {"llm_classified_skills.optional.label": rx},
                ]
            }
        )
    if days is not None:
        cutoff = (
            (datetime.now(timezone.utc) - timedelta(days=int(days))).date().isoformat()
        )
        gte = {"$gte": cutoff}
        clauses.append(
            {
                "$or": [
                    {"classifier_metadata.posted_date": gte},
                    {"classifier_metadata.date_posted": gte},
                ]
            }
        )

    if len(clauses) == 1:
        return dict(RANKED_JOBS_ACTIVE_FILTER)
    return {"$and": clauses}
