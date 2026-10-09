"""Mongo filter for the /jobs browse endpoint."""

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app.jobs.job_document import (
    CITY_FIELD,
    COUNTY_FIELD,
    PROVINCE_FIELD,
    RANKED_JOBS_ACTIVE_FILTER,
)


def build_jobs_browse_filter(
    *,
    search: str | None = None,
    category: str | None = None,
    employment_type: str | None = None,
    location: str | None = None,
    skills: str | None = None,
    days: int | None = None,
) -> dict[str, Any]:
    """Mongo filter for the /jobs browse endpoint, composed with ``is_active``.

    Every clause is optional; all supplied clauses are AND-ed together (a job must match
    all of them). Field paths target the raw ``classifier_metadata`` / ``llm_classified_skills``
    document so the filter is applied by Mongo before shaping. ``category`` and ``location``
    span several candidate field names because the stored data is not uniform.
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
    if category and category.strip():
        rx = {"$regex": re.escape(category.strip()), "$options": "i"}
        clauses.append(
            {
                "$or": [
                    {"classifier_metadata.category": rx},
                    {"classifier_metadata.sector": rx},
                    {"classifier_metadata.isco_occupation_group": rx},
                ]
            }
        )
    if employment_type and employment_type.strip():
        clauses.append({"classifier_metadata.employment_type": employment_type.strip()})
    if location and location.strip():
        rx = {"$regex": re.escape(location.strip()), "$options": "i"}
        clauses.append(
            {"$or": [{CITY_FIELD: rx}, {COUNTY_FIELD: rx}, {PROVINCE_FIELD: rx}]}
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
