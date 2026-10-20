"""Builders for the typed domain values tests pass across layers (jobs, occupations, users, timings)."""

from typing import Any

from app.domain.attributes import JobAttributes
from app.domain.job import Job
from app.domain.occupation import Occupation
from app.domain.skills import SkillReference
from app.domain.user import UserProfile
from app.domain.work_activities import WorkActivity
from app.jobs.repository import JobsLoadTiming, JobsPageTiming
from app.occupations.loader import OccupationsLoadTiming
from app.ranking.types import Candidate, MatchItem
from app.schemas import MatchRequest


def make_user(**request_fields: Any) -> UserProfile:
    """A user profile built from ``MatchRequest`` fields, so request defaults apply."""
    return UserProfile.from_request(MatchRequest(**request_fields))


def make_skills(*pairs: tuple[str, str]) -> list[SkillReference]:
    """Skill references from ``(id, label)`` pairs."""
    return [SkillReference(id=i, label=label) for i, label in pairs]


def make_work_activities(
    *rows: tuple[str, float, float] | tuple[str, float, float, str],
) -> list[WorkActivity]:
    """Work activities from ``(code, importance, level[, label])`` rows."""
    return [
        WorkActivity(
            code=r[0],
            importance=float(r[1]),
            level=float(r[2]),
            label=r[3] if len(r) > 3 else "",
        )
        for r in rows
    ]


def make_attributes(**levels: str) -> JobAttributes:
    """Job attributes from ``name=level_id`` pairs."""
    return JobAttributes(**levels)


def make_job(**overrides: Any) -> Job:
    """A minimal active job; any field can be overridden."""
    fields: dict[str, Any] = {
        "uuid": "job-1",
        "originUuid": "job-1",
        "opportunity_title": "Technical Manager",
        "location": "Nairobi",
        "city": "Nairobi",
        "province": "",
        "contract_type": "full_time",
        "closing_date": "",
        "opportunity_description": "",
        "essential_skills": [],
        "optional_skills": [],
        "skills": [],
        "skill_groups_origin_uuids": [],
        "attributes": JobAttributes(),
        "onet_work_activities": [],
    }
    fields.update(overrides)
    return Job(**fields)


def make_occupation(**overrides: Any) -> Occupation:
    """A minimal occupation row; any field can be overridden."""
    fields: dict[str, Any] = {
        "uuid": "occ-1",
        "originUuid": "occ-1",
        "occupation_label": "Nurse",
        "description": "",
        "county": "Nairobi",
        "essential_skills": [],
        "optional_skills": [],
        "attributes": JobAttributes(),
        "onet_work_activities": [],
        "included_tasks": "",
    }
    fields.update(overrides)
    return Occupation(**fields)


def jobs_load_timing(**overrides: Any) -> JobsLoadTiming:
    """A ``find_jobs_with_timing`` timing with zero durations."""
    fields: dict[str, Any] = {
        "mongo_ranked_find_ms": 0.0,
        "python_build_jobs_ms": 0.0,
        "n_ranked_raw": 0,
        "n_jobs": 0,
        "n_skipped_inactive": 0,
        "n_skipped_invalid": 0,
        "get_all_jobs_total_ms": 0.0,
        "jobs_find_use_projection": True,
    }
    fields.update(overrides)
    return JobsLoadTiming(**fields)


def jobs_page_timing(**overrides: Any) -> JobsPageTiming:
    """A ``get_jobs_page_with_timing`` timing with zero durations."""
    fields: dict[str, Any] = {
        "mongo_find_ms": 0.0,
        "python_build_jobs_ms": 0.0,
        "n_page_raw": 0,
        "n_jobs": 0,
        "n_skipped_inactive": 0,
        "n_skipped_invalid": 0,
        "has_more": False,
        "limit": 20,
        "total": None,
        "get_jobs_page_total_ms": 0.0,
    }
    fields.update(overrides)
    return JobsPageTiming(**fields)


def occupations_load_timing(**overrides: Any) -> OccupationsLoadTiming:
    """A cache-hit occupation load timing."""
    fields: dict[str, Any] = {
        "occupation_cache_hit": True,
        "occupation_source_read_ms": 0.0,
        "occupation_flatten_ms": 0.0,
        "n_occupation_rows": 0,
        "get_all_occupations_total_ms": 0.0,
    }
    fields.update(overrides)
    return OccupationsLoadTiming(**fields)


def make_candidate(
    item: MatchItem, *, rank: int = 1, cosine: float = 0.5, **overrides: Any
) -> Candidate:
    """A shortlisted candidate of ``item`` with no per-skill detail."""
    fields: dict[str, Any] = {
        "item": item,
        "rank": rank,
        "concat_cosine_similarity": cosine,
        "per_job_skill": [],
    }
    fields.update(overrides)
    return Candidate(**fields)
