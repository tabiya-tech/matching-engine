"""Typed input loading for the offline matching runners (BM25, cosine, hybrid, preference CLIs).

Every file or Mongo read is validated into the domain models here, so the runners never pass raw
dicts into the ranking layer:

* users: a JSON list or JSONL of ``MatchRequest`` objects -> ``UserProfile``
* jobs: a JSON list of either ranked-job documents (as stored in Mongo; mapped by
  ``build_job_from_ranked``) or flat job objects in the matching service's job shape
  (``JobFileRecord``) -> ``Job``
* Mongo: the same repository + location prefilter as ``POST /match`` -> ``Job``
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field

from app.domain.attributes import JobAttributes
from app.domain.base import SourceRecord
from app.domain.coercion import (
    OptionalFlag,
    OptionalLevel,
    OptionalText,
    OptionalVector,
    TextList,
)
from app.domain.job import Job
from app.domain.skills import SkillReference
from app.domain.user import UserProfile
from app.domain.work_activities import WorkActivityRecord, work_activities_from_records
from app.jobs.job_document import build_job_from_ranked
from app.jobs.repository import JobsLoadTiming
from app.schemas import MatchRequest

JobsSource = Literal["file", "mongo"]

# Keys only a stored ranked-job document has (the flat job shape never does).
_RANKED_DOCUMENT_KEYS = ("classifier_metadata", "llm_classified_skills", "job_id")


def backend_root() -> Path:
    """The ``backend/`` directory."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "backend":
            return parent
    raise RuntimeError(f"Could not locate 'backend/' from {here}")


def load_backend_dotenv() -> None:
    """Populate ``os.environ`` from ``backend/.env`` (no-op without python-dotenv)."""
    try:
        from dotenv import load_dotenv  # pylint: disable=import-outside-toplevel
    except ImportError:
        return
    p = backend_root() / ".env"
    if p.is_file():
        load_dotenv(p, override=False)


def _read_json_objects(path: Path) -> list[Mapping[str, object]]:
    """The objects of a JSON list file or a JSONL file."""
    if path.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, object]] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{ln}: invalid JSON ({e})") from e
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"{path}:{ln}: expected an object, got {type(obj).__name__}"
                    )
                rows.append(obj)
        return rows
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list, got {type(data).__name__}")
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(
                f"{path}[{i}]: expected an object, got {type(obj).__name__}"
            )
    return data


def user_from_record(record: Mapping[str, object]) -> UserProfile:
    """Validate one user record (``MatchRequest`` shape; extra keys ignored).

    :raises pydantic.ValidationError: If the record is not a valid ``MatchRequest``
    """
    return UserProfile.from_request(MatchRequest.model_validate(record))


def load_users(path: Path) -> list[UserProfile]:
    """Users from a JSON list or JSONL file of ``MatchRequest`` objects.

    :raises ValueError: If the file is not a list of objects
    :raises pydantic.ValidationError: If a record is not a valid ``MatchRequest``
    """
    return [user_from_record(r) for r in _read_json_objects(path)]


class JobFileSkill(SourceRecord):
    """A skill of a flat job record: ``{id, label}``."""

    id: OptionalText = Field(default=None, description="Taxonomy skill id.")
    label: OptionalText = Field(default=None, description="Skill label.")


class JobFileRecord(SourceRecord):
    """A job in the matching service's flat job shape (offline job files).

    Field names are the ``Job`` field names; the embedding may be stored as ``job_embedding`` /
    ``job_embedding_whitened`` and work activities under their upstream ``WA_*`` names.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    uuid: OptionalText = Field(default=None, description="Job id.")
    originUuid: OptionalText = Field(default=None, description="Stable source id.")
    job_fingerprint: OptionalText = Field(
        default=None, description="Content fingerprint."
    )
    opportunity_title: OptionalText = Field(default=None, description="Job title.")
    location: OptionalText = Field(default=None, description="Free-text location.")
    city: OptionalText = Field(default=None, description="City.")
    province: OptionalText = Field(default=None, description="Province/county.")
    employer: OptionalText = Field(default=None, description="Employer.")
    employment_type: OptionalText = Field(default=None, description="Employment type.")
    contract_type: OptionalText = Field(default=None, description="Contract type.")
    salary_text: OptionalText = Field(default=None, description="Salary text.")
    closing_date: OptionalText = Field(default=None, description="Closing date.")
    posted_date: OptionalText = Field(default=None, description="Posted date.")
    url: OptionalText = Field(default=None, description="Posting link.")
    opportunity_description: OptionalText = Field(
        default=None, description="Description."
    )
    opportunity_isco_occupation_group: OptionalText = Field(
        default=None, description="ISCO group label."
    )
    opportunity_isco_occupation_group_id: OptionalText = Field(
        default=None, description="ISCO group code."
    )
    related_occupation_id: OptionalText = Field(
        default=None, description="Related occupation."
    )
    category: OptionalText = Field(default=None, description="Sector/category.")
    source_platform: OptionalText = Field(default=None, description="Source platform.")
    essential_skills: list[JobFileSkill] = Field(
        default_factory=list, description="Essential skills."
    )
    optional_skills: list[JobFileSkill] = Field(
        default_factory=list, description="Optional skills."
    )
    skill_groups_origin_uuids: TextList = Field(
        default_factory=list, description="Skill-group origin UUIDs."
    )
    attributes: dict[str, str | bool | int | float | None] = Field(
        default_factory=dict, description="Attribute name -> level id."
    )
    onet_work_activities: list[WorkActivityRecord] = Field(
        default_factory=list, description="O*NET work activities."
    )
    requires_post_secondary: OptionalFlag = Field(
        default=None, description="Post-secondary requirement."
    )
    zqf_min: OptionalLevel = Field(default=None, description="Minimum ZQF level.")
    zqf_max: OptionalLevel = Field(default=None, description="Maximum ZQF level.")
    zqf_min_label: OptionalText = Field(
        default=None, description="Label of ``zqf_min``."
    )
    zqf_max_label: OptionalText = Field(
        default=None, description="Label of ``zqf_max``."
    )
    job_embedding: OptionalVector = Field(default=None, description="Concat embedding.")
    job_embedding_whitened: OptionalFlag = Field(
        default=None, description="True when the embedding is whitened."
    )

    def to_job(self) -> Job:
        """The record as a ``Job``."""
        uuid = self.uuid or ""
        essential = [
            SkillReference(id=s.id, label=s.label or "")
            for s in self.essential_skills
            if s.id
        ]
        optional = [
            SkillReference(id=s.id, label=s.label or "")
            for s in self.optional_skills
            if s.id
        ]
        skills: list[str] = []
        for s in essential + optional:
            if s.label and s.label not in skills:
                skills.append(s.label)
        flag = self.requires_post_secondary
        if flag is None:
            nested = self.attributes.get("requires_post_secondary")
            flag = nested if isinstance(nested, bool) else None
        embedding: np.ndarray | None = self.job_embedding
        return Job(
            uuid=uuid,
            originUuid=self.originUuid or self.job_fingerprint or uuid or None,
            job_fingerprint=self.job_fingerprint,
            opportunity_title=self.opportunity_title or "Unknown",
            location=self.location or "",
            city=self.city or "",
            province=self.province or "",
            employer=self.employer,
            employment_type=self.employment_type,
            contract_type=self.contract_type or "full_time",
            salary_text=self.salary_text,
            closing_date=self.closing_date or "",
            posted_date=self.posted_date,
            url=self.url,
            opportunity_description=self.opportunity_description or "",
            opportunity_isco_occupation_group=self.opportunity_isco_occupation_group,
            opportunity_isco_occupation_group_id=self.opportunity_isco_occupation_group_id,
            related_occupation_id=self.related_occupation_id,
            category=self.category,
            source_platform=self.source_platform,
            essential_skills=essential,
            optional_skills=optional,
            skills=skills,
            skill_groups_origin_uuids=list(self.skill_groups_origin_uuids),
            attributes=JobAttributes.from_levels(self.attributes),
            onet_work_activities=work_activities_from_records(
                self.onet_work_activities, source_id=f"job {uuid or '?'}"
            ),
            requires_post_secondary=flag,
            zqf_min=self.zqf_min,
            zqf_max=self.zqf_max,
            zqf_min_label=self.zqf_min_label,
            zqf_max_label=self.zqf_max_label,
            embedding=embedding.astype(np.float32) if embedding is not None else None,
            embedding_whitened=self.job_embedding_whitened is True,
        )


def job_from_record(record: Mapping[str, object]) -> Job | None:
    """Validate one job record: a stored ranked-job document or a flat job object.

    :return: The job, or None when a ranked-job document is inactive
    :raises pydantic.ValidationError: If the record has neither shape
    """
    if any(k in record for k in _RANKED_DOCUMENT_KEYS):
        return build_job_from_ranked(record)
    return JobFileRecord.model_validate(record).to_job()


def load_jobs_file(path: Path) -> list[Job]:
    """Jobs from a JSON list (inactive ranked documents are dropped).

    :raises ValueError: If the file is not a list of objects
    :raises pydantic.ValidationError: If a record is not a valid job
    """
    jobs = (job_from_record(r) for r in _read_json_objects(path))
    return [j for j in jobs if j is not None]


async def _async_fetch_jobs_mongo(
    users_for_filter: Sequence[UserProfile] | None,
) -> tuple[list[Job], JobsLoadTiming]:
    load_backend_dotenv()
    # pylint: disable=import-outside-toplevel
    from app.artifacts.repository import get_artifacts_repository
    from app.jobs.get_jobs_repository import standalone_jobs_repository
    from app.services.job_retrieval import retrieve_jobs_with_timing
    from app.services.location_query import location_hub_chains, resolve_location_keys

    location_keys = (
        resolve_location_keys(
            users_for_filter, location_hub_chains(get_artifacts_repository())
        )
        if users_for_filter
        else None
    )
    async with standalone_jobs_repository() as jobs_repository:
        return await retrieve_jobs_with_timing(jobs_repository, location_keys)


def fetch_jobs_from_mongo(
    users_for_filter: Sequence[UserProfile] | None,
) -> tuple[list[Job], JobsLoadTiming]:
    """Active jobs via the same repository and location filter as ``POST /match``."""
    return asyncio.run(_async_fetch_jobs_mongo(users_for_filter))


def load_jobs(
    jobs_source: JobsSource,
    jobs_path: Path | None,
    users: Sequence[UserProfile],
    mongo_filter_by_users: bool,
    *,
    log_prefix: str = "[offline]",
) -> tuple[list[Job], JobsLoadTiming | None]:
    """Jobs from a file or from Mongo, and the Mongo load timing (None for a file).

    :raises ValueError: If ``jobs_source`` is ``file`` and ``jobs_path`` is None
    """
    if jobs_source == "file":
        if jobs_path is None:
            raise ValueError("jobs_path is required when jobs_source is 'file'")
        return load_jobs_file(jobs_path), None
    jobs, timing = fetch_jobs_from_mongo(users if mongo_filter_by_users else None)
    if not mongo_filter_by_users:
        print(
            f"{log_prefix} mongo: loaded all active jobs (no per-user location filter)",
            file=sys.stderr,
        )
    return jobs, timing
