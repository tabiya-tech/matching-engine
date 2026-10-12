"""Job opportunity: an active job ad from the llm-reranker's ranked collection."""

import numpy as np
from pydantic import ConfigDict, Field

from app.domain.attributes import JobAttributes
from app.domain.base import DomainModel
from app.domain.skills import SkillReference
from app.domain.work_activities import WorkActivity


class Job(DomainModel):
    """One active job opportunity, as matched and as listed by ``GET /jobs``.

    Built from a stored ranked-job document by ``app.jobs.job_document.build_job_from_ranked``;
    the field-to-source mapping is in ``app/domain/README.md``. Field names follow the
    ``OpportunityRecommendation`` / ``JobListItem`` response fields they feed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    uuid: str = Field(description="Job id (``job_id`` of the stored document).")
    originUuid: str | None = Field(
        description="Stable id of the job in its source: the content fingerprint, else ``job_id``.",
    )
    job_fingerprint: str | None = Field(
        default=None,
        description="Content fingerprint of the ad (dedup key across scrapes).",
    )
    opportunity_title: str = Field(
        description="Job title; ``Unknown`` when the ad has none."
    )
    location: str = Field(
        description="``city`` and ``province`` joined by a space; empty when both are."
    )
    city: str = Field(description="City/town of the job; empty when unknown.")
    province: str = Field(description="Province/county of the job; empty when unknown.")
    employer: str | None = Field(default=None, description="Employer name.")
    employment_type: str | None = Field(
        default=None, description="Employment type as stated in the ad."
    )
    contract_type: str = Field(
        description="Employment type, defaulting to ``full_time`` when the ad has none."
    )
    salary_text: str | None = Field(
        default=None, description="Salary as stated in the ad."
    )
    closing_date: str = Field(
        description="Application closing date as stored; empty when absent."
    )
    posted_date: str | None = Field(
        default=None, description="Date the job was posted."
    )
    url: str | None = Field(
        default=None, description="Link to the job posting / application page."
    )
    opportunity_description: str = Field(
        description="Full job description; empty when absent."
    )
    opportunity_isco_occupation_group: str | None = Field(
        default=None, description="ISCO occupation group label."
    )
    opportunity_isco_occupation_group_id: str | None = Field(
        default=None, description="ISCO occupation group code."
    )
    related_occupation_id: str | None = Field(
        default=None,
        description="Occupation the job maps to; falls back to the ISCO group code.",
    )
    category: str | None = Field(
        default=None,
        description="Sector/category; falls back to the ISCO occupation group label.",
    )
    source_platform: str | None = Field(
        default=None, description="Platform the ad was scraped from."
    )
    essential_skills: list[SkillReference] = Field(
        description="Skills the ad requires, as ranked essential by the llm-reranker."
    )
    optional_skills: list[SkillReference] = Field(
        description="Skills the ad lists as nice-to-have, as ranked optional by the llm-reranker."
    )
    skills: list[str] = Field(
        description="De-duplicated skill labels, essential first then optional (jobs-board listing)."
    )
    skill_groups_origin_uuids: list[str] = Field(
        description="Skill-group origin UUIDs derived from the job's skills."
    )
    attributes: JobAttributes = Field(
        description="Preference-attribute levels of the job."
    )
    onet_work_activities: list[WorkActivity] = Field(
        description="Rated O*NET work activities of the job."
    )
    requires_post_secondary: bool | None = Field(
        default=None,
        description="True when the job requires post-secondary education; None when the ad does not say.",
    )
    zqf_min: int | None = Field(
        default=None,
        description="Minimum Zambia Qualifications Framework level (Zambia only).",
    )
    zqf_max: int | None = Field(
        default=None,
        description="Maximum Zambia Qualifications Framework level (Zambia only).",
    )
    zqf_min_label: str | None = Field(
        default=None, description="Label of ``zqf_min`` (Zambia only)."
    )
    zqf_max_label: str | None = Field(
        default=None, description="Label of ``zqf_max`` (Zambia only)."
    )
    embedding: np.ndarray | None = Field(
        default=None,
        repr=False,
        description=(
            "Stage-1 concat skill embedding (float32, ``EMBEDDING_DIM`` long); None when the "
            "document has none of the right size. Not returned by any endpoint."
        ),
    )
    embedding_whitened: bool = Field(
        default=False,
        description="True when ``embedding`` is already whitened by the llm-reranker (used as-is).",
    )

    @property
    def title(self) -> str:
        """The job title, so ranking code can treat jobs and occupations alike."""
        return self.opportunity_title
