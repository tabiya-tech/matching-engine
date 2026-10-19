"""Occupation: one taxonomy occupation in one county, from the committed occupation file."""

import numpy as np
from pydantic import ConfigDict, Field

from app.domain.attributes import JobAttributes
from app.domain.base import DomainModel
from app.domain.skills import SkillReference
from app.domain.work_activities import WorkActivity


class Occupation(DomainModel):
    """One occupation × county row of the occupation corpus.

    Built by ``app.occupations.flatten.flatten_occupations`` from the occupation file
    (``OCCUPATION_JSON_PATH``): every occupation is repeated once per county it has data for,
    because the attribute levels and expected demand vary by county. Skills and work activities
    are the same on every county row of an occupation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    uuid: str = Field(
        description="Row id: ``<code>_<county>``, or ``<code>`` when the occupation has no county data."
    )
    originUuid: str = Field(description="Occupation code from the occupation file.")
    occupation_label: str = Field(
        description="Preferred label of the occupation; ``Unknown`` when absent."
    )
    description: str = Field(description="Occupation description; empty when absent.")
    county: str = Field(
        description="County this row's attributes and demand apply to; empty when none."
    )
    location_keys: tuple[str, ...] = Field(
        default=(),
        description="Location keys of ``county`` (``app.location.keys``); decides location eligibility.",
    )
    essential_skills: list[SkillReference] = Field(
        description="Essential skills of the occupation."
    )
    optional_skills: list[SkillReference] = Field(
        description="Optional skills of the occupation."
    )
    attributes: JobAttributes = Field(
        description="Attribute levels and expected demand of the occupation in this county."
    )
    onet_work_activities: list[WorkActivity] = Field(
        description="Rated O*NET work activities of the occupation."
    )
    requires_post_secondary: bool | None = Field(
        default=None,
        description="True when the occupation requires post-secondary education; None when unknown.",
    )
    included_tasks: str = Field(
        description="Free-text list of the occupation's tasks (``(a) …; (b) …``); empty when absent.",
    )
    embedding: np.ndarray | None = Field(
        default=None,
        repr=False,
        description=(
            "Concat skill embedding from the occupation-embeddings NPZ, shared by every county "
            "row of the occupation; None until attached by ``attach_occupation_embeddings``."
        ),
    )
    embedding_whitened: bool = Field(
        default=False,
        description="True when ``embedding`` was whitened once at load (used as-is by stage-1 retrieval).",
    )

    @property
    def city(self) -> str:
        """The county, so location code can treat jobs and occupations alike."""
        return self.county

    @property
    def province(self) -> str:
        """The county, so location code can treat jobs and occupations alike."""
        return self.county

    @property
    def location(self) -> str:
        """The county, so location code can treat jobs and occupations alike."""
        return self.county

    @property
    def title(self) -> str:
        """The occupation label, so ranking code can treat jobs and occupations alike."""
        return self.occupation_label

    @property
    def skill_groups_origin_uuids(self) -> list[str]:
        """Always empty: the occupation file lists no skill groups."""
        return []
