"""Shape of the committed occupation file (``OCCUPATION_JSON_PATH``).

The file is a JSON list of ``OccupationFileEntry`` (one per occupation), exported from the taxonomy
with per-county attribute estimates and expected demand. ``OCCUPATION_FILE`` validates the whole
file at load time; ``app.occupations.flatten`` maps the entries to ``Occupation`` rows.
"""

from pydantic import Field, TypeAdapter

from app.domain.base import SourceRecord
from app.domain.coercion import OptionalFlag, OptionalText, TextList
from app.domain.work_activities import WorkActivityRecord


class OccupationInfo(SourceRecord):
    """The ``occupation`` block of an entry."""

    code: OptionalText = Field(
        default=None, description="Occupation code (taxonomy id)."
    )
    preferred_label: OptionalText = Field(default=None, description="Preferred label.")
    description: OptionalText = Field(
        default=None, description="Occupation description."
    )
    included_tasks: OptionalText = Field(
        default=None, description="Free-text task list, ``(a) …; (b) …``."
    )
    requires_post_secondary: OptionalFlag = Field(
        default=None,
        description="True when the occupation requires post-secondary education.",
    )


class SkillList(SourceRecord):
    """Parallel lists of skill UUIDs and labels (``skills.essential`` / ``skills.optional``)."""

    uuids: TextList = Field(default_factory=list, description="Skill UUIDs.")
    labels: list[OptionalText] = Field(
        default_factory=list,
        description="Skill labels, aligned with ``uuids``; may be shorter.",
    )


class OccupationSkills(SourceRecord):
    """The ``skills`` block of an entry."""

    essential: SkillList | None = Field(default=None, description="Essential skills.")
    optional: SkillList | None = Field(default=None, description="Optional skills.")


class AttributeEstimate(SourceRecord):
    """One attribute estimate in ``counties_data[].job_attributes.attributes`` (list form)."""

    attribute_name: OptionalText = Field(
        default=None, description="Attribute name, e.g. ``earnings_per_month``."
    )
    selected_level_id: OptionalText = Field(
        default=None, description="Estimated level id, e.g. ``earn_70k``."
    )


class CountyJobAttributes(SourceRecord):
    """``counties_data[].job_attributes``: attribute estimates as a list or a name→level mapping."""

    attributes: list[AttributeEstimate] | dict[str, OptionalText] | None = Field(
        default=None, description="Attribute estimates."
    )


class LaborDemand(SourceRecord):
    """``counties_data[].labor_demand``."""

    expected_demand: OptionalText = Field(
        default=None,
        description="Expected demand label, e.g. ``High Expected Demand``.",
    )


class CountyData(SourceRecord):
    """One county's data for an occupation (``counties_data[]``)."""

    county: OptionalText = Field(default=None, description="County name.")
    job_attributes: CountyJobAttributes | None = Field(
        default=None, description="Attribute estimates."
    )
    labor_demand: LaborDemand | None = Field(
        default=None, description="Expected labour-market demand."
    )


class OccupationFileEntry(SourceRecord):
    """One occupation in the occupation file."""

    occupation: OccupationInfo = Field(
        default_factory=OccupationInfo, description="Identity and description."
    )
    skills: OccupationSkills = Field(
        default_factory=OccupationSkills, description="Essential and optional skills."
    )
    counties_data: list[CountyData] = Field(
        default_factory=list,
        description="Per-county attributes and demand; empty for county-less entries.",
    )
    onet_work_activities: list[WorkActivityRecord] = Field(
        default_factory=list,
        description="O*NET work activities with importance and level.",
    )
    requires_post_secondary: OptionalFlag = Field(
        default=None,
        description="Entry-level fallback for ``occupation.requires_post_secondary`` (older exports).",
    )


OCCUPATION_FILE: TypeAdapter[list[OccupationFileEntry]] = TypeAdapter(
    list[OccupationFileEntry]
)
"""Validator for the whole occupation file."""
