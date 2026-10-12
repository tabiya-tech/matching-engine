"""Flatten the occupation file into the per-(occupation, county) ``Occupation`` rows matching scores.

Pure transformation (no I/O, no caching): ``app.occupations.loader`` reads and validates the file and
caches the result.
"""

from collections.abc import Sequence

from app.domain.attributes import JobAttributes
from app.domain.occupation import Occupation
from app.domain.skills import SkillReference
from app.domain.work_activities import work_activities_from_records
from app.occupations.occupation_file import CountyData, OccupationFileEntry, SkillList


def _skill_references(skills: SkillList | None) -> list[SkillReference]:
    """Zip occupation skill uuids with their labels ('' when the label list is shorter or blank)."""
    if skills is None:
        return []
    refs: list[SkillReference] = []
    for i, uuid in enumerate(skills.uuids):
        label = skills.labels[i] if i < len(skills.labels) else None
        refs.append(SkillReference(id=uuid, label=label or ""))
    return refs


def _county_attributes(county_data: CountyData) -> JobAttributes:
    """Attribute levels of one county (list or mapping form), plus its expected demand."""
    raw = county_data.job_attributes.attributes if county_data.job_attributes else None
    levels: dict[str, str | None] = {}
    if isinstance(raw, list):
        for estimate in raw:
            if estimate.attribute_name and estimate.selected_level_id:
                levels[estimate.attribute_name] = estimate.selected_level_id
    elif isinstance(raw, dict):
        levels = dict(raw)
    # Demand label so DemandScorer can read attributes["expected_demand"]
    # (engine-agnostic; powers score_breakdown.demand_* on /match).
    demand = (
        county_data.labor_demand.expected_demand if county_data.labor_demand else None
    )
    if demand:
        levels["expected_demand"] = demand
    return JobAttributes.from_levels(levels)


def flatten_occupations(entries: Sequence[OccupationFileEntry]) -> list[Occupation]:
    """One row per occupation × county (or one county-less row when the entry has no counties).

    Rows use the same ``SkillReference`` / ``WorkActivity`` types as jobs so the engines treat both
    alike. Work activities without a numeric importance and level are dropped.

    :param entries: The validated occupation file
    :return: The occupation rows, in file order
    """
    flattened: list[Occupation] = []
    for entry in entries:
        occ = entry.occupation
        code = occ.code or ""
        label = occ.preferred_label or "Unknown"

        # Post-secondary education gate (see app.ranking.retrieval):
        # occupation-level flag, applied to all of this occupation's county rows.
        requires_post_secondary = occ.requires_post_secondary
        if requires_post_secondary is None:
            requires_post_secondary = entry.requires_post_secondary

        work_activities = work_activities_from_records(
            entry.onet_work_activities, source_id=f"occupation {code or '?'}"
        )
        essential_skills = _skill_references(entry.skills.essential)
        optional_skills = _skill_references(entry.skills.optional)

        for county_data in entry.counties_data or [CountyData(county="")]:
            county = county_data.county or ""
            flattened.append(
                Occupation(
                    uuid=f"{code}_{county}" if county else code,
                    originUuid=code,
                    occupation_label=label,
                    description=occ.description or "",
                    county=county,
                    essential_skills=essential_skills,
                    optional_skills=optional_skills,
                    attributes=_county_attributes(county_data),
                    onet_work_activities=work_activities,
                    requires_post_secondary=requires_post_secondary,
                    # Occupation-specific tasks (sparse in source); formatter falls back to
                    # O*NET WA labels when absent. See app.matching.formatting._typical_tasks.
                    included_tasks=occ.included_tasks or "",
                )
            )

    return flattened
