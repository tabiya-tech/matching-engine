"""Build the occupation file (and occupation-group documents) from a Tabiya taxonomy export.

A taxonomy export is a directory of CSVs (``occupations.csv``, ``occupation_groups.csv``,
``occupation_hierarchy.csv``, ``occupation_to_skill_relations.csv``, ``skills.csv``,
``model_info.csv``), e.g. ``empujar/empujar-taxonomy/``.

The matching catalogue's "occupations" are the ISCO-08 **unit groups** (4-digit ``iscogroup``
codes), not the ESCO occupations under them. A unit group's essential (optional) skills are the
union of the essential (optional) skill relations of every occupation beneath it. Skill uuids are
the oldest ``UUIDHISTORY`` entry (the last line), which is the same across locales and is the id
the embedding artefacts use (see ``scripts/build_language_taxonomy.py``).

The export has no county attributes, expected demand, O*NET work activities, post-secondary flag
or task list. Those can be copied by code from an existing occupation file (``enrich_from``), and
counties can be given as plain names.

Pure (no I/O) except ``read_taxonomy_export``; driven by ``scripts/build_occupation_file.py``.
"""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from app.domain.base import DomainModel
from app.occupations.ingest import ParsedOccupation
from app.occupations.occupation_document import content_sha256
from app.occupations.occupation_file import SkillList

csv.field_size_limit(10_000_000)

UNIT_GROUP_TYPE = "iscogroup"
UNIT_GROUP_CODE_LENGTH = 4


class TaxonomyModelInfo(DomainModel):
    """The ``model_info.csv`` row: which taxonomy model and locale the export is."""

    name: str = Field(description="Model name, e.g. ``Tabiya (ESCO 1.1.1)``.")
    locale: str = Field(description="Locale, e.g. ``AR-es``.")
    version: str = Field(description="Model version; may be empty.")
    uuid: str = Field(description="Model uuid (oldest ``UUIDHISTORY`` entry).")


class TaxonomyGroup(DomainModel):
    """One ``occupation_groups.csv`` row."""

    id: str = Field(description="Export-local id (joins the hierarchy).")
    code: str = Field(description="Group code, e.g. ``2512`` or ``I31``.")
    group_type: str = Field(description="``iscogroup`` or ``localgroup``.")
    preferred_label: str = Field(description="Preferred label.")
    alt_labels: list[str] = Field(description="Alternative labels.")
    description: str = Field(description="Description; may be empty.")
    origin_uri: str = Field(description="Origin URI; may be empty.")
    uuid: str = Field(description="Oldest ``UUIDHISTORY`` entry.")


class TaxonomyOccupation(DomainModel):
    """One ``occupations.csv`` row (only the fields the build reads)."""

    id: str = Field(description="Export-local id.")
    code: str = Field(description="Occupation code, e.g. ``2512.1``.")
    occupation_type: str = Field(
        description="``escooccupation`` or ``localoccupation``."
    )


class TaxonomySkill(DomainModel):
    """One ``skills.csv`` row (only the fields the build reads)."""

    uuid: str = Field(description="Oldest ``UUIDHISTORY`` entry.")
    preferred_label: str = Field(description="Preferred label.")


class TaxonomyExport(DomainModel):
    """The parts of a taxonomy export the occupation build needs."""

    model_info: TaxonomyModelInfo | None = Field(
        description="The export's model, when ``model_info.csv`` is present."
    )
    groups: list[TaxonomyGroup] = Field(description="Occupation groups, in file order.")
    occupations: dict[str, TaxonomyOccupation] = Field(
        description="Occupations by export id."
    )
    skills: dict[str, TaxonomySkill] = Field(description="Skills by export id.")
    children: dict[str, list[str]] = Field(
        description="Hierarchy: parent export id -> child export ids, in file order."
    )
    essential_skills: dict[str, list[str]] = Field(
        description="Occupation export id -> essential skill export ids, in file order."
    )
    optional_skills: dict[str, list[str]] = Field(
        description="Occupation export id -> optional skill export ids, in file order."
    )


class SkillSetDiff(DomainModel):
    """How a built unit group's skills differ from the enrichment file's entry for it."""

    code: str
    missing_essential: int = Field(description="In the enrichment entry, not built.")
    extra_essential: int = Field(description="Built, not in the enrichment entry.")
    missing_optional: int
    extra_optional: int


class OccupationBuildReport(DomainModel):
    """What ``build_occupation_entries`` did, for the operator."""

    n_unit_groups: int = Field(
        description="Unit groups in the export (= entries built)."
    )
    codes_without_skills: list[str] = Field(description="Entries with no skills.")
    codes_without_description: list[str] = Field(
        description="Entries with no description."
    )
    codes_not_enriched: list[str] = Field(
        description="Entries the enrichment file has no entry for."
    )
    n_enrichment_unused: int = Field(
        description="Enrichment entries matching no built unit group."
    )
    n_occupations_outside_unit_groups: int = Field(
        description="Occupations under no unit group (their skills are not used)."
    )
    skill_diffs: list[SkillSetDiff] = Field(
        description="Entries whose skills differ from the enrichment entry's."
    )


def _lines(value: str) -> list[str]:
    return [line.strip() for line in (value or "").splitlines() if line.strip()]


def _oldest_uuid(uuid_history: str) -> str:
    history = _lines(uuid_history)
    return history[-1] if history else ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_taxonomy_export(directory: Path) -> TaxonomyExport:
    """Read the occupation-related CSVs of a taxonomy export.

    :raises FileNotFoundError: If a required CSV is missing
    :raises KeyError: If a required column is missing
    """
    model_info: TaxonomyModelInfo | None = None
    if (directory / "model_info.csv").is_file():
        rows = _read_csv(directory / "model_info.csv")
        if rows:
            row = rows[0]
            model_info = TaxonomyModelInfo(
                name=row["NAME"],
                locale=row["LOCALE"],
                version=row.get("VERSION", ""),
                uuid=_oldest_uuid(row.get("UUIDHISTORY", "")),
            )

    groups = [
        TaxonomyGroup(
            id=row["ID"],
            code=row["CODE"].strip(),
            group_type=row["GROUPTYPE"].strip(),
            preferred_label=row["PREFERREDLABEL"].strip(),
            alt_labels=_lines(row.get("ALTLABELS", "")),
            description=(row.get("DESCRIPTION") or "").strip(),
            origin_uri=(row.get("ORIGINURI") or "").strip(),
            uuid=_oldest_uuid(row.get("UUIDHISTORY", "")),
        )
        for row in _read_csv(directory / "occupation_groups.csv")
    ]
    occupations = {
        row["ID"]: TaxonomyOccupation(
            id=row["ID"],
            code=row["CODE"].strip(),
            occupation_type=row["OCCUPATIONTYPE"].strip(),
        )
        for row in _read_csv(directory / "occupations.csv")
    }
    skills = {
        row["ID"]: TaxonomySkill(
            uuid=_oldest_uuid(row["UUIDHISTORY"]),
            preferred_label=row["PREFERREDLABEL"].strip(),
        )
        for row in _read_csv(directory / "skills.csv")
    }
    children: dict[str, list[str]] = {}
    for row in _read_csv(directory / "occupation_hierarchy.csv"):
        children.setdefault(row["PARENTID"], []).append(row["CHILDID"])
    essential: dict[str, list[str]] = {}
    optional: dict[str, list[str]] = {}
    for row in _read_csv(directory / "occupation_to_skill_relations.csv"):
        relation = row["RELATIONTYPE"].strip().lower()
        target = essential if relation == "essential" else optional
        if relation in ("essential", "optional"):
            target.setdefault(row["OCCUPATIONID"], []).append(row["SKILLID"])
    return TaxonomyExport(
        model_info=model_info,
        groups=groups,
        occupations=occupations,
        skills=skills,
        children=children,
        essential_skills=essential,
        optional_skills=optional,
    )


def normalise_occupation_code(code: str) -> str:
    """Code used to match entries across files: ISCO codes lose leading zeros (``0110`` = ``110``)."""
    code = code.strip()
    return (code.lstrip("0") or "0") if code.isdigit() else code


def is_unit_group(group: TaxonomyGroup) -> bool:
    return (
        group.group_type == UNIT_GROUP_TYPE
        and len(group.code) == UNIT_GROUP_CODE_LENGTH
        and group.code.isdigit()
    )


def _descendant_occupations(export: TaxonomyExport, root_id: str) -> list[str]:
    """Export ids of every occupation under ``root_id``, depth-first in hierarchy order."""
    out: list[str] = []
    seen: set[str] = set()
    stack = list(reversed(export.children.get(root_id, [])))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node in export.occupations:
            out.append(node)
        stack.extend(reversed(export.children.get(node, [])))
    return out


def _skill_block(
    export: TaxonomyExport, skill_ids: Sequence[str]
) -> dict[str, list[str]]:
    uuids: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    for skill_id in skill_ids:
        skill = export.skills.get(skill_id)
        if skill is None or not skill.uuid or skill.uuid in seen:
            continue
        seen.add(skill.uuid)
        uuids.append(skill.uuid)
        labels.append(skill.preferred_label)
    return {"uuids": uuids, "labels": labels}


def _merge_skill_block(
    block: dict[str, list[str]],
    reference: SkillList | None,
    labels_by_uuid: Mapping[str, str],
) -> dict[str, list[str]]:
    """``block`` plus the reference's skills it lacks, labelled from the export when it has them."""
    if reference is None:
        return block
    uuids = list(block["uuids"])
    labels = list(block["labels"])
    seen = set(uuids)
    for i, uuid in enumerate(reference.uuids):
        if uuid in seen:
            continue
        seen.add(uuid)
        ref_label = reference.labels[i] if i < len(reference.labels) else None
        uuids.append(uuid)
        labels.append(labels_by_uuid.get(uuid) or ref_label or "")
    return {"uuids": uuids, "labels": labels}


def _skill_diff(
    code: str, built: Mapping[str, Mapping[str, list[str]]], reference: ParsedOccupation
) -> SkillSetDiff | None:
    ref = reference.entry.skills
    ref_ess = set(ref.essential.uuids if ref.essential else [])
    ref_opt = set(ref.optional.uuids if ref.optional else [])
    ess = set(built["essential"]["uuids"])
    opt = set(built["optional"]["uuids"])
    if ess == ref_ess and opt == ref_opt:
        return None
    return SkillSetDiff(
        code=code,
        missing_essential=len(ref_ess - ess),
        extra_essential=len(ess - ref_ess),
        missing_optional=len(ref_opt - opt),
        extra_optional=len(opt - ref_opt),
    )


def build_occupation_entries(
    export: TaxonomyExport,
    *,
    enrich_from: Sequence[ParsedOccupation] = (),
    counties: Sequence[str] = (),
    copy_counties: bool = False,
    merge_skills: bool = False,
) -> tuple[list[dict[str, Any]], OccupationBuildReport]:
    """One occupation-file entry per ISCO unit group of the export, in code order.

    :param enrich_from: An existing occupation file; its O*NET work activities, post-secondary
        flag and task list are copied onto the entry with the same (normalised) code
    :param counties: County / province names; each entry gets one ``counties_data`` row per name,
        with no attribute estimates or demand
    :param copy_counties: Copy ``counties_data`` from the enrichment entry instead (overrides
        ``counties`` for entries that have one)
    :param merge_skills: Add the enrichment entry's skills to the built ones (curated skills the
        taxonomy relations do not give), labelled from the export so the language stays the
        export's; ``skill_diffs`` still reports the difference before the merge
    :return: ``(entries, report)``; entries are plain JSON-ready dicts
    """
    enrich = {normalise_occupation_code(p.code): p for p in enrich_from}
    labels_by_uuid = {s.uuid: s.preferred_label for s in export.skills.values()}
    used_enrichment: set[str] = set()
    entries: list[dict[str, Any]] = []
    without_skills: list[str] = []
    without_description: list[str] = []
    not_enriched: list[str] = []
    diffs: list[SkillSetDiff] = []
    covered: set[str] = set()

    for group in sorted(
        (g for g in export.groups if is_unit_group(g)), key=lambda g: g.code
    ):
        occupation_ids = _descendant_occupations(export, group.id)
        covered.update(occupation_ids)
        essential_ids = [
            s for o in occupation_ids for s in export.essential_skills.get(o, [])
        ]
        optional_ids = [
            s for o in occupation_ids for s in export.optional_skills.get(o, [])
        ]
        skills = {
            "essential": _skill_block(export, essential_ids),
            "optional": _skill_block(export, optional_ids),
        }
        key = normalise_occupation_code(group.code)
        reference = enrich.get(key)
        if reference is not None:
            diff = _skill_diff(group.code, skills, reference)
            if diff is not None:
                diffs.append(diff)
            if merge_skills:
                ref_skills = reference.entry.skills
                skills = {
                    "essential": _merge_skill_block(
                        skills["essential"], ref_skills.essential, labels_by_uuid
                    ),
                    "optional": _merge_skill_block(
                        skills["optional"], ref_skills.optional, labels_by_uuid
                    ),
                }
        if not skills["essential"]["uuids"] and not skills["optional"]["uuids"]:
            without_skills.append(group.code)
        if not group.description:
            without_description.append(group.code)

        occupation: dict[str, Any] = {
            "code": group.code,
            "preferred_label": group.preferred_label,
            "alt_labels": group.alt_labels,
            "description": group.description,
            "included_tasks": None,
            "requires_post_secondary": None,
            "uuid": group.uuid,
            "origin_uri": group.origin_uri,
        }
        entry: dict[str, Any] = {
            "occupation": occupation,
            "skills": skills,
            "counties_data": [{"county": name} for name in counties],
            "onet_work_activities": [],
        }
        if reference is None:
            not_enriched.append(group.code)
        else:
            used_enrichment.add(key)
            ref_occ = reference.raw.get("occupation") or {}
            occupation["included_tasks"] = ref_occ.get("included_tasks")
            occupation["requires_post_secondary"] = (
                reference.entry.occupation.requires_post_secondary
                if reference.entry.occupation.requires_post_secondary is not None
                else reference.entry.requires_post_secondary
            )
            entry["onet_work_activities"] = list(
                reference.raw.get("onet_work_activities") or []
            )
            if copy_counties and reference.raw.get("counties_data"):
                entry["counties_data"] = list(reference.raw["counties_data"])
        entries.append(entry)

    report = OccupationBuildReport(
        n_unit_groups=len(entries),
        codes_without_skills=without_skills,
        codes_without_description=without_description,
        codes_not_enriched=not_enriched if enrich else [],
        n_enrichment_unused=len(set(enrich) - used_enrichment),
        n_occupations_outside_unit_groups=len(set(export.occupations) - covered),
        skill_diffs=diffs,
    )
    return entries, report


def _group_parents(export: TaxonomyExport) -> dict[str, str]:
    """Group export id -> parent group export id."""
    group_ids = {g.id for g in export.groups}
    return {
        child: parent
        for parent, kids in export.children.items()
        if parent in group_ids
        for child in kids
        if child in group_ids
    }


def group_ancestry(export: TaxonomyExport) -> dict[str, list[str]]:
    """Group code -> codes of the group and its ancestors, broadest first.

    Keyed by the export's code as is: normalising would merge ISCO levels (``1`` and ``01``).
    """
    by_id = {g.id: g for g in export.groups}
    parents = _group_parents(export)
    out: dict[str, list[str]] = {}
    for group in export.groups:
        chain: list[str] = []
        node: str | None = group.id
        while node is not None and node in by_id and len(chain) <= len(by_id):
            chain.append(by_id[node].code)
            node = parents.get(node)
        out[group.code] = list(reversed(chain))
    return out


def occupation_group_ancestry(export: TaxonomyExport) -> dict[str, list[str]]:
    """Normalised occupation code -> its group's ancestry (``group_ancestry``), broadest first.

    Only unit groups and non-numeric groups: their normalised codes cannot collide (unit groups
    all have ``UNIT_GROUP_CODE_LENGTH`` digits; other codes are not changed by normalising).
    """
    ancestry = group_ancestry(export)
    return {
        normalise_occupation_code(group.code): ancestry[group.code]
        for group in export.groups
        if is_unit_group(group) or not group.code.isdigit()
    }


def build_group_documents(export: TaxonomyExport) -> dict[str, dict[str, Any]]:
    """One ``MONGO_OCCUPATION_GROUPS_COLLECTION`` document per occupation group, keyed by code."""
    by_id = {g.id: g for g in export.groups}
    parents = _group_parents(export)
    ancestry = group_ancestry(export)
    docs: dict[str, dict[str, Any]] = {}
    for position, group in enumerate(export.groups):
        parent = by_id.get(parents.get(group.id, ""))
        body: dict[str, Any] = {
            "code": group.code,
            "group_type": group.group_type,
            "preferred_label": group.preferred_label,
            "alt_labels": group.alt_labels,
            "description": group.description,
            "origin_uri": group.origin_uri,
            "uuid": group.uuid,
            "parent_code": parent.code if parent else None,
            "ancestor_codes": ancestry[group.code][:-1],
            "is_unit_group": is_unit_group(group),
        }
        docs[group.code] = {
            "_id": group.code,
            **body,
            "position": position,
            "content_sha256": content_sha256(body, []),
        }
    return docs
