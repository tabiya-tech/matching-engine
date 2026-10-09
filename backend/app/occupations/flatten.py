"""Flatten the occupation taxonomy JSON into the per-(occupation, county) rows matching scores.

Pure transformation (no I/O, no caching): ``app.occupations.loader`` reads the file and caches the result.
"""

from typing import Any


def _occ_skill_pairs(uuids: list, labels: list) -> list[dict[str, str]]:
    """Zip occupation skill uuids with their labels (from the occupation JSON; '' if absent)."""
    pairs: list[dict[str, str]] = []
    for i, u in enumerate(uuids):
        lab = labels[i] if i < len(labels) else ""
        pairs.append({"id": str(u), "label": str(lab) if lab else ""})
    return pairs


def flatten_occupations(raw_occupations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per occupation × county (or one county-less row when the entry has no counties).

    Rows use the same ``{id, label}`` skill shape as job dicts so the engines treat both alike.
    """
    flattened = []
    for entry in raw_occupations:
        occ = entry.get("occupation", {})
        skills = entry.get("skills", {})
        ess_block = (
            skills.get("essential", {})
            if isinstance(skills.get("essential"), dict)
            else {}
        )
        opt_block = (
            skills.get("optional", {})
            if isinstance(skills.get("optional"), dict)
            else {}
        )
        ess_uuids = ess_block.get("uuids", []) or []
        opt_uuids = opt_block.get("uuids", []) or []
        ess_labels = ess_block.get("labels", []) or []
        opt_labels = opt_block.get("labels", []) or []
        counties = entry.get("counties_data", [])

        code = occ.get("code", "")
        label = occ.get("preferred_label", "Unknown")
        description = occ.get("description", "")

        # Post-secondary education gate (see app.services.education_eligibility):
        # occupation-level flag, applied to all of this occupation's county rows.
        requires_post_secondary = occ.get("requires_post_secondary")
        if requires_post_secondary is None:
            requires_post_secondary = entry.get("requires_post_secondary")

        raw_wa = entry.get("onet_work_activities", [])
        onet_wa = []
        for w in raw_wa:
            wc = w.get("WA_code")
            imp = w.get("WA_Importance", "")
            lvl = w.get("WA_Level", "")
            if wc and imp != "" and lvl != "":
                onet_wa.append(
                    {
                        "WA_code": wc,
                        "WA_label": w.get("WA_label", ""),
                        "WA_Importance": float(imp),
                        "WA_Level": float(lvl),
                    }
                )

        if not counties:
            counties = [{"county": "", "job_attributes": {}}]

        for cd in counties:
            county = cd.get("county", "")
            job_attrs = cd.get("job_attributes", {})
            attrs_raw = job_attrs.get("attributes", [])
            attributes = {}
            if isinstance(attrs_raw, list):
                for a in attrs_raw:
                    name = a.get("attribute_name")
                    val = a.get("selected_level_id")
                    if name and val:
                        attributes[name] = val
            elif isinstance(attrs_raw, dict):
                attributes = attrs_raw

            # Demand label so DemandScorer can read attributes["expected_demand"]
            # (engine-agnostic; powers score_breakdown.demand_* on /match_v4).
            expected_demand = (cd.get("labor_demand") or {}).get("expected_demand")
            if expected_demand:
                attributes = {**attributes, "expected_demand": expected_demand}

            # Wrap skills in the same {id, label} shape used by job dicts. Labels come
            # from the occupation JSON (skills.*.labels) when present, else empty; gap
            # analysis reads id directly without going through label resolution.
            flattened.append(
                {
                    "uuid": f"{code}_{county}" if county else code,
                    "originUuid": code,
                    "occupation_label": label,
                    "preferredLabel": label,
                    "description": description,
                    "location": county,
                    "city": county,
                    "province": county,
                    "essential_skills": _occ_skill_pairs(ess_uuids, ess_labels),
                    "optional_skills": _occ_skill_pairs(opt_uuids, opt_labels),
                    "skill_groups_origin_uuids": [],
                    "attributes": attributes,
                    "requires_post_secondary": requires_post_secondary,
                    "onet_work_activities": onet_wa,
                    # Occupation-specific tasks (sparse in source); formatter falls back to
                    # O*NET WA labels when absent. See match_v4_formatting._typical_tasks.
                    "included_tasks": occ.get("included_tasks") or "",
                }
            )

    return flattened
