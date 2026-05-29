"""Pure formatters: enriched v4 (Gemini-embeddings) recs -> MatchResponse rows.

No heavy deps (torch/Gemini) so these are unit-testable in isolation. They map the v4 engine's
per-item output (u_hat, p_hat, final_score, preference_details) plus a re-scored per-skill cosine
detail (`per_job_skill` from CosineSkillMatcher.score_pair) into OpportunityRecommendation /
OccupationRecommendation dicts (best-effort fidelity; Node2Vec-only fields left null).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def split_pref_details(details: Any) -> Tuple[List[dict], Optional[dict]]:
    """Split the unified scorer's `details` into (matched_preferences, work_activity_bws).

    Mirrors matching_service._split_pref_details but dependency-free.
    """
    standard: List[dict] = []
    wa_bws: Optional[dict] = None
    for d in details or []:
        if not isinstance(d, dict):
            continue
        if d.get("attribute") == "work_activity_bws":
            wa_bws = {
                "wa_score_sum": d.get("wa_score_sum", 0.0),
                "details": d.get("wa_details", []),
                "wa_aggregation": d.get("wa_aggregation"),
                "n_work_activities": d.get("n_work_activities"),
                "V_task": d.get("V_task"),
                "V_task_hat": d.get("V_task_hat"),
            }
        elif d.get("attribute"):
            # Any non-BWS attribute row is a MatchedPreference (unified DCE rows OR legacy rows).
            standard.append(d)
    return standard, wa_bws


def build_matched_skills(
    per_job_skill: Any,
    essential_ids: set,
    *,
    sim_threshold: float,
) -> Dict[str, List[dict]]:
    """Build a MatchedSkills-shaped dict from CosineSkillMatcher ``per_job_skill`` rows.

    ``essential_ids`` MUST be the matcher-RESOLVED ids of the item's essential skills (resolved via
    the same label->id function the matcher uses to populate ``job_skill_id``), so the split is in
    the same id space. Every ``per_job_skill`` row is a resolved essential ∪ optional skill: rows in
    ``essential_ids`` -> essential (with per-skill cosine + meets_threshold); the rest are optional
    (kept as OptionalSkillMatch when cosine clears the threshold). Skill groups are not computed by
    the Gemini engine (empty list).
    """
    essential: List[dict] = []
    optional: List[dict] = []
    for r in per_job_skill or []:
        if not isinstance(r, dict):
            continue
        jid = str(r.get("job_skill_id") or "")
        if not jid:
            continue
        sim = float(r.get("cosine_similarity") or 0.0)
        if jid in essential_ids:
            essential.append(
                {
                    "job_skill_id": jid,
                    "job_skill_label": r.get("job_skill_label"),
                    "best_user_skill_id": r.get("best_user_skill_id"),
                    "best_user_skill_label": r.get("best_user_skill_label"),
                    "similarity": round(sim, 4),
                    "meets_threshold": sim >= sim_threshold,
                }
            )
        elif sim >= sim_threshold:
            optional.append({"skill_id": jid, "skill_label": r.get("job_skill_label")})
    return {
        "essential_skill_matches": essential,
        "optional_exact_matches": optional,
        "skill_group_matches": [],
    }


def is_eligible_from_skills(essential_matches: List[dict], *, min_ess_share: float) -> bool:
    """Eligible if the share of essential skills meeting the threshold >= min_ess_share.

    No essential skills -> eligible (nothing to gate on). The post-secondary education gate is
    applied upstream during retrieval, so this is the only skill-side gate here.
    """
    if not essential_matches:
        return True
    met = sum(1 for m in essential_matches if m.get("meets_threshold"))
    return (met / len(essential_matches)) >= min_ess_share


def _score_breakdown(rec: Dict[str, Any]) -> Dict[str, Any]:
    sb = rec.get("score_breakdown") or {}
    return {
        "u_hat": rec.get("u_hat"),
        "p_hat": rec.get("p_hat"),
        "preference_score": rec.get("u_hat"),
        "preference_score_legacy": sb.get("preference_score_legacy"),
        # Node2Vec-only fields intentionally left null (see plan): p_hat_components,
        # total_skill_utility, skill_components, skill_diagnostics, skill_penalty_applied,
        # demand_score, demand_label.
    }


def _justification(matched_skills: Dict[str, List[dict]], matched_prefs: List[dict], final_score: float) -> str:
    parts: List[str] = []
    ess = [m for m in matched_skills["essential_skill_matches"] if m.get("meets_threshold")]
    if ess:
        labels = [str(m.get("job_skill_label") or m.get("job_skill_id")) for m in ess[:3]]
        parts.append("Matches key skills: " + ", ".join(labels) + ".")
    prefs = [p for p in matched_prefs if p.get("matched")]
    if prefs:
        labels = [str(p.get("attr_label") or p.get("attribute")) for p in prefs[:3]]
        parts.append("Fits preferences: " + ", ".join(labels) + ".")
    parts.append(f"Overall match score {round(float(final_score or 0.0), 2)}.")
    return " ".join(parts)


def build_opportunity_row(
    rec: Dict[str, Any],
    item: Dict[str, Any],
    per_job_skill: Any,
    essential_ids: set,
    *,
    rank: int,
    sim_threshold: float,
    min_ess_share: float,
) -> Dict[str, Any]:
    matched_skills = build_matched_skills(per_job_skill, essential_ids, sim_threshold=sim_threshold)
    matched_prefs, wa_bws = split_pref_details(rec.get("preference_details"))
    final_score = float(rec.get("final_score") or 0.0)
    return {
        "uuid": item.get("uuid"),
        "URL": item.get("url") or item.get("URL") or f"www.example.com/{item.get('uuid')}",
        "rank": rank,
        "opportunity_title": item.get("opportunity_title") or "",
        "opportunity_isco_occupation_group": item.get("opportunity_isco_occupation_group"),
        "opportunity_isco_occupation_group_id": item.get("opportunity_isco_occupation_group_id"),
        "location": item.get("location"),
        "employer": item.get("employer"),
        "employment_type": item.get("employment_type"),
        "salary_text": item.get("salary_text"),
        "required_education": item.get("required_education"),
        "required_experience": item.get("required_experience"),
        "closing_date": item.get("closing_date"),
        "is_eligible": is_eligible_from_skills(matched_skills["essential_skill_matches"], min_ess_share=min_ess_share),
        "justification": _justification(matched_skills, matched_prefs, final_score),
        "opportunity_description": item.get("opportunity_description") or item.get("contract_type", "full_time"),
        "contract_type": item.get("contract_type"),
        "final_score": round(final_score, 4),
        "score_breakdown": _score_breakdown(rec),
        "matched_skills": matched_skills,
        "matched_preferences": matched_prefs,
        "matched_work_activities": wa_bws,
    }


def build_occupation_row(
    rec: Dict[str, Any],
    item: Dict[str, Any],
    per_job_skill: Any,
    essential_ids: set,
    *,
    rank: int,
    sim_threshold: float,
    min_ess_share: float,
) -> Dict[str, Any]:
    matched_skills = build_matched_skills(per_job_skill, essential_ids, sim_threshold=sim_threshold)
    matched_prefs, wa_bws = split_pref_details(rec.get("preference_details"))
    final_score = float(rec.get("final_score") or 0.0)
    return {
        "uuid": item.get("uuid"),
        "originUuid": item.get("originUuid"),
        "rank": rank,
        "occupation_label": item.get("occupation_label") or item.get("preferredLabel") or "",
        "province": item.get("province"),
        "is_eligible": is_eligible_from_skills(matched_skills["essential_skill_matches"], min_ess_share=min_ess_share),
        "justification": _justification(matched_skills, matched_prefs, final_score),
        "occupation_description": item.get("occupation_description") or item.get("description"),
        "final_score": round(final_score, 4),
        "score_breakdown": _score_breakdown(rec),
        "matched_skills": matched_skills,
        "matched_preferences": matched_prefs,
        "matched_work_activities": wa_bws,
    }
