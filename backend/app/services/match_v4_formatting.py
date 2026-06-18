"""Pure formatters: enriched v4 (Gemini-embeddings) recs -> MatchResponse rows.

No heavy deps (torch/Gemini) so these are unit-testable in isolation. They map the v4 engine's
per-item output (u_hat, p_hat, final_score, preference_details) plus a re-scored per-skill cosine
detail (`per_job_skill` from CosineSkillMatcher.score_pair) into OpportunityRecommendation /
OccupationRecommendation dicts (best-effort fidelity; Node2Vec-only fields left null).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import V4_FULL_BADGE_PARTIAL, V4_FULL_BADGE_STRONG
from app.services.demand_score import DemandScorer
from app.services.preference_score_v1.levels import level_label, load_attribute_schema

# Engine-agnostic demand scorer (reads item attributes["expected_demand"]); torch-free.
_DEMAND_SCORER = DemandScorer()

# Max skills listed in a justification (the strongest by cosine similarity). Keeps the sentence
# readable when an item has many matched essential skills (real occupations can have 40+).
_MAX_JUSTIFICATION_SKILLS = 10


def _clamp01(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


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
        exact = bool(r.get("exact"))
        meets = exact or (sim >= sim_threshold)        # exact-id overlap always counts as "has it"
        tier = "exact" if exact else ("embedding" if sim >= sim_threshold else "none")
        if jid in essential_ids:
            essential.append(
                {
                    "job_skill_id": jid,
                    "job_skill_label": r.get("job_skill_label"),
                    "best_user_skill_id": r.get("best_user_skill_id"),
                    "best_user_skill_label": r.get("best_user_skill_label"),
                    "similarity": round(sim, 4),
                    "meets_threshold": meets,
                    "match_tier": tier,
                }
            )
        elif meets:
            optional.append({"skill_id": jid, "skill_label": r.get("job_skill_label")})
    return {
        "essential_skill_matches": essential,
        "optional_exact_matches": optional,
        "skill_group_matches": [],
    }


def essential_coverage(essential_matches: List[dict], n_essential_total: int) -> float:
    """Share of a job's essential skills the user meets, in [0,1]. Denominator is the job's TOTAL
    essential count, so unresolved essentials (absent from ``essential_matches``) count as not-met."""
    n = max(int(n_essential_total or 0), len(essential_matches or []))
    if n == 0:
        return 1.0
    met = sum(1 for m in (essential_matches or []) if m.get("meets_threshold"))
    return met / n


def unparsed_ranking_coverage(
    parsed_covs: List[float], *, override: float
) -> float:
    """Ranking coverage to assign a posting with NO parsed essential skills (unparsed).

    ``essential_coverage`` returns 1.0 for such postings, which gives them a demotion-free ride in
    the v4 Phase-2 ranking (``final *= coverage ** gamma``) despite zero verifiable skill overlap.
    Instead we treat an unparsed posting as a TYPICAL one:
      * ``override >= 0``  -> use that fixed value (1.0 restores the old free-passage behaviour).
      * else, if there are parsed coverages -> their mean (the live, per-shortlist average).
      * else (no parsed items at all) -> a neutral 0.5 fallback.
    Only the RANKING coverage is affected; the displayed essential_coverage is recomputed elsewhere.
    """
    if override >= 0:
        return float(override)
    if parsed_covs:
        return sum(parsed_covs) / len(parsed_covs)
    return 0.5


def skill_match_level(coverage: float, n_essential_total: int) -> str:
    """Graded badge from essential-coverage: strong / partial / weak ('unknown' if no essentials)."""
    if not n_essential_total:
        return "unknown"
    if coverage >= V4_FULL_BADGE_STRONG:
        return "strong"
    if coverage >= V4_FULL_BADGE_PARTIAL:
        return "partial"
    return "weak"


def is_eligible_from_skills(
    essential_matches: List[dict], *, n_essential_total: int, min_ess_share: float
) -> bool:
    """Eligible iff essential-coverage >= min_ess_share. No essential skills -> eligible (nothing to
    gate on). The post-secondary education gate is applied upstream during retrieval.
    """
    if not n_essential_total and not essential_matches:
        return True
    return essential_coverage(essential_matches, n_essential_total) >= min_ess_share


def _skill_components_from_cosine(
    per_job_skill: Any, essential_ids: set, *, sim_threshold: float
) -> Dict[str, Any]:
    """Approximate the legacy Node2Vec skill breakdown from v4 per-skill cosines.

    Reuses the SAME cosine signals v4 already computes (``per_job_skill[].cosine_similarity``,
    clamped to [0,1] in ``CosineSkillMatcher.score_pair``), so every numeric field is in [0,1]:
      - ``ess``: mean best cosine over essential skills (higher = better fit)
      - ``opt``: mean best cosine over optional (non-essential) rows
      - ``loc``/``grp``: ``None`` — v4 computes neither a location-similarity nor group recall
        (explicitly null, not 0, so "not computed" is never read as "no match")
      - ``total_skill_utility``: the essential-fit aggregate (skill-fit proxy, not ``p_hat``)
      - ``skill_penalty_applied``: essential gap share = share of essential below threshold
    Direction matches the legacy engine (utility/fit higher = better; penalty higher = worse),
    but the magnitudes are v4-derived approximations, not identical to the Node2Vec values.
    """
    ess_sims: List[float] = []
    opt_sims: List[float] = []
    for r in per_job_skill or []:
        if not isinstance(r, dict):
            continue
        jid = str(r.get("job_skill_id") or "")
        if not jid:
            continue
        sim = _clamp01(r.get("cosine_similarity") or 0.0)
        (ess_sims if jid in essential_ids else opt_sims).append(sim)

    ess = round(sum(ess_sims) / len(ess_sims), 4) if ess_sims else None
    opt = round(sum(opt_sims) / len(opt_sims), 4) if opt_sims else None
    penalty = (
        round(_clamp01(sum(1 for s in ess_sims if s < sim_threshold) / len(ess_sims)), 4)
        if ess_sims
        else None
    )
    return {
        "skill_components": {"loc": None, "ess": ess, "opt": opt, "grp": None},
        "total_skill_utility": ess,  # essential-fit proxy in [0,1]
        "skill_penalty_applied": penalty,
    }


def _score_breakdown(
    rec: Dict[str, Any],
    item: Dict[str, Any],
    per_job_skill: Any,
    essential_ids: set,
    *,
    sim_threshold: float,
) -> Dict[str, Any]:
    sb = rec.get("score_breakdown") or {}
    skills = _skill_components_from_cosine(per_job_skill, essential_ids, sim_threshold=sim_threshold)
    demand = _DEMAND_SCORER.calculate_score(item)
    present = bool(demand.get("present"))
    return {
        "u_hat": rec.get("u_hat"),
        "p_hat": rec.get("p_hat"),
        "p_hat_source": rec.get("p_hat_source"),  # 'concat_cosine_whitened' when Phase-2 demotion is on
        "preference_score": rec.get("u_hat"),
        "preference_score_legacy": sb.get("preference_score_legacy"),
        # v4 cosine approximations of the legacy Node2Vec skill breakdown (all in [0,1];
        # loc/grp null = not computed). See _skill_components_from_cosine.
        "total_skill_utility": skills["total_skill_utility"],
        "skill_components": skills["skill_components"],
        "skill_penalty_applied": skills["skill_penalty_applied"],
        # Engine-agnostic demand (reads item attributes["expected_demand"]); null when absent.
        "demand_score": demand.get("score") if present else None,
        "demand_label": demand.get("label") if present else None,
    }


def _salary_range(item: Dict[str, Any]) -> Optional[str]:
    """Occupation salary as the earnings-level label (e.g. 'earn_70k' -> '~70k')."""
    earn = (item.get("attributes") or {}).get("earnings_per_month")
    if not earn:
        return None
    lbl = level_label("earnings_per_month", earn, load_attribute_schema())
    return lbl if lbl and lbl != "—" else None


def _split_included_tasks(raw: str) -> List[str]:
    """Split freeform ``included_tasks`` text into a task list.

    Source uses lettered enumerations like ``"Tasks include -\\r\\n(a) ...; (b) ..."``: split on
    the ``(a)/(b)/...`` markers, else on newlines/semicolons; drop a leading "Tasks include".
    """
    text = (raw or "").replace("\r\n", "\n").strip()
    parts = re.split(r"\s*\([a-z]\)\s*", text)
    items = parts[1:] if len(parts) > 1 else re.split(r"[\n;]+", text)
    out: List[str] = []
    for it in items:
        s = re.sub(r"^tasks include\s*[-:]?\s*", "", it.strip(), flags=re.IGNORECASE).strip(" .;-")
        if s:
            out.append(s)
    return out


def _typical_tasks(item: Dict[str, Any], *, max_tasks: int = 8) -> List[str]:
    """Occupation tasks: prefer occupation-specific ``included_tasks``, else top O*NET WA labels."""
    raw = item.get("included_tasks")
    if isinstance(raw, str) and raw.strip():
        tasks = _split_included_tasks(raw)
        if tasks:
            return tasks[:max_tasks]
    wa_sorted = sorted(
        (w for w in (item.get("onet_work_activities") or []) if isinstance(w, dict) and w.get("WA_label")),
        key=lambda w: float(w.get("WA_Importance") or 0.0),
        reverse=True,
    )
    out: List[str] = []
    seen: set = set()
    for w in wa_sorted:
        lab = str(w.get("WA_label")).strip()
        if lab and lab not in seen:
            seen.add(lab)
            out.append(lab)
        if len(out) >= max_tasks:
            break
    return out


def _natural_list(items: List[str]) -> str:
    """['a','b','c'] -> 'a, b and c'; drops blanks. Empty list -> ''."""
    items = [str(i).strip() for i in items if str(i).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _humanize_demand(label: Any) -> Optional[str]:
    """'Very High Expected Demand' -> 'very high demand'; None when absent/unknown."""
    if not label:
        return None
    t = str(label).strip()
    if t.lower().endswith(" expected demand"):
        t = t[: -len(" expected demand")].strip()
    return f"{t.lower()} demand" if t else None


def _justification(
    matched_skills: Dict[str, List[dict]],
    matched_prefs: List[dict],
    item: Dict[str, Any],
    *,
    kind: str = "role",
) -> str:
    """Deterministic, second-person justification built from the match facts (no LLM).

    Draws on the user's own matched skills, the preferences they actually expressed (with the
    item's value), and the item's expected demand. Each part is omitted when absent, with a
    graceful fallback when nothing matched.
    """
    sentences: List[str] = []

    # 1. Skills — list the top matched essential skills by cosine similarity (deduped), up to
    # _MAX_JUSTIFICATION_SKILLS; prefer the user's own skill label, fall back to the item's.
    matched_ess = [m for m in matched_skills.get("essential_skill_matches", []) if m.get("meets_threshold")]
    matched_ess.sort(key=lambda m: float(m.get("similarity") or 0.0), reverse=True)
    names: List[str] = []
    seen: set = set()
    for m in matched_ess:
        nm = str(m.get("best_user_skill_label") or m.get("job_skill_label") or m.get("job_skill_id"))
        if nm and nm not in seen:
            seen.add(nm)
            names.append(nm)
        if len(names) >= _MAX_JUSTIFICATION_SKILLS:
            break
    if names:
        sentences.append(f"Strong match on your {_natural_list(names)} skills.")

    # 2. Preferences the user expressed (all of them), with the item's value where available.
    parts: List[str] = []
    for p in matched_prefs:
        if not p.get("matched"):
            continue
        lab = str(p.get("attr_label") or p.get("attribute") or "").strip().lower()
        val = p.get("job_value_label")
        parts.append(f"{lab} ({val})" if val and str(val) not in ("", "—") else lab)
    phrase = _natural_list(parts)
    if phrase:
        sentences.append(f"It fits your preferences for {phrase}.")

    # 3. Labour-market demand (occupations carry this; jobs usually don't).
    demand = _humanize_demand((item.get("attributes") or {}).get("expected_demand"))
    if demand:
        sentences.append(f"This {kind} is in {demand}.")

    if not sentences:
        return f"Recommended {kind} based on your overall profile."
    return " ".join(sentences)


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
    n_ess_total = len(item.get("essential_skills") or [])
    coverage = essential_coverage(matched_skills["essential_skill_matches"], n_ess_total)
    sb = _score_breakdown(rec, item, per_job_skill, essential_ids, sim_threshold=sim_threshold)
    sb["essential_coverage"] = round(coverage, 4)
    sb["skill_match_level"] = skill_match_level(coverage, n_ess_total)
    return {
        "uuid": item.get("uuid"),
        "originUuid": item.get("originUuid"),
        "URL": item.get("url") or item.get("URL") or f"www.example.com/{item.get('uuid')}",
        "rank": rank,
        "opportunity_title": item.get("opportunity_title") or "",
        "opportunity_isco_occupation_group": item.get("opportunity_isco_occupation_group"),
        "opportunity_isco_occupation_group_id": item.get("opportunity_isco_occupation_group_id"),
        "related_occupation_id": item.get("related_occupation_id"),
        "location": item.get("location"),
        "employer": item.get("employer"),
        "employment_type": item.get("employment_type"),
        "salary_text": item.get("salary_text"),
        "required_education": item.get("required_education"),
        "required_experience": item.get("required_experience"),
        "closing_date": item.get("closing_date"),
        "posted_date": item.get("posted_date"),
        "is_eligible": is_eligible_from_skills(
            matched_skills["essential_skill_matches"], n_essential_total=n_ess_total, min_ess_share=min_ess_share
        ),
        "justification": _justification(matched_skills, matched_prefs, item),
        "opportunity_description": item.get("opportunity_description") or item.get("contract_type", "full_time"),
        "contract_type": item.get("contract_type"),
        "final_score": round(final_score, 4),
        "score_breakdown": sb,
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
    n_ess_total = len(item.get("essential_skills") or [])
    coverage = essential_coverage(matched_skills["essential_skill_matches"], n_ess_total)
    sb = _score_breakdown(rec, item, per_job_skill, essential_ids, sim_threshold=sim_threshold)
    sb["essential_coverage"] = round(coverage, 4)
    sb["skill_match_level"] = skill_match_level(coverage, n_ess_total)
    return {
        "uuid": item.get("uuid"),
        "originUuid": item.get("originUuid"),
        "rank": rank,
        "occupation_label": item.get("occupation_label") or item.get("preferredLabel") or "",
        "province": item.get("province"),
        "is_eligible": is_eligible_from_skills(
            matched_skills["essential_skill_matches"], n_essential_total=n_ess_total, min_ess_share=min_ess_share
        ),
        "justification": _justification(matched_skills, matched_prefs, item),
        "occupation_description": item.get("occupation_description") or item.get("description"),
        "salary_range": _salary_range(item),
        "typical_tasks": _typical_tasks(item),
        "career_path_next_steps": [],  # no source in occupation data — see plan
        "final_score": round(final_score, 4),
        "score_breakdown": sb,
        "matched_skills": matched_skills,
        "matched_preferences": matched_prefs,
        "matched_work_activities": wa_bws,
    }
