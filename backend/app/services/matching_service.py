from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from app.config import (
    DEMAND_SCORE_MAPPING,
    GLOBAL_WEIGHTS,
    MATCH_APPLY_LOCATION_FILTER,
    MATCH_RESPONSE_SKILL_MIN_SCORE,
    MATCH_TOP_K_OCCUPATIONS,
    MATCH_TOP_K_OPPORTUNITIES,
    MATCH_TOP_K_SKILL_GAPS,
    SCORING_MODE,
)
from app.database import get_all_jobs, get_all_occupations
from app.match_timing_log import log_match_step
from app.services.education_eligibility import filter_jobs_by_education
from app.services.preference_score import PreferenceScorer
from app.services.skill_gap_analysis import analyze_skill_gaps
from app.services.skill_score import SkillScorer
from app.services.success_propensity import SuccessPropensityScorer

# Initialize scorers once at module level
scorer_skill = SkillScorer()
scorer_pref = PreferenceScorer()
scorer_success = SuccessPropensityScorer()


def _norm(v: Optional[str]) -> str:
    return str(v).strip().casefold() if v is not None else ""


def _split_pref_details(pref_details: list) -> tuple:
    """Separate standard preference entries from work-activity BWS entries."""
    standard = []
    wa_bws = None
    for d in pref_details:
        if d.get("attribute") == "work_activity_bws":
            wa_bws = {
                "wa_score_sum": d.get("wa_score_sum", 0.0),
                "details": d.get("wa_details", []),
            }
        else:
            standard.append(d)
    return standard, wa_bws


def _ms(t0: float) -> float:
    """Elapsed milliseconds since t0 = time.perf_counter()."""
    return (time.perf_counter() - t0) * 1000.0


def _filter_essential_skill_matches(match_details: dict) -> list[dict]:
    """Keep only essential skill matches whose similarity passes response threshold."""
    essential = match_details.get("essential_skill_matches", [])
    return [
        m for m in essential
        if float(m.get("similarity", 0.0)) >= MATCH_RESPONSE_SKILL_MIN_SCORE
    ]


def _filter_skill_gap_recommendations(skill_gaps: list[dict]) -> list[dict]:
    """Keep only skill-gap rows whose proximity passes response threshold."""
    return [
        g for g in skill_gaps
        if float(g.get("proximity_score", 0.0)) >= MATCH_RESPONSE_SKILL_MIN_SCORE
    ][:MATCH_TOP_K_SKILL_GAPS]


def _job_matches_user_location(job: Dict[str, Any], user: Dict[str, Any]) -> bool:
    """Lenient location match.
    - Always matches 'Remote' jobs
    - Matches if city or province match (case-insensitive, substring)
    """
    user_city = _norm(user.get("city"))
    user_province = _norm(user.get("province"))

    job_city = _norm(job.get("city"))
    job_province = _norm(job.get("province"))
    job_loc = _norm(job.get("location"))

    #  Always include Remote jobs
    if "remote" in job_city or "remote" in job_province or "remote" in job_loc:
        return True

    if not user_city or not user_province:
        return False

    # Check City Match (Lenient)
    if job_city and (user_city in job_city or job_city in user_city):
        return True

    # Check Province Match (Lenient)
    if job_province and (user_province in job_province or job_province in user_province):
        return True

    # Fallback to location string match
    if job_loc:
        return user_city in job_loc or user_province in job_loc

    return False


def _build_justification(match_details: dict, pref_details: list, demand_label: str, score: float) -> str:
    """Build a human-readable justification for a match."""
    top_prefs = [p for p in pref_details if p.get("matched", False)][:3]
    pref_text = "; ".join([
        f"{p['attribute'].replace('_', ' ').title()}: {p.get('job_value_label', 'N/A')}"
        for p in top_prefs
    ]) if top_prefs else ""

    top_skills = match_details.get("essential_skill_matches", [])[:2]
    skill_text = ". ".join([
        f"{s.get('best_user_skill_label', 'Unknown')} \u2194 {s.get('job_skill_label', 'Unknown')} (sim {s.get('similarity', 0)})"
        for s in top_skills if s.get('meets_threshold')
    ]) if top_skills else ""

    parts = []
    if skill_text:
        parts.append(f"Top skill matches: {skill_text}")
    if pref_text:
        parts.append(f"Top preference matches: {pref_text}")

    if demand_label:
        parts.append(f"Demand label: {demand_label}")

    return ". ".join(parts) if parts else f"Match Score: {round(float(score), 2)}"


def _create_score_breakdown_multiplicative(
    u_hat: float,
    p_hat_result: dict,
    skill_details: dict,
    pref_score_legacy: float,
    *,
    include_demand: bool = False,
) -> dict:
    """Interpretable score breakdown for the multiplicative (U*P) pipeline."""
    components = p_hat_result.get("components", {})
    breakdown = {
        "u_hat": round(float(u_hat), 4),
        "p_hat": round(float(p_hat_result.get("p_hat", 0.0)), 4),
        "p_hat_components": {
            "gate": round(float(components.get("gate", 0.0)), 4),
            "essential_fit": round(float(components.get("essential_fit", 0.0)), 4),
            "recruiter_readiness": round(float(components.get("recruiter_readiness", 0.0)), 4),
            "market_opportunity": round(float(components.get("market_opportunity", 0.0)), 4),
        },
        "total_skill_utility": round(float(skill_details.get("U_final", 0.0)), 4),
        "skill_components": skill_details.get("components", {"loc": 0.0, "ess": 0.0, "opt": 0.0, "grp": 0.0}),
        "skill_penalty_applied": round(float(skill_details.get("penalty", 0.0)), 4),
        "preference_score": round(float(pref_score_legacy), 4),
    }
    if include_demand:
        demand_label = p_hat_result.get("demand_label")
        breakdown["demand_score"] = round(float(DEMAND_SCORE_MAPPING.get(demand_label, 0.5)), 4) if demand_label else 0.5
        breakdown["demand_label"] = demand_label
    return breakdown


def _create_score_breakdown_additive(skill_details: dict, pref_score: float, demand_score: float, demand_label: str) -> dict:
    """Legacy additive score breakdown (kept for A/B testing)."""
    return {
        "total_skill_utility": round(float(skill_details.get("U_final", 0.0)), 4),
        "skill_components": skill_details.get("components", {"loc": 0.0, "ess": 0.0, "opt": 0.0, "grp": 0.0}),
        "skill_penalty_applied": round(float(skill_details.get("penalty", 0.0)), 4),
        "preference_score": round(float(pref_score), 4),
        "demand_score": round(float(demand_score), 4),
        "demand_label": demand_label,
    }


def _format_opportunity(item: dict, rank: int, recommendation: dict) -> dict:
    """Format a single opportunity recommendation."""
    skill_details = recommendation["skill_details"]
    match_details = recommendation["match_details"]
    scoring_mode = recommendation.get("scoring_mode", "multiplicative")
    pref_standard, wa_bws = _split_pref_details(recommendation["pref_details"])

    if scoring_mode == "multiplicative":
        breakdown = _create_score_breakdown_multiplicative(
            recommendation["u_hat"],
            recommendation["p_hat_result"],
            skill_details,
            recommendation["pref_details_score"],
        )
    else:
        breakdown = _create_score_breakdown_additive(
            skill_details,
            recommendation["pref_details_score"],
            recommendation.get("demand_score", 0.5),
            recommendation.get("demand_label", "Unknown"),
        )
    essential_matches = _filter_essential_skill_matches(match_details)

    return {
        "uuid": item.get("uuid"),
        "URL": item.get("url") or item.get("URL") or f"www.example.com/{item.get('uuid')}",
        "rank": rank,
        "opportunity_title": item.get("opportunity_title"),
        "opportunity_isco_occupation_group": item.get("opportunity_isco_occupation_group"),
        "opportunity_isco_occupation_group_id": item.get("opportunity_isco_occupation_group_id"),
        "location": item.get("location"),
        "employer": item.get("employer"),
        "employment_type": item.get("employment_type"),
        "salary_text": item.get("salary_text"),
        "required_education": item.get("required_education"),
        "required_experience": item.get("required_experience"),
        "closing_date": item.get("closing_date"),
        "is_eligible": bool(skill_details.get("is_eligible", True)),
        "justification": _build_justification(
            match_details,
            recommendation["pref_details"],
            recommendation.get("demand_label", "Unknown"),
            recommendation["score"],
        ),
        "opportunity_description": item.get("opportunity_description") or item.get("contract_type", "full_time"),
        "contract_type": item.get("contract_type"),
        "final_score": round(float(recommendation["score"]), 4),
        "score_breakdown": breakdown,
        "matched_skills": {
            "essential_skill_matches": essential_matches,
            "optional_exact_matches": match_details.get("optional_exact_matches", []),
            "skill_group_matches": match_details.get("skill_group_matches", []),
        },
        "matched_preferences": pref_standard,
        "matched_work_activities": wa_bws,
    }


def _format_opportunity_dashboard_row(item: dict, rank: int, recommendation: dict) -> dict:
    """Compact opportunity row for static comparison dashboards (embedded JSON in HTML).

    Unlike :func:`_format_opportunity`, includes *all* essential skill match rows in ``ms``,
    with ``meets_threshold`` driving the fourth tuple element (dashboard shows ✓/✗ per job skill).
    """
    skill_details = recommendation["skill_details"]
    match_details = recommendation["match_details"]
    scoring_mode = recommendation.get("scoring_mode", "multiplicative")
    _pref_standard, _wa_bws = _split_pref_details(recommendation["pref_details"])

    if scoring_mode == "multiplicative":
        breakdown = _create_score_breakdown_multiplicative(
            recommendation["u_hat"],
            recommendation["p_hat_result"],
            skill_details,
            recommendation["pref_details_score"],
        )
    else:
        breakdown = _create_score_breakdown_additive(
            skill_details,
            recommendation["pref_details_score"],
            recommendation.get("demand_score", 0.5),
            recommendation.get("demand_label", "Unknown"),
        )

    essentials = match_details.get("essential_skill_matches", []) or []
    ms: List[list] = []
    for m in essentials:
        u_lab = m.get("best_user_skill_label") or ""
        j_lab = m.get("job_skill_label") or ""
        sim = float(m.get("similarity", 0.0))
        ok = bool(m.get("meets_threshold", False))
        ms.append([u_lab, j_lab, round(sim, 4), ok])

    je = [str(m.get("job_skill_label") or "") for m in essentials]

    jo: List[str] = []
    for s in item.get("optional_skills") or []:
        if isinstance(s, dict):
            lab = (s.get("label") or "").strip()
            if lab:
                jo.append(lab)

    ph = breakdown.get("p_hat_components") or {}
    if isinstance(ph, dict):
        pg = float(ph.get("gate", 0.0))
        pe = float(ph.get("essential_fit", 0.0))
        pr = float(ph.get("recruiter_readiness", 0.0))
        pm = float(ph.get("market_opportunity", 0.0))
    else:
        pg = pe = pr = pm = 0.0

    sc = breakdown.get("skill_components") or {}
    if isinstance(sc, dict):
        sl = float(sc.get("loc", 0.0))
        se = float(sc.get("ess", 0.0))
        so = float(sc.get("opt", 0.0))
        sg = float(sc.get("grp", 0.0))
    else:
        sl = se = so = sg = 0.0

    u_hat = float(recommendation.get("u_hat", breakdown.get("u_hat", 0.5)))
    if scoring_mode == "multiplicative":
        p_hat = float(breakdown.get("p_hat", recommendation.get("p_hat_result", {}).get("p_hat", 0.0)))
    else:
        p_hat = float(recommendation.get("p_hat_result", {}).get("p_hat", 0.0))

    j_text = _build_justification(
        match_details,
        recommendation["pref_details"],
        recommendation.get("demand_label", "Unknown"),
        recommendation["score"],
    )

    url = item.get("url") or item.get("URL") or ""

    return {
        "r": rank,
        "t": item.get("opportunity_title") or "",
        "e": item.get("employer") or "",
        "l": item.get("location") or "",
        "el": bool(skill_details.get("is_eligible", True)),
        "f": round(float(recommendation["score"]), 4),
        "u": round(u_hat, 4),
        "p": round(p_hat, 4),
        "pg": round(pg, 4),
        "pe": round(pe, 4),
        "pr": round(pr, 4),
        "pm": round(pm, 4),
        "sl": round(sl, 4),
        "se": round(se, 4),
        "so": round(so, 4),
        "sg": round(sg, 4),
        "ms": ms,
        "j": j_text,
        "url": url,
        "je": je,
        "jo": jo,
    }


def _format_occupation(item: dict, rank: int, recommendation: dict) -> dict:
    """Format a single occupation recommendation."""
    skill_details = recommendation["skill_details"]
    match_details = recommendation["match_details"]
    scoring_mode = recommendation.get("scoring_mode", "multiplicative")
    pref_standard, wa_bws = _split_pref_details(recommendation["pref_details"])

    if scoring_mode == "multiplicative":
        breakdown = _create_score_breakdown_multiplicative(
            recommendation["u_hat"],
            recommendation["p_hat_result"],
            skill_details,
            recommendation["pref_details_score"],
            include_demand=True,
        )
    else:
        breakdown = _create_score_breakdown_additive(
            skill_details,
            recommendation["pref_details_score"],
            recommendation.get("demand_score", 0.5),
            recommendation.get("demand_label", "Unknown"),
        )
    essential_matches = _filter_essential_skill_matches(match_details)

    return {
        "uuid": item.get("uuid"),
        "originUuid": item.get("originUuid"),
        "rank": rank,
        "occupation_label": item.get("occupation_label") or item.get("preferredLabel"),
        "province": item.get("province"),
        "is_eligible": bool(skill_details.get("is_eligible", True)),
        "justification": _build_justification(
            match_details,
            recommendation["pref_details"],
            recommendation.get("demand_label", "Unknown"),
            recommendation["score"],
        ),
        "occupation_description": item.get("occupation_description") or item.get("description"),
        "final_score": round(float(recommendation["score"]), 4),
        "score_breakdown": breakdown,
        "matched_skills": {
            "essential_skill_matches": essential_matches,
            "optional_exact_matches": match_details.get("optional_exact_matches", []),
            "skill_group_matches": match_details.get("skill_group_matches", []),
        },
        "matched_preferences": pref_standard,
        "matched_work_activities": wa_bws,
    }


def _match_items(
    user: dict,
    items: list,
    item_type: str = "opportunity",
    top_k: int = 10,
    *,
    format_for_dashboard: bool = False,
) -> tuple[list, dict]:
    """
    Match user against items (opportunities or occupations) and return top recommendations.

    Returns (recommendations, timing_meta). timing_meta includes per-stage ms for
    u_hat (preference), skill+feasibility (embedding pair feeding p_hat), and
    p_hat (SuccessPropensityScorer) in multiplicative mode.

    Supports two scoring modes (controlled by config.SCORING_MODE):
      - "multiplicative": final_score = u_hat * p_hat  (paper-aligned)
      - "additive":       final_score = w1*S_skills + w2*S_pref + w3*S_demand  (legacy)
    """
    uid = str(user.get("user_id", "?"))
    n_in = len(items)

    t0 = time.perf_counter()
    if MATCH_APPLY_LOCATION_FILTER:
        items = [item for item in items if _job_matches_user_location(item, user)]
    # Post-secondary education gate (no-op for occupations, which lack the field).
    items = filter_jobs_by_education(user, items)
    filter_ms = _ms(t0)

    empty_timing: dict = {
        "item_type": item_type,
        "n_in": n_in,
        "n_after_filter": 0,
        "filter_ms": filter_ms,
        "n_scored": 0,
        "scoring_mode": SCORING_MODE,
        "u_hat_ms": 0.0,
        "skill_feasibility_ms": 0.0,
        "p_hat_ms": 0.0,
        "legacy_skill_ms": 0.0,
        "legacy_demand_ms": 0.0,
        "total_scoring_ms": 0.0,
        "format_recommendations_ms": 0.0,
    }

    if not items:
        log_match_step(
            "matching_service",
            "match_items (no matches after location filter)",
            user_id=uid,
            item_type=item_type,
            n_in=n_in,
            n_after_filter=0,
            filter_ms=filter_ms,
        )
        return [], empty_timing

    scoring_mode = SCORING_MODE

    # In legacy mode, import DemandScorer once (not per-item)
    _demand_scorer = None
    if scoring_mode != "multiplicative":
        from app.services.demand_score import DemandScorer
        _demand_scorer = DemandScorer()

    t_score = time.perf_counter()
    sum_u = 0.0
    sum_sk = 0.0
    sum_p = 0.0
    sum_leg_s = 0.0
    sum_leg_d = 0.0
    n_scored = 0
    # Calculate scores for all items
    recommendations: List[dict] = []
    for item in items:
        n_scored += 1
        t_u = time.perf_counter()
        pref = scorer_pref.calculate_score(user, item)
        sum_u += _ms(t_u)

        if scoring_mode == "multiplicative":
            # Single embedding / matmul pass for U + feasibility (p_hat)
            t_sk = time.perf_counter()
            skill, feasibility = scorer_skill.score_utility_and_feasibility(user, item)
            sum_sk += _ms(t_sk)
            t_p = time.perf_counter()
            p_hat_result = scorer_success.calculate_score(user, item, feasibility)
            sum_p += _ms(t_p)

            u_hat = pref.get("u_hat", 0.5)
            p_hat = p_hat_result.get("p_hat", 0.0)
            final_score = u_hat * p_hat

            recommendations.append({
                "item": item,
                "score": final_score,
                "scoring_mode": scoring_mode,
                "u_hat": u_hat,
                "p_hat_result": p_hat_result,
                "skill_details": skill,
                "match_details": feasibility.get("match_details", skill.get("match_details", {})),
                "pref_details": pref.get("details", []),
                "pref_details_score": pref.get("score", 0.0),
                "demand_label": p_hat_result.get("demand_label", "Unknown"),
            })
        else:
            # Legacy additive: skill utility only (no feasibility / p_hat)
            t_s = time.perf_counter()
            skill = scorer_skill.calculate_score(user, item)
            sum_leg_s += _ms(t_s)
            t_d = time.perf_counter()
            demand = _demand_scorer.calculate_score(item)
            sum_leg_d += _ms(t_d)

            w1 = GLOBAL_WEIGHTS["w1_skills"]
            w2 = GLOBAL_WEIGHTS["w2_preference"]
            w3 = GLOBAL_WEIGHTS["w3_market"]

            if not demand.get("present", False):
                # Demand absent — redistribute its weight to skills + prefs
                remaining = w1 + w2
                if remaining > 0:
                    w1 = w1 / remaining
                    w2 = w2 / remaining
                w3 = 0.0
                demand_score_val = 0.0
            else:
                demand_score_val = demand.get("score", 0.5)

            final_score = (
                w1 * skill.get("U_final", 0.0)
                + w2 * pref.get("score", 0.0)
                + w3 * demand_score_val
            )
            recommendations.append({
                "item": item,
                "score": final_score,
                "scoring_mode": scoring_mode,
                "u_hat": pref.get("u_hat", 0.5),
                "p_hat_result": {"p_hat": 0.0, "components": {}, "demand_label": demand.get("label", "Unknown")},
                "skill_details": skill,
                "match_details": skill.get("match_details", {}),
                "pref_details": pref.get("details", []),
                "pref_details_score": pref.get("score", 0.0),
                "demand_score": demand_score_val,
                "demand_label": demand.get("label", "Unknown"),
            })

    # Sort by final score; use demand as tie-breaker in multiplicative mode
    if scoring_mode == "multiplicative":
        recommendations.sort(
            key=lambda x: (
                x["score"],
                x["p_hat_result"].get("components", {}).get("market_opportunity", 0.0),
            ),
            reverse=True,
        )
    else:
        recommendations.sort(key=lambda x: x["score"], reverse=True)

    # Keep only top_k so we release large per-item dicts (skill_details, p_hat) for the rest
    recommendations = recommendations[:top_k]
    total_scoring_ms = _ms(t_score)

    t_fmt = time.perf_counter()
    # Format based on item type
    if item_type == "occupation":
        formatter = _format_occupation
    elif format_for_dashboard:
        formatter = _format_opportunity_dashboard_row
    else:
        formatter = _format_opportunity
    out = [formatter(r["item"], i, r) for i, r in enumerate(recommendations, 1)]
    format_recommendations_ms = _ms(t_fmt)

    timing: dict = {
        "item_type": item_type,
        "n_in": n_in,
        "n_after_filter": n_scored,
        "filter_ms": filter_ms,
        "n_scored": n_scored,
        "scoring_mode": scoring_mode,
        "u_hat_ms": round(sum_u, 2),
        "skill_feasibility_ms": round(sum_sk, 2),
        "p_hat_ms": round(sum_p, 2),
        "legacy_skill_ms": round(sum_leg_s, 2),
        "legacy_demand_ms": round(sum_leg_d, 2),
        "total_scoring_ms": round(total_scoring_ms, 2),
        "format_recommendations_ms": round(format_recommendations_ms, 2),
    }
    return out, timing


def match_user_with_data(
    user: dict,
    jobs: List[dict],
    occupations: List[dict],
) -> dict:
    """Synchronous matching using pre-fetched job and occupation lists (CPU-bound; run in a thread from async)."""
    user_id = user.get("user_id")
    if not user_id:
        raise ValueError("user must include user_id")

    uid = str(user_id)
    t_total = time.perf_counter()

    if not jobs and not occupations:
        log_match_step(
            "matching_service",
            "match_user_with_data (empty input)",
            user_id=uid,
            n_jobs=0,
            n_occ=0,
            total_ms=_ms(t_total),
        )
        return {
            "user_id": uid,
            "occupation_recommendations": [],
            "opportunity_recommendations": [],
            "skill_gap_recommendations": [],
        }

    t0 = time.perf_counter()
    opportunity_recommendations, opp_timing = _match_items(
        user, jobs, item_type="opportunity", top_k=MATCH_TOP_K_OPPORTUNITIES
    )
    t_opp = _ms(t0)

    t0 = time.perf_counter()
    occupation_recommendations, occ_timing = _match_items(
        user, occupations, item_type="occupation", top_k=MATCH_TOP_K_OCCUPATIONS
    )
    t_occ = _ms(t0)

    t0 = time.perf_counter()
    skill_gaps = analyze_skill_gaps(
        user,
        jobs,
        scorer_skill.engine,
        scorer_skill.skill_labels,
        top_k=MATCH_TOP_K_SKILL_GAPS,
        resolve_id=scorer_skill._resolve_label,
        timing_out=None,
    )
    skill_gaps = _filter_skill_gap_recommendations(skill_gaps)
    t_gaps = _ms(t0)

    total_ms = _ms(t_total)

    log_match_step(
        "matching_service",
        "match_user_with_data (phase totals vs pipeline)",
        user_id=uid,
        n_jobs=len(jobs),
        n_occupation_rows=len(occupations),
        opportunities_ms=t_opp,
        occupations_ms=t_occ,
        skill_gaps_ms=t_gaps,
        match_pipeline_total_ms=total_ms,
    )

    return {
        "user_id": uid,
        "occupation_recommendations": occupation_recommendations,
        "opportunity_recommendations": opportunity_recommendations,
        "skill_gap_recommendations": skill_gaps,
    }


def match_user_opportunities_for_dashboard(
    user: dict,
    jobs: List[dict],
    top_k: Optional[int] = None,
) -> List[dict]:
    """Run opportunity matching and return compact rows for ``comparison_dashboard``-style HTML.

    Each row matches the client's embedded schema (keys ``r``, ``t``, ``e``, ``ms``, …).
    """
    k = top_k if top_k is not None else MATCH_TOP_K_OPPORTUNITIES
    rows, _timing = _match_items(user, jobs, item_type="opportunity", top_k=k, format_for_dashboard=True)
    return rows


async def match_single_user(
    user: dict,
    jobs: Optional[List[dict]] = None,
    occupations: Optional[List[dict]] = None,
) -> dict:
    if jobs is None:
        jobs = await get_all_jobs()
    if occupations is None:
        occupations = await get_all_occupations()

    out = await asyncio.to_thread(match_user_with_data, user, jobs, occupations)
    return out
