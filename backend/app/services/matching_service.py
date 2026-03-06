from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.config import GLOBAL_WEIGHTS
from app.database import get_all_jobs, get_all_occupations
from app.services.demand_score import DemandScorer
from app.services.preference_score import PreferenceScorer
from app.services.skill_gap_analysis import analyze_skill_gaps
from app.services.skill_score import SkillScorer

# Initialize scorers once at module level
scorer_skill = SkillScorer()
scorer_pref = PreferenceScorer()
scorer_demand = DemandScorer()


def _norm(v: Optional[str]) -> str:
    return str(v).strip().casefold() if v is not None else ""

logger = logging.getLogger(__name__)

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
        f"{s.get('best_user_skill_label', 'Unknown')} ↔ {s.get('job_skill_label', 'Unknown')} (sim {s.get('similarity', 0)})"
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


def _create_score_breakdown(skill_details: dict, pref_score: float, demand_score: float, demand_label: str) -> dict:
    """Create the score breakdown dict."""
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
    match_details = skill_details.get("match_details", {})
    
    return {
        "uuid": item.get("uuid"),
        "URL": item.get("url") or item.get("URL") or f"www.example.com/{item.get('uuid')}",
        "rank": rank,
        "opportunity_title": item.get("opportunity_title"),
        "location": item.get("location"),
        "is_eligible": bool(skill_details.get("is_eligible", True)),
        "justification": _build_justification(
            match_details,
            recommendation["pref_details"],
            recommendation["demand_label"],
            recommendation["score"]
        ),
        "opportunity_description": item.get("opportunity_description") or item.get("contract_type", "full_time"),
        "contract_type": item.get("contract_type"),
        "final_score": round(float(recommendation["score"]), 4),
        "score_breakdown": _create_score_breakdown(
            skill_details,
            recommendation["pref_details_score"],
            recommendation["demand_score"],
            recommendation["demand_label"]
        ),
        "matched_skills": {
            "essential_skill_matches": match_details.get("essential_skill_matches", []),
            "optional_exact_matches": match_details.get("optional_exact_matches", []),
            "skill_group_matches": match_details.get("skill_group_matches", []),
        },
        "matched_preferences": recommendation["pref_details"],
    }


def _format_occupation(item: dict, rank: int, recommendation: dict) -> dict:
    """Format a single occupation recommendation."""
    skill_details = recommendation["skill_details"]
    match_details = skill_details.get("match_details", {})
    
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
            recommendation["demand_label"],
            recommendation["score"]
        ),
        "occupation_description": item.get("occupation_description") or item.get("description"),
        "final_score": round(float(recommendation["score"]), 4),
        "score_breakdown": _create_score_breakdown(
            skill_details,
            recommendation["pref_details_score"],
            recommendation["demand_score"],
            recommendation["demand_label"]
        ),
        "matched_skills": {
            "essential_skill_matches": match_details.get("essential_skill_matches", []),
            "optional_exact_matches": match_details.get("optional_exact_matches", []),
            "skill_group_matches": match_details.get("skill_group_matches", []),
        },
        "matched_preferences": recommendation["pref_details"],
    }


def _match_items(user: dict, items: list, item_type: str = "opportunity", top_k: int = 10):
    """
    Match user against items (opportunities or occupations) and return top recommendations.
    
    Args:
        user: User profile dict
        items: List of items to match against
        item_type: "opportunity" or "occupation"
        top_k: Number of top matches to return
    
    Returns:
        List of formatted recommendations
    """
    # Filter by location
    items = [item for item in items if _job_matches_user_location(item, user)]
    
    if not items:
        return []
    
    # Calculate scores for all items
    recommendations = []
    for item in items:
        skill = scorer_skill.calculate_score(user, item)
        pref = scorer_pref.calculate_score(user, item)
        demand = scorer_demand.calculate_score(item)
        
        final_score = (
            GLOBAL_WEIGHTS["w1_skills"] * skill.get("U_final", 0.0)
            + GLOBAL_WEIGHTS["w2_preference"] * pref.get("score", 0.0)
            + GLOBAL_WEIGHTS["w3_market"] * demand.get("score", 0.5)
        )
        
        recommendations.append({
            "item": item,
            "score": final_score,
            "skill_details": skill,
            "pref_details": pref.get("details", []),
            "pref_details_score": pref.get("score", 0.0),
            "demand_score": demand.get("score", 0.5),
            "demand_label": demand.get("label", "Unknown"),
        })
    
    # Sort and take top K
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    top = recommendations[:top_k]
    
    # Format based on item type
    formatter = _format_occupation if item_type == "occupation" else _format_opportunity
    return [formatter(r["item"], i, r) for i, r in enumerate(top, 1)]


async def match_single_user(user: dict):
    user_id = user.get("user_id")
    if not user_id:
        raise ValueError("user must include user_id")

    # Fetch all jobs and occupations from the database
    jobs = await get_all_jobs()
    logger.info(f"fetched {len(jobs)} jobs to match against")
    occupations = await get_all_occupations()
    logger.info(f"fetched {len(occupations)} occupations to match against")

    if not jobs and not occupations:
        return {
            "user_id": str(user_id),
            "occupation_recommendations": [],
            "opportunity_recommendations": [],
            "skill_gap_recommendations": [],
        }

    # Match against opportunities and occupations
    opportunity_recommendations = _match_items(user, jobs, item_type="opportunity", top_k=5)
    occupation_recommendations = _match_items(user, occupations, item_type="occupation", top_k=5)

    # Analyze skill gaps (use all jobs for better analysis)
    skill_gaps = analyze_skill_gaps(
        user,
        jobs,
        scorer_skill.engine,
        scorer_skill.skill_labels,
        top_k=5,
    )

    return {
        "user_id": str(user_id),
        "occupation_recommendations": occupation_recommendations,
        "opportunity_recommendations": opportunity_recommendations,
        "skill_gap_recommendations": skill_gaps,
    }
