from __future__ import annotations

from typing import Any, Dict, Optional

from app.config import GLOBAL_WEIGHTS
from app.database import get_all_jobs
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


async def match_single_user(user: dict):
    user_id = user.get("user_id")
    if not user_id:
        raise ValueError("user must include user_id")

    # Step 1: Fetch all jobs from the database
    jobs = await get_all_jobs()

    if not jobs:
        return {
            "user_id": str(user_id),
            "opportunity_recommendations": [],
            "skill_gap_recommendations": [],
        }

    # Filter by location
    jobs = [job for job in jobs if _job_matches_user_location(job, user)]

    recommendations = []

    for job in jobs:
        skill = scorer_skill.calculate_score(user, job)
        pref = scorer_pref.calculate_score(user, job)
        demand = scorer_demand.calculate_score(job)

        final_score = (
            GLOBAL_WEIGHTS["w1_skills"] * skill.get("U_final", 0.0)
            + GLOBAL_WEIGHTS["w2_preference"] * pref.get("score", 0.0)
            + GLOBAL_WEIGHTS["w3_market"] * demand.get("score", 0.5)
        )

        recommendations.append(
            {
                "job": job,
                "score": final_score,
                "skill_details": skill,
                "pref_details": pref.get("details", []),
                "pref_details_score": pref.get("score", 0.0),
                "demand_score": demand.get("score", 0.5),
                "demand_label": demand.get("label"),
            }
        )

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    top = recommendations[:10]

    formatted = []
    for i, r in enumerate(top, 1):
        skill_details = r["skill_details"]
        formatted.append(
            {
                "uuid": r["job"].get("uuid"),
                "originUuid": r["job"].get("originUuid"),
                "rank": i,
                "opportunity_title": r["job"].get("opportunity_title"),
                "location": r["job"].get("location"),
                "is_eligible": bool(skill_details.get("is_eligible", True)),
                "justification": f"Match Score: {round(float(r['score']), 2)}",
                "contract_type": r["job"].get("contract_type"),
                "final_score": round(float(r["score"]), 4),
                "score_breakdown": {
                    "total_skill_utility": round(float(skill_details.get("U_final", 0.0)), 4),
                    "skill_components": skill_details.get("components", {}),
                    "skill_penalty_applied": round(float(skill_details.get("penalty", 0.0)), 4),
                    "preference_score": round(float(r.get("pref_details_score", 0.0)), 4),
                    "demand_score": round(float(r.get("demand_score", 0.0)), 4),
                },
            }
        )

    skill_gaps = analyze_skill_gaps(
        user,
        jobs,
        scorer_skill.engine,
        scorer_skill.skill_labels,
        top_k=5,
    )

    return {
        "user_id": str(user_id),
        "opportunity_recommendations": formatted,
        "skill_gap_recommendations": skill_gaps,
    }
