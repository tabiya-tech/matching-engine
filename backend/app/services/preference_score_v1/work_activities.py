"""Part B — O*NET work activities (BWS × importance × level), mean over job activities."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.services.preference_score import PreferenceScorer


def work_activity_block(
    user_profile: dict,
    job_posting: dict,
) -> Tuple[float, Dict[str, Any]]:
    """
    S_wa = (1/N) Σ [ BWS(c) × (I_c/5) × (L_c/7) ].

    Returns (S_wa, detail dict for preference_details).
    """
    user_weights = user_profile.get("preference_vector", {}) or {}
    bws_scores = (
        user_profile.get("bws_scores")
        or user_weights.get("bws_scores")
        or {}
    )
    if PreferenceScorer.detect_bws_score_type(bws_scores) != "work_activity_id":
        return 0.0, {}

    wa_list = job_posting.get("onet_work_activities", []) or []
    if not wa_list:
        return 0.0, {}

    wa_details: List[dict] = []
    contributions: List[float] = []
    for wa in wa_list:
        wa_code = wa.get("WA_code")
        if not wa_code:
            continue
        wa_importance = float(wa.get("WA_Importance", 0) or 0)
        wa_level = float(wa.get("WA_Level", 0) or 0)
        user_bws = float(bws_scores.get(wa_code, 0.0))
        norm_importance = wa_importance / 5.0 if wa_importance else 0.0
        norm_level = wa_level / 7.0 if wa_level else 0.0
        wa_contribution = user_bws * norm_importance * norm_level
        contributions.append(wa_contribution)
        wa_details.append(
            {
                "wa_code": wa_code,
                "wa_label": str(wa.get("WA_label") or wa_code),
                "user_bws": user_bws,
                "wa_importance": wa_importance,
                "wa_level": wa_level,
                "norm_importance": round(norm_importance, 4),
                "norm_level": round(norm_level, 4),
                "wa_contribution": round(wa_contribution, 4),
            }
        )

    n = len(contributions)
    wa_score_sum = sum(contributions) / n if n else 0.0
    return wa_score_sum, {
        "attribute": "work_activity_bws",
        "wa_details": wa_details,
        "wa_score_sum": round(wa_score_sum, 4),
        "wa_aggregation": "mean",
        "n_work_activities": n,
    }
