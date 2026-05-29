"""Part B — O*NET work activities (BWS task utility) and the additive-RUM combination.

Shared by the preference scorers (``UnifiedPreferenceScorer`` and legacy ``PreferenceScorer``).

Additive-RUM integration (``BWS_INTEGRATION_MODE="additive_rum"``):
    V_task = Σ_c ŵ_c · β_c,  ŵ_c ∝ WA_Importance, Σŵ_c = 1 over the job's activities.
    β_c = bws_scores.get(WA_code, 0.0)  (HB posterior part-worth, ~[-2,2], 0 = neutral).
    u_hat = logistic( γ · [ α·Ṽ_task + (1-α)·Ṽ_dce ] ), each component harmonised to [-1,1].

``work_activity_block`` is the legacy (mean of BWS×Imp×Level) path, kept for
``BWS_INTEGRATION_MODE="legacy"``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from app.services.preference_score import PreferenceScorer
from .levels import attribute_label, ladder_position, level_label, resolve_schema_level_id


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _pref_value(user_profile: dict, attr_name: str) -> Optional[float]:
    """Per-attribute DCE value v_k = sigmoid(beta_k) in [0,1]; None if not supplied."""
    pv = user_profile.get("preference_vector", {}) or {}
    for src in (pv, user_profile):
        if attr_name in src:
            try:
                return float(src[attr_name])
            except (TypeError, ValueError):
                return None
    return None


def compute_dce_utility(
    user_profile: dict,
    job_posting: dict,
    schema: dict,
    *,
    logit_eps: float = 0.01,
    attr_scale: Optional[Dict[str, float]] = None,
    confidence: float = 1.0,
) -> Tuple[float, float, Dict[str, Any]]:
    """DCE-attribute utility from per-user betas recovered from the [0,1] contract.

    For each schema attribute with a user value v_k and a job level:
        beta_hat = logit(clamp(v_k, eps, 1-eps)) * scale_k     (v=0.5 -> 0)
        v_tilde  = ladder_position(reference->target) in [0,1]  (reference job -> 0)
        contribution = beta_hat * v_tilde
    V_dce = sum(contribution);  D = sum(|beta_hat|).
    V_dce_hat = confidence * clamp(V_dce / D, -1, 1)   (0 when D == 0).

    Returns (V_dce, V_dce_hat, detail dict). Direction comes from the sign of beta_hat,
    so no gain/cost orientation is applied; the schema ordering fixes each attribute's
    reference (0) and target (1).
    """
    scale = attr_scale or {}
    attr_names = [a.get("name") for a in (schema.get("attributes") or []) if a.get("name")]
    spec_by_name = {a["name"]: a for a in (schema.get("attributes") or []) if a.get("name")}

    v_dce = 0.0
    denom = 0.0
    rows: List[dict] = []
    for attr in attr_names:
        v = _pref_value(user_profile, attr)
        if v is None:
            continue
        raw_level = (job_posting.get("attributes", {}) or {}).get(attr)
        resolved = resolve_schema_level_id(attr, raw_level, schema)
        if resolved is None:
            continue  # job doesn't describe this attribute

        beta_hat = math.log(_clamp(v, logit_eps, 1.0 - logit_eps) / (1.0 - _clamp(v, logit_eps, 1.0 - logit_eps)))
        beta_hat *= float(scale.get(attr, 1.0))
        v_tilde = ladder_position(attr, raw_level, schema)
        contribution = beta_hat * v_tilde

        v_dce += contribution
        denom += abs(beta_hat)

        spec = spec_by_name.get(attr, {})
        levels = [lv["id"] for lv in spec.get("levels", [])]
        # MatchedPreference-compatible field names (attribute, job_value, user_weight,
        # beta, encoded_value, contribution, matched) so these rows flow straight into
        # the /match response and the /match_v4 dashboard (layer/on_job) unchanged.
        rows.append(
            {
                "attribute": attr,
                "attr_label": attribute_label(attr, schema),
                "job_value": resolved,
                "job_value_label": level_label(attr, raw_level, schema),  # level label, e.g. "~70k"
                "user_weight": round(v, 4),          # per-user [0,1] preference value v_k
                "beta": round(beta_hat, 4),          # recovered signed coefficient β̂_k
                "encoded_value": round(v_tilde, 4),  # graded ladder position ṽ_k
                "contribution": round(contribution, 4),
                "matched": contribution != 0.0,
                "reference_level": levels[0] if levels else None,
                "target_level": levels[-1] if levels else None,
                "on_job": True,
                "layer": "dce_attributes",
            }
        )

    v_dce_hat = (confidence * _clamp(v_dce / denom, -1.0, 1.0)) if denom > 0 else 0.0

    detail = {
        "attribute": "dce_utility",
        "dce_details": rows,
        "V_dce": round(v_dce, 4),
        "V_dce_hat": round(v_dce_hat, 4),
        "confidence_f": round(float(confidence), 4),
        "n_attributes": len(rows),
        "layer": "dce_attributes",
    }
    return v_dce, v_dce_hat, detail


def _bws_scores(user_profile: dict) -> dict:
    user_weights = user_profile.get("preference_vector", {}) or {}
    return user_profile.get("bws_scores") or user_weights.get("bws_scores") or {}


def compute_task_utility(
    user_profile: dict,
    job_posting: dict,
) -> Tuple[float, Dict[str, Any]]:
    """Importance-weighted BWS task utility.

    ``V_task = Σ_c ŵ_c · β_c`` with ``ŵ_c = WA_Importance_c / Σ WA_Importance`` (sum to 1).
    Level is intentionally NOT used (it is a skill-demand, not task-preference, signal).

    Returns ``(V_task ∈ [-2, 2], detail_dict)``; ``(0.0, {})`` when bws are not work-activity
    ids, the job has no activities, or total importance is zero.
    """
    bws_scores = _bws_scores(user_profile)
    if PreferenceScorer.detect_bws_score_type(bws_scores) != "work_activity_id":
        return 0.0, {}

    wa_list = job_posting.get("onet_work_activities", []) or []
    if not wa_list:
        return 0.0, {}

    # First pass: collect importance weights (the normaliser).
    rows: List[Dict[str, Any]] = []
    total_importance = 0.0
    for wa in wa_list:
        wa_code = wa.get("WA_code")
        if not wa_code:
            continue
        importance = float(wa.get("WA_Importance", 0) or 0)
        if importance <= 0:
            continue
        rows.append(
            {
                "wa_code": wa_code,
                "wa_label": str(wa.get("WA_label") or wa_code),
                "importance": importance,
                "wa_level": float(wa.get("WA_Level", 0) or 0),
                "beta": float(bws_scores.get(wa_code, 0.0)),
            }
        )
        total_importance += importance

    if total_importance <= 0 or not rows:
        return 0.0, {}

    v_task = 0.0
    wa_details: List[dict] = []
    for r in rows:
        weight = r["importance"] / total_importance  # ŵ_c, Σ = 1
        contribution = weight * r["beta"]
        v_task += contribution
        wa_details.append(
            {
                "wa_code": r["wa_code"],
                "wa_label": r["wa_label"],
                "user_bws": r["beta"],
                "wa_importance": r["importance"],
                "wa_level": r["wa_level"],
                "norm_importance": round(r["importance"] / 5.0, 4),  # display only
                "norm_level": round(r["wa_level"] / 7.0, 4),         # display only (unused in score)
                "weight": round(weight, 6),                          # ŵ_c (drives V_task)
                "beta": round(r["beta"], 4),                         # β_c
                "wa_contribution": round(contribution, 6),           # ŵ_c · β_c
            }
        )

    detail = {
        "attribute": "work_activity_bws",
        "wa_details": wa_details,
        "wa_score_sum": round(v_task, 4),  # == V_task
        "wa_aggregation": "importance_weighted",
        "n_work_activities": len(wa_details),
        "V_task": round(v_task, 4),
        "V_task_hat": round(_clamp(v_task / 2.0, -1.0, 1.0), 4),
    }
    return v_task, detail


def combine_utilities(
    v_task: float,
    v_dce: float,
    dce_normalizer: float,
    *,
    alpha: float,
    gamma: float,
    v_dce_already_harmonized: bool = False,
) -> Dict[str, Any]:
    """Additive-RUM combination of the task and DCE-attribute utilities into ``u_hat``.

    ``Ṽ_task = clamp(v_task/2, -1, 1)`` (β∈[-2,2], Σŵ=1 ⇒ v_task∈[-2,2]).
    ``Ṽ_dce``  = ``v_dce`` (if already harmonised to [-1,1]) else ``clamp(v_dce/dce_normalizer, -1, 1)``.
    ``V = γ·[α·Ṽ_task + (1-α)·Ṽ_dce]``;  ``u_hat = logistic(V)``.
    """
    v_task_h = _clamp(v_task / 2.0, -1.0, 1.0)
    if v_dce_already_harmonized:
        v_dce_h = _clamp(v_dce, -1.0, 1.0)
    elif dce_normalizer and dce_normalizer > 0:
        v_dce_h = _clamp(v_dce / dce_normalizer, -1.0, 1.0)
    else:
        v_dce_h = 0.0

    v = gamma * (alpha * v_task_h + (1.0 - alpha) * v_dce_h)
    if v >= 500:
        u_hat = 1.0
    elif v <= -500:
        u_hat = 0.0
    else:
        u_hat = 1.0 / (1.0 + math.exp(-v))

    return {
        "v_task": round(v_task, 4),
        "v_task_h": round(v_task_h, 4),
        "v_dce_h": round(v_dce_h, 4),
        "alpha": round(float(alpha), 4),
        "gamma": round(float(gamma), 4),
        "V": round(v, 4),
        "u_hat": round(u_hat, 4),
    }


def work_activity_block(
    user_profile: dict,
    job_posting: dict,
) -> Tuple[float, Dict[str, Any]]:
    """LEGACY (``BWS_INTEGRATION_MODE="legacy"``): S_wa = (1/N) Σ [ BWS(c) × (I_c/5) × (L_c/7) ].

    Returns (S_wa, detail dict for preference_details).
    """
    bws_scores = _bws_scores(user_profile)
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
