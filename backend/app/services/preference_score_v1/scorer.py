"""Unified preference scorer: DCE attribute utility + BWS task utility → additive-RUM u_hat.

Part A (DCE attributes): V_dce = Σ_k β̂_k · ṽ_k, with β̂_k recovered from the per-user
[0,1] contract via logit, and ṽ_k the graded ladder position (reference→target) from the
committed schema. See work_activities.compute_dce_utility.

Part B (BWS work activities): importance-weighted V_task (work_activities.compute_task_utility).

Combination: u_hat = logistic(γ·[α·Ṽ_task + (1-α)·Ṽ_dce]) (work_activities.combine_utilities).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from app.config import (
    BWS_ALPHA,
    BWS_GAIN_GAMMA,
    DCE_ATTR_SCALE,
    DCE_LOGIT_EPS,
    HYBRID_PREF_SCHEMA_PATH,
    HYBRID_PREF_VIGNETTES_FOR_FULL_CONFIDENCE,
)

from .levels import load_attribute_schema
from .work_activities import combine_utilities, compute_dce_utility, compute_task_utility

logger = logging.getLogger(__name__)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _confidence_f(user_profile: dict) -> float:
    """f ∈ [0,1] from explicit confidence or vignette count / full-confidence threshold.

    Defaults to 1.0 (no shrinkage) when the request carries no confidence signal.
    """
    for key in ("preference_confidence", "confidence", "pref_confidence"):
        v = user_profile.get(key)
        if v is not None:
            try:
                return _clamp01(float(v))
            except (TypeError, ValueError):
                pass
    pv = user_profile.get("preference_vector", {}) or {}
    for key in ("preference_confidence", "confidence"):
        v = pv.get(key)
        if v is not None:
            try:
                return _clamp01(float(v))
            except (TypeError, ValueError):
                pass
    n_vig = (
        user_profile.get("vignette_count")
        or user_profile.get("n_vignettes_completed")
        or pv.get("vignette_count")
        or pv.get("n_vignettes_completed")
    )
    if n_vig is not None:
        try:
            full = max(1, int(HYBRID_PREF_VIGNETTES_FOR_FULL_CONFIDENCE))
            return _clamp01(float(n_vig) / full)
        except (TypeError, ValueError):
            pass
    return 1.0


class UnifiedPreferenceScorer:
    """DCE-attribute utility + BWS task utility, combined via additive-RUM into u_hat."""

    def __init__(self, schema_path: Optional[str] = None):
        path = schema_path or (HYBRID_PREF_SCHEMA_PATH or None)
        try:
            self._schema = load_attribute_schema(path)
        except (OSError, ValueError) as e:
            # Never let a missing/invalid schema take down the whole matcher (this scorer is
            # built at matching_service import time). Degrade to BWS-only and log loudly.
            logger.error(
                "UnifiedPreferenceScorer: could not load attribute schema (%s). DCE attribute "
                "term DISABLED — scoring on BWS only. Commit/deploy job_attributes_schema.json "
                "(or set HYBRID_PREF_SCHEMA_PATH) to restore.",
                e,
            )
            self._schema = {"attributes": []}

    def calculate_score(
        self,
        user_profile: dict,
        job_posting: dict,
        *,
        include_work_activities: bool = True,
    ) -> dict:
        details: List[dict] = []

        # Part A — DCE attributes (signed, from logit-recovered per-user betas).
        f = _confidence_f(user_profile)
        _, v_dce_hat, dce_detail = compute_dce_utility(
            user_profile,
            job_posting,
            self._schema,
            logit_eps=DCE_LOGIT_EPS,
            attr_scale=DCE_ATTR_SCALE,
            confidence=f,
        )
        # Flat per-attribute rows (MatchedPreference-shaped) → matched_preferences / dashboard.
        details.extend(dce_detail["dce_details"])

        # Part B — BWS work activities (importance-weighted task utility).
        if include_work_activities:
            v_task, wa_detail = compute_task_utility(user_profile, job_posting)
            if wa_detail:
                details.append(wa_detail)
        else:
            v_task = 0.0

        # Combine (additive-RUM). v_dce_hat is already harmonised to [-1,1].
        comb = combine_utilities(
            v_task,
            v_dce_hat,
            1.0,
            alpha=BWS_ALPHA,
            gamma=BWS_GAIN_GAMMA,
            v_dce_already_harmonized=True,
        )

        bws_scores = (
            user_profile.get("bws_scores")
            or (user_profile.get("preference_vector") or {}).get("bws_scores")
            or {}
        )
        from app.services.preference_score import PreferenceScorer

        return {
            "u_hat": comb["u_hat"],
            "score": comb["u_hat"],
            "details": details,
            "S_attrs": round(v_dce_hat, 4),       # harmonised DCE utility ∈ [-1,1]
            "S_wa": comb["v_task_h"],             # harmonised task utility ∈ [-1,1]
            "raw": comb["V"],
            "V": comb["V"],
            "V_task": comb["v_task"],
            "V_task_hat": comb["v_task_h"],
            "V_dce": dce_detail["V_dce"],
            "V_dce_hat": comb["v_dce_h"],
            "confidence_f": round(f, 4),
            "alpha": comb["alpha"],
            "gamma": comb["gamma"],
            "bws_scores": bws_scores,
            "bws_score_type": PreferenceScorer.detect_bws_score_type(bws_scores),
            "top_10_bws": (
                user_profile.get("top_10_bws")
                or (user_profile.get("preference_vector") or {}).get("top_10_bws")
                or []
            ),
            "scoring_model": "unified_dce_bws_v1",
        }
