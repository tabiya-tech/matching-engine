"""Hybrid preference scoring (PDF spec): Part A attributes + Part B work activities."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from app.config import (
    HYBRID_PREF_SCHEMA_PATH,
    HYBRID_PREF_SIGMOID_FACTOR,
    HYBRID_PREF_VIGNETTES_FOR_FULL_CONFIDENCE,
)

from .levels import (
    attribute_label,
    attribute_orientation,
    job_level_to_vj,
    level_label,
    load_attribute_schema,
    resolve_schema_level_id,
)
from .work_activities import work_activity_block


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _user_weight(user_profile: dict, attr_name: str) -> float:
    attrs = user_profile.get("attributes") or {}
    entry = attrs.get(attr_name)
    if isinstance(entry, dict) and entry.get("importance") is not None:
        try:
            return _clamp01(float(entry["importance"]))
        except (TypeError, ValueError):
            pass
    pv = user_profile.get("preference_vector", {}) or {}
    try:
        return _clamp01(float(pv.get(attr_name, 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _confidence_f(user_profile: dict) -> float:
    """f from explicit confidence or vignette count / full-confidence threshold."""
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
    n_vig = user_profile.get("vignette_count") or pv.get("vignette_count")
    if n_vig is not None:
        try:
            full = max(1, int(HYBRID_PREF_VIGNETTES_FOR_FULL_CONFIDENCE))
            return _clamp01(float(n_vig) / full)
        except (TypeError, ValueError):
            pass
    return 1.0


class HybridPreferenceScorer:
    """Part A: V = Σ(wᵢ×Vⱼ)/Σwᵢ, S_attrs = V×f. Part B: mean BWS. u_hat = σ(raw×k)."""

    def __init__(self, schema_path: Optional[str] = None):
        path = schema_path or (HYBRID_PREF_SCHEMA_PATH or None)
        self._schema = load_attribute_schema(path)
        self._attr_names = [a["name"] for a in self._schema.get("attributes", [])]
        self._sigmoid_factor = float(HYBRID_PREF_SIGMOID_FACTOR)

    def calculate_score(
        self,
        user_profile: dict,
        job_posting: dict,
        *,
        include_work_activities: bool = True,
    ) -> dict:
        job_attrs = job_posting.get("attributes", {}) or {}
        details: List[dict] = []

        weighted_sum = 0.0
        weight_sum = 0.0
        for attr_name in self._attr_names:
            wi = _user_weight(user_profile, attr_name)
            if wi <= 0.0:
                continue
            level_raw = job_attrs.get(attr_name)
            if level_raw is None or not str(level_raw).strip():
                continue
            level_raw_s = str(level_raw).strip()
            resolved = resolve_schema_level_id(attr_name, level_raw_s, self._schema)
            if resolved is None:
                continue
            vj = job_level_to_vj(attr_name, level_raw_s, self._schema)
            term = wi * vj
            weighted_sum += term
            weight_sum += wi
            details.append(
                {
                    "attribute": attr_name,
                    "attr_label": attribute_label(attr_name, self._schema),
                    "job_value": level_raw_s,
                    "job_level_resolved": resolved,
                    "job_level_label": level_label(attr_name, level_raw_s, self._schema),
                    "user_weight": round(wi, 4),
                    "orientation": attribute_orientation(attr_name),
                    "encoded_value": round(vj, 4),
                    "contribution": round(term, 4),
                    "matched": vj > 0,
                    "on_job": True,
                    "layer": "dce_attributes",
                }
            )

        v_match = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        f = _confidence_f(user_profile)
        s_attrs = _clamp01(v_match * f)
        details.append(
            {
                "attribute": "dce_utility",
                "V": round(v_match, 4),
                "f": round(f, 4),
                "S_attrs": round(s_attrs, 4),
                "layer": "dce_attributes",
            }
        )

        if include_work_activities:
            s_wa, wa_detail = work_activity_block(user_profile, job_posting)
            s_wa = _clamp01(s_wa)
            if wa_detail:
                details.append(wa_detail)
        else:
            s_wa = 0.0

        raw = s_attrs + s_wa
        sigmoid_input = raw * self._sigmoid_factor
        if abs(sigmoid_input) >= 500:
            u_hat = 1.0 if sigmoid_input > 0 else 0.0
        else:
            u_hat = 1.0 / (1.0 + math.exp(-sigmoid_input))

        bws_scores = (
            user_profile.get("bws_scores")
            or (user_profile.get("preference_vector") or {}).get("bws_scores")
            or {}
        )
        from app.services.preference_score import PreferenceScorer

        return {
            "u_hat": round(u_hat, 4),
            "score": round(u_hat, 4),
            "details": details,
            "S_attrs": round(s_attrs, 4),
            "S_wa": round(s_wa, 4),
            "raw": round(raw, 4),
            "bws_scores": bws_scores,
            "bws_score_type": PreferenceScorer.detect_bws_score_type(bws_scores),
            "top_10_bws": (
                user_profile.get("top_10_bws")
                or (user_profile.get("preference_vector") or {}).get("top_10_bws")
                or []
            ),
            "scoring_model": "hybrid_preference_v1",
        }
