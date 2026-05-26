"""Ranker: combine u_hat (preferences) and p_hat (skill retrieval score)."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Tuple

from app.services.preference_score import PreferenceScorer
from app.config import FINAL_SCORE_COMBINER
from app.services.preference_score_v1.final_score import combine_final_score
from app.services.preference_score_v1.levels import attribute_label, load_attribute_schema


def user_preference_factors(user: Dict[str, Any]) -> List[Dict[str, Any]]:
    """User importance weights for dashboard (sorted high → low)."""
    schema = load_attribute_schema()
    pv = user.get("preference_vector") or {}
    rows: List[Dict[str, Any]] = []
    for spec in schema.get("attributes", []):
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name") or "")
        if not name:
            continue
        try:
            w = float(pv.get(name, 0.0))
        except (TypeError, ValueError):
            w = 0.0
        rows.append(
            {
                "attribute": name,
                "label": attribute_label(name, schema),
                "importance": round(max(0.0, min(1.0, w)), 4),
            }
        )
    rows.sort(key=lambda r: (-float(r["importance"]), r["label"]))
    return rows


def user_bws_summary(user: Dict[str, Any]) -> Dict[str, Any]:
    """Top BWS scores on work-activity codes for dashboard header."""
    pv = user.get("preference_vector") or {}
    bws_scores = user.get("bws_scores") or pv.get("bws_scores") or {}
    if not isinstance(bws_scores, dict) or not bws_scores:
        return {"has_bws": False, "score_type": None, "rows": []}

    score_type = PreferenceScorer.detect_bws_score_type(bws_scores)
    top_codes = list(user.get("top_10_bws") or pv.get("top_10_bws") or [])

    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _add(code: str) -> None:
        c = str(code).strip()
        if not c or c in seen:
            return
        seen.add(c)
        try:
            score = float(bws_scores.get(c, 0.0))
        except (TypeError, ValueError):
            score = 0.0
        rows.append({"wa_code": c, "bws": round(score, 2)})

    for code in top_codes[:10]:
        _add(code)
    for code, val in sorted(
        bws_scores.items(),
        key=lambda kv: (-abs(float(kv[1]) if kv[1] is not None else 0), str(kv[0])),
    ):
        if len(rows) >= 12:
            break
        _add(str(code))

    return {
        "has_bws": score_type == "work_activity_id",
        "score_type": score_type,
        "rows": rows,
    }


def work_activity_match_for_dashboard(details: Any) -> Dict[str, Any]:
    """Extract Part B (BWS × job work activities) for one job row."""
    for d in details or []:
        if not isinstance(d, dict) or d.get("attribute") != "work_activity_bws":
            continue
        wa_rows = []
        for w in d.get("wa_details") or []:
            if not isinstance(w, dict):
                continue
            wa_rows.append(
                {
                    "wa_code": w.get("wa_code"),
                    "wa_label": w.get("wa_label") or w.get("wa_code"),
                    "user_bws": w.get("user_bws"),
                    "wa_importance": w.get("wa_importance"),
                    "wa_level": w.get("wa_level"),
                    "wa_contribution": w.get("wa_contribution"),
                }
            )
        wa_rows.sort(
            key=lambda r: (-abs(float(r.get("wa_contribution") or 0)), str(r.get("wa_code") or ""))
        )
        return {
            "S_wa": d.get("wa_score_sum"),
            "n_work_activities": d.get("n_work_activities") or len(wa_rows),
            "wa_aggregation": d.get("wa_aggregation") or "mean",
            "rows": wa_rows[:25],
        }
    return {"S_wa": None, "n_work_activities": 0, "wa_aggregation": "mean", "rows": []}


def preference_details_for_dashboard(details: Any) -> List[Dict[str, Any]]:
    """Compact attribute-level preference rows for HTML (user weight vs job level)."""
    out: List[Dict[str, Any]] = []
    for d in details or []:
        if not isinstance(d, dict):
            continue
        if d.get("layer") != "dce_attributes":
            continue
        attr = d.get("attribute")
        if not attr or attr in ("dce_utility",):
            continue
        if d.get("on_job") is False:
            continue
        out.append(
            {
                "attribute": attr,
                "attr_label": d.get("attr_label") or attr,
                "user_weight": d.get("user_weight"),
                "job_value": d.get("job_value"),
                "job_level_resolved": d.get("job_level_resolved"),
                "job_level_label": d.get("job_level_label"),
                "encoded_value": d.get("encoded_value"),
                "contribution": d.get("contribution"),
                "orientation": d.get("orientation"),
            }
        )
    out.sort(key=lambda r: (-abs(float(r.get("contribution") or 0)), r["attr_label"]))
    return out


def p_hat_from_skill_rec(rec: Dict[str, Any]) -> Tuple[float, str]:
    """
    p_hat = raw cosine similarity from match_v3 stage-1 (not CE min–max score).

    ``cross_encoder_score`` is only used for CE *ordering*; it is per-user min–max
    in [0, 1] so the top row is often 1.0 and must not be used as p_hat.
    """
    cos = rec.get("concat_cosine_similarity")
    if cos is not None:
        return float(cos), "concat_cosine_similarity"
    mbc = rec.get("mean_best_cosine")
    if mbc is not None:
        return float(mbc), "mean_best_cosine"
    return 0.0, "mean_best_cosine"


def compute_final_score(
    pref_result: Dict[str, Any],
    p_hat: float,
    *,
    p_hat_source: str,
    combiner: str,
) -> Tuple[float, Dict[str, Any]]:
    """Compute final score from u_hat and p_hat (configurable)."""
    u_hat = float(pref_result.get("u_hat", 0.5))
    final = combine_final_score(u_hat, p_hat, combiner=combiner)  # type: ignore[arg-type]

    breakdown = {
        "scoring_mode": "multiplicative",
        "final_score_combiner": combiner,
        "u_hat": round(u_hat, 4),
        "p_hat": round(p_hat, 4),
        "p_hat_source": p_hat_source,
        "final_score": round(final, 4),
        "preference_score_legacy": round(float(pref_result.get("score", 0.0)), 4),
    }
    return final, breakdown


def enrich_recommendations_with_preferences(
    user: Dict[str, Any],
    recs: list[Dict[str, Any]],
    jobs_by_uuid: Dict[str, Dict[str, Any]],
    *,
    preference_scorer,
    include_work_activities: bool = True,
    final_score_combiner: str | None = None,
) -> list[Dict[str, Any]]:
    """
    Stage 3 only: compute u_hat per job, p_hat from stage 1–2 cosine, re-rank by u_hat × p_hat.

    Input ``recs`` must stay in CE order for ``cross_encoder_recommendations`` export;
    this function returns a new list sorted by ``final_score``.
    """
    scored: list[tuple[float, float, int, Dict[str, Any]]] = []

    for i, rec in enumerate(recs):
        uid = str(rec.get("job_uuid") or "")
        job = jobs_by_uuid.get(uid)
        if job is None:
            continue

        calc = preference_scorer.calculate_score
        if "include_work_activities" in inspect.signature(calc).parameters:
            pref = calc(user, job, include_work_activities=include_work_activities)
        else:
            pref = calc(user, job)
        p_hat, p_hat_source = p_hat_from_skill_rec(rec)
        combiner = (final_score_combiner or FINAL_SCORE_COMBINER).strip().lower()
        final, breakdown = compute_final_score(
            pref, p_hat, p_hat_source=p_hat_source, combiner=combiner
        )

        row = dict(rec)
        row["rank_cross_encoder"] = rec.get("rank")
        row["u_hat"] = pref.get("u_hat")
        row["preference_score"] = pref.get("score")
        row["preference_details"] = pref.get("details", [])
        row["preference_match_rows"] = preference_details_for_dashboard(pref.get("details"))
        row["work_activity_match"] = work_activity_match_for_dashboard(pref.get("details"))
        row["S_attrs"] = pref.get("S_attrs")
        row["S_wa"] = pref.get("S_wa")
        row["preference_include_work_activities"] = include_work_activities
        row["p_hat"] = round(p_hat, 4)
        row["p_hat_source"] = p_hat_source
        row["score_breakdown"] = breakdown
        row["final_score"] = breakdown["final_score"]
        scored.append((final, p_hat, i, row))

    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    out: list[Dict[str, Any]] = []
    for rank, (_fs, _ph, _i, row) in enumerate(scored, start=1):
        row["rank"] = rank
        out.append(row)
    return out
