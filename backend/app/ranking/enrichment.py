"""Ranker: combine u_hat (preferences) and p_hat (skill retrieval score)."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple

from app.ranking.demand import DemandScorer
from app.config import FINAL_SCORE_COMBINER, V4_FULL_COVERAGE_FLOOR
from app.ranking.preference.final_score import combine_final_score
from app.ranking.concat_embedding_text import job_skill_labels_for_concat
from app.ranking.location import job_matches_user_location

# Engine-agnostic demand scorer (reads item attributes["expected_demand"]); torch-free.
_DEMAND_SCORER = DemandScorer()


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
            key=lambda r: (
                -abs(float(r.get("wa_contribution") or 0)),
                str(r.get("wa_code") or ""),
            )
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
    include_demand: bool = False,
    demand_gamma: float = 0.0,
    p_hat_by_uuid: Optional[Dict[str, float]] = None,
    coverage_by_uuid: Optional[Dict[str, float]] = None,
    coverage_gamma: float = 0.0,
    location_tier_by_uuid: Optional[Dict[str, float]] = None,
) -> list[Dict[str, Any]]:
    """
    Stage 3 only: compute u_hat per job, p_hat from stage 1–2 cosine, re-rank by u_hat × p_hat.

    Input ``recs`` must stay in CE order for ``cross_encoder_recommendations`` export;
    this function returns a new list sorted by ``final_score``.

    ``include_demand`` (with ``demand_gamma`` > 0) applies an occupation-only labour-market tilt:
    ``final *= M ** demand_gamma`` where M = expected-demand score in [0,1] (neutral 1.0 when the
    item has no/unknown demand). Opportunities call with ``include_demand=False`` (the default),
    so their ranking is unaffected.
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
        # Phase-2 (V4_FULL_RANK_DEMOTE): override p_hat's skills-fit with the WHITENED+rescaled concat
        # cosine (de-anisotropised, discriminative). Still in [0,1]; final = u_hat x p_hat unchanged.
        if p_hat_by_uuid is not None and uid in p_hat_by_uuid:
            p_hat = max(0.0, min(1.0, float(p_hat_by_uuid[uid])))
            p_hat_source = "concat_cosine_whitened"
        combiner = (final_score_combiner or FINAL_SCORE_COMBINER).strip().lower()
        final, breakdown = compute_final_score(
            pref, p_hat, p_hat_source=p_hat_source, combiner=combiner
        )

        # Occupation-only demand tilt (opportunities pass include_demand=False): multiply the
        # final score by M**gamma — M = expected-demand score in [0,1], neutral 1.0 when the item
        # has no/unknown demand. Mirrors the legacy p_hat market factor; this re-ranks occupations.
        if include_demand and demand_gamma > 0:
            dres = _DEMAND_SCORER.calculate_score(job)
            if dres.get("present"):
                m = max(0.0, min(1.0, float(dres.get("score") or 0.0)))
                final = final * (m**demand_gamma)
                breakdown["final_score"] = round(final, 4)
                breakdown["demand_score"] = round(m, 4)
                breakdown["demand_label"] = dres.get("label")
                breakdown["demand_gamma"] = round(float(demand_gamma), 4)

        # Phase-2 achievability demotion: final *= floor + (1-floor) * essential_coverage**gamma (all
        # factors in [0,1], so final stays in range and the combination u_hat x p_hat is preserved —
        # coverage is just an extra [0,1] factor of p_hat). Demotes high-fit-but-missing-must-haves
        # items (woodworker x dev). The floor keeps a coverage of 0 a demotion rather than an
        # annihilation: without it every zero-coverage candidate ties at final_score 0.0, which drops
        # the u_hat x p_hat ordering and reports a non-zero breakdown beside a 0.0 score.
        if coverage_by_uuid is not None and coverage_gamma > 0:
            cov = coverage_by_uuid.get(uid)
            if cov is not None:
                cov = max(0.0, min(1.0, float(cov)))
                floor = max(0.0, min(1.0, V4_FULL_COVERAGE_FLOOR))
                factor = floor + (1.0 - floor) * cov**coverage_gamma
                final = final * factor
                breakdown["final_score"] = round(final, 4)
                breakdown["essential_coverage"] = round(cov, 4)
                breakdown["coverage_gamma"] = round(float(coverage_gamma), 4)
                breakdown["coverage_floor"] = round(floor, 4)
                breakdown["coverage_factor"] = round(factor, 4)

        # Urban-pull location tier (flat multiply): final *= tier, where tier in {1.0 local,
        # W_REGIONAL, W_NATIONAL, 0.0 off-chain}. The tier values are themselves the tuning knobs, so
        # no exponent. Local jobs preferred; off-chain (e.g. another batch user's locations) is an
        # exclusion, not a demotion — drop the candidate rather than return it scored 0.0.
        if location_tier_by_uuid is not None:
            tier = location_tier_by_uuid.get(uid)
            if tier is not None:
                tier = max(0.0, min(1.0, float(tier)))
                if tier <= 0.0:
                    continue
                final = final * tier
                breakdown["final_score"] = round(final, 4)
                breakdown["location_tier_factor"] = round(tier, 4)

        row = dict(rec)
        row["rank_cross_encoder"] = rec.get("rank")
        row["u_hat"] = pref.get("u_hat")
        row["preference_score"] = pref.get("score")
        row["preference_details"] = pref.get("details", [])
        row["preference_match_rows"] = preference_details_for_dashboard(
            pref.get("details")
        )
        row["work_activity_match"] = work_activity_match_for_dashboard(
            pref.get("details")
        )
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


def v3_recommendation_to_rec(
    row: Dict[str, Any],
    job_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Map one ``concat_gemini_ce_recommendations`` row to the batch runner shape.

    Scores use raw ``concat_cosine_similarity`` (match_v3). Skill display uses
    concat label lists only (no per-skill embedding pairs).
    """
    jid = str(row.get("job_uuid") or "").strip()
    cos = row.get("concat_cosine_similarity")
    try:
        cos_f = float(cos) if cos is not None else None
    except (TypeError, ValueError):
        cos_f = None

    job = job_index.get(jid)
    job_skills = job_skill_labels_for_concat(job) if job else []

    return {
        "rank": row.get("rank"),
        "job_uuid": jid,
        "job_title": row.get("opportunity_title") or row.get("job_title"),
        "employer": row.get("employer"),
        "location": row.get("location"),
        "concat_cosine_similarity": cos_f,
        "mean_best_cosine": round(cos_f, 4) if cos_f is not None else None,
        "cross_encoder_logit": row.get("cross_encoder_logit"),
        "cross_encoder_score": row.get("cross_encoder_score"),
        "rank_cosine": row.get("rank_cosine"),
        "job_concat_skills": job_skills,
    }


def enriched_recommendations(
    user,
    v3_row,
    item_index,
    pref_scorer,
    combiner,
    *,
    location_filter=True,
    location_user=None,
    include_demand: bool = False,
    demand_gamma: float = 0.0,
    p_hat_by_uuid: Optional[Dict[str, float]] = None,
    coverage_by_uuid: Optional[Dict[str, float]] = None,
    coverage_gamma: float = 0.0,
    location_tier_by_uuid: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """CE recs for one user -> preference-enriched, final-score-sorted recs (rich; with details).

    With ``location_filter`` (default), recs are first restricted to items matching a location via
    the same lenient rule ``/match`` uses (``job_matches_user_location``). ``location_user`` (a
    dict with city/province/location) overrides which location to filter by WITHOUT changing the
    ``user`` whose preferences drive ``u_hat`` — used for the random-county fallback. For
    occupations this keeps one row/code (a single county) with that county's attributes.

    Note: the post-secondary education gate is already applied upstream in stage-1 retrieval
    (it skips ineligible items before the top-k cutoff), so no education filtering is needed here.
    """
    ce_http = (v3_row or {}).get("concat_gemini_ce_recommendations") or []
    if location_filter and ce_http:
        loc = location_user or user
        ce_http = [
            r
            for r in ce_http
            if isinstance(r, dict)
            and job_matches_user_location(
                item_index.get(str(r.get("job_uuid") or "")) or {}, loc
            )
        ]
    ce_internal = [
        v3_recommendation_to_rec(r, item_index) for r in ce_http if isinstance(r, dict)
    ]
    if not ce_internal:
        return []
    return enrich_recommendations_with_preferences(
        user,
        ce_internal,
        item_index,
        preference_scorer=pref_scorer,
        include_work_activities=True,
        final_score_combiner=combiner,
        include_demand=include_demand,
        demand_gamma=demand_gamma,
        p_hat_by_uuid=p_hat_by_uuid,
        coverage_by_uuid=coverage_by_uuid,
        coverage_gamma=coverage_gamma,
        location_tier_by_uuid=location_tier_by_uuid,
    )
