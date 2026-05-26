"""match_v3 retrieval (Gemini concat cosine + CE) → hybrid preference final score.

Used by ``POST /match_v4`` — same response envelope as ``POST /match_v3`` with
``u_hat``, ``p_hat``, and ``final_score`` on each recommendation (ranked by final).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import FINAL_SCORE_COMBINER, PREFERENCE_SCORER_MODE
from app.services.gemini_ce_preference_matching.match_v3_bridge import v3_recommendation_to_rec
from app.services.gemini_ce_preference_matching.scoring import enrich_recommendations_with_preferences
from app.services.match_concat_gemini_ce_service import run_match_concat_gemini_ce
from app.services.preference_score_v1 import get_preference_scorer

__all__ = ["run_match_concat_gemini_ce_with_preferences"]


def _jobs_by_uuid(jobs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        uid = str(job.get("uuid") or job.get("_id") or "")
        if uid:
            out[uid] = job
    return out


def _preference_rec_to_http(row: Dict[str, Any]) -> Dict[str, Any]:
    sb = row.get("score_breakdown") or {}
    return {
        "rank": int(row.get("rank") or 0),
        "rank_cosine": row.get("rank_cosine"),
        "rank_cross_encoder": row.get("rank_cross_encoder"),
        "job_uuid": str(row.get("job_uuid") or ""),
        "opportunity_title": str(row.get("job_title") or row.get("opportunity_title") or ""),
        "employer": row.get("employer"),
        "location": row.get("location"),
        "URL": row.get("URL") or row.get("url"),
        "concat_cosine_similarity": row.get("concat_cosine_similarity"),
        "cross_encoder_logit": row.get("cross_encoder_logit"),
        "cross_encoder_score": row.get("cross_encoder_score"),
        "u_hat": row.get("u_hat"),
        "p_hat": row.get("p_hat"),
        "final_score": row.get("final_score"),
        "score_breakdown": sb if sb else None,
    }


def _final_formula(combiner: str) -> str:
    return "u_hat * p_hat" if combiner == "product" else "sqrt(u_hat * p_hat)"


def run_match_concat_gemini_ce_with_preferences(
    users: List[Dict[str, Any]],
    jobs: List[Dict[str, Any]],
    *,
    retrieve_top_k: int,
    final_top_k: int,
    mongo_timing: Optional[Dict[str, Any]] = None,
    final_score_combiner: Optional[str] = None,
    include_work_activities: bool = True,
) -> List[Dict[str, Any]]:
    """Return one dict per user (``MatchConcatGeminiCeResponse`` + preference fields)."""

    combiner = (final_score_combiner or FINAL_SCORE_COMBINER).strip().lower()
    if combiner not in ("product", "geometric_mean"):
        raise ValueError("final_score_combiner must be 'product' or 'geometric_mean'")

    v3_rows = run_match_concat_gemini_ce(
        users,
        jobs,
        retrieve_top_k=retrieve_top_k,
        final_top_k=final_top_k,
        mongo_timing=mongo_timing,
    )
    users_by_id = {str(u.get("user_id") or ""): u for u in users}
    job_index = _jobs_by_uuid(jobs)
    pref_scorer = get_preference_scorer()
    pref_cls = type(pref_scorer).__module__ + "." + type(pref_scorer).__name__

    out: List[Dict[str, Any]] = []
    for v3_row in v3_rows:
        uid = str(v3_row.get("user_id") or "")
        user = users_by_id.get(uid) or {}
        ce_http = v3_row.get("concat_gemini_ce_recommendations") or []
        ce_internal = [v3_recommendation_to_rec(r, job_index) for r in ce_http if isinstance(r, dict)]

        if ce_internal and user:
            enriched = enrich_recommendations_with_preferences(
                user,
                ce_internal,
                job_index,
                preference_scorer=pref_scorer,
                include_work_activities=include_work_activities,
                final_score_combiner=combiner,
            )
            recs = [_preference_rec_to_http(r) for r in enriched]
        else:
            recs = [
                {
                    **_preference_rec_to_http(v3_recommendation_to_rec(r, job_index)),
                    "final_score": None,
                    "u_hat": None,
                    "p_hat": r.get("concat_cosine_similarity"),
                }
                for r in ce_http
                if isinstance(r, dict)
            ]

        cfg = dict(v3_row.get("config_summary") or {})
        cfg.update(
            {
                "stage3": "combine_u_hat_and_p_hat",
                "stage3_scorer": "hybrid_preference_final",
                "preference_scorer_mode": PREFERENCE_SCORER_MODE,
                "preference_module": pref_cls,
                "final_score_combiner": combiner,
                "final_formula": _final_formula(combiner),
                "p_hat_source": "concat_cosine_similarity",
                "include_work_activities": include_work_activities,
            }
        )

        out.append(
            {
                "user_id": uid,
                "n_jobs_scored": v3_row.get("n_jobs_scored", 0),
                "n_jobs_active_loaded": v3_row.get("n_jobs_active_loaded", 0),
                "concat_gemini_ce_recommendations": recs,
                "config_summary": cfg,
            }
        )

    return out
