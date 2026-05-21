"""Bridge ``POST /match_v3`` output into gemini_ce_preference batch JSON rows."""

from __future__ import annotations

from typing import Any, Dict

from app.services.cross_encoder.concat_embedding_text import job_skill_labels_for_concat


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
