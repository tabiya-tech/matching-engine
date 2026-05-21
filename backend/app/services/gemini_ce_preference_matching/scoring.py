"""Multiplicative ranking: u_hat (preferences) × p_hat (skill retrieval score)."""

from __future__ import annotations

from typing import Any, Dict, Tuple


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


def compute_multiplicative_final_score(
    pref_result: Dict[str, Any],
    p_hat: float,
    *,
    p_hat_source: str,
) -> Tuple[float, Dict[str, Any]]:
    """``final_score = u_hat × p_hat``."""
    u_hat = float(pref_result.get("u_hat", 0.5))
    final = u_hat * p_hat

    breakdown = {
        "scoring_mode": "multiplicative",
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

        pref = preference_scorer.calculate_score(user, job)
        p_hat, p_hat_source = p_hat_from_skill_rec(rec)
        final, breakdown = compute_multiplicative_final_score(
            pref, p_hat, p_hat_source=p_hat_source
        )

        row = dict(rec)
        row["rank_cross_encoder"] = rec.get("rank")
        row["u_hat"] = pref.get("u_hat")
        row["preference_score"] = pref.get("score")
        row["preference_details"] = pref.get("details", [])
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
