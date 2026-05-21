"""match_v3 concat cosine + CE → u_hat × p_hat (p_hat = raw cosine, not CE 0–1).

Stage 1–2: same as ``POST /match_v3`` (:func:`~app.services.match_concat_gemini_ce_service.run_match_concat_gemini_ce`):
Gemini user concat × Mongo job vectors → **concat_cosine_similarity**, then CE rerank.

Stage 3: :class:`~app.services.preference_score.PreferenceScorer` → ``u_hat``;
``p_hat`` = ``concat_cosine_similarity``; ``final = u_hat × p_hat``.

Usage (from ``backend/``)::

    python -m app.services.gemini_ce_preference_matching.run_matching \\
        --users data/njila/njila_match_input.jsonl \\
        --from-mongo \\
        --retrieve-top-k 50 \\
        --final-top-k 10 \\
        --output output/results_gemini_ce_preference.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from app.config import (
    COSINE_CROSS_ENCODER_RETRIEVE_TOP_K,
    CROSS_ENCODER_BATCH_SIZE,
    CROSS_ENCODER_MODEL_NAME,
)
from app.services.cosine_similarity.run_cosine_matching import (
    _load_users,
    _trim_recommendations,
    load_jobs,
)
from app.services.cross_encoder.concat_embedding_text import user_skill_labels_for_concat
from app.services.cross_encoder.gemini_embeddings import EMBEDDING_DIM, MODEL_NAME as GEMINI_EMBEDDING_MODEL_NAME
from app.services.match_concat_gemini_ce_service import run_match_concat_gemini_ce
from app.services.preference_score import PreferenceScorer

from .match_v3_bridge import v3_recommendation_to_rec
from .scoring import enrich_recommendations_with_preferences


def _jobs_by_uuid(jobs: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for j in jobs:
        uid = str(j.get("uuid") or j.get("_id") or "")
        if uid:
            out[uid] = j
    return out


def run_pipeline(
    users_path: Path,
    jobs_source: Literal["file", "mongo"],
    jobs_path: Optional[Path],
    *,
    retrieve_top_k: int,
    final_top_k: int,
    output_path: Optional[Path],
    max_per_job_skills: int,
    mongo_filter_by_users: bool,
    cross_encoder_model: Optional[str],
    cross_encoder_batch_size: Optional[int],
) -> Dict[str, Any]:
    retrieve_top_k = max(1, int(retrieve_top_k))
    final_top_k = max(1, int(final_top_k))

    users = _load_users(users_path)
    jobs, mongo_timing = load_jobs(
        jobs_source, jobs_path, users, mongo_filter_by_users
    )
    job_index = _jobs_by_uuid(jobs)

    pref_scorer = PreferenceScorer()

    print(
        f"[gemini_ce_pref] users={len(users)} jobs={len(jobs)} "
        f"stage1=match_v3_concat_cosine retrieve_top_k={retrieve_top_k} "
        f"final_top_k={final_top_k} scoring=u_hat*p_hat p_hat=concat_cosine",
        file=sys.stderr,
    )

    v3_rows = run_match_concat_gemini_ce(
        users,
        jobs,
        retrieve_top_k=retrieve_top_k,
        final_top_k=retrieve_top_k,
        mongo_timing=mongo_timing,
    )

    results: List[Dict[str, Any]] = []

    for user, v3_row in zip(users, v3_rows):
        user_concat_skills = user_skill_labels_for_concat(user)

        ce_pool = [
            v3_recommendation_to_rec(r, job_index)
            for r in (v3_row.get("concat_gemini_ce_recommendations") or [])
            if isinstance(r, dict)
        ]

        # Column 1 export: CE order, p_hat = concat cosine only (no u_hat yet).
        ce_snapshot = copy.deepcopy(ce_pool)[:final_top_k]

        # Column 2: u_hat per job, then final = u_hat × p_hat, re-sorted.
        final_recs = enrich_recommendations_with_preferences(
            user,
            ce_pool,
            job_index,
            preference_scorer=pref_scorer,
        )
        final_recs = final_recs[:final_top_k]

        cfg_summary = v3_row.get("config_summary") or {}
        results.append({
            "user_id": user.get("user_id"),
            "city": user.get("city"),
            "province": user.get("province"),
            "n_user_concat_skills": len(user_concat_skills),
            "user_concat_skills": user_concat_skills,
            "n_jobs_scored": v3_row.get("n_jobs_scored"),
            "cross_encoder_recommendations": ce_snapshot,
            "recommendations": final_recs,
        })

    v3_cfg0 = (v3_rows[0].get("config_summary") if v3_rows else {}) or {}
    config: Dict[str, Any] = {
        "users_path": str(users_path),
        "jobs_source": jobs_source,
        "retrieve_top_k": retrieve_top_k,
        "final_top_k": final_top_k,
        "stage1_scorer": "match_v3_concat_gemini_cosine_mongo_job_vectors",
        "stage2_scorer": "cross_encoder_rerank",
        "stage3_scorer": "u_hat_times_p_hat",
        "scoring_mode": "multiplicative",
        "final_formula": "u_hat * p_hat",
        "p_hat_source": "concat_cosine_similarity",
        "preference_module": "app.services.preference_score.PreferenceScorer",
        "match_v3_service": "app.services.match_concat_gemini_ce_service",
        "gemini_user_embed_model": v3_cfg0.get("gemini_user_embed_model") or GEMINI_EMBEDDING_MODEL_NAME,
        "cross_encoder_model": v3_cfg0.get("cross_encoder_model") or cross_encoder_model or CROSS_ENCODER_MODEL_NAME,
        "cross_encoder_batch_size": int(
            cross_encoder_batch_size or CROSS_ENCODER_BATCH_SIZE
        ),
        "embedding_dim": v3_cfg0.get("embedding_dim") or EMBEDDING_DIM,
        "n_jobs_with_stage1_embedding": v3_cfg0.get("n_jobs_with_stage1_embedding"),
        "max_per_job_skills_in_output": max_per_job_skills,
    }
    if jobs_source == "file":
        config["jobs_path"] = str(jobs_path) if jobs_path else None
    else:
        config["mongo_filter_by_users"] = mongo_filter_by_users

    payload: Dict[str, Any] = {
        "config": config,
        "index_stats": {
            "n_jobs": len(jobs),
            "n_jobs_with_concat_embedding": v3_cfg0.get("n_jobs_with_stage1_embedding"),
            "embedding_dim": EMBEDDING_DIM,
        },
        "n_users": len(users),
        "n_jobs": len(jobs),
        "results": results,
    }
    if mongo_timing is not None:
        payload["mongo_timing"] = mongo_timing

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[gemini_ce_pref] wrote → {output_path}", file=sys.stderr)
    else:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    epilog = """\
Examples:
  %(prog)s --users data/njila/njila_match_input.jsonl --from-mongo \\
      --retrieve-top-k 50 --final-top-k 10 \\
      --output output/results_gemini_ce_preference.json

Requires GEMINI_API_KEY and jobs with job_embedding or concat_skill_embedding_gemini in Mongo.
"""
    p = argparse.ArgumentParser(
        description=(
            "match_v3 (concat Gemini cosine + CE) → u_hat × p_hat "
            "(p_hat = raw concat_cosine_similarity, not CE min–max score)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument("--users", required=True, type=Path, help="JSON or JSONL user records.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jobs", type=Path, help="Local jobs JSON list.")
    src.add_argument("--from-mongo", action="store_true", help="Load jobs from Mongo (.env).")

    p.add_argument(
        "--retrieve-top-k",
        type=int,
        default=COSINE_CROSS_ENCODER_RETRIEVE_TOP_K,
        help="match_v3 stage-1 shortlist / CE pool size per user.",
    )
    p.add_argument(
        "--final-top-k",
        type=int,
        default=10,
        help="Rows per user after u_hat × p_hat re-rank.",
    )
    p.add_argument("--output", type=Path, default=None, help="Write JSON; omit → stdout.")
    p.add_argument(
        "--max-per-job-skills",
        type=int,
        default=200,
        help="Cap per_job_skill rows per recommendation.",
    )
    p.add_argument(
        "--mongo-all-active",
        action="store_true",
        help="With --from-mongo: load all active jobs (no per-user location filter).",
    )
    p.add_argument("--cross-encoder-model", type=str, default=None)
    p.add_argument("--cross-encoder-batch-size", type=int, default=None)

    args = p.parse_args(argv)
    jobs_source: Literal["file", "mongo"] = "mongo" if args.from_mongo else "file"
    jobs_path = None if jobs_source == "mongo" else args.jobs

    if jobs_source == "file" and jobs_path is not None:
        jp = jobs_path.expanduser()
        if not jp.is_file():
            print(f"[gemini_ce_pref] jobs file not found: {jobs_path}", file=sys.stderr)
            return 2

    run_pipeline(
        users_path=args.users,
        jobs_source=jobs_source,
        jobs_path=jobs_path,
        retrieve_top_k=args.retrieve_top_k,
        final_top_k=args.final_top_k,
        output_path=args.output,
        max_per_job_skills=args.max_per_job_skills,
        mongo_filter_by_users=not args.mongo_all_active,
        cross_encoder_model=args.cross_encoder_model,
        cross_encoder_batch_size=args.cross_encoder_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
