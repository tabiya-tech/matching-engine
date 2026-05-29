"""Gemini concat user embedding × Mongo job vectors → cosine → cross-encoder.

Job vectors may come from:

* ``concat_skill_embedding_gemini.vector_bin`` (BSON float32 bytes), or
* ``job_embedding`` — array of ``embedding_dim`` floats on the ranked job document.

Used by public ``POST /match_v3`` with the same ``MatchRequest`` payload as ``POST /match``.
Cosine scores are only meaningful if ``job_embedding`` lives in the **same** space as the user
vector from ``gemini-embedding-001`` concat text (same dimension by default).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import (
    CROSS_ENCODER_BATCH_SIZE,
    CROSS_ENCODER_MODEL_NAME,
)
from app.services.cross_encoder.concat_embedding_text import (
    user_concat_embedding_text,
    user_skill_labels_for_concat,
)
from app.services.cross_encoder.gemini_embeddings import (
    EMBEDDING_DIM,
    MODEL_NAME as GEMINI_EMBEDDING_MODEL_NAME,
    embed_text_list,
    l2_normalize_rows,
)
from app.services.cross_encoder.reranker import CrossEncoderReranker, rerank_cosine_recommendations
from app.services.cosine_similarity.skill_score import CosineSkillMatcher
from app.services.education_eligibility import (
    job_requires_post_secondary,
    user_lacks_post_secondary,
)

__all__ = ["run_match_concat_gemini_ce", "preload_match_v3_models"]

logger = logging.getLogger(__name__)

# Heavy models cached for the process lifetime — CosineSkillMatcher torch-loads ~14k embedding
# rows + skills.csv; CrossEncoderReranker pulls HF weights on first predict. Rebuilding either
# per /match_v3 request is what produced the 80–100s requests in early testing.
_matcher_lock = threading.Lock()
_matcher_instance: Optional[CosineSkillMatcher] = None

_reranker_lock = threading.Lock()
_reranker_instance: Optional[CrossEncoderReranker] = None


def _get_matcher() -> CosineSkillMatcher:
    global _matcher_instance
    if _matcher_instance is None:
        with _matcher_lock:
            if _matcher_instance is None:
                _matcher_instance = CosineSkillMatcher()
    return _matcher_instance


def _get_reranker() -> CrossEncoderReranker:
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                inst = CrossEncoderReranker(
                    model_name=CROSS_ENCODER_MODEL_NAME,
                    batch_size=CROSS_ENCODER_BATCH_SIZE,
                )
                inst.warmup()
                _reranker_instance = inst
    return _reranker_instance


def preload_match_v3_models() -> Dict[str, float]:
    """Warm CosineSkillMatcher + CrossEncoder once (call from FastAPI lifespan to avoid per-request cost)."""

    import time

    t0 = time.perf_counter()
    _get_matcher()
    t1 = time.perf_counter()
    _get_reranker()
    t2 = time.perf_counter()
    return {
        "cosine_skill_matcher_ms": (t1 - t0) * 1000.0,
        "cross_encoder_ms": (t2 - t1) * 1000.0,
    }


def _gemini_api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or "").strip()


def _job_stage1_embedding_vector(job: Dict[str, Any]) -> Optional[np.ndarray]:
    """Prefer NPZ-sync BSON; fall back to ``job_embedding`` float list on the job doc."""

    sub = job.get("concat_skill_embedding_gemini")
    if isinstance(sub, dict):
        vb = sub.get("vector_bin")
        if vb is not None:
            raw = getattr(vb, "bytes", None) or bytes(vb)
            arr = np.frombuffer(raw, dtype=np.float32)
            if arr.size == EMBEDDING_DIM:
                return arr

    je = job.get("job_embedding")
    # Accept a float list (Mongo job docs) or a numpy array (occupation embeddings attached
    # in-process by app.database.attach_occupation_embeddings).
    if isinstance(je, np.ndarray):
        if je.ndim == 1 and je.size == EMBEDDING_DIM:
            return je.astype(np.float32, copy=False)
    elif isinstance(je, list) and je:
        arr = np.asarray(je, dtype=np.float32)
        if arr.ndim == 1 and arr.size == EMBEDDING_DIM:
            return arr
    return None


def _strip_job_vectors(job: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(job)
    out.pop("concat_skill_embedding_gemini", None)
    out.pop("job_embedding", None)
    return out


def _sorted_indices_desc(sim_row: np.ndarray) -> np.ndarray:
    return np.argsort(-sim_row, kind="stable")


def embed_user_unit_vectors(users: List[Dict[str, Any]]) -> np.ndarray:
    """Gemini concat embeddings for users, L2-normalised (float64 [n_users, EMBEDDING_DIM]).

    Lets a caller embed users ONCE and reuse the matrix across multiple corpora (jobs +
    occupations) via ``run_match_concat_gemini_ce(..., user_unit_vectors=...)``.
    """
    api_key = _gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set (required for user concat embeddings)")
    texts = []
    for u in users:
        t = user_concat_embedding_text(u).strip()
        texts.append(t if t else " ")
    u_emb = embed_text_list(texts, api_key=api_key, batch_size=100, sleep_s=0.12)
    if u_emb.shape[0] != len(users):
        raise RuntimeError("Gemini embed returned unexpected row count")
    return l2_normalize_rows(u_emb.astype(np.float32)).astype(np.float64)


def run_match_concat_gemini_ce(
    users: List[Dict[str, Any]],
    jobs: List[Dict[str, Any]],
    *,
    retrieve_top_k: int,
    final_top_k: int,
    mongo_timing: Optional[Dict[str, Any]] = None,
    user_unit_vectors: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Return one result dict per user (keys align with ``MatchConcatGeminiCeResponse``).

    ``user_unit_vectors`` (optional) supplies precomputed, L2-normalised user embeddings so the
    caller can embed users once and reuse them across corpora; if omitted they are embedded here.
    """

    if not users:
        return []
    rt = max(1, int(retrieve_top_k))
    fk = max(1, int(final_top_k))

    job_rows: List[Dict[str, Any]] = []
    vectors: List[np.ndarray] = []
    for j in jobs:
        v = _job_stage1_embedding_vector(j)
        if v is None:
            continue
        job_rows.append(j)
        vectors.append(v)

    n_with_emb = len(job_rows)
    n_active = len(jobs)

    for j in job_rows:
        j.pop("concat_skill_embedding_gemini", None)
        j.pop("job_embedding", None)

    # Post-secondary education gate: aligned with job_rows, used to skip candidates per user.
    job_requires_ps = [job_requires_post_secondary(j) for j in job_rows]

    if not job_rows:
        empty_summary = {
            "stage1": "concat_gemini_cosine_mongo_job_vectors",
            "stage2": "cross_encoder_rerank",
            "gemini_user_embed_model": GEMINI_EMBEDDING_MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "n_jobs_with_stage1_embedding": 0,
            "n_jobs_with_concat_gemini_embedding": 0,
            "n_jobs_active_loaded": n_active,
        }
        if mongo_timing:
            empty_summary["mongo_ranked_find_ms"] = mongo_timing.get("mongo_ranked_find_ms")
            empty_summary["jobs_retrieval_filter_applied"] = mongo_timing.get("jobs_retrieval_filter_applied")
        return [
            {
                "user_id": str(u.get("user_id") or ""),
                "n_jobs_scored": 0,
                "n_jobs_active_loaded": n_active,
                "concat_gemini_ce_recommendations": [],
                "config_summary": empty_summary,
            }
            for u in users
        ]

    j_mat = np.stack(vectors, axis=0).astype(np.float64)
    j_norm = l2_normalize_rows(j_mat.astype(np.float32)).astype(np.float64)
    jid_list = [str(j.get("uuid") or "") for j in job_rows]

    if user_unit_vectors is not None:
        u_norm = np.asarray(user_unit_vectors, dtype=np.float64)
        if u_norm.ndim != 2 or u_norm.shape[0] != len(users) or u_norm.shape[1] != EMBEDDING_DIM:
            raise RuntimeError(
                f"user_unit_vectors shape {u_norm.shape} != ({len(users)}, {EMBEDDING_DIM})"
            )
    else:
        u_norm = embed_user_unit_vectors(users)

    matcher = _get_matcher()
    reranker = _get_reranker()

    out_results: List[Dict[str, Any]] = []
    for i, user in enumerate(users):
        sim_row = (u_norm[i : i + 1] @ j_norm.T).reshape(-1)
        order = _sorted_indices_desc(sim_row)
        user_no_ps = user_lacks_post_secondary(user)

        cosine_recs: List[Dict[str, Any]] = []
        for ji in order:
            if user_no_ps and job_requires_ps[int(ji)]:
                continue  # job requires post-secondary education the user does not have
            jid = jid_list[int(ji)]
            job_obj = job_rows[int(ji)]
            job_plain = _strip_job_vectors(job_obj)
            concat_sim = float(sim_row[int(ji)])
            detail = matcher.score_pair(user, job_plain)
            detail = dict(detail)
            detail["concat_cosine_similarity"] = round(concat_sim, 6)
            detail["mean_best_cosine"] = round(concat_sim, 4)
            detail["min_best_cosine"] = round(concat_sim, 4)

            cosine_recs.append(
                {
                    "rank": len(cosine_recs) + 1,
                    "job_uuid": jid,
                    "job_title": job_plain.get("opportunity_title"),
                    "employer": job_plain.get("employer"),
                    "location": job_plain.get("location"),
                    **detail,
                }
            )
            if len(cosine_recs) >= rt:
                break

        for r_i, row in enumerate(cosine_recs, start=1):
            row["rank"] = r_i

        labels = user_skill_labels_for_concat(user)
        reranked = rerank_cosine_recommendations(
            labels,
            cosine_recs,
            reranker=reranker,
            final_top_k=fk,
        )

        recs: List[Dict[str, Any]] = []
        for row in reranked:
            recs.append(
                {
                    "rank": int(row.get("rank") or 0),
                    "rank_cosine": row.get("rank_cosine"),
                    "job_uuid": str(row.get("job_uuid") or ""),
                    "opportunity_title": str(row.get("job_title") or "") or "",
                    "employer": row.get("employer"),
                    "location": row.get("location"),
                    "URL": row.get("url") or row.get("URL"),
                    "concat_cosine_similarity": row.get("concat_cosine_similarity"),
                    "cross_encoder_logit": row.get("cross_encoder_logit"),
                    "cross_encoder_score": row.get("cross_encoder_score"),
                }
            )

        uid = str(user.get("user_id") or "")
        cfg = {
            "stage1": "concat_gemini_cosine_mongo_job_vectors",
            "stage2": "cross_encoder_rerank",
            "gemini_user_embed_model": GEMINI_EMBEDDING_MODEL_NAME,
            "cross_encoder_model": CROSS_ENCODER_MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "retrieve_top_k": rt,
            "final_top_k": fk,
            "n_jobs_with_stage1_embedding": n_with_emb,
            # Legacy key — counts jobs with BSON ``vector_bin`` or ``job_embedding`` array (same dim).
            "n_jobs_with_concat_gemini_embedding": n_with_emb,
            "n_jobs_active_loaded": n_active,
        }
        if mongo_timing:
            cfg["mongo_ranked_find_ms"] = mongo_timing.get("mongo_ranked_find_ms")
            cfg["jobs_retrieval_filter_applied"] = mongo_timing.get("jobs_retrieval_filter_applied")

        out_results.append(
            {
                "user_id": uid,
                "n_jobs_scored": n_with_emb,
                "n_jobs_active_loaded": n_active,
                "concat_gemini_ce_recommendations": recs,
                "config_summary": cfg,
            }
        )

    return out_results
