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
    V4_FULL_CONCAT_WHITENING_PATH,
    V4_FULL_EMBEDDING_MODEL_PATH,
    V4_FULL_RANK_DEMOTE,
    V4_FULL_WHITENED_GATE,
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
from app.services.cross_encoder.reranker import (
    CrossEncoderReranker,
    rerank_cosine_recommendations,
)
from app.languages import default_language
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
# One cross-encoder per language: the checkpoint has to understand the label text it scores.
_reranker_instances: Dict[str, CrossEncoderReranker] = {}


def _get_matcher() -> CosineSkillMatcher:
    global _matcher_instance
    if _matcher_instance is None:
        with _matcher_lock:
            if _matcher_instance is None:
                _matcher_instance = CosineSkillMatcher()
    return _matcher_instance


_v4_matcher_lock = threading.Lock()
_v4_matcher_instance: Optional[CosineSkillMatcher] = None


def _get_v4_matcher() -> CosineSkillMatcher:
    """v4-only matcher backed by the WHITENED skill artifact (de-anisotropised + rescaled). Separate
    singleton from ``_get_matcher`` so the per-skill GATE in /match_v4 is fixed without changing the
    shared matcher used by v2/v3 and the retrieval detail."""
    global _v4_matcher_instance
    if _v4_matcher_instance is None:
        with _v4_matcher_lock:
            if _v4_matcher_instance is None:
                _v4_matcher_instance = CosineSkillMatcher(
                    model_path=V4_FULL_EMBEDDING_MODEL_PATH
                )
    return _v4_matcher_instance


_concat_white_lock = threading.Lock()
_concat_white: Optional[Dict[str, Any]] = (
    None  # {mu,W,target} once loaded; {} if missing/disabled
)


def _get_concat_whitening() -> Optional[Dict[str, Any]]:
    """Lazy-load the concat-whitening artifact (mu, W=Sigma^-1/2, target) for the Phase-2 whitened
    p_hat skills-fit. Returns None if the artifact is absent (caller then falls back to the raw cosine)."""
    global _concat_white
    if _concat_white is None:
        with _concat_white_lock:
            if _concat_white is None:
                path = V4_FULL_CONCAT_WHITENING_PATH
                if path and os.path.exists(path):
                    z = np.load(path)
                    mu = z["mu"].astype(np.float64)
                    W = z["W"].astype(np.float64)
                    target = float(z["target"])
                    if (
                        mu.shape[0] != EMBEDDING_DIM
                        or W.shape[0] != EMBEDDING_DIM
                        or target <= 0
                    ):
                        # Dim/target mismatch (e.g. embedding model changed without rebuilding the
                        # artifact). Disable rather than risk a mid-request matmul error / bad rescale.
                        logger.error(
                            "concat whitening artifact %s incompatible (mu_dim=%d W_dim=%d target=%.4f, "
                            "expected dim=%d, target>0); whitened p_hat disabled",
                            path,
                            mu.shape[0],
                            W.shape[0],
                            target,
                            EMBEDDING_DIM,
                        )
                        _concat_white = {}
                    else:
                        _concat_white = {"mu": mu, "W": W, "target": target}
                        logger.info(
                            "loaded concat whitening artifact %s (target=%.4f)",
                            path,
                            target,
                        )
                else:
                    logger.warning(
                        "concat whitening artifact not found at %s; whitened p_hat disabled",
                        path,
                    )
                    _concat_white = {}
    return _concat_white or None


def whiten_concat_rows(vecs: np.ndarray) -> np.ndarray:
    """L2-normalise rows, apply the concat whitening ((.-mu)@W), re-normalise -> unit whitened rows.
    If the artifact is unavailable, returns the L2-normalised rows unchanged."""
    v = l2_normalize_rows(np.asarray(vecs, dtype=np.float64))
    cw = _get_concat_whitening()
    if cw is None:
        return v
    return l2_normalize_rows((v - cw["mu"]) @ cw["W"])


def concat_rescale_target() -> float:
    """p99 rescale target for the whitened concat cosine (0.0 if the artifact is unavailable)."""
    cw = _get_concat_whitening()
    return cw["target"] if cw else 0.0


def _get_reranker() -> CrossEncoderReranker:
    """Cross-encoder for the deployment's language, loaded on first use and then reused.

    Keyed by language rather than a single global so that a process whose
    ``TARGET_LANGUAGE`` changes (tests, a script) does not reuse the wrong checkpoint.
    """
    lang = default_language()
    existing = _reranker_instances.get(lang)
    if existing is not None:
        return existing
    with _reranker_lock:
        existing = _reranker_instances.get(lang)
        if existing is not None:
            return existing
        inst = CrossEncoderReranker(
            batch_size=CROSS_ENCODER_BATCH_SIZE,
            language=lang,
        )
        inst.warmup()
        _reranker_instances[lang] = inst
        return inst


def preload_match_v3_models() -> Dict[str, float]:
    """Warm CosineSkillMatcher + CrossEncoder once (call from FastAPI lifespan to avoid per-request cost)."""

    import time

    t0 = time.perf_counter()
    _get_matcher()
    t1 = time.perf_counter()
    if V4_FULL_WHITENED_GATE:
        _get_v4_matcher()  # warm the whitened v4 gate matrix so the first /match_v4 doesn't pay the load
    if V4_FULL_RANK_DEMOTE:
        _get_concat_whitening()  # warm the concat-whitening artifact for the Phase-2 whitened p_hat
        # Log the in-process artifact's sha256 + target so ops can confirm they MATCH the DB's recorded
        # whitening.artifact_sha256 / target. If they ever diverge, whitened-user (in-process) vs
        # job_embedding (DB) would be an inconsistent cosine — this is the one hard dependency.
        try:
            import hashlib

            _sha = hashlib.sha256(open(V4_FULL_CONCAT_WHITENING_PATH, "rb").read()).hexdigest()
            logger.info(
                "concat whitening artifact: path=%s sha256=%s target=%.6f (must match DB job whitening)",
                V4_FULL_CONCAT_WHITENING_PATH,
                _sha,
                concat_rescale_target(),
            )
        except OSError:
            logger.warning(
                "concat whitening artifact not readable at %s; DB-whitened jobs will be degraded.",
                V4_FULL_CONCAT_WHITENING_PATH,
            )
    t1b = time.perf_counter()
    _get_reranker()
    t2 = time.perf_counter()
    return {
        "cosine_skill_matcher_ms": (t1 - t0) * 1000.0,
        "v4_whitened_matcher_ms": (t1b - t1) * 1000.0,
        "cross_encoder_ms": (t2 - t1b) * 1000.0,
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


def _job_is_prewhitened(job: Dict[str, Any]) -> bool:
    """True iff this job's stage-1 embedding is ALREADY whitened on the DB side (set by
    build_job_dict_from_ranked from llm_reranker_meta.embedding.whitening.enabled). Occupations and
    offline jobs lack the flag -> raw (whitened in-process)."""
    return bool(job.get("job_embedding_whitened"))


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
        raise ValueError(
            "GEMINI_API_KEY is not set (required for user concat embeddings)"
        )
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
    apply_location_tier: bool = False,
) -> List[Dict[str, Any]]:
    """Return one result dict per user (keys align with ``MatchConcatGeminiCeResponse``).

    ``user_unit_vectors`` (optional) supplies precomputed, L2-normalised user embeddings so the
    caller can embed users once and reuse them across corpora; if omitted they are embedded here.

    The language is the deployment's (``TARGET_LANGUAGE``); it selects the cross-encoder
    checkpoint for the stage-2 rerank, whose passages are skill-label text. Stage-1 retrieval is
    language-neutral: the skill embeddings are shared across languages (see
    ``services/skill_label_packs``).
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

    # Tiered urban-pull (v4 opportunities only): weight each job's stage-1 cosine by the user's
    # location tier (local=1.0 > regional hub > national hub; off-chain=0) BEFORE the retrieve_top_k
    # cutoff, so relevant local jobs survive the funnel instead of being drowned by the (much larger)
    # national-hub supply. Off-chain jobs (tier 0) are skipped at retrieval entirely. Soft: an
    # irrelevant local job still loses to a much-better hub job.
    _hub_chains = None
    _loc_w_reg = _loc_w_nat = 1.0
    if apply_location_tier:
        from app.config import (
            LOCATION_HUB_CHAINS_PATH,
            LOCATION_TIER_W_NATIONAL,
            LOCATION_TIER_W_REGIONAL,
        )
        from app.services.location_tiers import load_hub_chains

        _hub_chains = load_hub_chains(LOCATION_HUB_CHAINS_PATH)
        _loc_w_reg, _loc_w_nat = LOCATION_TIER_W_REGIONAL, LOCATION_TIER_W_NATIONAL

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
            empty_summary["mongo_ranked_find_ms"] = mongo_timing.get(
                "mongo_ranked_find_ms"
            )
            empty_summary["jobs_retrieval_filter_applied"] = mongo_timing.get(
                "jobs_retrieval_filter_applied"
            )
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
        if (
            u_norm.ndim != 2
            or u_norm.shape[0] != len(users)
            or u_norm.shape[1] != EMBEDDING_DIM
        ):
            raise RuntimeError(
                f"user_unit_vectors shape {u_norm.shape} != ({len(users)}, {EMBEDDING_DIM})"
            )
    else:
        u_norm = embed_user_unit_vectors(users)

    lang = default_language()
    matcher = _get_matcher()
    reranker = _get_reranker()

    # Whitened-space stage-1 retrieval. The concat artifact (same one the DB used to whiten
    # job_embedding) is present in practice, so we rank in the de-anisotropised whitened space (the
    # meaningful signal; raw concat cosine sd ~0.02). Jobs already whitened on the DB side are used
    # as-is; RAW vectors (occupations, offline, not-yet-whitened jobs) are whitened in-process once
    # (numerically identical to the DB result — same artifact). When the artifact is unavailable
    # (target==0) we fall back to the legacy raw cosine and log loudly (DB-whitened jobs degrade).
    _whiten_target = concat_rescale_target()
    if _whiten_target > 0:
        j_used = j_norm.copy()
        raw_idx = [k for k, jr in enumerate(job_rows) if not _job_is_prewhitened(jr)]
        if raw_idx:
            j_used[raw_idx] = whiten_concat_rows(j_mat[raw_idx])
        u_used = whiten_concat_rows(u_norm)
    else:
        if any(_job_is_prewhitened(jr) for jr in job_rows):
            logger.error(
                "concat whitening artifact unavailable but DB job_embedding is whitened; stage-1 "
                "cosine will be raw-user vs whitened-job (degraded). Ship concat_whitening_gemini.npz."
            )
        j_used, u_used = j_norm, u_norm

    out_results: List[Dict[str, Any]] = []
    for i, user in enumerate(users):
        sim_row = (u_used[i : i + 1] @ j_used.T).reshape(-1)
        # Location-tier weighting of the stage-1 ranking (urban-pull). Rank by cosine * tier so local
        # jobs are favoured for the shortlist; keep the RAW cosine for the stored similarity downstream.
        loc_tier_vec = None
        if _hub_chains is not None:
            county = user.get("province") or user.get("city") or ""
            loc_tier_vec = np.array(
                [
                    _hub_chains.tier_factor_for_job(
                        jr, county, w_regional=_loc_w_reg, w_national=_loc_w_nat
                    )
                    for jr in job_rows
                ],
                dtype=float,
            )
            rank_row = sim_row * loc_tier_vec
        else:
            rank_row = sim_row
        order = _sorted_indices_desc(rank_row)
        user_no_ps = user_lacks_post_secondary(user)

        cosine_recs: List[Dict[str, Any]] = []
        for ji in order:
            if user_no_ps and job_requires_ps[int(ji)]:
                continue  # job requires post-secondary education the user does not have
            if loc_tier_vec is not None and loc_tier_vec[int(ji)] <= 0.0:
                continue  # off-chain location for this user: excluded at retrieval
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
            "cross_encoder_model": reranker.model_name,
            "language": lang,
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
            cfg["jobs_retrieval_filter_applied"] = mongo_timing.get(
                "jobs_retrieval_filter_applied"
            )

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
