"""Gemini concat user embedding × job/occupation vectors → whitened cosine shortlist → cross-encoder.

Item vectors may come from:

* ``concat_skill_embedding_gemini.vector_bin`` (BSON float32 bytes), or
* ``job_embedding`` — array of ``embedding_dim`` floats on the ranked job document (or the
  occupation vectors attached by ``app.occupations.loader.attach_occupation_embeddings``).

Cosine scores are only meaningful if ``job_embedding`` lives in the **same** space as the user
vector from ``gemini-embedding-001`` concat text (same dimension by default).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app import observability
from app.artifacts.repository import get_artifacts_repository
from app.clients.gemini_embedding_client import EMBEDDING_DIM
from app.clients.gemini_embedding_client import (
    MODEL_NAME as GEMINI_EMBEDDING_MODEL_NAME,
)
from app.config import (
    LOCATION_HUB_CHAINS_PATH,
    LOCATION_TIER_W_NATIONAL,
    LOCATION_TIER_W_REGIONAL,
)
from app.languages import default_language
from app.ranking.rerank import CrossEncoderRerank
from app.ranking.retrieval import (
    Stage1Retriever,
    l2_normalize_rows,
    user_concat_embedding_text,
    user_skill_labels_for_concat,
)
from app.server_dependencies.model_dependencies import (
    get_concat_whitener,
    get_cross_encoder_client,
    get_gemini_embedding_client,
    get_skill_matcher,
)

__all__ = ["run_match_concat_gemini_ce", "embed_user_unit_vectors"]

logger = logging.getLogger(__name__)

_RETRIEVER = Stage1Retriever(embedding_dim=EMBEDDING_DIM)
_RERANK = CrossEncoderRerank()


def embed_user_unit_vectors(users: list[dict[str, Any]]) -> np.ndarray:
    """Gemini concat embeddings for users, L2-normalised (float64 [n_users, EMBEDDING_DIM]).

    Lets a caller embed users ONCE and reuse the matrix across multiple corpora (jobs +
    occupations) via ``run_match_concat_gemini_ce(..., user_unit_vectors=...)``.
    """
    client = get_gemini_embedding_client()
    client.ensure_configured()
    with observability.stage("embedding", n_users=len(users), model=client.model_name):
        texts = []
        for u in users:
            t = user_concat_embedding_text(u).strip()
            texts.append(t if t else " ")
        u_emb = client.embed_texts(texts, batch_size=100, sleep_s=0.12)
        if u_emb.shape[0] != len(users):
            raise RuntimeError("Gemini embed returned unexpected row count")
        return l2_normalize_rows(u_emb.astype(np.float32)).astype(np.float64)


def run_match_concat_gemini_ce(
    users: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    *,
    retrieve_top_k: int,
    final_top_k: int,
    mongo_timing: dict[str, Any] | None = None,
    user_unit_vectors: np.ndarray | None = None,
    apply_location_tier: bool = False,
    corpus: str = "jobs",
) -> list[dict[str, Any]]:
    """Return one result dict per user with ``concat_gemini_ce_recommendations``.

    ``user_unit_vectors`` (optional) supplies precomputed, L2-normalised user embeddings so the
    caller can embed users once and reuse them across corpora; if omitted they are embedded here.
    ``corpus`` only labels the ``shortlist`` / ``rerank`` trace spans (``jobs`` / ``occupations``).

    The language is the deployment's (``TARGET_LANGUAGE``); it selects the cross-encoder
    checkpoint for the stage-2 rerank, whose passages are skill-label text. Stage-1 retrieval is
    language-neutral: the skill embeddings are shared across languages (see
    ``ranking/skills``).
    """
    if not users:
        return []
    rt = max(1, int(retrieve_top_k))
    fk = max(1, int(final_top_k))
    embedding_model = GEMINI_EMBEDDING_MODEL_NAME
    embedding_dim = EMBEDDING_DIM

    with observability.stage("shortlist", corpus=corpus, n_users=len(users)) as sl:
        stage1 = _RETRIEVER.prepare(jobs)
        n_with_emb = len(stage1.rows)
        n_active = stage1.n_loaded

        # Tiered urban-pull (v4 opportunities only): weight each job's stage-1 cosine by the user's
        # location tier (local=1.0 > regional hub > national hub; off-chain=0) BEFORE the retrieve_top_k
        # cutoff, so relevant local jobs survive the funnel instead of being drowned by the (much larger)
        # national-hub supply. Off-chain jobs (tier 0) are skipped at retrieval entirely. Soft: an
        # irrelevant local job still loses to a much-better hub job.
        _hub_chains = None
        _loc_w_reg = _loc_w_nat = 1.0
        if apply_location_tier:
            _hub_chains = get_artifacts_repository().load_hub_chains(
                LOCATION_HUB_CHAINS_PATH
            )
            _loc_w_reg, _loc_w_nat = (
                LOCATION_TIER_W_REGIONAL,
                LOCATION_TIER_W_NATIONAL,
            )

        if not stage1.rows:
            empty_summary = {
                "stage1": "concat_gemini_cosine_mongo_job_vectors",
                "stage2": "cross_encoder_rerank",
                "gemini_user_embed_model": embedding_model,
                "embedding_dim": embedding_dim,
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

        if user_unit_vectors is not None:
            u_norm = np.asarray(user_unit_vectors, dtype=np.float64)
            if (
                u_norm.ndim != 2
                or u_norm.shape[0] != len(users)
                or u_norm.shape[1] != embedding_dim
            ):
                raise RuntimeError(
                    f"user_unit_vectors shape {u_norm.shape} != ({len(users)}, {embedding_dim})"
                )
        else:
            u_norm = embed_user_unit_vectors(users)

        lang = default_language()
        matcher = get_skill_matcher()
        reranker = get_cross_encoder_client()

        shortlists = _RETRIEVER.shortlist(
            users,
            stage1,
            u_norm,
            matcher=matcher,
            whitener=get_concat_whitener(),
            retrieve_top_k=rt,
            hub_chains=_hub_chains,
            w_regional=_loc_w_reg,
            w_national=_loc_w_nat,
        )
        observability.update_observation(
            sl,
            metadata={
                "n_candidates_with_embedding": n_with_emb,
                "n_candidates_loaded": n_active,
            },
        )

    out_results: list[dict[str, Any]] = []
    with observability.stage(
        "rerank", corpus=corpus, n_users=len(users), model=reranker.model_name
    ):
        for user, cosine_recs in zip(users, shortlists):
            labels = user_skill_labels_for_concat(user)
            pairs = _RERANK.build_pairs(labels, cosine_recs, final_top_k=fk)
            scores = reranker.predict_scores(pairs) if pairs else []
            reranked = _RERANK.apply(cosine_recs, scores, final_top_k=fk)

            recs: list[dict[str, Any]] = []
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
                "gemini_user_embed_model": embedding_model,
                "cross_encoder_model": reranker.model_name,
                "language": lang,
                "embedding_dim": embedding_dim,
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
