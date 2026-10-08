"""Gemini concat user embedding × job/occupation vectors → whitened cosine shortlist → cross-encoder.

Item vectors may come from:

* ``concat_skill_embedding_gemini.vector_bin`` (BSON float32 bytes), or
* ``job_embedding`` — array of ``embedding_dim`` floats on the ranked job document (or the
  occupation vectors attached by ``OccupationsRepository.attach_embeddings``).

Cosine scores are only meaningful if ``job_embedding`` lives in the **same** space as the user
vector from ``gemini-embedding-001`` concat text (same dimension by default).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app import observability
from app.artifacts.repository import IArtifactsRepository
from app.clients.cross_encoder_client import ICrossEncoderClient
from app.clients.gemini_embedding_client import IGeminiEmbeddingClient
from app.config import (
    LOCATION_HUB_CHAINS_PATH,
    LOCATION_TIER_W_NATIONAL,
    LOCATION_TIER_W_REGIONAL,
)
from app.languages import default_language
from app.ranking.concat_embedding_text import (
    user_concat_embedding_text,
    user_skill_labels_for_concat,
)
from app.ranking.rerank import ICrossEncoderRerank
from app.ranking.retrieval import IStage1Retriever
from app.ranking.skill_matcher import CosineSkillMatcher
from app.ranking.vectors import l2_normalize_rows
from app.ranking.whitening import ConcatWhitener

logger = logging.getLogger(__name__)


class IConcatCrossEncoderEngine(ABC):
    """Interface for the two-stage (concat cosine → cross-encoder) candidate engine."""

    @abstractmethod
    def embed_users(self, users: List[Dict[str, Any]]) -> np.ndarray:
        """
        Gemini concat embeddings for users, L2-normalised (float64 ``[n_users, dim]``).

        Lets a caller embed users ONCE and reuse the matrix across corpora (jobs + occupations).

        :raises ValueError: If the embedding client is not configured
        :raises RuntimeError: If the embedding call fails or returns the wrong row count
        """
        raise NotImplementedError()

    @abstractmethod
    def run(
        self,
        users: List[Dict[str, Any]],
        items: List[Dict[str, Any]],
        *,
        retrieve_top_k: int,
        final_top_k: int,
        mongo_timing: Optional[Dict[str, Any]] = None,
        user_unit_vectors: Optional[np.ndarray] = None,
        apply_location_tier: bool = False,
        corpus: str = "jobs",
    ) -> List[Dict[str, Any]]:
        """
        Shortlists and reranks ``items`` for every user.

        Strips the stage-1 vector fields off ``items`` in place.

        :param users: The users (MatchRequest dicts)
        :param items: Jobs or occupation rows
        :param retrieve_top_k: Stage-1 shortlist size per user
        :param final_top_k: Stage-2 slate size per user
        :param mongo_timing: Job-load timing echoed into ``config_summary``
        :param user_unit_vectors: Precomputed ``embed_users`` output; embedded here if omitted
        :param apply_location_tier: Weight stage-1 by the user's location tier (urban-pull)
        :param corpus: Labels the ``shortlist`` / ``rerank`` trace spans
        :return: One result dict per user with ``concat_gemini_ce_recommendations``
        :raises RuntimeError: If ``user_unit_vectors`` has the wrong shape
        """
        raise NotImplementedError()


class ConcatCrossEncoderEngine(IConcatCrossEncoderEngine):
    """The language is the deployment's (``TARGET_LANGUAGE``); it selects the cross-encoder
    checkpoint for the stage-2 rerank, whose passages are skill-label text. Stage-1 retrieval is
    language-neutral: the skill embeddings are shared across languages (see
    ``ranking/skill_label_packs``).

    Heavy models arrive as providers and are resolved on first use, inside the caller's thread.
    """

    def __init__(
        self,
        *,
        embedding_client: IGeminiEmbeddingClient,
        cross_encoder_provider: Callable[[], ICrossEncoderClient],
        matcher_provider: Callable[[], CosineSkillMatcher],
        whitener_provider: Callable[[], ConcatWhitener],
        artifacts_repository: IArtifactsRepository,
        retriever: IStage1Retriever,
        rerank: ICrossEncoderRerank,
    ):
        self._embedding_client = embedding_client
        self._cross_encoder_provider = cross_encoder_provider
        self._matcher_provider = matcher_provider
        self._whitener_provider = whitener_provider
        self._artifacts_repository = artifacts_repository
        self._retriever = retriever
        self._rerank = rerank
        self._logger = logging.getLogger(self.__class__.__name__)

    def embed_users(self, users: List[Dict[str, Any]]) -> np.ndarray:
        client = self._embedding_client
        client.ensure_configured()
        with observability.stage(
            "embedding", n_users=len(users), model=client.model_name
        ):
            texts = []
            for u in users:
                t = user_concat_embedding_text(u).strip()
                texts.append(t if t else " ")
            u_emb = client.embed_texts(texts, batch_size=100, sleep_s=0.12)
            if u_emb.shape[0] != len(users):
                raise RuntimeError("Gemini embed returned unexpected row count")
            return l2_normalize_rows(u_emb.astype(np.float32)).astype(np.float64)

    def run(
        self,
        users: List[Dict[str, Any]],
        items: List[Dict[str, Any]],
        *,
        retrieve_top_k: int,
        final_top_k: int,
        mongo_timing: Optional[Dict[str, Any]] = None,
        user_unit_vectors: Optional[np.ndarray] = None,
        apply_location_tier: bool = False,
        corpus: str = "jobs",
    ) -> List[Dict[str, Any]]:
        if not users:
            return []
        rt = max(1, int(retrieve_top_k))
        fk = max(1, int(final_top_k))
        embedding_model = self._embedding_client.model_name
        embedding_dim = self._embedding_client.embedding_dim

        with observability.stage("shortlist", corpus=corpus, n_users=len(users)) as sl:
            stage1 = self._retriever.prepare(items)
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
                _hub_chains = self._artifacts_repository.load_hub_chains(
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
                u_norm = self.embed_users(users)

            lang = default_language()
            matcher = self._matcher_provider()
            reranker = self._cross_encoder_provider()

            shortlists = self._retriever.shortlist(
                users,
                stage1,
                u_norm,
                matcher=matcher,
                whitener=self._whitener_provider(),
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

        out_results: List[Dict[str, Any]] = []
        with observability.stage(
            "rerank", corpus=corpus, n_users=len(users), model=reranker.model_name
        ):
            for user, cosine_recs in zip(users, shortlists):
                labels = user_skill_labels_for_concat(user)
                pairs = self._rerank.build_pairs(labels, cosine_recs, final_top_k=fk)
                scores = reranker.predict_scores(pairs) if pairs else []
                reranked = self._rerank.apply(cosine_recs, scores, final_top_k=fk)

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
                            "concat_cosine_similarity": row.get(
                                "concat_cosine_similarity"
                            ),
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
                    cfg["mongo_ranked_find_ms"] = mongo_timing.get(
                        "mongo_ranked_find_ms"
                    )
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
