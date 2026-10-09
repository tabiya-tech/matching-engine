"""Matching service: ``POST /match`` — occupations + opportunities + skill gaps per user.

Runs the v4 engine (Gemini concat cosine -> cross-encoder rerank -> u_hat x p_hat) over BOTH jobs
and occupations using a single shared user embedding, reuses the Node2Vec skill-gap analysis, and
assembles one ``MatchResponse`` per user. Per-item detail is best-effort from the engine outputs
(see ``app.matching.formatting``).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from app import observability
from app.artifacts.repository import IArtifactsRepository
from app.clients.cross_encoder_client import ICrossEncoderClient
from app.clients.gemini_embedding_client import IGeminiEmbeddingClient
from app.config import (
    FINAL_SCORE_COMBINER,
    LOCATION_HUB_CHAINS_PATH,
    LOCATION_TIER_ENABLED,
    LOCATION_TIER_W_NATIONAL,
    LOCATION_TIER_W_REGIONAL,
    MATCH_TOP_K_SKILL_GAPS,
    MATCH_V2_MAX_USERS_PER_REQUEST,
    MATCH_V4_DISABLE_OCCUPATIONS,
    MATCH_V4_FINAL_TOP_K,
    MATCH_V4_OCC_DEMAND_GAMMA,
    MATCH_V4_RETRIEVE_TOP_K,
    MATCH_V4_TOP_K_OCCUPATIONS,
    SKILL_RESCALE_TARGET,
    V4_FULL_COVERAGE_GAMMA,
    V4_FULL_MIN_ESS_SHARE,
    V4_FULL_RANK_DEMOTE,
    V4_FULL_SIM_THRESHOLD,
    V4_FULL_UNPARSED_COVERAGE,
    V4_FULL_WHITENED_GATE,
)
from app.jobs.repository import IJobsRepository
from app.languages import default_language
from app.match_timing_log import log_match_step
from app.matching import formatting as fmt
from app.matching.errors import InvalidMatchRequestError
from app.matching.types import MatchOptions
from app.occupations.repository import IOccupationsRepository
from app.ranking.rerank import ICrossEncoderRerank
from app.ranking.retrieval import (
    ConcatWhitener,
    IStage1Retriever,
    index_by_uuid,
    is_prewhitened,
    l2_normalize_rows,
    stage1_vector,
    user_concat_embedding_text,
    user_matches_any_county,
    user_skill_labels_for_concat,
)
from app.ranking.scoring import (
    RankOverrides,
    enriched_recommendations,
    skill_detail,
)
from app.ranking.skill_gaps import skill_gaps_for
from app.ranking.skills import CosineSkillMatcher, SkillScorer
from app.schemas import MatchRequest, MatchResponse

logger = logging.getLogger(__name__)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _retrieval_trace_meta(
    jobs: list[dict[str, Any]],
    jobs_timing: dict[str, Any],
    occ: list[dict[str, Any]],
    occ_timing: dict[str, Any],
) -> dict[str, Any]:
    """Counts + Mongo / occupation-cache timings for the ``retrieval`` trace span (no job content)."""
    meta: dict[str, Any] = {"n_jobs": len(jobs), "n_occupation_rows": len(occ)}
    for timing in (jobs_timing or {}, occ_timing or {}):
        for k, v in timing.items():
            if isinstance(v, (int, float, bool)):
                meta[k] = v
    return meta


class IMatchingService(ABC):
    """Interface for matching users to occupations, opportunities and skill gaps."""

    @abstractmethod
    async def match(
        self, payload: list[MatchRequest], options: MatchOptions
    ) -> list[MatchResponse]:
        """
        Matches every user in ``payload``; loads the jobs and occupations it needs.

        :param payload: One MatchRequest per user
        :param options: Per-request tuning (None fields take the server defaults)
        :return: One MatchResponse per user, in payload order
        :raises InvalidMatchRequestError: If the payload is empty, has too many users, or
            ``final_score_combiner`` is not ``product`` / ``geometric_mean``
        :raises ValueError: If the engine rejects an input (e.g. no Gemini API key)
        :raises Exception: If any other error occurs
        """
        raise NotImplementedError()

    @abstractmethod
    def rank(
        self,
        users: list[dict[str, Any]],
        jobs: list[dict[str, Any]],
        occupations: list[dict[str, Any]],
        *,
        retrieve_top_k: int,
        final_top_k: int,
        final_score_combiner: str | None = None,
        skill_gap_top_k: int = MATCH_TOP_K_SKILL_GAPS,
        mongo_timing: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ranks already-loaded jobs and occupations for each user (CPU-bound; call off the event loop).

        :return: One MatchResponse-shaped dict per user
        :raises ValueError: If ``final_score_combiner`` is invalid or the engine rejects an input
        """
        raise NotImplementedError()

    @abstractmethod
    def embed_users(self, users: list[dict[str, Any]]) -> np.ndarray:
        """
        Gemini concat embeddings for users, L2-normalised (float64 ``[n_users, dim]``).

        Lets a caller embed users ONCE and reuse the matrix across corpora (jobs + occupations).

        :raises ValueError: If the embedding client is not configured
        :raises RuntimeError: If the embedding call fails or returns the wrong row count
        """
        raise NotImplementedError()

    @abstractmethod
    def shortlist_and_rerank(
        self,
        users: list[dict[str, Any]],
        items: list[dict[str, Any]],
        *,
        retrieve_top_k: int,
        final_top_k: int,
        mongo_timing: dict[str, Any] | None = None,
        user_unit_vectors: np.ndarray | None = None,
        apply_location_tier: bool = False,
        corpus: str = "jobs",
    ) -> list[dict[str, Any]]:
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


class MatchingService(IMatchingService):
    """The deployment's ``TARGET_LANGUAGE`` selects the stage-2 cross-encoder checkpoint, whose
    passages are skill-label text. Stage-1 retrieval and the per-skill gate are
    language-neutral: every language's labels resolve into the same shared embedding space
    (``ranking/skill_label_packs``).

    Heavy models arrive as providers and are resolved on first use, inside the worker thread.
    """

    def __init__(
        self,
        *,
        jobs_repository: IJobsRepository,
        occupations_repository: IOccupationsRepository,
        artifacts_repository: IArtifactsRepository,
        embedding_client: IGeminiEmbeddingClient,
        cross_encoder_provider: Callable[[], ICrossEncoderClient],
        retrieval_matcher_provider: Callable[[], CosineSkillMatcher],
        gate_matcher_provider: Callable[[], CosineSkillMatcher],
        whitener_provider: Callable[[], ConcatWhitener],
        skill_scorer_provider: Callable[[], SkillScorer],
        preference_scorer_provider: Callable[[], Any],
        retriever: IStage1Retriever,
        rerank: ICrossEncoderRerank,
    ):
        self._jobs_repository = jobs_repository
        self._occupations_repository = occupations_repository
        self._artifacts_repository = artifacts_repository
        self._embedding_client = embedding_client
        self._cross_encoder_provider = cross_encoder_provider
        # Per-skill detail attached to each stage-1 row (the shared, non-whitened matcher).
        self._retrieval_matcher_provider = retrieval_matcher_provider
        # Per-skill GATE for matched skills / coverage (whitened by default).
        self._gate_matcher_provider = gate_matcher_provider
        self._whitener_provider = whitener_provider
        self._skill_scorer_provider = skill_scorer_provider
        self._preference_scorer_provider = preference_scorer_provider
        self._retriever = retriever
        self._rerank = rerank
        self._embedding_dim = embedding_client.embedding_dim
        self._logger = logging.getLogger(self.__class__.__name__)

    async def _load_occupations(self):
        """Occupation corpus (+ concat embeddings), as ``(rows, timing)``.

        Returns ``([], {})`` without touching disk or the embedding cache when
        ``MATCH_V4_DISABLE_OCCUPATIONS`` is set — the flag must short-circuit the load too, not just
        the response rows.
        """
        if MATCH_V4_DISABLE_OCCUPATIONS:
            return [], {}
        occ, timing = await self._occupations_repository.load_with_timing()
        return self._occupations_repository.attach_embeddings(occ), timing

    def embed_users(self, users: list[dict[str, Any]]) -> np.ndarray:
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

    def shortlist_and_rerank(
        self,
        users: list[dict[str, Any]],
        items: list[dict[str, Any]],
        *,
        retrieve_top_k: int,
        final_top_k: int,
        mongo_timing: dict[str, Any] | None = None,
        user_unit_vectors: np.ndarray | None = None,
        apply_location_tier: bool = False,
        corpus: str = "jobs",
    ) -> list[dict[str, Any]]:
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
            matcher = self._retrieval_matcher_provider()
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

        out_results: list[dict[str, Any]] = []
        with observability.stage(
            "rerank", corpus=corpus, n_users=len(users), model=reranker.model_name
        ):
            for user, cosine_recs in zip(users, shortlists):
                labels = user_skill_labels_for_concat(user)
                pairs = self._rerank.build_pairs(labels, cosine_recs, final_top_k=fk)
                scores = reranker.predict_scores(pairs) if pairs else []
                reranked = self._rerank.apply(cosine_recs, scores, final_top_k=fk)

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

    async def match(
        self, payload: list[MatchRequest], options: MatchOptions
    ) -> list[MatchResponse]:
        t_req = time.perf_counter()
        if len(payload) > MATCH_V2_MAX_USERS_PER_REQUEST:
            raise InvalidMatchRequestError(
                f"Too many users in one request (max {MATCH_V2_MAX_USERS_PER_REQUEST})."
            )
        if not payload:
            raise InvalidMatchRequestError(
                "Request body must be a non-empty JSON array."
            )

        users = [u.model_dump() for u in payload]
        rt = (
            options.retrieve_top_k
            if options.retrieve_top_k is not None
            else MATCH_V4_RETRIEVE_TOP_K
        )
        ft = (
            options.final_top_k
            if options.final_top_k is not None
            else MATCH_V4_FINAL_TOP_K
        )
        combiner = (options.final_score_combiner or "").strip().lower() or None
        if combiner is not None and combiner not in ("product", "geometric_mean"):
            raise InvalidMatchRequestError(
                "final_score_combiner must be 'product' or 'geometric_mean'"
            )
        skill_gap_top_k = options.skill_gap_top_k

        observability.set_request_users(
            users,
            retrieve_top_k=rt,
            final_top_k=ft,
            final_score_combiner=combiner,
            skill_gap_top_k=skill_gap_top_k,
        )

        t_fetch = time.perf_counter()
        with observability.stage("retrieval") as span:
            (jobs, mongo_timing), (occ, occ_timing) = await asyncio.gather(
                self._jobs_repository.find_active(users),
                self._load_occupations(),
            )
            observability.update_observation(
                span,
                metadata=_retrieval_trace_meta(jobs, mongo_timing, occ, occ_timing),
            )
        fetch_wall_ms = _ms(t_fetch)

        t_score = time.perf_counter()
        raw = await asyncio.to_thread(
            self.rank,
            users,
            jobs,
            occ,
            retrieve_top_k=rt,
            final_top_k=ft,
            final_score_combiner=combiner,
            skill_gap_top_k=skill_gap_top_k
            if skill_gap_top_k is not None
            else MATCH_TOP_K_SKILL_GAPS,
            mongo_timing=mongo_timing,
        )
        score_ms = _ms(t_score)

        with observability.stage("formatting", step="response_model"):
            out: list[MatchResponse] = [MatchResponse(**row) for row in raw]

        log_match_step(
            "http /match",
            "request (summary)",
            n_users=len(users),
            n_jobs=len(jobs),
            n_occupation_rows=len(occ),
            fetch_parallel_wall_ms=fetch_wall_ms,
            scoring_thread_pool_ms=score_ms,
            request_total_ms=_ms(t_req),
        )
        return out

    def _location_tier_overrides(
        self,
        user: dict[str, Any],
        v3_row: dict[str, Any] | None,
        item_index: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Per-uuid location-tier multiplier for a user's job shortlist (urban-pull Part B).

        local=1.0, regional hub=W_REGIONAL, national hub=W_NATIONAL, remote=1.0, off-chain=0.0. Returns ``{}``
        (a no-op) when the feature is disabled or the hub-chain data is unavailable. Runs independent of the
        Phase-2 coverage demotion (no whitening artifact needed)."""
        if not LOCATION_TIER_ENABLED:
            return {}
        hc = self._artifacts_repository.load_hub_chains(LOCATION_HUB_CHAINS_PATH)
        if hc is None:
            return {}
        county = user.get("province") or user.get("city") or ""
        tiers: dict[str, float] = {}
        for r in (v3_row or {}).get("concat_gemini_ce_recommendations") or []:
            if not isinstance(r, dict):
                continue
            uuid = str(r.get("job_uuid") or "")
            if not uuid or uuid in tiers:
                continue
            item = item_index.get(uuid)
            if not item:
                continue
            tiers[uuid] = hc.tier_factor_for_job(
                item,
                county,
                w_regional=LOCATION_TIER_W_REGIONAL,
                w_national=LOCATION_TIER_W_NATIONAL,
            )
        return tiers

    def rank(
        self,
        users: list[dict[str, Any]],
        jobs: list[dict[str, Any]],
        occupations: list[dict[str, Any]],
        *,
        retrieve_top_k: int,
        final_top_k: int,
        final_score_combiner: str | None = None,
        skill_gap_top_k: int = MATCH_TOP_K_SKILL_GAPS,
        mongo_timing: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        combiner = (final_score_combiner or FINAL_SCORE_COMBINER).strip().lower()
        if combiner not in ("product", "geometric_mean"):
            raise ValueError(
                "final_score_combiner must be 'product' or 'geometric_mean'"
            )
        if not users:
            return []

        # Per-deployment kill-switch: with MATCH_V4_DISABLE_OCCUPATIONS every user gets an empty
        # occupation_recommendations list and NO occupation work runs at all (no stage-1 retrieval, no CE
        # rerank, no Phase-2 overrides). Dropping the corpus here makes every occupation-derived structure
        # below empty; the remaining guards skip the calls that would otherwise run on an empty corpus.
        occupations_enabled = not MATCH_V4_DISABLE_OCCUPATIONS
        if not occupations_enabled:
            occupations = []

        u_norm = self.embed_users(users)  # embed users ONCE, reuse for both corpora
        pref_scorer = self._preference_scorer_provider()
        # Per-skill GATE matcher: whitened (default) or — via the kill-switch — the legacy raw matcher.
        matcher = self._gate_matcher_provider()

        job_index = index_by_uuid(jobs)
        occ_index = index_by_uuid(occupations)

        # Phase-2 (V4_FULL_RANK_DEMOTE) ranking inputs. Snapshot the concat embeddings NOW — stage-1
        # retrieval pops job_embedding off these dicts in place — and whiten the user vectors once.
        # When the toggle is off these stay empty and ranking is the Phase-1 behaviour.
        # Safety: if the toggle is on but the concat-whitening artifact is unavailable/incompatible
        # (target==0), DON'T half-apply Phase 2 (raw p_hat x coverage is count-biased — see notes); fall
        # back to pure Phase-1 (annotation-only, ranking unchanged) and log loudly.
        demote_active = (
            V4_FULL_RANK_DEMOTE and self._whitener_provider().rescale_target() > 0
        )
        if V4_FULL_RANK_DEMOTE and not demote_active:
            logger.error(
                "V4_FULL_RANK_DEMOTE is on but the concat-whitening artifact is unavailable; "
                "falling back to Phase-1 (no demotion, raw p_hat). Build/ship the artifact to enable Phase 2."
            )
        job_concat: dict[
            str, tuple
        ] = {}  # uuid -> (stage1_vector, is_already_whitened)
        occ_concat: dict[str, tuple] = {}
        u_white_by_uid: dict[str, np.ndarray] = {}
        rank_overrides: RankOverrides | None = None
        if demote_active:
            whitener = self._whitener_provider()
            rank_overrides = RankOverrides(
                matcher=matcher,
                whitener=whitener,
                whitened_gate=V4_FULL_WHITENED_GATE,
                sim_threshold=V4_FULL_SIM_THRESHOLD,
                unparsed_coverage=V4_FULL_UNPARSED_COVERAGE,
            )
            with observability.stage("preference_scoring", step="phase2_inputs"):
                for j in jobs:
                    v = stage1_vector(j, dim=self._embedding_dim)
                    if v is not None:
                        # (vector, is_already_whitened) — DB-whitened jobs are consumed directly in Phase-2;
                        # raw vectors are whitened in-process. Same artifact => identical result either way.
                        job_concat[str(j.get("uuid") or "")] = (v, is_prewhitened(j))
                for o in occupations:
                    v = stage1_vector(o, dim=self._embedding_dim)
                    if v is not None:
                        # occupations are whitened once at cache-load (consumed directly) or raw (whitened
                        # in-process) — the flag set by OccupationsRepository.attach_embeddings tells which.
                        occ_concat[str(o.get("uuid") or "")] = (v, is_prewhitened(o))
                u_white = whitener.whiten_rows(u_norm)
                u_white_by_uid = {
                    str(u.get("user_id") or ""): u_white[i] for i, u in enumerate(users)
                }

        job_v3 = self.shortlist_and_rerank(
            users,
            jobs,
            retrieve_top_k=retrieve_top_k,
            final_top_k=final_top_k,
            mongo_timing=mongo_timing,
            user_unit_vectors=u_norm,
            # Urban-pull: weight the stage-1 cosine by location tier so relevant local jobs survive the
            # retrieve_top_k cutoff (occupations keep their own county scoping, so not applied there).
            apply_location_tier=LOCATION_TIER_ENABLED,
        )
        # Occupations are flattened into 4 identical-embedding county-rows per code (the fixed sample
        # counties Kilifi/Kitui/Mombasa/Nairobi). The per-user location filter (below) keeps only the
        # user's own county row per code, so the shortlist/CE slate must be wide enough that ~top_k
        # distinct codes survive: size at top_k * 4 (counties) * 2 (buffer). De-dup by code remains a
        # safety net.
        occ_breadth = max(retrieve_top_k, final_top_k, MATCH_V4_TOP_K_OCCUPATIONS * 8)
        occ_v3 = (
            self.shortlist_and_rerank(
                users,
                occupations,
                retrieve_top_k=occ_breadth,
                final_top_k=occ_breadth,
                user_unit_vectors=u_norm,
                corpus="occupations",
            )
            if occupations_enabled
            else []
        )
        job_v3_by_uid = {str(r.get("user_id") or ""): r for r in job_v3}
        occ_v3_by_uid = {str(r.get("user_id") or ""): r for r in occ_v3}

        # Available occupation counties (Kilifi/Kitui/Mombasa/Nairobi). Safety net: if a user's province
        # matches none of them, fall back to a random available county so occupations still return.
        occ_counties = sorted(
            {str(o.get("province")) for o in occupations if o.get("province")}
        )

        def _skill_detail(user, item):
            return skill_detail(
                matcher, user, item, whitened_gate=V4_FULL_WHITENED_GATE
            )

        out: list[dict[str, Any]] = []
        cov_gamma = V4_FULL_COVERAGE_GAMMA if demote_active else 0.0
        for user in users:
            uid = str(user.get("user_id") or "")

            # Opportunities. Jobs keep the existing location scoping (Mongo prefilter via
            # JobsRepository.find_active(users)). Instead of a hard python location filter, urban-pull
            # applies a per-user SOFT location tier (local=1.0 > regional hub > national hub; off-chain=0)
            # as a final_score multiplier — local jobs preferred, hub jobs surface when better/needed, and
            # off-chain jobs (e.g. another batch user's locations) are dropped. Always-on (independent of
            # the Phase-2 coverage demotion); {} no-op when LOCATION_TIER_ENABLED is off.
            with observability.stage("preference_scoring", corpus="jobs"):
                # Phase-2 ranking overrides (whitened+rescaled p_hat + coverage demotion); empty dicts when
                # the toggle is off, in which case enriched_recommendations falls back to the stage-1 concat
                # cosine p_hat (itself whitened, just unrescaled) with no demotion (Phase 1).
                job_p, job_cov, job_det = ({}, {}, {})
                if rank_overrides is not None:
                    job_p, job_cov, job_det = rank_overrides.compute(
                        user,
                        job_v3_by_uid.get(uid),
                        job_index,
                        job_concat,
                        u_white_by_uid.get(uid),
                    )
                job_tiers = self._location_tier_overrides(
                    user, job_v3_by_uid.get(uid), job_index
                )
                job_recs = enriched_recommendations(
                    user,
                    job_v3_by_uid.get(uid),
                    job_index,
                    pref_scorer,
                    combiner,
                    location_filter=False,
                    p_hat_by_uuid=job_p,
                    coverage_by_uuid=job_cov,
                    coverage_gamma=cov_gamma,
                    location_tier_by_uuid=job_tiers,
                )
            opportunities: list[dict[str, Any]] = []
            with observability.stage("formatting", corpus="jobs"):
                for rec in job_recs:
                    item = job_index.get(str(rec.get("job_uuid") or ""))
                    if not item:
                        continue
                    per, ess_ids = job_det.get(
                        str(rec.get("job_uuid") or "")
                    ) or _skill_detail(user, item)
                    opportunities.append(
                        fmt.build_opportunity_row(
                            rec,
                            item,
                            per,
                            ess_ids,
                            rank=len(opportunities) + 1,
                            sim_threshold=V4_FULL_SIM_THRESHOLD,
                            min_ess_share=V4_FULL_MIN_ESS_SHARE,
                        )
                    )

            # Occupations: filter to the user's county; if the user's province matches no occupation
            # county, fall back to a random available county (location filter only — the user's real
            # preferences still drive u_hat). Then dedupe by code, keep best-ranked, take top-k.
            # Skipped entirely (empty list) when MATCH_V4_DISABLE_OCCUPATIONS is set.
            occupations_out: list[dict[str, Any]] = []
            if occupations_enabled:
                with observability.stage("preference_scoring", corpus="occupations"):
                    occ_p, occ_cov, occ_det = ({}, {}, {})
                    if rank_overrides is not None:
                        occ_p, occ_cov, occ_det = rank_overrides.compute(
                            user,
                            occ_v3_by_uid.get(uid),
                            occ_index,
                            occ_concat,
                            u_white_by_uid.get(uid),
                        )
                    loc_user = None
                    if occ_counties and not user_matches_any_county(user, occ_counties):
                        fallback = random.choice(occ_counties)
                        loc_user = {
                            "city": fallback,
                            "province": fallback,
                            "location": fallback,
                        }
                        logger.warning(
                            "User %r province=%r matches no occupation county %s; using random fallback county %r.",
                            uid,
                            user.get("province"),
                            occ_counties,
                            fallback,
                        )
                    occ_recs = enriched_recommendations(
                        user,
                        occ_v3_by_uid.get(uid),
                        occ_index,
                        pref_scorer,
                        combiner,
                        location_user=loc_user,
                        include_demand=True,
                        demand_gamma=MATCH_V4_OCC_DEMAND_GAMMA,
                        p_hat_by_uuid=occ_p,
                        coverage_by_uuid=occ_cov,
                        coverage_gamma=cov_gamma,
                    )
                with observability.stage("formatting", corpus="occupations"):
                    seen_codes: set = set()
                    for rec in occ_recs:
                        item = occ_index.get(str(rec.get("job_uuid") or ""))
                        if not item:
                            continue
                        code = str(item.get("originUuid") or item.get("uuid") or "")
                        if not code or code in seen_codes:
                            continue
                        seen_codes.add(code)
                        per, ess_ids = occ_det.get(
                            str(rec.get("job_uuid") or "")
                        ) or _skill_detail(user, item)
                        occupations_out.append(
                            fmt.build_occupation_row(
                                rec,
                                item,
                                per,
                                ess_ids,
                                rank=len(occupations_out) + 1,
                                sim_threshold=V4_FULL_SIM_THRESHOLD,
                                min_ess_share=V4_FULL_MIN_ESS_SHARE,
                            )
                        )
                        if len(occupations_out) >= MATCH_V4_TOP_K_OCCUPATIONS:
                            break

            with observability.stage("skill_gaps"):
                skill_gaps = skill_gaps_for(
                    user,
                    jobs,
                    skill_gap_top_k,
                    scorer=self._skill_scorer_provider(),
                    rescale_target=SKILL_RESCALE_TARGET,
                )

            out.append(
                {
                    "user_id": uid,
                    "occupation_recommendations": occupations_out,
                    "opportunity_recommendations": opportunities,
                    "skill_gap_recommendations": skill_gaps,
                }
            )

        return out
