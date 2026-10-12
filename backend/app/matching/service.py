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
from collections.abc import Callable, Sequence
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
from app.domain.job import Job
from app.domain.occupation import Occupation
from app.domain.user import UserProfile
from app.jobs.repository import IJobsRepository, JobsLoadTiming
from app.languages import default_language
from app.match_timing_log import log_match_step
from app.services.job_retrieval import retrieve_jobs_with_timing
from app.matching import formatting as fmt
from app.matching.errors import InvalidMatchRequestError
from app.matching.types import MatchOptions
from app.occupations.loader import (
    OccupationsLoadTiming,
    attach_occupation_embeddings,
    get_all_occupations_with_timing,
)
from app.ranking.rerank import ICrossEncoderRerank
from app.ranking.retrieval import (
    ConcatWhitener,
    IStage1Retriever,
    Place,
    is_prewhitened,
    l2_normalize_rows,
    stage1_vector,
    user_concat_embedding_text,
    user_matches_any_county,
    user_skill_labels_for_concat,
)
from app.ranking.scoring import (
    IPreferenceScorer,
    RankOverrides,
    enriched_recommendations,
    skill_detail,
)
from app.ranking.skill_gaps import skill_gaps_for
from app.ranking.skills import CosineSkillMatcher, SkillScorer
from app.ranking.types import (
    Candidate,
    Corpus,
    MatchItem,
    RankOverrideSet,
    RetrievalSummary,
    UserShortlist,
)
from app.schemas import (
    MatchRequest,
    MatchResponse,
    OccupationRecommendation,
    OpportunityRecommendation,
)

logger = logging.getLogger(__name__)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _retrieval_trace_meta(
    jobs: Sequence[Job],
    jobs_timing: JobsLoadTiming,
    occ: Sequence[Occupation],
    occ_timing: OccupationsLoadTiming | None,
) -> dict[str, float | int | bool]:
    """Counts + Mongo / occupation-cache timings for the ``retrieval`` trace span (no job content)."""
    meta: dict[str, float | int | bool] = {
        "n_jobs": len(jobs),
        "n_occupation_rows": len(occ),
    }
    for timing in (jobs_timing, occ_timing):
        if timing is None:
            continue
        for k, v in timing.model_dump().items():
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
        users: Sequence[UserProfile],
        jobs: Sequence[Job],
        occupations: Sequence[Occupation],
        *,
        retrieve_top_k: int,
        final_top_k: int,
        final_score_combiner: str | None = None,
        skill_gap_top_k: int = MATCH_TOP_K_SKILL_GAPS,
        mongo_timing: JobsLoadTiming | None = None,
    ) -> list[MatchResponse]:
        """
        Ranks already-loaded jobs and occupations for each user (CPU-bound; call off the event loop).

        :param users: The users to match, in response order
        :param jobs: The job pool (already location-prefiltered)
        :param occupations: The occupation corpus with embeddings attached; ignored when
            ``MATCH_V4_DISABLE_OCCUPATIONS`` is set
        :param retrieve_top_k: Stage-1 shortlist size per user
        :param final_top_k: Cross-encoder pool size and maximum opportunities per user
        :param final_score_combiner: ``product`` or ``geometric_mean``; None = ``FINAL_SCORE_COMBINER``
        :param skill_gap_top_k: Skill-gap recommendations per user
        :param mongo_timing: Timing of the job load, echoed into the retrieval summary
        :return: One MatchResponse per user, in input order
        :raises ValueError: If ``final_score_combiner`` is invalid or the engine rejects an input
        """
        raise NotImplementedError()

    @abstractmethod
    def embed_users(self, users: Sequence[UserProfile]) -> np.ndarray:
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
        users: Sequence[UserProfile],
        items: Sequence[MatchItem],
        *,
        retrieve_top_k: int,
        final_top_k: int,
        mongo_timing: JobsLoadTiming | None = None,
        user_unit_vectors: np.ndarray | None = None,
        apply_location_tier: bool = False,
        corpus: Corpus = "jobs",
    ) -> list[UserShortlist]:
        """
        Shortlists and reranks ``items`` for every user.

        :param users: The users
        :param items: Jobs or occupation rows
        :param retrieve_top_k: Stage-1 shortlist size per user
        :param final_top_k: Stage-2 slate size per user
        :param mongo_timing: Job-load timing echoed into ``config_summary``
        :param user_unit_vectors: Precomputed ``embed_users`` output; embedded here if omitted
        :param apply_location_tier: Weight stage-1 by the user's location tier (urban-pull)
        :param corpus: Labels the ``shortlist`` / ``rerank`` trace spans
        :return: One reranked slate per user, in input order
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
        artifacts_repository: IArtifactsRepository,
        embedding_client: IGeminiEmbeddingClient,
        cross_encoder_provider: Callable[[], ICrossEncoderClient],
        retrieval_matcher_provider: Callable[[], CosineSkillMatcher],
        gate_matcher_provider: Callable[[], CosineSkillMatcher],
        whitener_provider: Callable[[], ConcatWhitener],
        skill_scorer_provider: Callable[[], SkillScorer],
        preference_scorer_provider: Callable[[], IPreferenceScorer],
        retriever: IStage1Retriever,
        rerank: ICrossEncoderRerank,
    ):
        self._jobs_repository = jobs_repository
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

    async def _load_occupations(
        self,
    ) -> tuple[list[Occupation], OccupationsLoadTiming | None]:
        """Occupation corpus (+ concat embeddings), as ``(rows, timing)``.

        Returns ``([], None)`` without touching disk or the embedding cache when
        ``MATCH_V4_DISABLE_OCCUPATIONS`` is set — the flag must short-circuit the load too, not just
        the response rows.
        """
        if MATCH_V4_DISABLE_OCCUPATIONS:
            return [], None
        occ, timing = await get_all_occupations_with_timing()
        return attach_occupation_embeddings(occ), timing

    def embed_users(self, users: Sequence[UserProfile]) -> np.ndarray:
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
        users: Sequence[UserProfile],
        items: Sequence[MatchItem],
        *,
        retrieve_top_k: int,
        final_top_k: int,
        mongo_timing: JobsLoadTiming | None = None,
        user_unit_vectors: np.ndarray | None = None,
        apply_location_tier: bool = False,
        corpus: Corpus = "jobs",
    ) -> list[UserShortlist]:
        if not users:
            return []
        rt = max(1, int(retrieve_top_k))
        fk = max(1, int(final_top_k))
        embedding_model = self._embedding_client.model_name
        embedding_dim = self._embedding_client.embedding_dim
        timing_fields = {
            "mongo_ranked_find_ms": mongo_timing.mongo_ranked_find_ms
            if mongo_timing
            else None,
            "jobs_retrieval_filter_applied": mongo_timing.jobs_retrieval_filter_applied
            if mongo_timing
            else None,
        }

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
                empty_summary = RetrievalSummary(
                    stage1="concat_gemini_cosine_mongo_job_vectors",
                    stage2="cross_encoder_rerank",
                    gemini_user_embed_model=embedding_model,
                    embedding_dim=embedding_dim,
                    n_jobs_with_stage1_embedding=0,
                    n_jobs_active_loaded=n_active,
                    **timing_fields,
                )
                return [
                    UserShortlist(
                        user_id=u.user_id or "",
                        n_jobs_scored=0,
                        n_jobs_active_loaded=n_active,
                        candidates=[],
                        config_summary=empty_summary,
                    )
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

        summary = RetrievalSummary(
            stage1="concat_gemini_cosine_mongo_job_vectors",
            stage2="cross_encoder_rerank",
            gemini_user_embed_model=embedding_model,
            cross_encoder_model=reranker.model_name,
            language=lang,
            embedding_dim=embedding_dim,
            retrieve_top_k=rt,
            final_top_k=fk,
            n_jobs_with_stage1_embedding=n_with_emb,
            n_jobs_active_loaded=n_active,
            **timing_fields,
        )
        out_results: list[UserShortlist] = []
        with observability.stage(
            "rerank", corpus=corpus, n_users=len(users), model=reranker.model_name
        ):
            for user, shortlist in zip(users, shortlists):
                labels = user_skill_labels_for_concat(user)
                pairs = self._rerank.build_pairs(labels, shortlist, final_top_k=fk)
                scores = reranker.predict_scores(pairs) if pairs else []
                out_results.append(
                    UserShortlist(
                        user_id=user.user_id or "",
                        n_jobs_scored=n_with_emb,
                        n_jobs_active_loaded=n_active,
                        candidates=self._rerank.apply(
                            shortlist, scores, final_top_k=fk
                        ),
                        config_summary=summary,
                    )
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

        users = [UserProfile.from_request(u) for u in payload]
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
                retrieve_jobs_with_timing(self._jobs_repository, users),
                self._load_occupations(),
            )
            observability.update_observation(
                span,
                metadata=_retrieval_trace_meta(jobs, mongo_timing, occ, occ_timing),
            )
        fetch_wall_ms = _ms(t_fetch)

        t_score = time.perf_counter()
        out: list[MatchResponse] = await asyncio.to_thread(
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
        self, user: UserProfile, shortlist: UserShortlist | None
    ) -> dict[str, float]:
        """Per-uuid location-tier multiplier for a user's job shortlist (urban-pull Part B).

        local=1.0, regional hub=W_REGIONAL, national hub=W_NATIONAL, remote=1.0, off-chain=0.0. Returns ``{}``
        (a no-op) when the feature is disabled or the hub-chain data is unavailable. Runs independent of the
        Phase-2 coverage demotion (no whitening artifact needed)."""
        if not LOCATION_TIER_ENABLED or shortlist is None:
            return {}
        hc = self._artifacts_repository.load_hub_chains(LOCATION_HUB_CHAINS_PATH)
        if hc is None:
            return {}
        county = user.province or user.city or ""
        tiers: dict[str, float] = {}
        for c in shortlist.candidates:
            if not c.uuid or c.uuid in tiers:
                continue
            tiers[c.uuid] = hc.tier_factor_for_job(
                Place.of_item(c.item),
                county,
                w_regional=LOCATION_TIER_W_REGIONAL,
                w_national=LOCATION_TIER_W_NATIONAL,
            )
        return tiers

    def rank(
        self,
        users: Sequence[UserProfile],
        jobs: Sequence[Job],
        occupations: Sequence[Occupation],
        *,
        retrieve_top_k: int,
        final_top_k: int,
        final_score_combiner: str | None = None,
        skill_gap_top_k: int = MATCH_TOP_K_SKILL_GAPS,
        mongo_timing: JobsLoadTiming | None = None,
    ) -> list[MatchResponse]:
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

        # Phase-2 (V4_FULL_RANK_DEMOTE) ranking inputs: the concat embeddings by uuid, and the user
        # vectors whitened once. When the toggle is off these stay empty and ranking is the Phase-1
        # behaviour.
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
        # uuid -> (stage1_vector, is_already_whitened)
        job_concat: dict[str, tuple[np.ndarray, bool]] = {}
        occ_concat: dict[str, tuple[np.ndarray, bool]] = {}
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
                # (vector, is_already_whitened) — DB-whitened jobs are consumed directly in Phase-2;
                # raw vectors are whitened in-process. Same artifact => identical result either way.
                # Occupations are whitened once at cache-load (consumed directly) or raw (whitened
                # in-process) — the flag set by attach_occupation_embeddings tells which.
                for concat, corpus_items in (
                    (job_concat, jobs),
                    (occ_concat, occupations),
                ):
                    for it in corpus_items:
                        v = stage1_vector(it, dim=self._embedding_dim)
                        if v is not None:
                            concat[it.uuid] = (v, is_prewhitened(it))
                u_white = whitener.whiten_rows(u_norm)
                u_white_by_uid = {
                    u.user_id or "": u_white[i] for i, u in enumerate(users)
                }

        job_slates = self.shortlist_and_rerank(
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
        occ_slates = (
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
        job_slate_by_uid = {r.user_id: r for r in job_slates}
        occ_slate_by_uid = {r.user_id: r for r in occ_slates}

        # Available occupation counties (Kilifi/Kitui/Mombasa/Nairobi). Safety net: if a user's province
        # matches none of them, fall back to a random available county so occupations still return.
        occ_counties = sorted({o.province for o in occupations if o.province})

        out: list[MatchResponse] = []
        cov_gamma = V4_FULL_COVERAGE_GAMMA if demote_active else 0.0
        for user in users:
            uid = user.user_id or ""

            # Opportunities. Jobs keep the existing location scoping (Mongo prefilter via
            # retrieve_jobs_with_timing(users)). Instead of a hard python location filter, urban-pull
            # applies a per-user SOFT location tier (local=1.0 > regional hub > national hub; off-chain=0)
            # as a final_score multiplier — local jobs preferred, hub jobs surface when better/needed, and
            # off-chain jobs (e.g. another batch user's locations) are dropped. Always-on (independent of
            # the Phase-2 coverage demotion); {} no-op when LOCATION_TIER_ENABLED is off.
            job_slate = job_slate_by_uid.get(uid)
            with observability.stage("preference_scoring", corpus="jobs"):
                # Phase-2 ranking overrides (whitened+rescaled p_hat + coverage demotion); empty when
                # the toggle is off, in which case enriched_recommendations falls back to the stage-1 concat
                # cosine p_hat (itself whitened, just unrescaled) with no demotion (Phase 1).
                job_over = self._overrides(
                    rank_overrides, user, job_slate, job_concat, u_white_by_uid.get(uid)
                )
                job_recs = enriched_recommendations(
                    user,
                    job_slate.candidates if job_slate else [],
                    pref_scorer,
                    combiner,
                    location_filter=False,
                    p_hat_by_uuid=job_over.p_hat,
                    coverage_by_uuid=job_over.coverage,
                    coverage_gamma=cov_gamma,
                    location_tier_by_uuid=self._location_tier_overrides(
                        user, job_slate
                    ),
                )
            opportunities: list[OpportunityRecommendation] = []
            with observability.stage("formatting", corpus="jobs"):
                for rec in job_recs:
                    item = rec.item
                    if not isinstance(item, Job):
                        continue
                    detail = job_over.skill_details.get(rec.uuid) or skill_detail(
                        matcher, user, item, whitened_gate=V4_FULL_WHITENED_GATE
                    )
                    opportunities.append(
                        fmt.build_opportunity_row(
                            item,
                            detail,
                            rank=len(opportunities) + 1,
                            final_score=rec.final_score,
                            scored=rec,
                            sim_threshold=V4_FULL_SIM_THRESHOLD,
                            min_ess_share=V4_FULL_MIN_ESS_SHARE,
                        )
                    )

            # Occupations: filter to the user's county; if the user's province matches no occupation
            # county, fall back to a random available county (location filter only — the user's real
            # preferences still drive u_hat). Then dedupe by code, keep best-ranked, take top-k.
            # Skipped entirely (empty list) when MATCH_V4_DISABLE_OCCUPATIONS is set.
            occupations_out: list[OccupationRecommendation] = []
            if occupations_enabled:
                occ_slate = occ_slate_by_uid.get(uid)
                with observability.stage("preference_scoring", corpus="occupations"):
                    occ_over = self._overrides(
                        rank_overrides,
                        user,
                        occ_slate,
                        occ_concat,
                        u_white_by_uid.get(uid),
                    )
                    loc_user: Place | None = None
                    if occ_counties and not user_matches_any_county(user, occ_counties):
                        fallback = random.choice(occ_counties)
                        loc_user = Place.region(fallback)
                        logger.warning(
                            "User %r matches no occupation county %s; using random fallback county %r.",
                            uid,
                            occ_counties,
                            fallback,
                        )
                    occ_recs = enriched_recommendations(
                        user,
                        occ_slate.candidates if occ_slate else [],
                        pref_scorer,
                        combiner,
                        location_user=loc_user,
                        include_demand=True,
                        demand_gamma=MATCH_V4_OCC_DEMAND_GAMMA,
                        p_hat_by_uuid=occ_over.p_hat,
                        coverage_by_uuid=occ_over.coverage,
                        coverage_gamma=cov_gamma,
                    )
                with observability.stage("formatting", corpus="occupations"):
                    seen_codes: set[str] = set()
                    for rec in occ_recs:
                        item = rec.item
                        if not isinstance(item, Occupation):
                            continue
                        code = item.originUuid or item.uuid
                        if not code or code in seen_codes:
                            continue
                        seen_codes.add(code)
                        detail = occ_over.skill_details.get(rec.uuid) or skill_detail(
                            matcher, user, item, whitened_gate=V4_FULL_WHITENED_GATE
                        )
                        occupations_out.append(
                            fmt.build_occupation_row(
                                item,
                                detail,
                                rank=len(occupations_out) + 1,
                                final_score=rec.final_score,
                                scored=rec,
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
                MatchResponse(
                    user_id=uid,
                    occupation_recommendations=occupations_out,
                    opportunity_recommendations=opportunities,
                    skill_gap_recommendations=skill_gaps,
                )
            )

        return out

    @staticmethod
    def _overrides(
        rank_overrides: RankOverrides | None,
        user: UserProfile,
        slate: UserShortlist | None,
        concat_by_uuid: dict[str, tuple[np.ndarray, bool]],
        u_white_vec: np.ndarray | None,
    ) -> RankOverrideSet:
        """Phase-2 overrides of one user's slate; empty when Phase 2 is off or there is no slate."""
        if rank_overrides is None or slate is None:
            return RankOverrideSet()
        candidates: list[Candidate] = slate.candidates
        return rank_overrides.compute(user, candidates, concat_by_uuid, u_white_vec)
