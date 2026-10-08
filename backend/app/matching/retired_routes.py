"""Retired matching handlers (legacy ``/match``, ``/experiments/v2|v3|v5/match``).

None of these is registered on a router; ``app.matching.routes`` serves ``POST /match``. They are kept
as they were when retired.
"""

import asyncio
import logging
import time
from typing import Annotated, Any, Dict, List, Optional

from fastapi import Body, Depends, HTTPException, Query

from app import observability
from app.config import (
    COSINE_CROSS_ENCODER_RETRIEVE_TOP_K,
    MATCH_TOP_K_SKILL_GAPS,
    MATCH_V2_HYBRID_TOP_K,
    MATCH_V2_MAX_USERS_PER_REQUEST,
    MATCH_V4_DISABLE_OCCUPATIONS,
    MATCH_V4_FINAL_TOP_K,
    MATCH_V4_RETRIEVE_TOP_K,
)
from app.jobs.repository import JobsRepository
from app.match_timing_log import log_match_step
from app.matching.examples import (
    MATCH_BODY_DESCRIPTION,
    MATCH_BODY_EXAMPLE,
    MATCH_V5_BODY_EXAMPLE,
)
from app.matching.get_matching_service import get_matching_service
from app.matching.service import IMatchingService, _retrieval_trace_meta
from app.occupations.get_occupations_repository import get_occupations_repository
from app.ranking.education import zqf_annotation
from app.schemas import (
    MatchRequest,
    MatchRequestV5,
    MatchResponse,
    MatchResponseV5,
    MatchV2JobRecommendation,
)
from app.server_dependencies.db_dependencies import get_jobs_db
from app.services.match_v2_full_service import run_match_v2_full
from app.services.match_v3_full_service import run_match_v3_full
from app.services.matching_service import match_user_with_data

logger = logging.getLogger(__name__)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


async def get_all_jobs_with_timing(users=None):
    return await JobsRepository(db=get_jobs_db()).find_active(users)


async def get_all_occupations_with_timing():
    return await get_occupations_repository().load_with_timing()


def attach_occupation_embeddings(occupations):
    return get_occupations_repository().attach_embeddings(occupations)


async def _load_v4_occupations():
    """Occupation corpus (+ concat embeddings) for the v4 engine, as ``(rows, timing)``.

    Returns ``([], {})`` without touching disk or the embedding cache when
    ``MATCH_V4_DISABLE_OCCUPATIONS`` is set — the v4/v5 routes are the only occupation consumers of
    this corpus, so the flag must short-circuit the load too, not just the response rows.
    """
    if MATCH_V4_DISABLE_OCCUPATIONS:
        return [], {}
    occ, timing = await get_all_occupations_with_timing()
    return attach_occupation_embeddings(occ), timing


def _jobs_by_uuid(job_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for job in job_list:
        uid = str(job.get("uuid") or job.get("_id") or "")
        if uid:
            out[uid] = job
    return out


def _fused_rows_to_match_v2_jobs(
    fused_rows: List[Dict[str, Any]],
    job_index: Dict[str, Dict[str, Any]],
) -> List[MatchV2JobRecommendation]:
    recs: List[MatchV2JobRecommendation] = []
    for row in fused_rows:
        ju = str(row.get("job_uuid") or "")
        jb = job_index.get(ju) or {}
        url = jb.get("url") or jb.get("URL")
        fs = row.get("fusion_score")
        if fs is None:
            fs = row.get("weighted_minmax_fusion") or 0.0
        recs.append(
            MatchV2JobRecommendation(
                rank=int(row.get("rank") or 0),
                job_uuid=ju,
                opportunity_title=str(row.get("job_title") or ""),
                employer=row.get("employer"),
                location=row.get("location"),
                URL=url,
                fusion_score=float(fs),
                bm25_norm_within_candidates=row.get("bm25_norm_within_candidates"),
                cos_norm_within_candidates=row.get("cos_norm_within_candidates"),
                mean_best_cosine_raw=row.get("mean_best_cosine_raw"),
                bm25_score_raw=row.get("bm25_score_raw"),
                matched_skills=[str(x) for x in (row.get("matched_skills") or [])],
                matched_skills_cosine=[
                    str(x) for x in (row.get("matched_skills_cosine") or [])
                ],
            )
        )
    return recs


def _execute_hybrid_http(
    users: List[Dict[str, Any]],
    jobs: List[Dict[str, Any]],
    *,
    fusion_top_k: int,
    alpha_on_cosine: float,
) -> Dict[str, Any]:
    from app.services.hybrid_scoring.run_bm25_cosine_hybrid import (
        hybrid_match_users_with_jobs,
    )

    return hybrid_match_users_with_jobs(
        users,
        jobs,
        col_display_k=fusion_top_k,
        alpha_on_cosine=alpha_on_cosine,
    )


# Retired: not registered on any router; ``app.matching.routes`` serves POST /match.
async def match_legacy(
    payload: Annotated[
        List[MatchRequest],
        Body(..., description=MATCH_BODY_DESCRIPTION, example=MATCH_BODY_EXAMPLE),
    ],
):
    """Match one or more users. Body is a JSON array of MatchRequest (use length 1 for a single user)."""

    try:
        # One Mongo + one occupation load per request; run each user in a thread pool
        # (CPU-bound scoring) so concurrent requests are not stuck behind one GIL.
        t_req = time.perf_counter()
        users = [u.model_dump() for u in payload]
        n_users = len(users)
        observability.set_request_users(users)

        # Mongo ping runs at app startup (warmup_on_startup), not here — avoids multi-second noise per request.
        t_fetch = time.perf_counter()
        with observability.stage("retrieval") as span:
            (jobs, jobs_timing), (occ, occ_timing) = await asyncio.gather(
                get_all_jobs_with_timing(users=users),
                get_all_occupations_with_timing(),
            )
            observability.update_observation(
                span, metadata=_retrieval_trace_meta(jobs, jobs_timing, occ, occ_timing)
            )
        fetch_parallel_wall_ms = _ms(t_fetch)
        t_score = time.perf_counter()
        tasks = [asyncio.to_thread(match_user_with_data, u, jobs, occ) for u in users]
        results = await asyncio.gather(*tasks)
        scoring_ms = _ms(t_score)

        log_match_step(
            "http /match",
            "request (summary)",
            n_users=n_users,
            n_jobs=len(jobs),
            n_occupation_rows=len(occ),
            fetch_parallel_wall_ms=fetch_parallel_wall_ms,
            scoring_thread_pool_ms=scoring_ms,
            request_total_ms=_ms(t_req),
        )
        return results
    except ValueError as e:
        logger.exception(e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        logger.exception(e)
        raise e
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e.__class__.__name__}"
        )


# Retired: not registered on any router; ``app.matching.routes`` serves POST /match.
async def match_v2(
    payload: Annotated[
        List[MatchRequest],
        Body(..., description=MATCH_BODY_DESCRIPTION, example=MATCH_BODY_EXAMPLE),
    ],
    fusion_top_k: Optional[int] = Query(
        None,
        ge=1,
        le=500,
        description=(
            "Max hybrid fused opportunities per user (pool min–max ranking). "
            f"Defaults to MATCH_V2_HYBRID_TOP_K ({MATCH_V2_HYBRID_TOP_K})."
        ),
    ),
    alpha_on_cosine: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Fusion weight on normalised cosine; BM25 receives (1−α). "
            "Overrides env HYBRID_ALPHA_ON_COSINE when set."
        ),
    ),
    skill_gap_top_k: Optional[int] = Query(
        None,
        ge=1,
        le=50,
        description="Number of skill-gap recommendations. Default: MATCH_TOP_K_SKILL_GAPS.",
    ),
):
    """Hybrid BM25 × cosine-skill embeddings, returned in the full ``MatchResponse`` shape.

    **Matching formula is unchanged from the original v2 engine** (BM25 × embedding-cosine pool
    min–max fusion); only the response is reshaped to match ``POST /match_v4``: opportunities,
    occupations and skill gaps. Opportunities load **all active jobs** (``is_active`` only) — no
    per-user Mongo location prefilter — and occupations are scored with the **same** hybrid engine
    over the occupation corpus (county-scoped like v4). ``final_score`` is the hybrid fusion score;
    v4-only ``u_hat``/``p_hat``/preference fields are empty (the v2 engine produces no such signal).

    Does **not** require ``x-api-key`` (temporary; gated separately from ``POST /match``).
    """

    from app.services.hybrid_scoring.run_bm25_cosine_hybrid import (
        _alpha_on_cosine_from_env,
    )

    try:
        t_req = time.perf_counter()
        if len(payload) > MATCH_V2_MAX_USERS_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Too many users in one request (max {MATCH_V2_MAX_USERS_PER_REQUEST}).",
            )
        if not payload:
            raise HTTPException(
                status_code=400, detail="Request body must be a non-empty JSON array."
            )

        users = [u.model_dump() for u in payload]
        n_users = len(users)
        fk = fusion_top_k if fusion_top_k is not None else MATCH_V2_HYBRID_TOP_K

        env_alpha, _env_key = _alpha_on_cosine_from_env()
        alpha = (
            alpha_on_cosine
            if alpha_on_cosine is not None
            else (env_alpha if env_alpha is not None else 0.5)
        )

        observability.set_request_users(
            users,
            fusion_top_k=fk,
            alpha_on_cosine=alpha,
            skill_gap_top_k=skill_gap_top_k,
        )

        t_fetch = time.perf_counter()
        # Full active catalog (no union location filter) + occupation corpus, in parallel.
        with observability.stage("retrieval") as span:
            (jobs, mongo_timing), (occ, occ_timing) = await asyncio.gather(
                get_all_jobs_with_timing(users=None),
                get_all_occupations_with_timing(),
            )
            observability.update_observation(
                span,
                metadata=_retrieval_trace_meta(jobs, mongo_timing, occ, occ_timing),
            )
        fetch_wall_ms = _ms(t_fetch)

        t_score = time.perf_counter()
        raw = await asyncio.to_thread(
            run_match_v2_full,
            users,
            jobs,
            occ,
            fusion_top_k=fk,
            alpha_on_cosine=alpha,
            skill_gap_top_k=skill_gap_top_k
            if skill_gap_top_k is not None
            else MATCH_TOP_K_SKILL_GAPS,
        )
        score_ms = _ms(t_score)

        with observability.stage("formatting", step="response_model"):
            out: List[MatchResponse] = [MatchResponse(**row) for row in raw]

        log_match_step(
            "http /experiments/v2/match",
            "request (summary)",
            n_users=n_users,
            n_jobs=len(jobs),
            n_occupation_rows=len(occ),
            fetch_parallel_wall_ms=fetch_wall_ms,
            scoring_thread_pool_ms=score_ms,
            request_total_ms=_ms(t_req),
        )
        return out

    except HTTPException:
        raise
    except ImportError as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail="Hybrid matching requires optional dependency rank-bm25 (pip install rank-bm25).",
        ) from e
    except ValueError as e:
        logger.exception(e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e.__class__.__name__}"
        )


# Retired: not registered on any router; ``app.matching.routes`` serves POST /match.
async def match_v3(
    payload: Annotated[
        List[MatchRequest],
        Body(
            ...,
            description=(
                MATCH_BODY_DESCRIPTION
                + " When JOBS_RETRIEVAL_FILTER is on, city/province must overlap job locations in Mongo."
            ),
            example=MATCH_BODY_EXAMPLE,
        ),
    ],
    retrieve_top_k: Optional[int] = Query(
        None,
        ge=1,
        le=500,
        description=(
            "Stage-1 concat cosine shortlist size. "
            f"Default: COSINE_CROSS_ENCODER_RETRIEVE_TOP_K ({COSINE_CROSS_ENCODER_RETRIEVE_TOP_K})."
        ),
    ),
    final_top_k: Optional[int] = Query(
        None,
        ge=1,
        le=200,
        description="Stage-2 cross-encoder slate size after rerank. Default: 30.",
    ),
    skill_gap_top_k: Optional[int] = Query(
        None,
        ge=1,
        le=50,
        description="Number of skill-gap recommendations. Default: MATCH_TOP_K_SKILL_GAPS.",
    ),
):
    """Gemini concat-cosine → CE rerank, returned in the full ``MatchResponse`` shape.

    **Matching logic is unchanged from the original v3 engine** (Gemini user concat embedding ×
    Mongo job vectors → cross-encoder rerank); only the response is reshaped to match
    ``POST /match_v4``: opportunities, occupations and skill gaps. Occupations are scored with the
    **same** v3 engine over the occupation corpus (county-scoped like v4). ``final_score`` is the
    concat cosine similarity (whitened space when the concat artifact is present); v4-only
    ``u_hat``/``p_hat``/preference fields are empty (the v3 engine produces no such signal).

    **Database:** reads active jobs via ``MONGO_URL``, ``MONGO_DB_NAME``, ``MONGO_JOBS_COLLECTION``.
    Stage-1 vectors may come from ``concat_skill_embedding_gemini.vector_bin`` **or** a numeric
    ``job_embedding`` array of length **3072** (same dim as ``gemini-embedding-001`` user vectors).

    **Does not** require ``x-api-key``. Users are embedded with ``GEMINI_API_KEY``.
    """
    try:
        t_req = time.perf_counter()
        if len(payload) > MATCH_V2_MAX_USERS_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Too many users in one request (max {MATCH_V2_MAX_USERS_PER_REQUEST}).",
            )
        if not payload:
            raise HTTPException(
                status_code=400, detail="Request body must be a non-empty JSON array."
            )

        users = [u.model_dump() for u in payload]
        rt = (
            retrieve_top_k
            if retrieve_top_k is not None
            else COSINE_CROSS_ENCODER_RETRIEVE_TOP_K
        )
        ft = final_top_k if final_top_k is not None else 30
        observability.set_request_users(
            users, retrieve_top_k=rt, final_top_k=ft, skill_gap_top_k=skill_gap_top_k
        )

        t_fetch = time.perf_counter()
        with observability.stage("retrieval") as span:
            (jobs, mongo_timing), (occ, occ_timing) = await asyncio.gather(
                get_all_jobs_with_timing(users=users),
                get_all_occupations_with_timing(),
            )
            occ = attach_occupation_embeddings(occ)
            observability.update_observation(
                span,
                metadata=_retrieval_trace_meta(jobs, mongo_timing, occ, occ_timing),
            )
        fetch_wall_ms = _ms(t_fetch)

        t_score = time.perf_counter()
        raw = await asyncio.to_thread(
            run_match_v3_full,
            users,
            jobs,
            occ,
            retrieve_top_k=rt,
            final_top_k=ft,
            skill_gap_top_k=skill_gap_top_k
            if skill_gap_top_k is not None
            else MATCH_TOP_K_SKILL_GAPS,
        )
        score_ms = _ms(t_score)

        with observability.stage("formatting", step="response_model"):
            out: List[MatchResponse] = [MatchResponse(**row) for row in raw]

        log_match_step(
            "http /experiments/v3/match",
            "request (summary)",
            n_users=len(users),
            n_jobs=len(jobs),
            n_occupation_rows=len(occ),
            fetch_parallel_wall_ms=fetch_wall_ms,
            scoring_thread_pool_ms=score_ms,
            request_total_ms=_ms(t_req),
        )
        return out

    except HTTPException:
        raise
    except ValueError as e:
        logger.exception(e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e.__class__.__name__}"
        ) from e


# ---------------------------------------------------------------------------
# Experiment: /experiments/v5/match
# ---------------------------------------------------------------------------


# Retired: not registered on any router; ``app.matching.routes`` serves POST /match.
async def match_v5(
    payload: Annotated[
        List[MatchRequestV5],
        Body(
            ...,
            description=(
                "Same body as ``POST /match_v4`` plus ``zqf_level`` (optional int). "
                "Returns opportunities annotated with ``zqf_eligible``, ``zqf_gap``, "
                "and ZQF labels from Mongo job ``classifier_metadata``. "
                "For Zambia deployments use ``zqf_level`` only; "
                "``any_post_secondary_educ`` is the Kenya post-secondary gate (optional, omit for Zambia)."
            ),
            example=MATCH_V5_BODY_EXAMPLE,
        ),
    ],
    retrieve_top_k: Optional[int] = Query(
        None,
        ge=1,
        le=500,
        description=f"Stage-1 concat cosine shortlist size. Default: {COSINE_CROSS_ENCODER_RETRIEVE_TOP_K}.",
    ),
    final_top_k: Optional[int] = Query(
        None,
        ge=1,
        le=200,
        description="CE pool size and max preference-ranked rows returned. Default: 30.",
    ),
    final_score_combiner: Optional[str] = Query(
        None,
        description="How to combine u_hat and p_hat: 'product' or 'geometric_mean'.",
    ),
    skill_gap_top_k: Optional[int] = Query(
        None,
        ge=1,
        le=50,
        description="Number of skill-gap recommendations.",
    ),
    matching_service: IMatchingService = Depends(get_matching_service),
):
    """Experiment: matching with ZQF education annotation on opportunities.

    Runs the full matching pipeline then annotates each opportunity with ``zqf_eligible``
    and ``zqf_gap`` based on the user's ``zqf_level`` and the job's ``zqf_min``.
    """
    try:
        t_req = time.perf_counter()
        if len(payload) > MATCH_V2_MAX_USERS_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Too many users in one request (max {MATCH_V2_MAX_USERS_PER_REQUEST}).",
            )
        if not payload:
            raise HTTPException(
                status_code=400, detail="Request body must be a non-empty JSON array."
            )

        users = [u.model_dump() for u in payload]
        rt = retrieve_top_k if retrieve_top_k is not None else MATCH_V4_RETRIEVE_TOP_K
        ft = final_top_k if final_top_k is not None else MATCH_V4_FINAL_TOP_K
        combiner = (final_score_combiner or "").strip().lower() or None
        if combiner is not None and combiner not in ("product", "geometric_mean"):
            raise HTTPException(
                status_code=400,
                detail="final_score_combiner must be 'product' or 'geometric_mean'",
            )

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
                get_all_jobs_with_timing(users=users),
                _load_v4_occupations(),
            )
            observability.update_observation(
                span,
                metadata=_retrieval_trace_meta(jobs, mongo_timing, occ, occ_timing),
            )
        fetch_wall_ms = _ms(t_fetch)

        job_uuid_index = _jobs_by_uuid(jobs)

        t_score = time.perf_counter()
        raw = await asyncio.to_thread(
            matching_service.rank,
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

        with observability.stage(
            "formatting", step="zqf_annotation_and_response_model"
        ):
            for row, user in zip(raw, users):
                user_zqf = user.get("zqf_level")
                for opp in row.get("opportunity_recommendations") or []:
                    job = job_uuid_index.get(str(opp.get("uuid") or ""))
                    job_zqf_min = job.get("zqf_min") if job else None
                    eligible, gap = zqf_annotation(user_zqf, job_zqf_min)
                    opp["zqf_eligible"] = eligible
                    opp["zqf_gap"] = gap
                    opp["zqf_min_label"] = job.get("zqf_min_label") if job else None
                    opp["zqf_max_label"] = job.get("zqf_max_label") if job else None

            out: List[MatchResponseV5] = [MatchResponseV5(**row) for row in raw]

        log_match_step(
            "http /experiments/v5/match",
            "request (summary)",
            n_users=len(users),
            n_jobs=len(jobs),
            n_occupation_rows=len(occ),
            fetch_parallel_wall_ms=fetch_wall_ms,
            scoring_thread_pool_ms=score_ms,
            request_total_ms=_ms(t_req),
        )
        return out

    except HTTPException:
        raise
    except ValueError as e:
        logger.exception(e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e.__class__.__name__}"
        ) from e
