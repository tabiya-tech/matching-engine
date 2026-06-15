import asyncio
import logging
import time
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel

from fastapi import APIRouter, Body, HTTPException, Depends, Query
from fastapi.security import APIKeyHeader

from app.schemas import (
    MatchRequest,
    MatchResponse,
    MatchV2JobRecommendation,
    MatchRequestV5,
    MatchResponseV5,
    JobsPage,
    JobsStats,
)
from app.config import (
    MATCH_V2_HYBRID_TOP_K,
    MATCH_V2_MAX_USERS_PER_REQUEST,
    COSINE_CROSS_ENCODER_RETRIEVE_TOP_K,
    MATCH_V4_RETRIEVE_TOP_K,
    MATCH_V4_FINAL_TOP_K,
    MATCH_TOP_K_SKILL_GAPS,
    JOBS_PAGE_DEFAULT_LIMIT,
    JOBS_PAGE_MAX_LIMIT,
    DEBUG_MODE,
)
from app.database import (
    attach_occupation_embeddings,
    get_all_jobs_with_timing,
    get_all_occupations_with_timing,
    get_jobs_page_with_timing,
    get_jobs_stats,
    InvalidCursor,
)
from app.match_timing_log import log_match_step
from app.services.matching_service import match_user_with_data
from app.services.match_v2_full_service import run_match_v2_full
from app.services.match_v3_full_service import run_match_v3_full
from app.services.match_v4_full_service import run_match_v4_full

api_key_auth = APIKeyHeader(
    scheme_name="gcp_api_key", name="x-api-key", auto_error=True
)

router = APIRouter(dependencies=[Depends(api_key_auth)])
# Public: /experiments/v2/match (BM25×cosine), /experiments/v3/match (Gemini+CE), /match_v4 (Gemini+CE+preference final).
router_public = APIRouter()
logger = logging.getLogger(__name__)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


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


class Health(BaseModel):
    status: str


# Swagger default for Kenya / post-secondary endpoints (/match, /match_v2, /match_v3, /match_v4).
_MATCH_BODY_EXAMPLE: List[Dict[str, Any]] = [
    {
        "user_id": "u1",
        "city": "Nairobi",
        "province": "Nairobi",
        "any_post_secondary_educ": 1,
        "skills_vector": {
            "top_skills": [
                {
                    "originUUID": "00000000-0000-4000-8000-000000000001",
                    "preferredLabel": "customer service",
                    "proficiency": 0.8,
                }
            ]
        },
        "skill_groups_origin_uuids": [],
        "preference_vector": {
            "earnings_per_month": 0,
            "physical_demand": 0,
            "social_interaction": 0,
            "career_growth": 0,
        },
    }
]

_MATCH_BODY_DESCRIPTION = (
    "JSON **array** of MatchRequest (one object per user). "
    "``any_post_secondary_educ``: ``0`` = no post-secondary (jobs with "
    "``requires_post_secondary`` are filtered out), ``1`` = has post-secondary, "
    "omit to disable the education gate."
)

# Swagger default for /experiments/v5/match (Zambia: ZQF annotation on opportunities).
_MATCH_V5_BODY_EXAMPLE: List[Dict[str, Any]] = [
    {
        "user_id": "u1",
        "city": "Lusaka",
        "province": "Lusaka",
        "zqf_level": 4,
        "skills_vector": {
            "top_skills": [
                {
                    "originUUID": "00000000-0000-4000-8000-000000000001",
                    "preferredLabel": "prepare bakery products",
                    "proficiency": 0.85,
                },
                {
                    "originUUID": "00000000-0000-4000-8000-000000000002",
                    "preferredLabel": "bake goods",
                    "proficiency": 0.78,
                },
            ]
        },
        "skill_groups_origin_uuids": [],
        "preference_vector": {
            "earnings_per_month": 0.6,
            "physical_demand": 0.5,
            "social_interaction": 0.5,
            "career_growth": 0.6,
        },
    }
]


@router.get("/health")
async def health() -> Health:
    return Health(status="ok")


@router.get(
    "/jobs",
    tags=["jobs"],
    operation_id="list_jobs",
    response_model=JobsPage,
    responses={
        400: {
            "description": "Bad Request - invalid cursor",
            "content": {
                "application/json": {"example": {"detail": "invalid cursor"}}
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {"example": {"detail": "Internal server error"}}
            },
        },
    },
)
async def list_jobs(
    cursor: Optional[str] = Query(
        None,
        description=(
            "Opaque pagination cursor returned as ``next_cursor`` by the previous "
            "response. Omit to fetch the first page."
        ),
    ),
    limit: int = Query(
        JOBS_PAGE_DEFAULT_LIMIT,
        ge=1,
        le=JOBS_PAGE_MAX_LIMIT,
        description=f"Page size (1–{JOBS_PAGE_MAX_LIMIT}). Default {JOBS_PAGE_DEFAULT_LIMIT}.",
    ),
    search: Optional[str] = Query(None, description="Case-insensitive search on the job title."),
    category: Optional[str] = Query(None, description="Filter by sector/category (matches category, sector, or ISCO group)."),
    employment_type: Optional[str] = Query(None, description="Filter by employment type (exact match)."),
    location: Optional[str] = Query(None, description="Case-insensitive filter on city/county/province."),
    skills: Optional[str] = Query(None, description="Case-insensitive filter on a skill label of the opportunity."),
    days: Optional[int] = Query(None, ge=1, le=3650, description="Only jobs posted within the last N days."),
    include_total: bool = Query(False, description="When true, include the total count of jobs matching the filters."),
):
    """
    Browse active jobs with cursor-based pagination and optional filters.

    Reads from the same Mongo collection and through the same shaping as the matched-jobs
    endpoints (CORE-418), so a browsed job and a matched job are the same object minus the
    per-user scoring fields. Results are ordered newest-first (``_id`` descending) and the
    keyset cursor is stable under concurrent inserts. Supplied filters are AND-ed together;
    pass ``include_total=true`` to also receive the total count for the active filter set.
    """
    try:
        t_req = time.perf_counter()
        jobs, next_cursor, total, timing = await get_jobs_page_with_timing(
            cursor=cursor,
            limit=limit,
            search=search,
            category=category,
            employment_type=employment_type,
            location=location,
            skills=skills,
            days=days,
            include_total=include_total,
        )
        log_match_step(
            "http /jobs",
            "request (summary)",
            n_jobs=len(jobs),
            has_more=timing.get("has_more"),
            limit=timing.get("limit"),
            total=total,
            request_total_ms=_ms(t_req),
        )
        return JobsPage(items=jobs, next_cursor=next_cursor, total=total)
    except InvalidCursor as e:
        logger.warning("Invalid /jobs cursor: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e.__class__.__name__}"
        )


@router.get(
    "/jobs/stats",
    tags=["jobs"],
    operation_id="jobs_stats",
    response_model=JobsStats,
    responses={
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {"example": {"detail": "Internal server error"}}
            },
        },
    },
)
async def jobs_stats() -> JobsStats:
    """Aggregate counts over the active jobs catalog: total jobs, distinct sectors, distinct platforms."""
    try:
        t_req = time.perf_counter()
        stats = await get_jobs_stats()
        log_match_step(
            "http /jobs/stats",
            "request (summary)",
            total=stats.total,
            sectors=stats.sectors,
            platforms=stats.platforms,
            request_total_ms=_ms(t_req),
        )
        return stats
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e.__class__.__name__}"
        )


@router.post(
    "/match",
    tags=["matching"],
    operation_id="match",
    response_model=List[MatchResponse],
    responses={
        400: {
            "description": "Bad Request - invalid payload content",
            "content": {
                "application/json": {"example": {"detail": "user must include user_id"}}
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {"example": {"detail": "Internal server error"}}
            },
        },
    },
)
async def match(
    payload: Annotated[
        List[MatchRequest],
        Body(..., description=_MATCH_BODY_DESCRIPTION, example=_MATCH_BODY_EXAMPLE),
    ],
):
    """Match one or more users. Body is a JSON array of MatchRequest (use length 1 for a single user)."""

    try:
        # One Mongo + one occupation load per request; run each user in a thread pool
        # (CPU-bound scoring) so concurrent requests are not stuck behind one GIL.
        t_req = time.perf_counter()
        users = [u.model_dump() for u in payload]
        n_users = len(users)

        # Mongo ping runs at app startup (warmup_on_startup), not here — avoids multi-second noise per request.
        t_fetch = time.perf_counter()
        (jobs, _), (occ, _) = await asyncio.gather(
            get_all_jobs_with_timing(users=users),
            get_all_occupations_with_timing(),
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


@router_public.post(
    "/experiments/v2/match",
    tags=["experiments"],
    operation_id="match_v2",
    response_model=List[MatchResponse],
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def match_v2(
    payload: Annotated[
        List[MatchRequest],
        Body(..., description=_MATCH_BODY_DESCRIPTION, example=_MATCH_BODY_EXAMPLE),
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

        t_fetch = time.perf_counter()
        # Full active catalog (no union location filter) + occupation corpus, in parallel.
        (jobs, _mongo_timing), (occ, _occ_timing) = await asyncio.gather(
            get_all_jobs_with_timing(users=None),
            get_all_occupations_with_timing(),
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


@router_public.post(
    "/experiments/v3/match",
    tags=["experiments"],
    operation_id="match_v3",
    response_model=List[MatchResponse],
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def match_v3(
    payload: Annotated[
        List[MatchRequest],
        Body(
            ...,
            description=(
                _MATCH_BODY_DESCRIPTION
                + " When JOBS_RETRIEVAL_FILTER is on, city/province must overlap job locations in Mongo."
            ),
            example=_MATCH_BODY_EXAMPLE,
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
    raw concat cosine similarity; v4-only ``u_hat``/``p_hat``/preference fields are empty (the v3
    engine produces no such signal).

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

        t_fetch = time.perf_counter()
        (jobs, _mongo_timing), (occ, _occ_timing) = await asyncio.gather(
            get_all_jobs_with_timing(users=users),
            get_all_occupations_with_timing(),
        )
        occ = attach_occupation_embeddings(occ)
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


@router_public.post(
    "/match_v4",
    tags=["matching"],
    operation_id="match_v4",
    response_model=List[MatchResponse],
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def match_v4(
    payload: Annotated[
        List[MatchRequest],
        Body(
            ...,
            description=(
                _MATCH_BODY_DESCRIPTION
                + " Preference scoring uses ``PREFERENCE_SCORER_MODE`` "
                "(default ``unified``: DCE attributes + BWS)."
            ),
            example=_MATCH_BODY_EXAMPLE,
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
        description="CE pool size and max preference-ranked rows returned. Default: 30.",
    ),
    final_score_combiner: Optional[str] = Query(
        None,
        description=(
            "How to combine u_hat and p_hat: ``product`` (u_hat × p_hat) or "
            "``geometric_mean`` (√(u_hat × p_hat)). Defaults to env FINAL_SCORE_COMBINER."
        ),
    ),
    skill_gap_top_k: Optional[int] = Query(
        None,
        ge=1,
        le=50,
        description="Number of skill-gap recommendations. Default: MATCH_TOP_K_SKILL_GAPS.",
    ),
):
    """Full `MatchResponse` (occupations + opportunities + skill-gaps) via the Gemini engine.

    Same JSON body as ``POST /match_v3``. Opportunities **and** occupations are matched with the v4
    Gemini concat-cosine → cross-encoder → ``u_hat × p_hat`` engine (occupations use precomputed
    concat embeddings); skill gaps reuse the existing analysis. Returns ``List[MatchResponse]``.

    Does **not** require ``x-api-key``. Uses ``GEMINI_API_KEY`` for user embeddings.
    """
    if DEBUG_MODE:
        print("matching.v4.request=")
        for item in payload:
            print(item.model_dump_json())
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

        t_fetch = time.perf_counter()
        (jobs, mongo_timing), (occ, _occ_timing) = await asyncio.gather(
            get_all_jobs_with_timing(users=users),
            get_all_occupations_with_timing(),
        )
        occ = attach_occupation_embeddings(occ)
        fetch_wall_ms = _ms(t_fetch)

        t_score = time.perf_counter()
        raw = await asyncio.to_thread(
            run_match_v4_full,
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

        out: List[MatchResponse] = [MatchResponse(**row) for row in raw]

        log_match_step(
            "http /match_v4",
            "request (summary)",
            n_users=len(users),
            n_jobs=len(jobs),
            n_occupation_rows=len(occ),
            fetch_parallel_wall_ms=fetch_wall_ms,
            scoring_thread_pool_ms=score_ms,
            request_total_ms=_ms(t_req),
        )

        if DEBUG_MODE:
            print("matching.v4.response=")
            for _item in out:
                print(_item.model_dump_json())

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


def _zqf_annotation(user_zqf, job_zqf_min):
    """(zqf_eligible, zqf_gap) or (None, None) when either side is missing."""
    if user_zqf is not None and isinstance(job_zqf_min, (int, float)):
        jmin = int(job_zqf_min)
        ulevel = int(user_zqf)
        return (ulevel >= jmin, abs(ulevel - jmin))
    return (None, None)


@router_public.post(
    "/experiments/v5/match",
    tags=["experiments"],
    operation_id="match_v5_experiment",
    response_model=List[MatchResponseV5],
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
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
            example=_MATCH_V5_BODY_EXAMPLE,
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

        t_fetch = time.perf_counter()
        (jobs, mongo_timing), (occ, _occ_timing) = await asyncio.gather(
            get_all_jobs_with_timing(users=users),
            get_all_occupations_with_timing(),
        )
        occ = attach_occupation_embeddings(occ)
        fetch_wall_ms = _ms(t_fetch)

        job_uuid_index = _jobs_by_uuid(jobs)

        t_score = time.perf_counter()
        raw = await asyncio.to_thread(
            run_match_v4_full,
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

        for row, user in zip(raw, users):
            user_zqf = user.get("zqf_level")
            for opp in row.get("opportunity_recommendations") or []:
                job = job_uuid_index.get(str(opp.get("uuid") or ""))
                job_zqf_min = job.get("zqf_min") if job else None
                eligible, gap = _zqf_annotation(user_zqf, job_zqf_min)
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
