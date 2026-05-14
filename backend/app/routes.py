import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.security import APIKeyHeader

from app.schemas import MatchRequest, MatchResponse, MatchV2Response, MatchV2JobRecommendation
from app.config import MATCH_V2_HYBRID_TOP_K, MATCH_V2_MAX_USERS_PER_REQUEST
from app.database import get_all_jobs_with_timing, get_all_occupations_with_timing
from app.match_timing_log import log_match_step
from app.services.matching_service import match_user_with_data

api_key_auth = APIKeyHeader(
    scheme_name="gcp_api_key",
    name="x-api-key",
    auto_error=True
)

router = APIRouter(dependencies=[Depends(api_key_auth)])
# Temporary: hybrid endpoint without API key (parity with keyed /match TBD).
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
                matched_skills_cosine=[str(x) for x in (row.get("matched_skills_cosine") or [])],
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
    from app.services.hybrid_scoring.run_bm25_cosine_hybrid import hybrid_match_users_with_jobs

    return hybrid_match_users_with_jobs(
        users,
        jobs,
        col_display_k=fusion_top_k,
        alpha_on_cosine=alpha_on_cosine,
    )

class Health(BaseModel):
    status: str

@router.get("/health")
async def health() -> Health:
    return Health(status="ok")

@router.post(
    "/match",
    tags=["matching"],
    response_model=List[MatchResponse],
    responses={
        400: {
            "description": "Bad Request - invalid payload content",
            "content": {"application/json": {"example": {"detail": "user must include user_id"}}},
        },
        500: {
            "description": "Internal Server Error",
            "content": {"application/json": {"example": {"detail": "Internal server error"}}},
        },
    },
)
async def match(payload: List[MatchRequest]):
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
        raise HTTPException(status_code=500, detail=f"Internal server error: {e.__class__.__name__}")


@router_public.post(
    "/match_v2",
    tags=["matching"],
    response_model=List[MatchV2Response],
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def match_v2(
    payload: List[MatchRequest],
    fusion_top_k: Optional[int] = Query(
        None,
        ge=1,
        le=500,
        description=(
            "Max hybrid fused rows per user (pool min–max ranking). "
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
):
    """Hybrid BM25 × cosine-skill embeddings: fused rankings over the Mongo job corpus.

    Loads **all active jobs** (``is_active`` only) — **no per-user Mongo location prefilter** and no
    retrieval sort/limit, so BM25 indexes align with ``JOBS_RETRIEVAL_FILTER=0`` / CLI ``--mongo-all-active``.
    This avoids the filtered job slice used by ``POST /match``.

    Does **not** run occupations, skill gaps, or ``SkillScorer`` / ``p_hat``.
    Returns ``column_fused_weighted_minmax`` style rows as ``hybrid_recommendations``.
    Does **not** require ``x-api-key`` (temporary; gated separately from ``POST /match``).
    """

    from app.services.hybrid_scoring.run_bm25_cosine_hybrid import _alpha_on_cosine_from_env

    try:
        t_req = time.perf_counter()
        if len(payload) > MATCH_V2_MAX_USERS_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Too many users in one request (max {MATCH_V2_MAX_USERS_PER_REQUEST}).",
            )
        if not payload:
            raise HTTPException(status_code=400, detail="Request body must be a non-empty JSON array.")

        users = [u.model_dump() for u in payload]
        n_users = len(users)
        fk = fusion_top_k if fusion_top_k is not None else MATCH_V2_HYBRID_TOP_K

        env_alpha, _env_key = _alpha_on_cosine_from_env()
        alpha = alpha_on_cosine if alpha_on_cosine is not None else (env_alpha if env_alpha is not None else 0.5)

        t_fetch = time.perf_counter()
        # Full active catalog — no union location filter or retrieval cap on this endpoint.
        jobs, mongo_timing = await get_all_jobs_with_timing(users=None)
        fetch_wall_ms = _ms(t_fetch)

        if len(jobs) == 0:
            empty_summary = {"fusion": "weighted_minmax_on_candidate_pool_bm25_cosine_only", "n_jobs": 0}
            return [
                MatchV2Response(
                    user_id=str(u.get("user_id") or ""),
                    n_jobs_scored=0,
                    hybrid_recommendations=[],
                    hybrid_config_summary=empty_summary,
                )
                for u in users
            ]

        t_score = time.perf_counter()
        envelope = await asyncio.to_thread(_execute_hybrid_http, users, jobs, fusion_top_k=fk, alpha_on_cosine=alpha)
        hybrid_ms = _ms(t_score)

        job_index = _jobs_by_uuid(jobs)
        idx = envelope.get("index_stats") or {}
        cfg_slice = {
            k: envelope["config"].get(k)
            for k in (
                "scorer",
                "fusion",
                "alpha_on_cosine_skill",
                "variant_bm25",
                "bm25_pool_k",
                "cosine_pool_k",
                "column_display_rows",
            )
            if k in (envelope.get("config") or {})
        }
        cfg_slice["n_jobs"] = idx.get("n_jobs", len(jobs))
        cfg_slice["embedding_dim"] = idx.get("embedding_dim")
        if mongo_timing is not None:
            cfg_slice["mongo_retrieval_filter_applied"] = mongo_timing.get("jobs_retrieval_filter_applied")
            cfg_slice["mongo_ranked_find_ms"] = mongo_timing.get("mongo_ranked_find_ms")

        out: List[MatchV2Response] = []
        for row in envelope.get("results") or []:
            fused = row.get("column_fused_weighted_minmax") or []
            uid = str(row.get("user_id") or "")
            out.append(
                MatchV2Response(
                    user_id=uid,
                    n_jobs_scored=len(jobs),
                    hybrid_recommendations=_fused_rows_to_match_v2_jobs(fused, job_index),
                    hybrid_config_summary=cfg_slice,
                )
            )

        log_match_step(
            "http /match_v2",
            "request (summary)",
            n_users=n_users,
            n_jobs=len(jobs),
            n_occupation_rows=0,
            fetch_parallel_wall_ms=fetch_wall_ms,
            scoring_thread_pool_ms=hybrid_ms,
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
        raise HTTPException(status_code=500, detail=f"Internal server error: {e.__class__.__name__}")
