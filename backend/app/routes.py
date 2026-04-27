import asyncio
import logging
import time

from pydantic import BaseModel
from typing import List

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import APIKeyHeader

from app.schemas import MatchRequest, MatchResponse
from app.database import get_all_jobs_with_timing, get_all_occupations_with_timing
from app.match_timing_log import log_match_step
from app.services.matching_service import match_user_with_data

api_key_auth = APIKeyHeader(
    scheme_name="gcp_api_key",
    name="x-api-key",
    auto_error=True
)

router = APIRouter(dependencies=[Depends(api_key_auth)])
logger = logging.getLogger(__name__)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

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
