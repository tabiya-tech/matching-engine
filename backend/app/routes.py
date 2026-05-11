import asyncio
import logging
import time
import app.config as c

from pydantic import BaseModel
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import APIKeyHeader

from app.schemas import MatchRequest, MatchResponse
from app.database import get_all_jobs_with_timing, get_all_occupations_with_timing, get_test_users
from app.match_timing_log import log_match_step
from app.matching_config_store import (
    fetch_mongo_routing_document,
    load_and_apply_mongo_routing_config,
    load_effective_matching_settings,
    save_mongo_routing_config,
    save_override_flat,
)
from app.matching_runtime import DEFAULT_TUNABLE_FLAT, TUNABLE_KEYS, build_effective_settings
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


@router.get("/config", tags=["config"])
async def get_matching_config():
    settings = await load_effective_matching_settings()
    return {
        "effective": settings.to_effective_flat(),
        "sources": settings.sources,
    }


@router.get("/config/mongo", tags=["config"])
async def get_mongo_config():
    runtime = await load_and_apply_mongo_routing_config()
    doc = await fetch_mongo_routing_document()
    if not doc:
        return runtime
    vals = doc.get("values") or {}
    return {
        "MONGO_DB_NAME": str(vals.get("MONGO_DB_NAME") or runtime["MONGO_DB_NAME"]),
        "MONGO_JOBS_COLLECTION": str(vals.get("MONGO_JOBS_COLLECTION") or runtime["MONGO_JOBS_COLLECTION"]),
    }


@router.post("/config/mongo", tags=["config"])
async def post_mongo_config(body: Dict[str, Any]):
    keys = {
        "MONGO_DB_NAME",
        "MONGO_JOBS_COLLECTION",
    }
    updates = {k: str(v).strip() for k, v in body.items() if k in keys and str(v).strip()}
    if not updates:
        raise HTTPException(status_code=422, detail="No valid mongo config keys provided")
    for k, v in updates.items():
        setattr(c, k, v)
    await save_mongo_routing_config(updates)
    return {"ok": True, "mongo_config": await get_mongo_config()}


@router.get("/test-users", tags=["testing"])
async def get_dynamic_test_users(limit: int = 500):
    try:
        users = await get_test_users(limit=limit)
        return {"users": users, "count": len(users)}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Failed to load test users: {e.__class__.__name__}")


@router.post("/config", tags=["config"])
async def post_matching_config(body: Dict[str, Any]):
    user_flat = {str(k): str(v).strip() for k, v in body.items() if k in TUNABLE_KEYS}
    try:
        build_effective_settings(user_flat)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    merged = {**DEFAULT_TUNABLE_FLAT, **user_flat}
    await save_override_flat(merged)
    settings = await load_effective_matching_settings()
    return {
        "ok": True,
        "effective": settings.to_effective_flat(),
        "sources": settings.sources,
    }


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
        await load_and_apply_mongo_routing_config()
        # One Mongo + one occupation load per request; run each user in a thread pool
        # (CPU-bound scoring) so concurrent requests are not stuck behind one GIL.
        t_req = time.perf_counter()
        users = [u.model_dump() for u in payload]
        n_users = len(users)

        # Mongo ping runs at app startup (warmup_on_startup), not here — avoids multi-second noise per request.
        t_fetch = time.perf_counter()
        (jobs, _), (occ, _), _ = await asyncio.gather(
            get_all_jobs_with_timing(users=users),
            get_all_occupations_with_timing(),
            load_effective_matching_settings(),
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
