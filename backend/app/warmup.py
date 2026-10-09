"""One-time startup work, run from the FastAPI lifespan (never per request).

Each step is toggled by an env flag (see ``.env.example``) and is non-fatal: a failure is logged and
the app still starts, so a slow or briefly unreachable Mongo does not keep the container from serving.
"""

import asyncio
import logging
import os
import time

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.occupations.loader import get_all_occupations_with_timing
from app.occupations.work_activities import load_wa_lookup
from app.server_dependencies.db_dependencies import (
    MatchingDBProvider,
    check_mongo_health,
)

logger = logging.getLogger(__name__)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _env_warmup_flag(name: str, default: bool = True) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


async def warmup_on_startup(jobs_db: AsyncIOMotorDatabase) -> None:
    """Ping Mongo, ensure indexes, and preload heavy one-time caches.

    Toggle with ``MONGO_WARMUP_ON_STARTUP``, ``ENSURE_INDEXES_ON_STARTUP``, ``WARMUP_OCCUPATIONS_CACHE``,
    ``WARMUP_WA_LOOKUP``, ``WARMUP_MATCH_V3_MODELS``. WA lookup defaults to off — jobs use pre-enriched
    ``onet_work_activities``.
    """
    if _env_warmup_flag("MONGO_WARMUP_ON_STARTUP", True):
        t0 = time.perf_counter()
        if await check_mongo_health(jobs_db.client):
            logger.info("Mongo warmup: ping ok (%.2f ms)", _ms(t0))
        else:
            logger.error("Mongo warmup: ping failed (%.2f ms)", _ms(t0))
    else:
        logger.info("Mongo warmup skipped (MONGO_WARMUP_ON_STARTUP=0)")

    if _env_warmup_flag("ENSURE_INDEXES_ON_STARTUP", True):
        try:
            await MatchingDBProvider.initialize_jobs_mongo_db(jobs_db, logger)
        except Exception:
            # initialize_jobs_mongo_db already logged the traceback.
            logger.warning("Ensuring jobs indexes failed; continuing startup")
    else:
        logger.info("Index creation skipped (ENSURE_INDEXES_ON_STARTUP=0)")

    if _env_warmup_flag("WARMUP_OCCUPATIONS_CACHE", True):
        t0 = time.perf_counter()
        try:
            await get_all_occupations_with_timing()
            logger.info("Occupation cache warmup: ok (%.2f ms)", _ms(t0))
        except Exception:
            logger.exception("Occupation cache warmup failed")
    else:
        logger.info("Occupation cache warmup skipped (WARMUP_OCCUPATIONS_CACHE=0)")

    if _env_warmup_flag("WARMUP_WA_LOOKUP", False):
        try:
            load_wa_lookup()
            logger.info("WA taxonomy lookup: built at startup")
        except Exception:
            logger.exception("WA lookup warmup failed")
    else:
        logger.info(
            "WA lookup warmup skipped (enriched jobs carry onet_work_activities; set WARMUP_WA_LOOKUP=1 to force)"
        )

    if _env_warmup_flag("WARMUP_MATCH_V3_MODELS", False):
        try:
            from app.server_dependencies.model_dependencies import preload_models

            t0 = time.perf_counter()
            timings = await asyncio.to_thread(preload_models)
            logger.info(
                "/match_v3 model warmup: ok (total %.2f ms; matcher %.2f ms, cross-encoder %.2f ms)",
                _ms(t0),
                timings.get("cosine_skill_matcher_ms", 0.0),
                timings.get("cross_encoder_ms", 0.0),
            )
        except Exception:
            logger.exception("/match_v3 model warmup failed")
    else:
        logger.info(
            "/match_v3 model warmup skipped (set WARMUP_MATCH_V3_MODELS=1 to preload CosineSkillMatcher + CrossEncoder at startup)"
        )
