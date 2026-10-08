"""Start-up warmup: Mongo ping, jobs indexes, occupation corpus and (optionally) model preload."""

import asyncio
import logging
import os
import time

from app.jobs.repository import JobsRepository
from app.occupations.get_occupations_repository import get_occupations_repository
from app.server_dependencies.db_dependencies import get_jobs_db

logger = logging.getLogger(__name__)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _env_warmup_flag(name: str, default: bool = True) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


async def warmup_on_startup() -> None:
    """Ping Mongo and preload heavy one-time caches. Called from FastAPI lifespan (not per /match).

    Toggle with ``MONGO_WARMUP_ON_STARTUP``, ``WARMUP_OCCUPATIONS_CACHE``, ``WARMUP_WA_LOOKUP``
    (see ``.env.example``). WA lookup defaults to off — jobs use pre-enriched ``onet_work_activities``.
    """
    jobs_repository = JobsRepository(db=get_jobs_db())
    occupations_repository = get_occupations_repository()
    if _env_warmup_flag("MONGO_WARMUP_ON_STARTUP", True):
        t0 = time.perf_counter()
        try:
            await jobs_repository.ping()
            logger.info("Mongo warmup: ping ok (%.2f ms)", _ms(t0))
        except Exception:
            logger.exception("Mongo warmup: ping failed")
    else:
        logger.info("Mongo warmup skipped (MONGO_WARMUP_ON_STARTUP=0)")

    if _env_warmup_flag("ENSURE_INDEXES_ON_STARTUP", True):
        try:
            await jobs_repository.ensure_indexes()
        except Exception:
            logger.exception("Ensuring jobs indexes failed")
    else:
        logger.info("Index creation skipped (ENSURE_INDEXES_ON_STARTUP=0)")

    if _env_warmup_flag("WARMUP_OCCUPATIONS_CACHE", True):
        t0 = time.perf_counter()
        try:
            await occupations_repository.load_with_timing()
            logger.info("Occupation cache warmup: ok (%.2f ms)", _ms(t0))
        except Exception:
            logger.exception("Occupation cache warmup failed")
    else:
        logger.info("Occupation cache warmup skipped (WARMUP_OCCUPATIONS_CACHE=0)")

    if _env_warmup_flag("WARMUP_WA_LOOKUP", False):
        try:
            occupations_repository.load_wa_lookup()
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
