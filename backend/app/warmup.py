"""One-time startup work, run from the FastAPI lifespan (never per request).

Each step is toggled by an env flag (see ``.env.example``) and is non-fatal: a failure is logged and
the app still starts, so a slow or briefly unreachable Mongo does not keep the container from serving.
"""

import asyncio
import logging
import os
import time

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import (
    JOBS_RETRIEVAL_FILTER,
    STAGE1_RETRIEVAL_BACKEND,
)
from app.jobs.job_document import LOCATION_KEYS_FIELD, RANKED_JOBS_ACTIVE_FILTER
from app.jobs.mongo_repository import MongoJobsRepository
from app.occupations.loader import get_all_occupations_with_timing
from app.occupations.work_activities import load_wa_lookup
from app.server_dependencies.database_collections import Collections
from app.server_dependencies.db_dependencies import (
    MatchingDBProvider,
    check_mongo_health,
)
from app.server_dependencies.vector_search import (
    JOBS_VECTOR_INDEX,
    Stage1Coverage,
    VectorIndexSpec,
    vector_index_status,
)

logger = logging.getLogger(__name__)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _env_warmup_flag(name: str, default: bool = True) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


async def _warn_jobs_without_location_keys(jobs_db: AsyncIOMotorDatabase) -> None:
    """Log active jobs that /match cannot load because ``location_keys`` was never written.

    The location filter has no fallback for them; run ``tools.location.backfill_location_keys``.
    """
    if not JOBS_RETRIEVAL_FILTER:
        return
    try:
        n_missing = await jobs_db.get_collection(Collections.JOBS).count_documents(
            {**RANKED_JOBS_ACTIVE_FILTER, LOCATION_KEYS_FIELD: {"$exists": False}}
        )
    except Exception:
        logger.exception("Counting active jobs without location_keys failed")
        return
    if n_missing:
        logger.warning(
            "%d active jobs have no location_keys and are never matched; run "
            "python -m tools.location.backfill_location_keys",
            n_missing,
        )


def log_stage1_coverage(coverage: Stage1Coverage) -> None:
    """Log how many documents the stage-1 vector search can see, and loudly what it cannot."""
    logger.info(
        "stage-1 vector search coverage of %s: %d/%d searchable "
        "(other whitening artifact %d, not whitened %d, no embedding %d; artifact %s)",
        coverage.collection,
        coverage.n_searchable,
        coverage.n_total,
        coverage.n_other_artifact,
        coverage.n_not_whitened,
        coverage.n_no_embedding,
        coverage.artifact_sha256[:12],
    )
    if coverage.n_other_artifact:
        logger.error(
            "%d documents in %s were whitened with another artifact than this service's (%s); "
            "they are excluded from matching, never mixed in. Re-whiten them with the deployed "
            "artifact, or deploy the artifact they were made with.",
            coverage.n_other_artifact,
            coverage.collection,
            coverage.artifact_sha256[:12],
        )
    if coverage.n_not_whitened or coverage.n_no_embedding:
        logger.warning(
            "%d documents in %s are excluded from matching until backfilled "
            "(%d not whitened, %d without an embedding).",
            coverage.n_not_whitened + coverage.n_no_embedding,
            coverage.collection,
            coverage.n_not_whitened,
            coverage.n_no_embedding,
        )


async def _check_vector_index(db: AsyncIOMotorDatabase, spec: VectorIndexSpec) -> None:
    try:
        status = await vector_index_status(db, spec)
    except Exception as e:
        logger.error(
            "stage-1 vector index %s on %s could not be checked (%s: %s); is Atlas Vector "
            "Search available on this cluster?",
            spec.name,
            spec.collection,
            type(e).__name__,
            e,
        )
        return
    if not status.exists:
        logger.error(
            "stage-1 vector index %s is missing on %s; /match will fail. Create it with "
            "`python -m scripts.vector_search_indexes --apply`.",
            spec.name,
            spec.collection,
        )
    elif not status.queryable:
        logger.error(
            "stage-1 vector index %s on %s is not queryable yet (status %s).",
            spec.name,
            spec.collection,
            status.status,
        )
    elif not status.definition_matches:
        logger.warning(
            "stage-1 vector index %s on %s differs from the definition in code; update it with "
            "`python -m scripts.vector_search_indexes --apply`.",
            spec.name,
            spec.collection,
        )
    else:
        logger.info(
            "stage-1 vector index %s on %s: %s",
            spec.name,
            spec.collection,
            status.status,
        )


async def check_stage1_vector_search(jobs_db: AsyncIOMotorDatabase) -> None:
    """``STAGE1_RETRIEVAL_BACKEND=db``: check the jobs vector index and the whitening version.

    Hashes the whitening artifact (cached for every request after), checks the index exists and is
    queryable, and logs how many active jobs are searchable and how many are excluded because they
    were whitened with another artifact, not whitened, or not embedded.
    """
    from app.server_dependencies.model_dependencies import (
        get_concat_whitening_sha256,
    )

    artifact_sha256 = await asyncio.to_thread(get_concat_whitening_sha256)
    if artifact_sha256 is None:
        logger.error(
            "STAGE1_RETRIEVAL_BACKEND=db but the concat whitening artifact is unreadable; "
            "/match cannot whiten users and will fail."
        )
        return
    await _check_vector_index(jobs_db, JOBS_VECTOR_INDEX)
    log_stage1_coverage(
        await MongoJobsRepository(jobs_db).count_stage1_coverage(artifact_sha256)
    )


async def warmup_on_startup(jobs_db: AsyncIOMotorDatabase) -> None:
    """Ping Mongo, ensure indexes, and preload heavy one-time caches.

    Toggle with ``MONGO_WARMUP_ON_STARTUP``, ``ENSURE_INDEXES_ON_STARTUP``, ``WARMUP_OCCUPATIONS_CACHE``,
    ``WARMUP_WA_LOOKUP``, ``WARMUP_MATCH_V3_MODELS``, ``CHECK_STAGE1_VECTOR_SEARCH``. WA lookup
    defaults to off — jobs use pre-enriched ``onet_work_activities``.
    """
    if _env_warmup_flag("MONGO_WARMUP_ON_STARTUP", True):
        t0 = time.perf_counter()
        if await check_mongo_health(jobs_db.client):
            logger.info("Mongo warmup: ping ok (%.2f ms)", _ms(t0))
            await _warn_jobs_without_location_keys(jobs_db)
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

    if STAGE1_RETRIEVAL_BACKEND == "db" and _env_warmup_flag(
        "CHECK_STAGE1_VECTOR_SEARCH", True
    ):
        try:
            await check_stage1_vector_search(jobs_db)
        except Exception:
            logger.exception("Stage-1 vector search check failed")

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
