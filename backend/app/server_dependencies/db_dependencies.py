"""Single owner of the MongoDB client: creation, index initialization, and shutdown.

The client is created once in the FastAPI lifespan (``app.main``), handed to repositories through
FastAPI dependencies (``Depends(MatchingDBProvider.get_jobs_db)``), and closed on shutdown. Nothing
else in the app constructs a Mongo client. See ``app/server_dependencies/README.md``.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel

from .database_collections import Collections


def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    return int((os.getenv(name) or str(default)).strip() or str(default))


@dataclass(frozen=True)
class MongoDbSettings:
    mongo_url: str
    database_name: str | None
    server_selection_timeout_ms: int = 30000
    max_pool_size: int = 50
    min_pool_size: int = 0
    tls_allow_invalid_certificates: bool = False
    tls_ca_file: str = ""

    @classmethod
    def from_env(cls) -> "MongoDbSettings":
        mongo_url = os.getenv("MONGO_URL")
        if not mongo_url:
            raise ValueError("MONGO_URL environment variable is not set")
        return cls(
            mongo_url=mongo_url,
            database_name=os.getenv("MONGO_DB_NAME"),
            server_selection_timeout_ms=_env_int(
                "MONGO_SERVER_SELECTION_TIMEOUT_MS", 30000
            ),
            max_pool_size=_env_int("MONGO_MAX_POOL_SIZE", 50),
            min_pool_size=_env_int("MONGO_MIN_POOL_SIZE", 0),
            # Both names are accepted (dev-only; skips certificate verification).
            tls_allow_invalid_certificates=_env_flag(
                "MONGO_TLS_ALLOW_INVALID_CERTIFICATES"
            )
            or _env_flag("MONGO_TLS_INSECURE"),
            tls_ca_file=(os.getenv("MONGO_TLS_CA_FILE") or "").strip(),
        )


def _looks_like_tls_mongodb(uri: str) -> bool:
    u = uri.lower().strip()
    return (
        "mongodb+srv://" in u
        or "tls=true" in u
        or "tls = true" in u
        or "ssl=true" in u
        or "ssl = true" in u
    )


def _certifi_ca_file(logger: logging.Logger) -> str | None:
    try:
        import certifi

        return certifi.where()
    except ImportError:
        logger.warning(
            "TLS MongoDB URL detected but certifi not installed — install certifi or set "
            "MONGO_TLS_CA_FILE for certificate verification "
            "(or use MONGO_TLS_ALLOW_INVALID_CERTIFICATES=1 for local dev only)."
        )
        return None


def build_client_kwargs(
    settings: MongoDbSettings, logger: logging.Logger
) -> dict[str, Any]:
    """Pool, timeout and TLS options for ``AsyncIOMotorClient``.

    Atlas (mongodb+srv) uses TLS, and on some macOS/Python installs the default CA store is empty
    (``SSL: CERTIFICATE_VERIFY_FAILED``). Precedence:

    * ``MONGO_TLS_ALLOW_INVALID_CERTIFICATES=1`` / ``MONGO_TLS_INSECURE=1`` — skip verification (dev only).
    * ``MONGO_TLS_CA_FILE=/path.pem`` — explicit CA bundle; the literal ``certifi`` means ``certifi.where()``.
    * Otherwise, if the URI looks TLS, use ``certifi.where()`` when certifi is installed.
    """
    kwargs: dict[str, Any] = {
        "serverSelectionTimeoutMS": settings.server_selection_timeout_ms,
        "maxPoolSize": max(1, settings.max_pool_size),
    }
    if settings.min_pool_size > 0:
        kwargs["minPoolSize"] = settings.min_pool_size

    if settings.tls_allow_invalid_certificates:
        kwargs["tlsAllowInvalidCertificates"] = True
        logger.warning(
            "MongoDB TLS certificate verification disabled (not for production)."
        )
    elif settings.tls_ca_file.lower() == "certifi":
        ca = _certifi_ca_file(logger)
        if ca:
            kwargs["tlsCAFile"] = ca
    elif settings.tls_ca_file:
        kwargs["tlsCAFile"] = settings.tls_ca_file
    elif _looks_like_tls_mongodb(settings.mongo_url):
        ca = _certifi_ca_file(logger)
        if ca:
            kwargs["tlsCAFile"] = ca
    return kwargs


def _get_jobs_db(
    settings: MongoDbSettings, logger: logging.Logger
) -> AsyncIOMotorDatabase:
    """
    Decouples the client creation from the database provider.
    This allows to mock the database creation in tests, instead of mocking the database provider.
    """
    return AsyncIOMotorClient(
        settings.mongo_url, **build_client_kwargs(settings, logger)
    )[settings.database_name]


async def check_mongo_health(client: AsyncIOMotorClient) -> bool:
    try:
        result = await client.admin.command("ping")
        return result.get("ok") == 1.0
    except Exception:
        return False


# Indexes on the jobs collection that the /jobs (browse, stats) and /match queries rely on.
#
# MongoDB uses ONE index per query (index intersection is rarely chosen by the planner, and never
# when a sort is present), so these are not combined at query time — each is sized for a specific
# access pattern:
#
#   * {is_active: 1, _id: -1}  — the workhorse. The browse endpoint always filters is_active, sorts
#     by _id desc, and seeks with a keyset (_id < cursor); this one index serves the equality, the
#     sort, AND the cursor range together (also covers count_documents(is_active...) and the
#     active-jobs load behind /match). Without it every such call is a full scan + in-memory sort.
#   * {is_active: 1, employment_type: 1, _id: -1}  — the only exact-match browse filter. The trailing
#     _id key lets an employment_type-filtered browse use the index for the equality AND the sort.
#   * {is_active: 1, <category|isco_group|source_platform>: 1}  — back the distinct() calls in
#     /jobs/stats (used by those, not by the sorted browse query).
#
# Substring/regex filters (title, category, location, skills) cannot use a B-tree index in any
# combination — they are always applied as residual filters. Speeding those up further would need
# an Atlas Search / ``$text`` index plus a query change.
JOBS_INDEX_MODELS = [
    IndexModel([("is_active", ASCENDING), ("_id", DESCENDING)], name="is_active_-_id"),
    IndexModel(
        [
            ("is_active", ASCENDING),
            ("classifier_metadata.employment_type", ASCENDING),
            ("_id", DESCENDING),
        ],
        name="is_active_employment_type_-_id",
    ),
    IndexModel(
        [("is_active", ASCENDING), ("classifier_metadata.category", ASCENDING)],
        name="is_active_category",
    ),
    IndexModel(
        [
            ("is_active", ASCENDING),
            ("classifier_metadata.isco_occupation_group", ASCENDING),
        ],
        name="is_active_isco_group",
    ),
    IndexModel(
        [("is_active", ASCENDING), ("classifier_metadata.source_platform", ASCENDING)],
        name="is_active_source_platform",
    ),
]


class MatchingDBProvider:
    """
    Provides the jobs database instance. Holds the only MongoDB client in the process.
    """

    _jobs_mongo_db: AsyncIOMotorDatabase | None = None
    _lock = asyncio.Lock()
    _logger = logging.getLogger(__qualname__)

    @staticmethod
    def _get_settings() -> MongoDbSettings:
        # Defer reading the settings until the first time they are needed.
        # Otherwise, the settings would be read at import time, which breaks tests and scripts that
        # import the app without a MONGO_URL.
        return MongoDbSettings.from_env()

    @staticmethod
    async def initialize_jobs_mongo_db(
        jobs_db: AsyncIOMotorDatabase, logger: logging.Logger
    ) -> list[str]:
        """Create (idempotently) the indexes the jobs queries need. Returns the ensured index names.

        ``create_indexes`` is a no-op for indexes that already exist with the same spec, so this is
        safe to run on every startup.
        """
        try:
            t0 = time.perf_counter()
            created = await jobs_db.get_collection(Collections.JOBS).create_indexes(
                JOBS_INDEX_MODELS
            )
            logger.info(
                "Ensured %d indexes on %s in %.2f ms: %s",
                len(created),
                Collections.JOBS,
                (time.perf_counter() - t0) * 1000.0,
                created,
            )
            return created
        except Exception as e:
            logger.exception(e)
            raise e

    @classmethod
    async def get_jobs_db(cls) -> AsyncIOMotorDatabase:
        """Return the jobs database, creating the client on first call.

        Creating the client does no I/O (Motor connects lazily), so this never blocks on Mongo;
        connectivity is checked by the startup warmup instead (``app.warmup``).
        """
        if (
            cls._jobs_mongo_db is None
        ):  # Check if the database instance has been created
            async with (
                cls._lock
            ):  # Ensure that only one coroutine is creating the database instance
                if cls._jobs_mongo_db is None:  # Double-check after acquiring the lock
                    settings = cls._get_settings()
                    cls._jobs_mongo_db = _get_jobs_db(settings, cls._logger)
                    cls._logger.info(
                        "Created MongoDB client for database: %s",
                        settings.database_name,
                    )
        return cls._jobs_mongo_db

    @classmethod
    def close(cls) -> None:
        """Close the client and forget it, so a later ``get_jobs_db`` opens a fresh one."""
        if cls._jobs_mongo_db is not None:
            cls._jobs_mongo_db.client.close()
            cls._jobs_mongo_db = None
            cls._logger.info("Closed MongoDB client")

    @staticmethod
    def clear_cache() -> None:
        """
        Forget the cached database instance without closing it.

        This is useful for testing purposes to ensure that the database instance is re-created.
        """
        MatchingDBProvider._jobs_mongo_db = None
