"""MongoDB client for the jobs database.

One ``AsyncIOMotorClient`` per process, built at import from ``MONGO_URL`` / ``MONGO_DB_NAME`` and
the ``MONGO_*`` pool / TLS settings. Repositories receive the database through ``get_jobs_db``.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

load_dotenv()

# Load from environment
MONGO_URL = os.getenv("MONGO_URL")
DATABASE_NAME = os.getenv("MONGO_DB_NAME")

if not MONGO_URL:
    raise ValueError("MONGO_URL environment variable is not set")

_mongo_sel_ms = int(
    (os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS") or "30000").strip() or "30000"
)
_mongo_max_pool = int((os.getenv("MONGO_MAX_POOL_SIZE") or "50").strip() or "50")
_mongo_min_pool = int((os.getenv("MONGO_MIN_POOL_SIZE") or "0").strip() or "0")
_mongo_client_kwargs: dict[str, Any] = {
    "serverSelectionTimeoutMS": _mongo_sel_ms,
    "maxPoolSize": max(1, _mongo_max_pool),
}
if _mongo_min_pool > 0:
    _mongo_client_kwargs["minPoolSize"] = _mongo_min_pool


def _looks_like_tls_mongodb(uri: str) -> bool:
    u = uri.lower().strip()
    return (
        "mongodb+srv://" in u
        or "tls=true" in u
        or "tls = true" in u
        or "ssl=true" in u
        or "ssl = true" in u
    )


def _configure_mongodb_tls(kwargs: dict[str, Any]) -> None:
    """Atlas and other TLS backends need a CA bundle. macOS/Python.org installs often lack one.

    * ``MONGO_TLS_ALLOW_INVALID_CERTIFICATES=1`` — dev-only; skips verification (unsafe).
    * ``MONGO_TLS_CA_FILE=/path.pem`` — explicit CA bundle path.
    * Otherwise, if URI looks TLS and ``certifi`` is installed, use ``certifi.where()``.
      (Typically present as a transitive dep of ``requests`` / ``httpx``.)
    """
    if not MONGO_URL or not _looks_like_tls_mongodb(MONGO_URL):
        return

    allow_invalid = os.getenv(
        "MONGO_TLS_ALLOW_INVALID_CERTIFICATES", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    if allow_invalid:
        kwargs["tlsAllowInvalidCertificates"] = True
        logger.warning(
            "MONGO_TLS_ALLOW_INVALID_CERTIFICATES is set — TLS verification disabled (not for production)."
        )
        return

    ca_explicit = os.getenv("MONGO_TLS_CA_FILE", "").strip()
    if ca_explicit:
        kwargs["tlsCAFile"] = ca_explicit
        return

    try:
        import certifi

        kwargs["tlsCAFile"] = certifi.where()
    except ImportError:
        logger.warning(
            "TLS MongoDB URL detected but certifi not installed — install certifi or set "
            "MONGO_TLS_CA_FILE for certificate verification "
            "(or use MONGO_TLS_ALLOW_INVALID_CERTIFICATES=1 for local dev only)."
        )


_configure_mongodb_tls(_mongo_client_kwargs)


def _mongo_tls_client_options() -> dict[str, Any]:
    """Extra Motor/PyMongo TLS options from env.

    Atlas (mongodb+srv) uses TLS. On some macOS/Python installs the default CA
    store is empty or incomplete and you get::

        SSL: CERTIFICATE_VERIFY_FAILED / unable to get local issuer certificate

    """
    extra: dict[str, Any] = {}
    insecure = (os.getenv("MONGO_TLS_INSECURE") or "").strip().lower()
    if insecure in ("1", "true", "yes", "on"):
        extra["tlsAllowInvalidCertificates"] = True
        return extra

    ca_raw = (os.getenv("MONGO_TLS_CA_FILE") or "").strip()
    if not ca_raw:
        return extra
    if ca_raw.lower() == "certifi":
        import certifi

        extra["tlsCAFile"] = certifi.where()
        return extra
    extra["tlsCAFile"] = ca_raw
    return extra


_mongo_client_kwargs.update(_mongo_tls_client_options())

client = AsyncIOMotorClient(MONGO_URL, **_mongo_client_kwargs)
db = client[DATABASE_NAME]

logger = logging.getLogger(__name__)


def get_jobs_db() -> AsyncIOMotorDatabase:
    """The process-wide jobs database."""
    return db


async def close_mongo_connection():
    client.close()
