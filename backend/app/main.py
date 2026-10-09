import asyncio
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.jobs.routes import add_jobs_routes
from app.match_timing_log import init_match_timing_log
from app.matching.routes import add_matching_routes
from app.observability import (
    MatchTracingMiddleware,
    init_tracing,
    shutdown_tracing,
    tracing_config_from_env,
)
from app.server_dependencies.auth import api_key_auth
from app.server_dependencies.model_dependencies import (
    get_cross_encoder_client,
    get_skill_scorer,
)

load_dotenv()
init_match_timing_log()

logger = logging.getLogger(__name__)


def _warmup_non_blocking() -> bool:
    v = (os.getenv("WARMUP_NON_BLOCKING") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Mongo ping + occupation JSON + WA lookup once at startup so /match does not pay cold-connection cost each time."""
    from app.server_dependencies.warmup import warmup_on_startup

    init_tracing(tracing_config_from_env())
    # Load the skill-gap engine before serving so the first /match does not pay the load.
    get_skill_scorer()

    async def _warmup_safe() -> None:
        try:
            await warmup_on_startup()
        except Exception:
            logger.exception("Startup warmup failed")

    if _warmup_non_blocking():
        asyncio.create_task(_warmup_safe())
        logger.info(
            "Startup: warmup scheduled in background (WARMUP_NON_BLOCKING=1); "
            "first /match may still pay part of cold cost until warmup finishes"
        )
    else:
        await _warmup_safe()
    get_cross_encoder_client()
    yield
    # Flush buffered traces before Cloud Run tears the instance down.
    shutdown_tracing()


def _openapi_servers() -> list[dict[str, str]]:
    """Swagger UI \"Try it out\" base URL.

    Default ``/`` = same host + port as the page serving ``/docs`` (works when uvicorn uses
    ``--port 8081`` etc. without setting ``PORT`` in env). Override with ``OPENAPI_SERVER_URL``
    behind a gateway or for a fixed public base URL.
    """
    explicit = (os.getenv("OPENAPI_SERVER_URL") or "").strip().rstrip("/")
    if explicit:
        return [{"url": explicit, "description": "Configured (OPENAPI_SERVER_URL)"}]
    return [
        {
            "url": "/",
            "description": "Same origin as /docs (recommended for local Swagger)",
        }
    ]


app = FastAPI(title="Matching Service", lifespan=lifespan, servers=_openapi_servers())

logging.basicConfig(level=logging.INFO)

# CORS for browser clients (Swagger /docs, local frontends).
# Dev: allow any localhost / 127.0.0.1 / 0.0.0.0 origin on any port (regex). Otherwise no Origin = no preflight (curl etc.).
# Prod override: set CORS_ALLOW_ORIGINS (comma-separated full origins) in the environment.
_allow_explicit = (os.getenv("CORS_ALLOW_ORIGINS") or "").strip()
if _allow_explicit:
    _origins = [o.strip() for o in _allow_explicit.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# One Langfuse trace per /match* request (a no-op unless MATCHING_ENABLE_TRACING is on).
# Added last, so it is outermost and times the whole request.
app.add_middleware(MatchTracingMiddleware)

router = APIRouter(dependencies=[Depends(api_key_auth)])


class Health(BaseModel):
    status: str


@router.get("/health")
async def health() -> Health:
    return Health(status="ok")


add_jobs_routes(router)
add_matching_routes(router)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    # Default 127.0.0.1 so /docs URLs work in the browser; use UVICORN_HOST=0.0.0.0 for Docker/LAN.
    _host = (os.getenv("UVICORN_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    _port = int((os.getenv("PORT") or "8080").strip() or "8080")
    uvicorn.run(app, host=_host, port=_port)
