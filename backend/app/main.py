import asyncio
import logging
import logging.config
import os
from contextlib import asynccontextmanager, suppress

import yaml
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import DEBUG_MODE
from app.constants.errors import (
    FORBIDDEN_RESPONSE_DOC,
    INTERNAL_ERROR_RESPONSE_DOC,
    HTTPErrorResponse,
)
from app.jobs.routes import add_jobs_routes
from app.matching.routes import add_matching_routes
from app.observability import (
    MatchTracingMiddleware,
    init_tracing,
    shutdown_tracing,
    tracing_config_from_env,
)
from app.server_dependencies.auth import api_key_auth
from app.server_dependencies.db_dependencies import MatchingDBProvider
from app.server_dependencies.model_dependencies import (
    get_cross_encoder_client,
    get_skill_scorer,
)
from app.warmup import warmup_on_startup

load_dotenv()


def setup_logging() -> None:
    """JSON logs on stdout (``logging.cfg.yaml``); ``DEBUG_MODE`` lowers the root level to DEBUG."""
    config_file = os.path.join(os.path.dirname(__file__), "logging.cfg.yaml")
    with open(config_file, encoding="utf-8") as stream:
        logging.config.dictConfig(yaml.safe_load(stream))
    if DEBUG_MODE:
        logging.getLogger().setLevel(logging.DEBUG)


setup_logging()
logger = logging.getLogger(__name__)


def _warmup_non_blocking() -> bool:
    v = (os.getenv("WARMUP_NON_BLOCKING") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Owns the MongoDB client for the app's lifetime and runs the one-time warmup.

    The client is created here (before any request) and closed on shutdown; routes reach it only
    through ``Depends(get_jobs_repository)``. Warmup (ping, indexes, occupation cache, models) runs
    once so /match does not pay cold-start cost on each request.
    """
    init_tracing(tracing_config_from_env())
    # Load the skill-gap engine before serving so the first /match does not pay the load.
    get_skill_scorer()

    jobs_db = await MatchingDBProvider.get_jobs_db()

    async def _warmup_safe() -> None:
        try:
            await warmup_on_startup(jobs_db)
        except Exception:
            logger.exception("Startup warmup failed")

    warmup_task = None
    if _warmup_non_blocking():
        warmup_task = asyncio.create_task(_warmup_safe())
        logger.info(
            "Startup: warmup scheduled in background (WARMUP_NON_BLOCKING=1); "
            "first /match may still pay part of cold cost until warmup finishes"
        )
    else:
        await _warmup_safe()
    get_cross_encoder_client()
    yield
    # A background warmup still running at shutdown would otherwise use the client after it is closed.
    if warmup_task is not None and not warmup_task.done():
        warmup_task.cancel()
        with suppress(asyncio.CancelledError):
            await warmup_task
    # Flush buffered traces before Cloud Run tears the instance down.
    shutdown_tracing()
    MatchingDBProvider.close()


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


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=HTTPErrorResponse(detail=detail).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 as ``HTTPErrorResponse``: FastAPI's error list flattened to one ``detail`` string.

    Only ``loc`` and ``msg`` are kept — never the rejected ``input``, which can be profile data.
    """
    errors = exc.errors()
    detail = "; ".join(
        f"{'.'.join(str(part) for part in err.get('loc', ()))}: {err.get('msg', '')}"
        for err in errors
    )
    logger.warning(
        "Request validation failed {errors=%s}",
        [
            (".".join(str(p) for p in err.get("loc", ())), err.get("type"))
            for err in errors
        ],
    )
    return _error_response(422, detail or "Request validation failed.")


@app.exception_handler(Exception)
async def _unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Safety net for anything a route or dependency did not map to an ``HTTPException``.

    Not logged here: Starlette re-raises after this response is sent and uvicorn logs the traceback.
    """
    return _error_response(500, f"Internal server error: {exc.__class__.__name__}")


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


@router.get(
    "/health",
    responses={403: FORBIDDEN_RESPONSE_DOC, 500: INTERNAL_ERROR_RESPONSE_DOC},
)
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
