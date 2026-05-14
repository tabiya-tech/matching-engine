import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.match_timing_log import init_match_timing_log
from app.routes import router, router_public


load_dotenv()
init_match_timing_log()

logger = logging.getLogger(__name__)


def _warmup_non_blocking() -> bool:
    v = (os.getenv("WARMUP_NON_BLOCKING") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Mongo ping + occupation JSON + WA lookup once at startup so /match does not pay cold-connection cost each time."""
    from app.database import warmup_on_startup

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
    yield


app = FastAPI(title="Matching Service", lifespan=lifespan)

logging.basicConfig(level=logging.INFO)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(router_public)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
