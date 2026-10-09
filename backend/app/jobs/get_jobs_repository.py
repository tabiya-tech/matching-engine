from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.jobs.mongo_repository import MongoJobsRepository
from app.jobs.repository import IJobsRepository
from app.server_dependencies.db_dependencies import MatchingDBProvider


async def get_jobs_repository(
    jobs_db: AsyncIOMotorDatabase = Depends(MatchingDBProvider.get_jobs_db),
) -> IJobsRepository:
    # Built per request rather than cached: it only holds a collection handle, and a cached instance
    # would keep pointing at a closed client after the provider is closed and reopened (tests, scripts).
    return MongoJobsRepository(jobs_db)


@asynccontextmanager
async def standalone_jobs_repository() -> AsyncIterator[IJobsRepository]:
    """Jobs repository for code running outside the FastAPI app (CLI scripts, offline evaluation).

    Opens the provider's client for the duration of the block and closes it on exit, so the client
    is bound to the caller's event loop. Never use this inside the app — routes get the repository
    through ``Depends(get_jobs_repository)`` and the lifespan owns the client.
    """
    jobs_db = await MatchingDBProvider.get_jobs_db()
    try:
        yield MongoJobsRepository(jobs_db)
    finally:
        MatchingDBProvider.close()
