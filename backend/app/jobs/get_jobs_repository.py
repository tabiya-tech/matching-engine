"""Dependency provider for the singleton JobsRepository."""

import asyncio

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.jobs.repository import IJobsRepository, JobsRepository
from app.server_dependencies.db_dependencies import get_jobs_db

_jobs_repository_singleton: IJobsRepository | None = None
_jobs_repository_lock = asyncio.Lock()


async def get_jobs_repository(
    db: AsyncIOMotorDatabase = Depends(get_jobs_db),
) -> IJobsRepository:
    """Return the process-wide JobsRepository, instantiating it on first use."""
    global _jobs_repository_singleton  # pylint: disable=global-statement

    if _jobs_repository_singleton is None:
        async with _jobs_repository_lock:
            if _jobs_repository_singleton is None:
                _jobs_repository_singleton = JobsRepository(db=db)

    return _jobs_repository_singleton
