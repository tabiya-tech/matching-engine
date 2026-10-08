"""Dependency provider for the singleton JobsService."""

import asyncio

from fastapi import Depends

from app.jobs.get_jobs_repository import get_jobs_repository
from app.jobs.repository import IJobsRepository
from app.jobs.service import IJobsService, JobsService

_jobs_service_singleton: IJobsService | None = None
_jobs_service_lock = asyncio.Lock()


async def get_jobs_service(
    jobs_repository: IJobsRepository = Depends(get_jobs_repository),
) -> IJobsService:
    """Return the process-wide JobsService, instantiating it on first use."""
    global _jobs_service_singleton  # pylint: disable=global-statement

    if _jobs_service_singleton is None:
        async with _jobs_service_lock:
            if _jobs_service_singleton is None:
                _jobs_service_singleton = JobsService(jobs_repository=jobs_repository)

    return _jobs_service_singleton
