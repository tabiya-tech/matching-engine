"""Dependency provider for the JobsService."""

from fastapi import Depends

from app.jobs.get_jobs_repository import get_jobs_repository
from app.jobs.repository import IJobsRepository
from app.jobs.service import IJobsService, JobsService


async def get_jobs_service(
    jobs_repository: IJobsRepository = Depends(get_jobs_repository),
) -> IJobsService:
    # Built per request, like the jobs repository it wraps.
    return JobsService(jobs_repository=jobs_repository)
