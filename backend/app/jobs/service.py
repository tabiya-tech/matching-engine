"""Service for browsing the jobs catalog."""

import logging
import time
from abc import ABC, abstractmethod

from app.jobs.repository import IJobsRepository
from app.jobs.types import JobsBrowseQuery
from app.match_timing_log import log_match_step
from app.schemas import JobsPage, JobsStats


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class IJobsService(ABC):
    """Interface for browsing active jobs."""

    @abstractmethod
    async def browse(self, query: JobsBrowseQuery) -> JobsPage:
        """
        One page of active jobs, newest first.

        :param query: Cursor, page size and filters
        :return: The page, the next cursor (None on the last page) and the total when requested
        :raises InvalidCursor: If ``query.cursor`` is malformed
        :raises Exception: If any other error occurs
        """
        raise NotImplementedError()

    @abstractmethod
    async def stats(self) -> JobsStats:
        """
        Aggregate counts over the active jobs catalog.

        :raises Exception: If any error occurs
        """
        raise NotImplementedError()


class JobsService(IJobsService):
    def __init__(self, *, jobs_repository: IJobsRepository):
        self._jobs_repository = jobs_repository
        self._logger = logging.getLogger(self.__class__.__name__)

    async def browse(self, query: JobsBrowseQuery) -> JobsPage:
        t_req = time.perf_counter()
        (
            jobs,
            next_cursor,
            total,
            timing,
        ) = await self._jobs_repository.get_jobs_page_with_timing(
            cursor=query.cursor,
            limit=query.limit,
            search=query.search,
            category=query.category,
            employment_type=query.employment_type,
            location=query.location,
            skills=query.skills,
            days=query.days,
            include_total=query.include_total,
        )
        log_match_step(
            "http /jobs",
            "request (summary)",
            n_jobs=len(jobs),
            has_more=timing.get("has_more"),
            limit=timing.get("limit"),
            total=total,
            request_total_ms=_ms(t_req),
        )
        return JobsPage(items=jobs, next_cursor=next_cursor, total=total)

    async def stats(self) -> JobsStats:
        t_req = time.perf_counter()
        stats = await self._jobs_repository.get_jobs_stats()
        log_match_step(
            "http /jobs/stats",
            "request (summary)",
            total=stats.total,
            sectors=stats.sectors,
            platforms=stats.platforms,
            request_total_ms=_ms(t_req),
        )
        return stats
