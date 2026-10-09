"""Job pool for a /match request: applies the location prefilter policy, then reads via the repository."""

from collections.abc import Sequence
from typing import Any

from app.config import JOBS_RETRIEVAL_FILTER, JOBS_RETRIEVAL_LIMIT
from app.jobs.job_document import RANKED_JOBS_ACTIVE_FILTER
from app.jobs.repository import IJobsRepository
from app.services.location_query import build_mongo_filter_active_and_location


async def retrieve_jobs_with_timing(
    jobs_repository: IJobsRepository, users: Sequence[dict] | None = None
) -> tuple[list[dict], dict[str, Any]]:
    """Load the active jobs to match ``users`` against; returns ``(jobs, timing)``.

    If ``JOBS_RETRIEVAL_FILTER`` is true and ``users`` is non-empty, the query also ORs per-user
    location clauses (see ``build_mongo_filter_active_and_location``), sorts newest-first, and applies
    ``JOBS_RETRIEVAL_LIMIT``. With no ``users`` (or the filter off), every active job is loaded with no
    sort or cap.

    timing keys: those of ``IJobsRepository.find_jobs_with_timing`` plus ``jobs_retrieval_filter_applied``.
    """
    query_filter: dict[str, Any] | None = None
    if JOBS_RETRIEVAL_FILTER and users:
        built = build_mongo_filter_active_and_location(users)
        if built is not None and built != RANKED_JOBS_ACTIVE_FILTER:
            query_filter = built
    jobs, timing = await jobs_repository.find_jobs_with_timing(
        query_filter, limit=JOBS_RETRIEVAL_LIMIT
    )
    timing["jobs_retrieval_filter_applied"] = query_filter is not None
    return jobs, timing
