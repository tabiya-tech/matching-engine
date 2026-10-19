"""Job pool for a /match request: applies the location filter policy, then reads via the repository."""

from collections.abc import Sequence
from typing import Any

from app.config import JOBS_RETRIEVAL_FILTER, JOBS_RETRIEVAL_LIMIT
from app.domain.job import Job
from app.jobs.repository import IJobsRepository, JobsLoadTiming
from app.location.keys import UserLocationKeys
from app.services.location_query import build_location_filter


async def retrieve_jobs_with_timing(
    jobs_repository: IJobsRepository,
    location_keys: Sequence[UserLocationKeys] | None = None,
) -> tuple[list[Job], JobsLoadTiming]:
    """Load the active jobs to match the users with ``location_keys`` against; returns ``(jobs, timing)``.

    If ``JOBS_RETRIEVAL_FILTER`` is true and ``location_keys`` is non-empty, only jobs having one of
    the users' location keys are loaded (``build_location_filter``; indexed), newest first, at most
    ``JOBS_RETRIEVAL_LIMIT``. With no keys (or the filter off), every active job is loaded with no
    sort or cap.

    The returned timing is the repository's, with ``jobs_retrieval_filter_applied`` set.

    :param location_keys: The requesting users' eligible keys (``resolve_location_keys``)
    """
    query_filter: dict[str, Any] | None = None
    if JOBS_RETRIEVAL_FILTER and location_keys:
        query_filter = build_location_filter(location_keys)
    jobs, timing = await jobs_repository.find_jobs_with_timing(
        query_filter, limit=JOBS_RETRIEVAL_LIMIT
    )
    return jobs, timing.model_copy(
        update={"jobs_retrieval_filter_applied": query_filter is not None}
    )
