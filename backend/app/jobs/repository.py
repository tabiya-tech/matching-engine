from abc import ABC, abstractmethod
from typing import Any

from app.schemas import JobListItem, JobsStats


class IJobsRepository(ABC):
    """
    Interface for the Jobs Repository.
    Allows to mock the repository in tests.
    """

    @abstractmethod
    async def find_jobs_with_timing(
        self, query_filter: dict[str, Any] | None = None, *, limit: int = 0
    ) -> tuple[list[dict], dict[str, Any]]:
        """
        Load active jobs as flat job dicts (see ``build_job_dict_from_ranked``).

        Args:
            query_filter: Mongo filter to apply. ``None`` loads every active job, unsorted and uncapped.
                A filter is expected to already include ``is_active`` (the caller composes it); results
                are then sorted newest-first.
            limit: Cap on documents read when ``query_filter`` is given; ``0`` means no cap.

        Returns:
            ``(jobs, timing)`` where timing holds the Mongo read and Python build durations and counts.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_jobs_page_with_timing(
        self,
        cursor: str | None = None,
        limit: int = 20,
        *,
        search: str | None = None,
        category: str | None = None,
        employment_type: str | None = None,
        location: str | None = None,
        skills: str | None = None,
        days: int | None = None,
        include_total: bool = False,
    ) -> tuple[list[JobListItem], str | None, int | None, dict[str, Any]]:
        """
        Cursor-paginated, filterable browse over active jobs.

        Returns:
            ``(jobs, next_cursor, total, timing)``. Raises ``InvalidCursor`` if ``cursor`` is malformed.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_jobs_stats(self) -> JobsStats:
        """Aggregate counts over the active jobs catalog."""
        raise NotImplementedError
