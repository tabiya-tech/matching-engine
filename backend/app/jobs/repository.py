from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field

from app.domain.base import DomainModel
from app.domain.job import Job
from app.schemas import JobListItem, JobsStats
from app.server_dependencies.vector_search import Stage1Coverage, VectorSearchQuery


class JobsLoadTiming(DomainModel):
    """Durations and counts of one ``find_jobs_with_timing`` call (logged and traced, never returned)."""

    mongo_ranked_find_ms: float = Field(
        description="Time reading the documents from Mongo."
    )
    python_build_jobs_ms: float = Field(
        description="Time validating and mapping documents to jobs."
    )
    n_ranked_raw: int = Field(description="Documents read.")
    n_jobs: int = Field(description="Jobs returned.")
    n_skipped_inactive: int = Field(
        description="Documents skipped because ``is_active`` is False."
    )
    n_skipped_invalid: int = Field(
        description="Documents skipped because they failed validation."
    )
    get_all_jobs_total_ms: float = Field(description="Total time of the call.")
    jobs_find_use_projection: bool = Field(
        description="Whether the find used ``RANKED_JOB_FIND_PROJECTION``."
    )
    jobs_retrieval_filter_applied: bool = Field(
        default=False,
        description="Whether the /match location prefilter was applied (set by ``retrieve_jobs_with_timing``).",
    )


class JobsPageTiming(DomainModel):
    """Durations and counts of one ``get_jobs_page_with_timing`` call."""

    mongo_find_ms: float = Field(description="Time reading the page from Mongo.")
    python_build_jobs_ms: float = Field(
        description="Time validating and mapping documents to jobs."
    )
    n_page_raw: int = Field(
        description="Documents read for the page (without the look-ahead one)."
    )
    n_jobs: int = Field(description="Jobs on the page.")
    n_skipped_inactive: int = Field(
        description="Documents skipped because ``is_active`` is False."
    )
    n_skipped_invalid: int = Field(
        description="Documents skipped because they failed validation."
    )
    has_more: bool = Field(description="Whether another page exists.")
    limit: int = Field(description="Page size used.")
    total: int | None = Field(
        description="Jobs matching the filters; None unless requested."
    )
    get_jobs_page_total_ms: float = Field(description="Total time of the call.")


class JobSearchHit(DomainModel):
    """One job returned by the stage-1 vector search."""

    job: Job = Field(description="The job, without its embedding.")
    cosine: float = Field(
        description="Whitened concat cosine between the query and the job's ``job_embedding``."
    )


class JobsSearchResult(DomainModel):
    """The hits of one ``search_stage1`` call, best first, with what was dropped and how long it took."""

    hits: list[JobSearchHit] = Field(description="Valid active jobs, best first.")
    n_returned: int = Field(description="Documents the search returned.")
    n_skipped_inactive: int = Field(
        description="Documents skipped because ``is_active`` is False."
    )
    n_skipped_invalid: int = Field(
        description="Documents skipped because they failed validation."
    )
    search_ms: float = Field(description="Time running the search and reading hits.")


class IJobsRepository(ABC):
    """
    Interface for the Jobs Repository.
    Allows to mock the repository in tests.
    """

    @abstractmethod
    async def find_jobs_with_timing(
        self, query_filter: dict[str, Any] | None = None, *, limit: int = 0
    ) -> tuple[list[Job], JobsLoadTiming]:
        """
        Load active jobs, validated and mapped from their stored documents (see ``build_job_from_ranked``).

        Documents that fail validation are skipped and logged, not raised.

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
        location: list[str] | None = None,
        skills: str | None = None,
        days: int | None = None,
        include_total: bool = False,
    ) -> tuple[list[JobListItem], str | None, int | None, JobsPageTiming]:
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

    @abstractmethod
    async def search_stage1(self, query: VectorSearchQuery) -> JobsSearchResult:
        """
        Stage-1 vector search over the whitened ``job_embedding`` (``JOBS_VECTOR_INDEX``).

        ``query.filter`` is applied inside the search; the caller composes it (``is_active``, the
        whitening artifact, location and education). Hits are mapped through
        ``build_job_from_ranked`` without the embedding; invalid documents are logged and skipped.

        Args:
            query: The whitened user vector, the pre-filter and how many jobs to return.

        Returns:
            The hits, best first (at most ``query.limit``).

        Raises:
            pymongo.errors.OperationFailure: If the index is missing or the filter uses an
                undeclared field.
        """
        raise NotImplementedError

    @abstractmethod
    async def count_stage1_coverage(self, artifact_sha256: str) -> Stage1Coverage:
        """
        Count active jobs by vector-search eligibility for the given whitening artifact.

        Args:
            artifact_sha256: SHA-256 of the service's ``V4_FULL_CONCAT_WHITENING_PATH``.
        """
        raise NotImplementedError
