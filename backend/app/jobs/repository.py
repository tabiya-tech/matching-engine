"""Repository for the jobs collection (``MONGO_JOBS_COLLECTION``)."""

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import (
    JOBS_FIND_USE_PROJECTION,
    JOBS_RETRIEVAL_FILTER,
    JOBS_RETRIEVAL_LIMIT,
    MONGO_JOBS_COLLECTION,
)
from app.jobs.documents import (
    JOBS_INDEX_MODELS,
    JOBS_PAGE_SORT,
    RANKED_JOB_FIND_PROJECTION,
    RANKED_JOBS_ACTIVE_FILTER,
    _decode_jobs_cursor,
    _encode_jobs_cursor,
    build_job_dict_from_ranked,
    build_jobs_browse_filter,
    build_mongo_filter_active_and_location,
)
from app.schemas import JobListItem, JobsStats

logger = logging.getLogger(__name__)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class IJobsRepository(ABC):
    """Interface for reading the jobs collection."""

    @abstractmethod
    async def find_active(
        self, users: Sequence[dict] | None = None
    ) -> tuple[list[dict], dict[str, Any]]:
        """
        Loads active jobs as flat job dicts (see ``build_job_dict_from_ranked``).

        If ``JOBS_RETRIEVAL_FILTER`` is true and ``users`` is non-empty, the query also ORs
        per-user location clauses, sorts by ``_id`` descending and applies ``JOBS_RETRIEVAL_LIMIT``.

        :param users: Users whose locations scope the query, or None for every active job
        :return: ``(jobs, timing)``; timing keys: mongo_ranked_find_ms, python_build_jobs_ms,
            n_ranked_raw, n_jobs, n_skipped_inactive, get_all_jobs_total_ms,
            jobs_retrieval_filter_applied, jobs_find_use_projection
        :raises Exception: If any database error occurs
        """
        raise NotImplementedError()

    @abstractmethod
    async def browse_page(
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
        Cursor-paginated, filterable browse over active jobs, newest first.

        Pagination is keyset-based on ``_id`` descending; ``cursor`` wraps the last ``_id`` of the
        previous page. Supplied filters are AND-ed (see ``build_jobs_browse_filter``).

        :return: ``(jobs, next_cursor, total, timing)``; ``next_cursor`` is None on the last page,
            ``total`` is None unless ``include_total`` is set
        :raises InvalidCursor: If ``cursor`` is malformed
        :raises Exception: If any database error occurs
        """
        raise NotImplementedError()

    @abstractmethod
    async def stats(self) -> JobsStats:
        """
        Aggregate counts over the active jobs catalog.

        :return: Total jobs, distinct sectors and distinct source platforms
        :raises Exception: If any database error occurs
        """
        raise NotImplementedError()

    @abstractmethod
    async def ensure_indexes(self) -> list[str]:
        """
        Creates (idempotently) the indexes the jobs queries need.

        :return: The ensured index names
        :raises Exception: If any database error occurs
        """
        raise NotImplementedError()

    @abstractmethod
    async def ping(self) -> None:
        """
        Pings the database.

        :raises Exception: If the database cannot be reached
        """
        raise NotImplementedError()


class JobsRepository(IJobsRepository):
    def __init__(self, *, db: AsyncIOMotorDatabase):
        self._db = db
        self._logger = logging.getLogger(self.__class__.__name__)

    async def find_active(
        self, users: Sequence[dict] | None = None
    ) -> tuple[list[dict], dict[str, Any]]:
        t_total = time.perf_counter()
        t0 = time.perf_counter()
        filt: dict[str, Any] = RANKED_JOBS_ACTIVE_FILTER
        retrieval_applied = False
        if JOBS_RETRIEVAL_FILTER and users:
            built = build_mongo_filter_active_and_location(users)
            if built is not None and built != RANKED_JOBS_ACTIVE_FILTER:
                filt = built
                retrieval_applied = True
        col = self._db[MONGO_JOBS_COLLECTION]
        if JOBS_FIND_USE_PROJECTION:
            cursor = col.find(filt, RANKED_JOB_FIND_PROJECTION)
        else:
            cursor = col.find(filt)
        if retrieval_applied:
            cursor = cursor.sort([("_id", -1)])
            if JOBS_RETRIEVAL_LIMIT > 0:
                cursor = cursor.limit(JOBS_RETRIEVAL_LIMIT)
        ranked_docs = [d async for d in cursor]
        mongo_ranked_find_ms = _ms(t0)

        t0 = time.perf_counter()
        jobs: list[dict] = []
        skipped = 0
        for rd in ranked_docs:
            built = build_job_dict_from_ranked(rd)
            if built is None:
                skipped += 1
                continue
            jobs.append(built)

        python_build_jobs_ms = _ms(t0)
        total_ms = _ms(t_total)
        logger.info(
            "Loaded %d active jobs from %s (matched=%d, skipped_in_build=%d)",
            len(jobs),
            MONGO_JOBS_COLLECTION,
            len(ranked_docs),
            skipped,
        )
        return jobs, {
            "mongo_ranked_find_ms": mongo_ranked_find_ms,
            "python_build_jobs_ms": python_build_jobs_ms,
            "n_ranked_raw": len(ranked_docs),
            "n_jobs": len(jobs),
            "n_skipped_inactive": skipped,
            "get_all_jobs_total_ms": total_ms,
            "jobs_retrieval_filter_applied": retrieval_applied,
            "jobs_find_use_projection": JOBS_FIND_USE_PROJECTION,
        }

    async def browse_page(
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
        t_total = time.perf_counter()
        limit = max(1, int(limit))

        base_filt = build_jobs_browse_filter(
            search=search,
            category=category,
            employment_type=employment_type,
            location=location,
            skills=skills,
            days=days,
        )
        filt: dict[str, Any] = dict(base_filt)
        if cursor:
            # Compose the keyset seek with the (possibly compound) filter without clobbering it.
            filt = {"$and": [base_filt, {"_id": {"$lt": _decode_jobs_cursor(cursor)}}]}

        col = self._db[MONGO_JOBS_COLLECTION]
        projection = RANKED_JOB_FIND_PROJECTION if JOBS_FIND_USE_PROJECTION else None
        # Fetch limit+1 so we can tell whether another page exists.
        t0 = time.perf_counter()
        query = col.find(filt, projection) if projection else col.find(filt)
        query = query.sort(JOBS_PAGE_SORT).limit(limit + 1)
        raw_docs = [d async for d in query]
        mongo_find_ms = _ms(t0)

        total: int | None = None
        if include_total:
            total = await col.count_documents(base_filt)

        has_more = len(raw_docs) > limit
        page_docs = raw_docs[:limit]

        t0 = time.perf_counter()
        jobs: list[JobListItem] = []
        skipped = 0
        for rd in page_docs:
            built = build_job_dict_from_ranked(rd)
            if built is None:
                skipped += 1
                continue
            jobs.append(
                JobListItem(
                    uuid=built.get("uuid"),
                    originUuid=built.get("originUuid"),
                    url=built.get("url"),
                    opportunity_title=built.get("opportunity_title", "No title"),
                    opportunity_isco_occupation_group=built.get(
                        "opportunity_isco_occupation_group"
                    ),
                    opportunity_isco_occupation_group_id=built.get(
                        "opportunity_isco_occupation_group_id"
                    ),
                    related_occupation_id=built.get("related_occupation_id"),
                    location=built.get("location"),
                    city=built.get("city"),
                    province=built.get("province"),
                    employer=built.get("employer"),
                    employment_type=built.get("employment_type"),
                    contract_type=built.get("contract_type"),
                    salary_text=built.get("salary_text"),
                    closing_date=built.get("closing_date"),
                    posted_date=built.get("posted_date"),
                    opportunity_description=built.get("opportunity_description"),
                    # Consumer-contract fields (Compass jobs board)=built.get("# Consumer-contract fields (Compass jobs board),
                    # posting was scraped from, and the flat list of skill labels for this opportunity.
                    category=built.get("category"),
                    source_platform=built.get("source_platform"),
                    skills=built.get("skills", []),
                )
            )
        python_build_ms = _ms(t0)

        next_cursor = (
            _encode_jobs_cursor(page_docs[-1]["_id"])
            if has_more and page_docs
            else None
        )

        return (
            jobs,
            next_cursor,
            total,
            {
                "mongo_find_ms": mongo_find_ms,
                "python_build_jobs_ms": python_build_ms,
                "n_page_raw": len(page_docs),
                "n_jobs": len(jobs),
                "n_skipped_inactive": skipped,
                "has_more": has_more,
                "limit": limit,
                "total": total,
                "get_jobs_page_total_ms": _ms(t_total),
            },
        )

    async def stats(self) -> JobsStats:
        col = self._db[MONGO_JOBS_COLLECTION]
        total = await col.count_documents(RANKED_JOBS_ACTIVE_FILTER)

        raw_categories = await col.distinct(
            "classifier_metadata.category", RANKED_JOBS_ACTIVE_FILTER
        )
        if not raw_categories:
            raw_categories = await col.distinct(
                "classifier_metadata.isco_occupation_group", RANKED_JOBS_ACTIVE_FILTER
            )
        sectors = len(
            {str(c).strip().lower() for c in raw_categories if str(c).strip()}
        )

        platforms_set: set = set()
        for field in (
            "classifier_metadata.source_platform",
            "classifier_metadata.source",
            "classifier_metadata.platform",
        ):
            for p in await col.distinct(field, RANKED_JOBS_ACTIVE_FILTER):
                if str(p).strip():
                    platforms_set.add(str(p).strip().lower())

        return JobsStats(total=total, sectors=sectors, platforms=len(platforms_set))

    async def ensure_indexes(self) -> list[str]:
        t0 = time.perf_counter()
        col = self._db[MONGO_JOBS_COLLECTION]
        created = await col.create_indexes(JOBS_INDEX_MODELS)
        logger.info(
            "Ensured %d indexes on %s in %.2f ms: %s",
            len(created),
            MONGO_JOBS_COLLECTION,
            _ms(t0),
            created,
        )
        return created

    async def ping(self) -> None:
        await self._db.command("ping")
