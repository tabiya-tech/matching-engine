import logging
import time
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import JOBS_FIND_USE_PROJECTION
from app.jobs.browse_filter import build_jobs_browse_filter
from app.jobs.job_document import (
    RANKED_JOB_FIND_PROJECTION,
    RANKED_JOBS_ACTIVE_FILTER,
    build_job_dict_from_ranked,
)
from app.jobs.pagination import JOBS_PAGE_SORT, decode_jobs_cursor, encode_jobs_cursor
from app.jobs.repository import IJobsRepository
from app.schemas import JobListItem, JobsStats
from app.server_dependencies.database_collections import Collections


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class MongoJobsRepository(IJobsRepository):
    """
    Mongo implementation of ``IJobsRepository``: reads the ranked/enriched job ads written by the llm-reranker.
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self._collection = db.get_collection(Collections.JOBS)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _find(self, query_filter: dict[str, Any]):
        if JOBS_FIND_USE_PROJECTION:
            return self._collection.find(query_filter, RANKED_JOB_FIND_PROJECTION)
        return self._collection.find(query_filter)

    async def find_jobs_with_timing(
        self, query_filter: dict[str, Any] | None = None, *, limit: int = 0
    ) -> tuple[list[dict], dict[str, Any]]:
        t_total = time.perf_counter()
        t0 = time.perf_counter()
        cursor = self._find(
            query_filter if query_filter is not None else RANKED_JOBS_ACTIVE_FILTER
        )
        if query_filter is not None:
            cursor = cursor.sort([("_id", -1)])
            if limit > 0:
                cursor = cursor.limit(limit)
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

        self._logger.info(
            "Loaded %d active jobs from %s (matched=%d, skipped_in_build=%d)",
            len(jobs),
            Collections.JOBS,
            len(ranked_docs),
            skipped,
        )
        return jobs, {
            "mongo_ranked_find_ms": mongo_ranked_find_ms,
            "python_build_jobs_ms": python_build_jobs_ms,
            "n_ranked_raw": len(ranked_docs),
            "n_jobs": len(jobs),
            "n_skipped_inactive": skipped,
            "get_all_jobs_total_ms": _ms(t_total),
            "jobs_find_use_projection": JOBS_FIND_USE_PROJECTION,
        }

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
        """Reads through the same ``build_job_dict_from_ranked`` shaping as ``find_jobs_with_timing``
        (the matched-jobs data source), so browse and match return identical job objects.

        Pagination is keyset-based on ``_id`` descending (newest first, stable under inserts):
        ``cursor`` is an opaque token wrapping the last ``_id`` of the previous page, and the
        next page is ``_id < cursor_id``. One extra document is fetched to compute ``has_more``
        and the next cursor without a second round-trip. Optional ``search``/``category``/
        ``employment_type``/``location``/``skills``/``days`` filters narrow the catalog
        (see ``build_jobs_browse_filter``); when ``include_total`` is set, the total count of
        the filtered catalog (ignoring the cursor) is returned for client-side pagination UIs.
        ``next_cursor`` is ``None`` on the last page; ``total`` is ``None`` unless ``include_total``
        is set.
        """
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
            filt = {"$and": [base_filt, {"_id": {"$lt": decode_jobs_cursor(cursor)}}]}

        # Fetch limit+1 so we can tell whether another page exists.
        t0 = time.perf_counter()
        query = self._find(filt).sort(JOBS_PAGE_SORT).limit(limit + 1)
        raw_docs = [d async for d in query]
        mongo_find_ms = _ms(t0)

        total: int | None = None
        if include_total:
            total = await self._collection.count_documents(base_filt)

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
                    # Consumer-contract fields (Compass jobs board): sector/category, the platform the
                    # posting was scraped from, and the flat list of skill labels for this opportunity.
                    category=built.get("category"),
                    source_platform=built.get("source_platform"),
                    skills=built.get("skills", []),
                )
            )
        python_build_ms = _ms(t0)

        next_cursor = (
            encode_jobs_cursor(page_docs[-1]["_id"]) if has_more and page_docs else None
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

    async def get_jobs_stats(self) -> JobsStats:
        """``sectors`` counts distinct, case-insensitively-deduplicated categories (falling back
        to the ISCO occupation group label, matching ``build_job_dict_from_ranked``);
        ``platforms`` counts distinct source platforms.
        """
        col = self._collection
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
