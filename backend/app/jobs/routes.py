"""Routes for browsing the jobs catalog: ``GET /jobs`` and ``GET /jobs/stats``."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import JOBS_PAGE_DEFAULT_LIMIT, JOBS_PAGE_MAX_LIMIT
from app.jobs.errors import InvalidCursor
from app.jobs.get_jobs_service import get_jobs_service
from app.jobs.service import IJobsService
from app.jobs.types import JobsBrowseQuery
from app.schemas import JobsPage, JobsStats

logger = logging.getLogger(__name__)


def add_jobs_routes(router: APIRouter) -> None:
    """
    Adds the jobs routes to the router.

    :param router: The authenticated service router
    """

    @router.get(
        "/jobs",
        tags=["jobs"],
        operation_id="list_jobs",
        response_model=JobsPage,
        responses={
            400: {
                "description": "Bad Request - invalid cursor",
                "content": {
                    "application/json": {"example": {"detail": "invalid cursor"}}
                },
            },
            500: {
                "description": "Internal Server Error",
                "content": {
                    "application/json": {"example": {"detail": "Internal server error"}}
                },
            },
        },
    )
    async def list_jobs(
        cursor: Optional[str] = Query(
            None,
            description=(
                "Opaque pagination cursor returned as ``next_cursor`` by the previous "
                "response. Omit to fetch the first page."
            ),
        ),
        limit: int = Query(
            JOBS_PAGE_DEFAULT_LIMIT,
            ge=1,
            le=JOBS_PAGE_MAX_LIMIT,
            description=f"Page size (1–{JOBS_PAGE_MAX_LIMIT}). Default {JOBS_PAGE_DEFAULT_LIMIT}.",
        ),
        search: Optional[str] = Query(
            None, description="Case-insensitive search on the job title."
        ),
        category: Optional[str] = Query(
            None,
            description="Filter by sector/category (matches category, sector, or ISCO group).",
        ),
        employment_type: Optional[str] = Query(
            None, description="Filter by employment type (exact match)."
        ),
        location: Optional[str] = Query(
            None, description="Case-insensitive filter on city/county/province."
        ),
        skills: Optional[str] = Query(
            None,
            description="Case-insensitive filter on a skill label of the opportunity.",
        ),
        days: Optional[int] = Query(
            None, ge=1, le=3650, description="Only jobs posted within the last N days."
        ),
        include_total: bool = Query(
            False,
            description="When true, include the total count of jobs matching the filters.",
        ),
        jobs_service: IJobsService = Depends(get_jobs_service),
    ):
        """
        Browse active jobs with cursor-based pagination and optional filters.

        Reads from the same Mongo collection and through the same shaping as the matched-jobs
        endpoints (CORE-418), so a browsed job and a matched job are the same object minus the
        per-user scoring fields. Results are ordered newest-first (``_id`` descending) and the
        keyset cursor is stable under concurrent inserts. Supplied filters are AND-ed together;
        pass ``include_total=true`` to also receive the total count for the active filter set.
        """
        try:
            return await jobs_service.browse(
                JobsBrowseQuery(
                    cursor=cursor,
                    limit=limit,
                    search=search,
                    category=category,
                    employment_type=employment_type,
                    location=location,
                    skills=skills,
                    days=days,
                    include_total=include_total,
                )
            )
        except InvalidCursor as e:
            logger.warning("Invalid /jobs cursor: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(e)
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {e.__class__.__name__}"
            )

    @router.get(
        "/jobs/stats",
        tags=["jobs"],
        operation_id="jobs_stats",
        response_model=JobsStats,
        responses={
            500: {
                "description": "Internal Server Error",
                "content": {
                    "application/json": {"example": {"detail": "Internal server error"}}
                },
            },
        },
    )
    async def jobs_stats(
        jobs_service: IJobsService = Depends(get_jobs_service),
    ) -> JobsStats:
        """Aggregate counts over the active jobs catalog: total jobs, distinct sectors, distinct platforms."""
        try:
            return await jobs_service.stats()
        except Exception as e:
            logger.exception(e)
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {e.__class__.__name__}"
            )
