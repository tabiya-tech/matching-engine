"""Tests for the /match job-pool policy (location prefilter) that sits above the jobs repository."""

import asyncio
from unittest.mock import AsyncMock, patch

from app.jobs.job_document import RANKED_JOBS_ACTIVE_FILTER
from app.jobs.repository import IJobsRepository
from app.services import job_retrieval
from app.services.location_query import (
    build_mongo_filter_active_and_location,
    location_or_clauses_for_one_user,
)

_NAIROBI_USER = {"user_id": "u1", "city": "Nairobi", "province": "Nairobi"}


def _repository() -> AsyncMock:
    repository = AsyncMock(spec=IJobsRepository)
    repository.find_jobs_with_timing.side_effect = lambda *_a, **_kw: (
        [{"uuid": "j1"}],
        {"n_jobs": 1},
    )
    return repository


class TestRetrieveJobsWithTiming:
    def test_without_users_loads_every_active_job(self):
        repository = _repository()

        jobs, timing = asyncio.run(
            job_retrieval.retrieve_jobs_with_timing(repository, None)
        )

        assert repository.find_jobs_with_timing.await_args.args == (None,)
        assert jobs == [{"uuid": "j1"}]
        assert timing["jobs_retrieval_filter_applied"] is False

    def test_with_users_applies_the_location_prefilter_and_limit(self):
        repository = _repository()

        with (
            patch.object(job_retrieval, "JOBS_RETRIEVAL_FILTER", True),
            patch.object(job_retrieval, "JOBS_RETRIEVAL_LIMIT", 123),
        ):
            _, timing = asyncio.run(
                job_retrieval.retrieve_jobs_with_timing(repository, [_NAIROBI_USER])
            )

        call = repository.find_jobs_with_timing.await_args
        assert call.args == (build_mongo_filter_active_and_location([_NAIROBI_USER]),)
        assert call.kwargs == {"limit": 123}
        assert timing["jobs_retrieval_filter_applied"] is True

    def test_prefilter_switched_off_loads_every_active_job(self):
        repository = _repository()

        with patch.object(job_retrieval, "JOBS_RETRIEVAL_FILTER", False):
            _, timing = asyncio.run(
                job_retrieval.retrieve_jobs_with_timing(repository, [_NAIROBI_USER])
            )

        assert repository.find_jobs_with_timing.await_args.args == (None,)
        assert timing["jobs_retrieval_filter_applied"] is False


class TestLocationQuery:
    def test_no_users_means_no_filter(self):
        assert build_mongo_filter_active_and_location([]) is None

    def test_filter_is_active_and_any_location_clause(self):
        actual = build_mongo_filter_active_and_location([_NAIROBI_USER])

        assert actual["$and"][0] == RANKED_JOBS_ACTIVE_FILTER
        assert {
            "classifier_metadata.city": {"$regex": "nairobi", "$options": "i"}
        } in actual["$and"][1]["$or"]

    def test_user_without_city_and_province_only_matches_remote_jobs(self):
        actual = location_or_clauses_for_one_user({"user_id": "u1"})

        assert actual == [
            {"classifier_metadata.city": {"$regex": "remote", "$options": "i"}},
            {"classifier_metadata.county": {"$regex": "remote", "$options": "i"}},
        ]
