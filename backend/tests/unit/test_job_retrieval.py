"""Tests for the /match job-pool policy (location filter) that sits above the jobs repository."""

import asyncio
from unittest.mock import AsyncMock, patch

from app.jobs.repository import IJobsRepository
from app.location.keys import HubChains, user_location_keys
from app.services import job_retrieval, location_query
from app.services.location_query import build_location_filter
from tests.domain_factories import jobs_load_timing, make_job, make_user

_NAIROBI_KEYS = user_location_keys(
    make_user(user_id="u1", city="Nairobi", province="Nairobi")
)
_JOB = make_job(uuid="j1")


def _repository() -> AsyncMock:
    repository = AsyncMock(spec=IJobsRepository)
    repository.find_jobs_with_timing.side_effect = lambda *_a, **_kw: (
        [_JOB],
        jobs_load_timing(n_jobs=1),
    )
    return repository


class TestRetrieveJobsWithTiming:
    def test_without_users_loads_every_active_job(self):
        repository = _repository()

        jobs, timing = asyncio.run(
            job_retrieval.retrieve_jobs_with_timing(repository, None)
        )

        assert repository.find_jobs_with_timing.await_args.args == (None,)
        assert jobs == [_JOB]
        assert timing.jobs_retrieval_filter_applied is False

    def test_with_users_applies_the_location_filter_and_limit(self):
        repository = _repository()

        with (
            patch.object(job_retrieval, "JOBS_RETRIEVAL_FILTER", True),
            patch.object(job_retrieval, "JOBS_RETRIEVAL_LIMIT", 123),
        ):
            _, timing = asyncio.run(
                job_retrieval.retrieve_jobs_with_timing(repository, [_NAIROBI_KEYS])
            )

        call = repository.find_jobs_with_timing.await_args
        assert call.args == (build_location_filter([_NAIROBI_KEYS]),)
        assert call.kwargs == {"limit": 123}
        assert timing.jobs_retrieval_filter_applied is True

    def test_filter_switched_off_loads_every_active_job(self):
        repository = _repository()

        with patch.object(job_retrieval, "JOBS_RETRIEVAL_FILTER", False):
            _, timing = asyncio.run(
                job_retrieval.retrieve_jobs_with_timing(repository, [_NAIROBI_KEYS])
            )

        assert repository.find_jobs_with_timing.await_args.args == (None,)
        assert timing.jobs_retrieval_filter_applied is False


class TestLocationQuery:
    def test_filter_is_active_and_location_keys_in(self):
        assert build_location_filter([_NAIROBI_KEYS]) == {
            "is_active": True,
            "location_keys": {"$in": ["nairobi", "remote"]},
        }

    def test_filter_is_the_union_of_the_batch_users_keys(self):
        kitui = user_location_keys(
            make_user(user_id="u2", city="Kitui", province="Kitui")
        )

        actual = build_location_filter([_NAIROBI_KEYS, kitui])

        assert actual["location_keys"] == {"$in": ["kitui", "nairobi", "remote"]}

    def test_user_without_city_and_province_only_matches_remote_jobs(self):
        keys = user_location_keys(make_user(user_id="u1"))

        assert build_location_filter([keys])["location_keys"] == {"$in": ["remote"]}

    def test_hub_chains_only_when_location_tiers_are_on(self):
        artifacts = AsyncMock()
        artifacts.load_hub_chains = lambda _path: HubChains("nairobi", None, None)

        with patch.object(location_query, "LOCATION_TIER_ENABLED", False):
            assert location_query.location_hub_chains(artifacts) is None
        with patch.object(location_query, "LOCATION_TIER_ENABLED", True):
            assert location_query.location_hub_chains(artifacts) is not None

    def test_resolved_keys_widen_to_the_hub_chain(self):
        users = [make_user(user_id="u1", city="Kitui", province="Kitui")]

        (keys,) = location_query.resolve_location_keys(
            users, HubChains("nairobi", None, None)
        )

        assert keys.keys == {"remote", "kitui", "nairobi"}
