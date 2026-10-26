"""MATCH_V4_DISABLE_LOCATION kill-switch (see config.MATCH_V4_DISABLE_LOCATION).

The flag must remove every location step on ``/match_v4``: no hub-chain lookup, no location-key
restriction passed into job retrieval/shortlisting, and no occupation county filter. It takes
priority over LOCATION_TIER_ENABLED (hub chains are never even loaded when it's on).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from app.jobs.repository import IJobsRepository
from app.location.keys import HubChains
from app.matching import service as svc
from app.matching.service import MatchingService
from app.matching.types import MatchOptions
from app.schemas import MatchRequest
from app.services import location_query
from tests.domain_factories import (
    jobs_load_timing,
    make_job,
    make_occupation,
    make_user,
)


def _service() -> MatchingService:
    return MatchingService(
        jobs_repository=MagicMock(),
        artifacts_repository=MagicMock(),
        embedding_client=MagicMock(embedding_dim=4),
        cross_encoder_provider=MagicMock,
        retrieval_matcher_provider=MagicMock,
        gate_matcher_provider=MagicMock,
        whitener_provider=MagicMock,
        skill_scorer_provider=MagicMock,
        preference_scorer_provider=MagicMock,
        retriever=MagicMock(),
        rerank=MagicMock(),
    )


def _run(users, jobs, occupations, *, disable_location: bool):
    """MatchingService.rank with the ML stack stubbed; returns (rows, shortlist_mock, artifacts)."""
    service = _service()
    artifacts = MagicMock()
    artifacts.load_hub_chains.return_value = HubChains("nairobi", None, None)
    service._artifacts_repository = artifacts
    shortlist = MagicMock(return_value=[])
    with (
        patch.object(svc, "MATCH_V4_DISABLE_OCCUPATIONS", False),
        patch.object(svc, "MATCH_V4_DISABLE_LOCATION", disable_location),
        patch.object(location_query, "MATCH_V4_DISABLE_LOCATION", disable_location),
        patch.object(location_query, "LOCATION_TIER_ENABLED", True),
        patch.object(svc, "V4_FULL_RANK_DEMOTE", False),
        patch.object(svc, "skill_gaps_for", return_value=[]),
        patch.object(service, "embed_users", return_value=np.zeros((len(users), 4))),
        patch.object(service, "shortlist_and_rerank", shortlist),
    ):
        rows = service.rank(users, jobs, occupations, retrieve_top_k=10, final_top_k=5)
    return rows, shortlist, artifacts


class TestHubChainsShortCircuit:
    def test_disabled_never_loads_hub_chains(self):
        user = make_user(user_id="u1", city="Nairobi", province="Nairobi")
        _, _, artifacts = _run([user], [], [], disable_location=True)

        artifacts.load_hub_chains.assert_not_called()

    def test_enabled_loads_hub_chains_when_location_tiers_are_on(self):
        user = make_user(user_id="u1", city="Nairobi", province="Nairobi")
        _, _, artifacts = _run([user], [], [], disable_location=False)

        artifacts.load_hub_chains.assert_called_once()


class TestJobShortlistLocationKeys:
    def test_disabled_passes_no_location_keys_to_the_job_shortlist(self):
        user = make_user(user_id="u1", city="Nairobi", province="Nairobi")
        job = make_job(uuid="j1", city="Kitui", province="Kitui")
        _, shortlist, _ = _run([user], [job], [], disable_location=True)

        job_call = next(
            c
            for c in shortlist.call_args_list
            if c.kwargs.get("corpus") != "occupations"
        )
        assert job_call.kwargs["location_keys"] is None

    def test_enabled_passes_tier_keys_to_the_job_shortlist(self):
        user = make_user(user_id="u1", city="Nairobi", province="Nairobi")
        job = make_job(uuid="j1", city="Kitui", province="Kitui")
        _, shortlist, _ = _run([user], [job], [], disable_location=False)

        job_call = next(
            c
            for c in shortlist.call_args_list
            if c.kwargs.get("corpus") != "occupations"
        )
        assert job_call.kwargs["location_keys"] is not None


class TestOccupationCountyFilter:
    def test_disabled_skips_the_occupation_county_filter(self):
        user = make_user(user_id="u1", city="Nairobi", province="Nairobi")
        occupations = [
            make_occupation(uuid="occ_Nairobi", originUuid="occ", county="Nairobi"),
            make_occupation(uuid="occ_Kitui", originUuid="occ", county="Kitui"),
        ]
        rows, _, _ = _run([user], [], occupations, disable_location=True)

        # shortlist_and_rerank is stubbed to return [] (no candidates), so there's nothing to
        # score either way here; this test only exercises that rank() doesn't raise while
        # computing loc_keys=None for the occupation call site (see test_matching_service_rank.py
        # for the end-to-end assertion that the county filter is actually bypassed).
        assert rows[0].user_id == "u1"


class TestAsyncMatchInMemoryJobPool:
    """MatchingService.match() with no db_retriever (in-memory job pool, retrieve_jobs_with_timing)
    — the other branch from test_matching_service_db_stage1.py, which only covers db_retriever."""

    @staticmethod
    def _service(jobs_repository: IJobsRepository) -> MatchingService:
        return MatchingService(
            jobs_repository=jobs_repository,
            artifacts_repository=MagicMock(),
            embedding_client=MagicMock(model_name="m", embedding_dim=4),
            cross_encoder_provider=MagicMock,
            retrieval_matcher_provider=MagicMock,
            gate_matcher_provider=MagicMock,
            whitener_provider=MagicMock,
            skill_scorer_provider=MagicMock,
            preference_scorer_provider=MagicMock,
            retriever=MagicMock(),
            rerank=MagicMock(),
        )

    def _match(self, *, disable_location: bool):
        jobs_repository = AsyncMock(spec=IJobsRepository)
        jobs_repository.find_jobs_with_timing.return_value = (
            [],
            jobs_load_timing(n_jobs=0),
        )
        service = self._service(jobs_repository)
        with (
            patch.object(svc, "MATCH_V4_DISABLE_OCCUPATIONS", True),
            patch.object(svc, "MATCH_V4_DISABLE_LOCATION", disable_location),
            patch.object(location_query, "MATCH_V4_DISABLE_LOCATION", disable_location),
            patch.object(location_query, "LOCATION_TIER_ENABLED", True),
            patch.object(svc, "V4_FULL_RANK_DEMOTE", False),
            patch.object(service, "embed_users", return_value=np.zeros((1, 4))),
            patch.object(service, "shortlist_and_rerank", return_value=[]),
        ):
            asyncio.run(
                service.match(
                    [
                        MatchRequest.model_validate(
                            {
                                "user_id": "u1",
                                "city": "Nairobi",
                                "province": "Nairobi",
                                "preference_vector": {"earnings_per_month": 0.9},
                            }
                        )
                    ],
                    MatchOptions(retrieve_top_k=10, final_top_k=10),
                )
            )
        return jobs_repository

    def test_disabled_loads_jobs_with_no_location_restriction(self):
        jobs_repository = self._match(disable_location=True)

        assert jobs_repository.find_jobs_with_timing.await_args.args == (None,)

    def test_enabled_restricts_jobs_to_the_users_location_keys(self):
        jobs_repository = self._match(disable_location=False)

        (query_filter,) = jobs_repository.find_jobs_with_timing.await_args.args
        assert query_filter is not None
