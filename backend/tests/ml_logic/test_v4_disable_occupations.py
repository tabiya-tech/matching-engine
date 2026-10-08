"""MATCH_V4_DISABLE_OCCUPATIONS kill-switch (see config.MATCH_V4_DISABLE_OCCUPATIONS).

The flag must do two things, not one: return an empty ``occupation_recommendations`` list AND skip
every piece of occupation work (corpus load, stage-1 retrieval + CE rerank in the engine).
Opportunities and skill gaps must be untouched.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.matching import service as svc
from app.matching.service import MatchingService


def _service(*, engine=None, occupations_repository=None) -> MatchingService:
    """MatchingService with the ML stack and the repositories faked."""
    return MatchingService(
        jobs_repository=MagicMock(),
        occupations_repository=occupations_repository or MagicMock(),
        artifacts_repository=MagicMock(),
        engine=engine or MagicMock(),
        gate_matcher_provider=MagicMock,
        whitener_provider=MagicMock,
        skill_scorer_provider=MagicMock,
        preference_scorer_provider=MagicMock,
        embedding_dim=4,
    )


@pytest.fixture()
def v4_user():
    return {
        "user_id": "u1",
        "city": "Nairobi",
        "province": "Nairobi",
        "skills_vector": {"top_skills": []},
        "preference_vector": {},
    }


@pytest.fixture()
def occupation_rows():
    return [
        {"uuid": "occ-1", "originUuid": "code-1", "province": "Nairobi"},
        {"uuid": "occ-2", "originUuid": "code-2", "province": "Kitui"},
    ]


def _run(users, jobs, occupations, *, disabled: bool):
    """MatchingService.rank with the ML stack stubbed; returns (rows, retrieval_mock)."""
    engine = MagicMock()
    engine.embed_users.return_value = np.zeros((len(users), 4))
    engine.run.return_value = []
    with (
        patch.object(svc, "MATCH_V4_DISABLE_OCCUPATIONS", disabled),
        patch.object(svc, "V4_FULL_RANK_DEMOTE", False),
        patch.object(svc, "skill_gaps_for", return_value=[]),
    ):
        rows = _service(engine=engine).rank(
            users, jobs, occupations, retrieve_top_k=10, final_top_k=5
        )
    return rows, engine.run


class TestEngineKillSwitch:
    def test_disabled_returns_no_occupations(self, v4_user, occupation_rows):
        rows, _ = _run([v4_user], [], occupation_rows, disabled=True)
        assert rows[0]["occupation_recommendations"] == []

    def test_disabled_skips_the_occupation_retrieval_pass(
        self, v4_user, occupation_rows
    ):
        _, retrieval = _run([v4_user], [], occupation_rows, disabled=True)
        # Jobs only — the occupation corpus never reaches stage-1 retrieval / the cross-encoder.
        assert retrieval.call_count == 1

    def test_enabled_still_runs_the_occupation_retrieval_pass(
        self, v4_user, occupation_rows
    ):
        _, retrieval = _run([v4_user], [], occupation_rows, disabled=False)
        assert retrieval.call_count == 2

    def test_disabled_leaves_opportunities_and_skill_gaps_shaped(
        self, v4_user, occupation_rows
    ):
        rows, _ = _run([v4_user], [], occupation_rows, disabled=True)
        assert rows[0]["user_id"] == "u1"
        assert rows[0]["opportunity_recommendations"] == []
        assert rows[0]["skill_gap_recommendations"] == []


class TestServiceCorpusLoad:
    def test_disabled_skips_the_corpus_load(self):
        repository = MagicMock()
        repository.load_with_timing = AsyncMock()
        with patch.object(svc, "MATCH_V4_DISABLE_OCCUPATIONS", True):
            occ, timing = asyncio.run(
                _service(occupations_repository=repository)._load_occupations()
            )
        assert occ == []
        assert timing == {}
        repository.load_with_timing.assert_not_called()
        repository.attach_embeddings.assert_not_called()

    def test_enabled_loads_and_embeds_the_corpus(self):
        repository = MagicMock()
        repository.load_with_timing = AsyncMock(
            return_value=([{"uuid": "occ-1"}], {"occupation_cache_hit": True})
        )
        repository.attach_embeddings.side_effect = lambda rows: rows
        with patch.object(svc, "MATCH_V4_DISABLE_OCCUPATIONS", False):
            occ, timing = asyncio.run(
                _service(occupations_repository=repository)._load_occupations()
            )
        assert [o["uuid"] for o in occ] == ["occ-1"]
        assert timing["occupation_cache_hit"] is True
        repository.attach_embeddings.assert_called_once()
