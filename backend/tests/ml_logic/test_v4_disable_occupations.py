"""MATCH_V4_DISABLE_OCCUPATIONS kill-switch (see config.MATCH_V4_DISABLE_OCCUPATIONS).

The flag must do two things, not one: return an empty ``occupation_recommendations`` list AND skip
every piece of occupation work (corpus load in the route, stage-1 retrieval + CE rerank in the
engine). Opportunities and skill gaps must be untouched.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app import routes
from app.services import match_v4_full_service as svc


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
    """run_match_v4_full with the ML stack stubbed; returns (rows, retrieval_mock)."""
    retrieval = MagicMock(return_value=[])
    with (
        patch.object(svc, "MATCH_V4_DISABLE_OCCUPATIONS", disabled),
        patch.object(svc, "V4_FULL_RANK_DEMOTE", False),
        patch.object(svc, "run_match_concat_gemini_ce", retrieval),
        patch.object(
            svc, "embed_user_unit_vectors", return_value=np.zeros((len(users), 4))
        ),
        patch.object(svc, "get_preference_scorer", return_value=MagicMock()),
        patch.object(svc, "_get_v4_matcher", return_value=MagicMock()),
        patch.object(svc, "_get_matcher", return_value=MagicMock()),
        patch.object(svc, "_skill_gaps_for", return_value=[]),
    ):
        rows = svc.run_match_v4_full(
            users, jobs, occupations, retrieve_top_k=10, final_top_k=5
        )
    return rows, retrieval


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


class TestRouteCorpusLoad:
    def test_disabled_skips_the_corpus_load(self):
        loader = MagicMock()
        attach = MagicMock()
        with (
            patch.object(routes, "MATCH_V4_DISABLE_OCCUPATIONS", True),
            patch.object(routes, "get_all_occupations_with_timing", loader),
            patch.object(routes, "attach_occupation_embeddings", attach),
        ):
            occ, timing = asyncio.run(routes._load_v4_occupations())
        assert occ == []
        assert timing == {}
        loader.assert_not_called()
        attach.assert_not_called()

    def test_enabled_loads_and_embeds_the_corpus(self):
        async def _loader():
            return [{"uuid": "occ-1"}], {"occupation_cache_hit": True}

        attach = MagicMock(side_effect=lambda rows: rows)
        with (
            patch.object(routes, "MATCH_V4_DISABLE_OCCUPATIONS", False),
            patch.object(routes, "get_all_occupations_with_timing", _loader),
            patch.object(routes, "attach_occupation_embeddings", attach),
        ):
            occ, timing = asyncio.run(routes._load_v4_occupations())
        assert [o["uuid"] for o in occ] == ["occ-1"]
        assert timing["occupation_cache_hit"] is True
        attach.assert_called_once()
