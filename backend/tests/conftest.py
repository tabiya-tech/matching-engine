"""Shared fixtures for the test suite.

Data-validation and schema tests import Pydantic models directly (no app startup).
Smoke tests need the FastAPI app running with mocked infrastructure.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure backend/ is on sys.path so `app.*` imports resolve.
_BACKEND = str(Path(__file__).resolve().parent.parent)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# ---------------------------------------------------------------------------
# Environment: set BEFORE any app module is imported so config.py and the DB
# provider's settings resolve. No test opens a real Mongo connection.
# ---------------------------------------------------------------------------
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB_NAME", "test")

from app.matching.service import MatchingService  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures for smoke tests (mocked FastAPI app)
# ---------------------------------------------------------------------------


def _mock_match_response(user, *_args, **_kwargs):
    """Return a minimal MatchResponse-shaped dict without any ML dependencies."""
    return {
        "user_id": user.get("user_id", "unknown"),
        "opportunity_recommendations": [],
        "occupation_recommendations": [],
        "skill_gap_recommendations": [],
    }


def _mock_run_match_full(users, *_args, **_kwargs):
    """Batch matcher stub used by the POST /match route."""
    return [_mock_match_response(u) for u in users]


class _StubMatchingService(MatchingService):
    """The real request handling (validation, retrieval, response model) over a stubbed engine."""

    async def _load_occupations(self):
        return [], {}

    def rank(self, users, *_args, **_kwargs):
        return _mock_run_match_full(users)


@pytest.fixture()
def mocked_jobs_repository():
    """Stand-in for the jobs repository injected into routes; set side effects per test."""
    from app.jobs.repository import IJobsRepository

    repository = AsyncMock(spec=IJobsRepository)
    repository.find_jobs_with_timing.side_effect = lambda *_a, **_kw: ([], {})
    return repository


@pytest.fixture()
def test_client(mocked_jobs_repository):
    """TestClient with mocked DB, Gemini, and model loading.

    Uses a context-manager so the FastAPI lifespan actually executes. The provider's client factory
    is patched (no Mongo connection); the services receive ``mocked_jobs_repository``, and the
    matching engine is stubbed.
    """
    from fastapi.testclient import TestClient

    from app.jobs.get_jobs_repository import get_jobs_repository
    from app.main import app
    from app.matching.get_matching_service import get_matching_service
    from app.server_dependencies.db_dependencies import MatchingDBProvider

    matching_service = _StubMatchingService(
        jobs_repository=mocked_jobs_repository,
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

    MatchingDBProvider.clear_cache()
    patches = [
        patch(
            "app.server_dependencies.db_dependencies._get_jobs_db",
            return_value=MagicMock(),
        ),
        patch("app.main.warmup_on_startup", new_callable=AsyncMock),
        patch("app.main.get_cross_encoder_client", return_value=MagicMock()),
        patch("app.main.get_skill_scorer", return_value=MagicMock()),
    ]
    for p in patches:
        p.start()

    app.dependency_overrides[get_jobs_repository] = lambda: mocked_jobs_repository
    app.dependency_overrides[get_matching_service] = lambda: matching_service
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_jobs_repository, None)
    app.dependency_overrides.pop(get_matching_service, None)

    for p in patches:
        p.stop()
    MatchingDBProvider.clear_cache()
