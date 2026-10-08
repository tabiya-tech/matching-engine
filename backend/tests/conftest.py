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
# Environment: set BEFORE any app module is imported so db_dependencies.py and
# config.py don't crash on missing MONGO_URL.
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

    def rank(self, users, *_args, **_kwargs):
        return _mock_run_match_full(users)


def _fake_jobs_repository() -> MagicMock:
    repository = MagicMock()
    repository.find_active = AsyncMock(return_value=([], {}))
    repository.browse_page = AsyncMock(return_value=([], None, None, {}))
    repository.stats = AsyncMock()
    return repository


def _fake_occupations_repository() -> MagicMock:
    repository = MagicMock()
    repository.load_with_timing = AsyncMock(return_value=([], {}))
    repository.attach_embeddings.side_effect = lambda rows: rows
    return repository


@pytest.fixture()
def jobs_repository():
    """The fake jobs repository behind ``test_client`` (set side effects on it per test)."""
    return _fake_jobs_repository()


@pytest.fixture()
def test_client(jobs_repository):
    """TestClient with mocked DB, Gemini, and model loading.

    Uses a context-manager so the FastAPI lifespan actually executes.
    """
    from fastapi.testclient import TestClient

    from app.jobs.get_jobs_service import get_jobs_service
    from app.jobs.service import JobsService
    from app.main import app
    from app.matching.get_matching_service import get_matching_service

    matching_service = _StubMatchingService(
        jobs_repository=jobs_repository,
        occupations_repository=_fake_occupations_repository(),
        artifacts_repository=MagicMock(),
        engine=MagicMock(),
        gate_matcher_provider=MagicMock,
        whitener_provider=MagicMock,
        skill_scorer_provider=MagicMock,
        preference_scorer_provider=MagicMock,
        embedding_dim=4,
    )
    app.dependency_overrides[get_matching_service] = lambda: matching_service
    app.dependency_overrides[get_jobs_service] = lambda: JobsService(
        jobs_repository=jobs_repository
    )

    patches = [
        patch(
            "app.server_dependencies.warmup.warmup_on_startup", new_callable=AsyncMock
        ),
        patch("app.main.get_cross_encoder_client", return_value=MagicMock()),
        patch("app.main.get_skill_scorer", return_value=MagicMock()),
    ]
    for p in patches:
        p.start()

    with TestClient(app) as client:
        yield client

    for p in patches:
        p.stop()
    app.dependency_overrides = {}
