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
    """Batch matcher stub used by v2/v3/v4/v5 HTTP routes."""
    return [_mock_match_response(u) for u in users]


async def _mock_occupations(*_a, **_kw):
    return ([], {})


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
    is patched (no Mongo connection), and routes receive ``mocked_jobs_repository``.
    """
    # Import target modules first so patch() can resolve the attribute paths.
    import app.main  # noqa: F401
    import app.routes  # noqa: F401
    import app.services.matching_service  # noqa: F401
    from app.jobs.get_jobs_repository import get_jobs_repository
    from app.server_dependencies.db_dependencies import MatchingDBProvider

    MatchingDBProvider.clear_cache()
    patches = [
        patch(
            "app.server_dependencies.db_dependencies._get_jobs_db",
            return_value=MagicMock(),
        ),
        patch("app.main.warmup_on_startup", new_callable=AsyncMock),
        patch("app.main.get_cross_encoder_client", return_value=MagicMock()),
        patch(
            "app.routes.get_all_occupations_with_timing", side_effect=_mock_occupations
        ),
        patch("app.routes.attach_occupation_embeddings", side_effect=lambda x: x),
        patch("app.routes.match_user_with_data", side_effect=_mock_match_response),
        patch("app.routes.run_match_v2_full", side_effect=_mock_run_match_full),
        patch("app.routes.run_match_v3_full", side_effect=_mock_run_match_full),
        patch("app.routes.run_match_v4_full", side_effect=_mock_run_match_full),
    ]
    for p in patches:
        p.start()

    from fastapi.testclient import TestClient
    from app.main import app

    app.dependency_overrides[get_jobs_repository] = lambda: mocked_jobs_repository
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_jobs_repository, None)

    for p in patches:
        p.stop()
    MatchingDBProvider.clear_cache()
