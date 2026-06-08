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
# Environment: set BEFORE any app module is imported so database.py and
# config.py don't crash on missing MONGO_URL.
# ---------------------------------------------------------------------------
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB_NAME", "test")


# ---------------------------------------------------------------------------
# Fixtures for data-validation tests (pure Pydantic, no app startup)
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_user():
    """Minimal valid MatchRequest dict."""
    return {
        "user_id": "test-u1",
        "city": "Johannesburg",
        "province": "Gauteng",
        "skills_vector": {
            "top_skills": [
                {
                    "originUUID": "00000000-0000-4000-8000-000000000001",
                    "preferredLabel": "customer service",
                    "proficiency": 0.8,
                }
            ]
        },
        "skill_groups_origin_uuids": [],
        "preference_vector": {
            "earnings_per_month": 0.5,
            "physical_demand": 0.5,
            "social_interaction": 0.5,
            "career_growth": 0.5,
        },
    }


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


async def _mock_jobs(*_a, **_kw):
    return ([], {})


async def _mock_occupations(*_a, **_kw):
    return ([], {})


@pytest.fixture()
def test_client():
    """TestClient with mocked DB, Gemini, and model loading.

    Uses a context-manager so the FastAPI lifespan actually executes.
    """
    # Import target modules first so patch() can resolve the attribute paths.
    import app.database  # noqa: F401
    import app.routes  # noqa: F401
    import app.services.match_concat_gemini_ce_service  # noqa: F401
    import app.services.matching_service  # noqa: F401

    patches = [
        patch("app.database.warmup_on_startup", new_callable=AsyncMock),
        patch(
            "app.services.match_concat_gemini_ce_service._get_reranker",
            return_value=MagicMock(),
        ),
        patch("app.routes.get_all_jobs_with_timing", side_effect=_mock_jobs),
        patch(
            "app.routes.get_all_occupations_with_timing", side_effect=_mock_occupations
        ),
        patch("app.routes.attach_occupation_embeddings", side_effect=lambda x: x),
        patch("app.routes.match_user_with_data", side_effect=_mock_match_response),
    ]
    for p in patches:
        p.start()

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        yield client

    for p in patches:
        p.stop()
