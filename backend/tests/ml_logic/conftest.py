"""Shared fixtures for ML / matching logic tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def base_user():
    return {
        "user_id": "ml-u1",
        "city": "Lusaka",
        "province": "Lusaka",
        "skills_vector": {
            "top_skills": [
                {
                    "originUUID": "00000000-0000-4000-8000-000000000001",
                    "preferredLabel": "manage staff",
                }
            ]
        },
        "preference_vector": {"earnings_per_month": 0.5},
    }


@pytest.fixture()
def remote_job():
    return {
        "uuid": "job-remote",
        "city": "Remote",
        "province": "",
        "location": "Remote",
        "requires_post_secondary": False,
    }


@pytest.fixture()
def local_job():
    return {
        "uuid": "job-local",
        "city": "Lusaka",
        "province": "Lusaka",
        "location": "Lusaka",
        "requires_post_secondary": False,
    }


@pytest.fixture()
def foreign_job():
    return {
        "uuid": "job-foreign",
        "city": "Nairobi",
        "province": "Nairobi",
        "location": "Nairobi",
        "requires_post_secondary": False,
    }


@pytest.fixture()
def mock_engine():
    return MockSimilarityEngine()


class MockSimilarityEngine:
    """Tiny embedding engine for skill-gap tests (no model files)."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def _rows(self, skill_ids):
        if not skill_ids:
            return np.empty((0, self.dim), dtype=np.float32)
        rows = []
        for sid in skill_ids:
            seed = abs(hash(str(sid))) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            n = np.linalg.norm(v)
            rows.append(v / n if n > 0 else v)
        return np.stack(rows, axis=0)
