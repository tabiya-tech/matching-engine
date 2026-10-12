"""Shared fixtures for ML / matching logic tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.domain.job import Job
from app.domain.user import UserProfile
from tests.domain_factories import make_job, make_user


@pytest.fixture()
def base_user() -> UserProfile:
    return make_user(
        user_id="ml-u1",
        city="Lusaka",
        province="Lusaka",
        skills_vector={
            "top_skills": [
                {
                    "originUUID": "00000000-0000-4000-8000-000000000001",
                    "preferredLabel": "manage staff",
                }
            ]
        },
        preference_vector={"earnings_per_month": 0.5},
    )


def _job_at(uuid: str, city: str, province: str, location: str) -> Job:
    return make_job(
        uuid=uuid,
        city=city,
        province=province,
        location=location,
        requires_post_secondary=False,
    )


@pytest.fixture()
def remote_job() -> Job:
    return _job_at("job-remote", "Remote", "", "Remote")


@pytest.fixture()
def local_job() -> Job:
    return _job_at("job-local", "Lusaka", "Lusaka", "Lusaka")


@pytest.fixture()
def foreign_job() -> Job:
    return _job_at("job-foreign", "Nairobi", "Nairobi", "Nairobi")


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
