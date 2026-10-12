"""Fixtures for component-level scorer tests."""

from __future__ import annotations

import pytest

from app.domain.job import Job
from app.domain.user import UserProfile
from app.server_dependencies.model_dependencies import build_skill_matcher
from tests.domain_factories import make_job, make_skills, make_user


@pytest.fixture(scope="module")
def cosine_matcher():
    """Load once per module — embedding matrix is large but cached."""
    return build_skill_matcher()


def _user_with_skills(labels) -> UserProfile:
    return make_user(
        user_id="meta-u1",
        skills_vector={
            "top_skills": [
                {
                    "originUUID": f"00000000-0000-4000-8000-{i:012x}",
                    "preferredLabel": lab,
                }
                for i, lab in enumerate(labels)
            ]
        },
    )


def _job_with_essential(labels) -> Job:
    return make_job(
        uuid="job-1",
        essential_skills=make_skills(
            *[(f"id-{i}", lab) for i, lab in enumerate(labels)]
        ),
    )
