"""Fixtures for component-level scorer tests."""

from __future__ import annotations

import pytest

from app.services.cosine_similarity.skill_score import CosineSkillMatcher


@pytest.fixture(scope="module")
def cosine_matcher():
    """Load once per module — embedding matrix is large but cached."""
    return CosineSkillMatcher()


def _user_with_skills(labels):
    return {
        "user_id": "meta-u1",
        "skills_vector": {
            "top_skills": [
                {
                    "originUUID": f"00000000-0000-4000-8000-{i:012x}",
                    "preferredLabel": lab,
                }
                for i, lab in enumerate(labels)
            ]
        },
    }


def _job_with_essential(labels):
    return {
        "uuid": "job-1",
        "essential_skills": [{"label": lab} for lab in labels],
        "optional_skills": [],
    }
