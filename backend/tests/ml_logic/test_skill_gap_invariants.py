"""Skill-gap analysis invariants."""

from app.config import MATCH_RESPONSE_SKILL_MIN_SCORE
from app.services.matching_service import (
    _filter_skill_gap_recommendations,
    _skill_gap_candidate_pool_k,
)
from app.services.skill_gap_analysis import analyze_skill_gaps


def _job_with_skills(uuid: str, essential, optional=None):
    return {
        "uuid": uuid,
        "essential_skills": [{"id": s, "label": s} for s in essential],
        "optional_skills": [{"id": s, "label": s} for s in (optional or [])],
    }


class TestSkillGapInvariants:
    def test_empty_user_skills_returns_empty_list(self, mock_engine):
        engine = mock_engine
        user = {"skills_vector": {"top_skills": []}}
        jobs = [_job_with_skills("j1", ["skill-b"])]
        out = analyze_skill_gaps(
            user,
            jobs,
            engine,
            {"skill-b": "skill-b"},
            top_k=5,
            resolve_id=lambda x: x,
        )
        assert out == []

    def test_does_not_recommend_skills_user_already_has(self, mock_engine):
        engine = mock_engine
        user = {
            "skills_vector": {
                "top_skills": [{"preferredLabel": "skill-a"}],
            }
        }
        jobs = [
            _job_with_skills("j1", ["skill-a"], ["skill-b"]),
            _job_with_skills("j2", ["skill-c"]),
        ]
        out = analyze_skill_gaps(
            user,
            jobs,
            engine,
            {
                "skill-a": "skill-a",
                "skill-b": "skill-b",
                "skill-c": "skill-c",
            },
            top_k=10,
            resolve_id=lambda x: x,
        )
        returned_ids = {row["skill_id"] for row in out}
        assert "skill-a" not in returned_ids

    def test_respects_top_k(self, mock_engine):
        engine = mock_engine
        user = {
            "skills_vector": {
                "top_skills": [{"preferredLabel": "skill-a"}],
            }
        }
        jobs = [
            _job_with_skills("j1", [], [f"skill-{i}" for i in range(20)]),
        ]
        top_k = 3
        out = analyze_skill_gaps(
            user,
            jobs,
            engine,
            {f"skill-{i}": f"skill-{i}" for i in range(20)},
            top_k=top_k,
            resolve_id=lambda x: x,
        )
        filtered = _filter_skill_gap_recommendations(out, top_k=top_k)
        assert len(filtered) <= top_k

    def test_proximity_filter_drops_below_threshold(self):
        gaps = [
            {"skill_id": "s1", "proximity_score": MATCH_RESPONSE_SKILL_MIN_SCORE},
            {
                "skill_id": "s2",
                "proximity_score": MATCH_RESPONSE_SKILL_MIN_SCORE - 0.01,
            },
        ]
        out = _filter_skill_gap_recommendations(gaps, top_k=5)
        assert len(out) == 1
        assert out[0]["skill_id"] == "s1"

    def test_candidate_pool_wider_than_requested_top_k(self):
        assert _skill_gap_candidate_pool_k(3) >= 3
        assert _skill_gap_candidate_pool_k(3) > 3
