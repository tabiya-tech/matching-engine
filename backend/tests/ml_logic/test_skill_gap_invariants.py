"""Skill-gap analysis invariants."""

from app.config import MATCH_RESPONSE_SKILL_MIN_SCORE
from app.ranking.skill_gaps import (
    analyze_skill_gaps,
)
from app.ranking.skill_gaps import (
    filter_skill_gap_recommendations as _filter_skill_gap_recommendations,
)
from app.ranking.skill_gaps import (
    skill_gap_candidate_pool_k as _skill_gap_candidate_pool_k,
)
from app.schemas import SkillGapRecommendation
from tests.domain_factories import make_job, make_skills, make_user


def _job_with_skills(uuid: str, essential, optional=None):
    return make_job(
        uuid=uuid,
        essential_skills=make_skills(*[(s, s) for s in essential]),
        optional_skills=make_skills(*[(s, s) for s in (optional or [])]),
    )


def _user(*labels: str):
    return make_user(
        skills_vector={
            "top_skills": [{"originUUID": lab, "preferredLabel": lab} for lab in labels]
        }
    )


def _gap(skill_id: str, proximity: float) -> SkillGapRecommendation:
    return SkillGapRecommendation(
        skill_id=skill_id,
        skill_label=skill_id,
        proximity_score=proximity,
        job_unlock_count=1,
        combined_score=proximity,
        reasoning="",
    )


class TestSkillGapInvariants:
    def test_empty_user_skills_returns_empty_list(self, mock_engine):
        engine = mock_engine
        user = _user()
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
        user = _user("skill-a")
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
        returned_ids = {row.skill_id for row in out}
        assert "skill-a" not in returned_ids

    def test_respects_top_k(self, mock_engine):
        engine = mock_engine
        user = _user("skill-a")
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
            _gap("s1", MATCH_RESPONSE_SKILL_MIN_SCORE),
            _gap("s2", MATCH_RESPONSE_SKILL_MIN_SCORE - 0.01),
        ]
        out = _filter_skill_gap_recommendations(gaps, top_k=5)
        assert len(out) == 1
        assert out[0].skill_id == "s1"

    def test_candidate_pool_wider_than_requested_top_k(self):
        assert _skill_gap_candidate_pool_k(3) >= 3
        assert _skill_gap_candidate_pool_k(3) > 3
