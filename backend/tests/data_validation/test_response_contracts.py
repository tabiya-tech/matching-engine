"""Data-validation tests for response Pydantic models.

These import schemas directly -- no app startup, no mocking.
Guards the full MatchResponseV5 contract shown in Swagger so any
field rename, removal, or type change is caught immediately.
"""

import pytest
from pydantic import ValidationError

from app.schemas import (
    MatchResponse,
    MatchResponseV5,
    OpportunityRecommendation,
    OpportunityRecommendationV5,
    OccupationRecommendation,
    SkillGapRecommendation,
    ScoreBreakdown,
    MatchedSkills,
    MatchedSkill,
    MatchedPreference,
    SkillComponents,
    PHatComponents,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _minimal_opportunity(**overrides):
    defaults = dict(
        uuid="j1",
        rank=1,
        opportunity_title="Test Job",
        is_eligible=True,
        justification="good fit",
        final_score=0.85,
        score_breakdown=ScoreBreakdown(),
        matched_skills=MatchedSkills(),
    )
    defaults.update(overrides)
    return defaults


def _minimal_occupation(**overrides):
    defaults = dict(
        uuid="occ1",
        rank=1,
        occupation_label="Baker",
        is_eligible=True,
        justification="relevant",
        final_score=0.75,
        score_breakdown=ScoreBreakdown(),
        matched_skills=MatchedSkills(),
    )
    defaults.update(overrides)
    return defaults


def _minimal_skill_gap(**overrides):
    defaults = dict(
        skill_id="s1",
        skill_label="Python",
        proximity_score=0.9,
        job_unlock_count=3,
        combined_score=0.85,
        reasoning="high demand",
    )
    defaults.update(overrides)
    return defaults


# ── MatchResponse top-level ──────────────────────────────────────────────────


class TestMatchResponseContract:
    """user_id is required; the three recommendation lists default to []."""

    def test_user_id_required(self):
        with pytest.raises(ValidationError):
            MatchResponse()

    def test_minimal_valid(self):
        resp = MatchResponse(user_id="u1")
        assert resp.user_id == "u1"
        assert resp.opportunity_recommendations == []
        assert resp.occupation_recommendations == []
        assert resp.skill_gap_recommendations == []


# ── OpportunityRecommendation ────────────────────────────────────────────────


class TestOpportunityRecommendationContract:
    """All required fields from the Swagger schema must be present and typed."""

    def test_required_fields_reject_missing(self):
        for field in (
            "uuid",
            "rank",
            "opportunity_title",
            "is_eligible",
            "justification",
            "final_score",
            "score_breakdown",
            "matched_skills",
        ):
            kwargs = _minimal_opportunity()
            del kwargs[field]
            with pytest.raises(ValidationError, match=field):
                OpportunityRecommendation(**kwargs)

    def test_full_opportunity(self):
        opp = OpportunityRecommendation(
            **_minimal_opportunity(
                originUuid="orig1",
                URL="https://example.com/job/1",
                opportunity_isco_occupation_group="Bakers",
                opportunity_isco_occupation_group_id="7512",
                related_occupation_id="occ-123",
                location="Lusaka",
                employer="ACME Corp",
                employment_type="full-time",
                salary_text="K5000/month",
                required_education="Certificate",
                required_experience="1 year",
                closing_date="2026-07-01",
                posted_date="2026-06-01",
                opportunity_description="Baking role",
                contract_type="permanent",
            )
        )
        assert opp.uuid == "j1"
        assert opp.rank == 1
        assert opp.opportunity_title == "Test Job"
        assert opp.is_eligible is True
        assert opp.justification == "good fit"
        assert opp.final_score == 0.85
        assert opp.location == "Lusaka"
        assert opp.employer == "ACME Corp"

    def test_optional_fields_default_none(self):
        opp = OpportunityRecommendation(**_minimal_opportunity())
        for field in (
            "originUuid",
            "URL",
            "opportunity_isco_occupation_group",
            "opportunity_isco_occupation_group_id",
            "related_occupation_id",
            "location",
            "employer",
            "employment_type",
            "salary_text",
            "required_education",
            "required_experience",
            "closing_date",
            "posted_date",
            "opportunity_description",
            "contract_type",
            "matched_work_activities",
        ):
            assert getattr(opp, field) is None, f"{field} should default to None"
        assert opp.matched_preferences == []


# ── OccupationRecommendation ────────────────────────────────────────────────


class TestOccupationRecommendationContract:
    """Guards occupation recommendation required fields and defaults."""

    def test_required_fields_reject_missing(self):
        for field in (
            "uuid",
            "rank",
            "occupation_label",
            "is_eligible",
            "justification",
            "final_score",
            "score_breakdown",
            "matched_skills",
        ):
            kwargs = _minimal_occupation()
            del kwargs[field]
            with pytest.raises(ValidationError, match=field):
                OccupationRecommendation(**kwargs)

    def test_full_occupation(self):
        occ = OccupationRecommendation(
            **_minimal_occupation(
                originUuid="orig-occ1",
                province="Lusaka",
                occupation_description="Bakes bread",
                salary_range="K3000-K5000",
                typical_tasks=["mix dough", "operate oven"],
                career_path_next_steps=["Head Baker", "Pastry Chef"],
            )
        )
        assert occ.uuid == "occ1"
        assert occ.occupation_label == "Baker"
        assert occ.province == "Lusaka"
        assert occ.typical_tasks == ["mix dough", "operate oven"]
        assert occ.career_path_next_steps == ["Head Baker", "Pastry Chef"]

    def test_optional_fields_default(self):
        occ = OccupationRecommendation(**_minimal_occupation())
        assert occ.originUuid is None
        assert occ.province is None
        assert occ.occupation_description is None
        assert occ.salary_range is None
        assert occ.typical_tasks == []
        assert occ.career_path_next_steps == []
        assert occ.matched_preferences == []
        assert occ.matched_work_activities is None


# ── SkillGapRecommendation ───────────────────────────────────────────────────


class TestSkillGapRecommendationContract:
    """Every field on SkillGapRecommendation is required."""

    def test_required_fields_reject_missing(self):
        for field in (
            "skill_id",
            "skill_label",
            "proximity_score",
            "job_unlock_count",
            "combined_score",
            "reasoning",
        ):
            kwargs = _minimal_skill_gap()
            del kwargs[field]
            with pytest.raises(ValidationError, match=field):
                SkillGapRecommendation(**kwargs)

    def test_full_skill_gap(self):
        sg = SkillGapRecommendation(**_minimal_skill_gap())
        assert sg.skill_id == "s1"
        assert sg.skill_label == "Python"
        assert sg.proximity_score == 0.9
        assert sg.job_unlock_count == 3
        assert sg.combined_score == 0.85
        assert sg.reasoning == "high demand"


# ── ScoreBreakdown ───────────────────────────────────────────────────────────


class TestScoreBreakdownContract:
    """ScoreBreakdown fields must exist and default to None."""

    def test_all_fields_default_none(self):
        sb = ScoreBreakdown()
        assert sb.u_hat is None
        assert sb.p_hat is None
        assert sb.p_hat_components is None
        assert sb.total_skill_utility is None
        assert sb.skill_components is None
        assert sb.skill_diagnostics is None
        assert sb.skill_penalty_applied is None
        assert sb.preference_score is None
        assert sb.preference_score_legacy is None
        assert sb.demand_score is None
        assert sb.demand_label is None

    def test_populated_breakdown(self):
        sb = ScoreBreakdown(
            u_hat=0.72,
            p_hat=0.65,
            p_hat_components=PHatComponents(
                gate=0.8,
                essential_fit=0.7,
                recruiter_readiness=0.6,
                market_opportunity=0.5,
            ),
            skill_components=SkillComponents(loc=0.1, ess=0.5, opt=0.2, grp=0.1),
        )
        assert sb.u_hat == 0.72
        assert sb.p_hat == 0.65
        assert sb.p_hat_components.gate == 0.8
        assert sb.skill_components.ess == 0.5


# ── MatchedSkills ────────────────────────────────────────────────────────────


class TestMatchedSkillsContract:
    """MatchedSkills sub-lists must default to [] and accept items."""

    def test_defaults_empty(self):
        ms = MatchedSkills()
        assert ms.essential_skill_matches == []
        assert ms.optional_exact_matches == []
        assert ms.skill_group_matches == []

    def test_essential_skill_match_fields(self):
        m = MatchedSkill(
            job_skill_id="js1",
            job_skill_label="baking",
            best_user_skill_id="us1",
            best_user_skill_label="prepare bakery products",
            similarity=0.92,
            meets_threshold=True,
        )
        assert m.job_skill_id == "js1"
        assert m.similarity == 0.92
        assert m.meets_threshold is True

    def test_essential_skill_required_fields(self):
        with pytest.raises(ValidationError):
            MatchedSkill()


# ── MatchedPreference ────────────────────────────────────────────────────────


class TestMatchedPreferenceContract:
    """Guards the matched_preferences items inside recommendations."""

    def test_full_preference(self):
        mp = MatchedPreference(
            attribute="earnings_per_month",
            job_value="K5000",
            job_value_label="Above average",
            user_weight=0.77,
            beta=0.5,
            encoded_value=0.8,
            contribution=0.4,
            matched=True,
        )
        assert mp.attribute == "earnings_per_month"
        assert mp.user_weight == 0.77
        assert mp.matched is True

    def test_required_fields_reject_missing(self):
        for field in (
            "attribute",
            "user_weight",
            "beta",
            "encoded_value",
            "contribution",
            "matched",
        ):
            kwargs = dict(
                attribute="earnings",
                user_weight=0.5,
                beta=0.3,
                encoded_value=0.6,
                contribution=0.2,
                matched=True,
            )
            del kwargs[field]
            with pytest.raises(ValidationError, match=field):
                MatchedPreference(**kwargs)


# ── V5 extensions ────────────────────────────────────────────────────────────


class TestMatchResponseV5Contract:
    """MatchResponseV5 must mirror MatchResponse + ZQF-annotated opportunities."""

    def test_v5_top_level_structure(self):
        resp = MatchResponseV5(user_id="u1")
        assert resp.user_id == "u1"
        assert resp.opportunity_recommendations == []
        assert resp.occupation_recommendations == []
        assert resp.skill_gap_recommendations == []

    def test_v5_user_id_required(self):
        with pytest.raises(ValidationError):
            MatchResponseV5()

    def test_v5_opportunity_has_zqf_fields(self):
        opp = OpportunityRecommendationV5(
            **_minimal_opportunity(
                zqf_eligible=True,
                zqf_gap=2,
                zqf_min_label="Diploma / Technician",
                zqf_max_label="Bachelor's Degree",
            )
        )
        assert opp.zqf_eligible is True
        assert opp.zqf_gap == 2
        assert opp.zqf_min_label == "Diploma / Technician"
        assert opp.zqf_max_label == "Bachelor's Degree"

    def test_v5_zqf_fields_default_none(self):
        opp = OpportunityRecommendationV5(**_minimal_opportunity())
        assert opp.zqf_eligible is None
        assert opp.zqf_gap is None
        assert opp.zqf_min_label is None
        assert opp.zqf_max_label is None

    def test_v5_inherits_all_opportunity_fields(self):
        opp = OpportunityRecommendationV5(
            **_minimal_opportunity(location="Solwezi", employer="ACME")
        )
        assert opp.uuid == "j1"
        assert opp.rank == 1
        assert opp.opportunity_title == "Test Job"
        assert opp.location == "Solwezi"
        assert opp.employer == "ACME"
