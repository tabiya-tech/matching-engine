"""Tests for the v4 full-response formatters (ranked candidate -> recommendation models)."""

import pytest

from app.matching import formatting as fmt
from app.ranking import scoring as coverage
from app.ranking.types import (
    AttributePreference,
    FinalScoreBreakdown,
    PreferenceScore,
    ScoredCandidate,
    SkillDetail,
    SkillMatchDetail,
    WorkActivityContribution,
    WorkActivityUtility,
)
from app.schemas import MatchedSkill, MatchedSkills, MatchResponse
from tests.domain_factories import (
    make_attributes,
    make_candidate,
    make_job,
    make_occupation,
    make_skills,
    make_work_activities,
)


def _skill(jid, label, uid=None, ulabel=None, sim=0.0):
    return SkillMatchDetail(
        job_skill_id=jid,
        job_skill_label=label,
        best_user_skill_id=uid,
        best_user_skill_label=ulabel,
        cosine_similarity=sim,
    )


def _per_job_skill():
    return [
        _skill("E1", "welding", "U1", "metalwork", 0.82),
        _skill("E2", "blueprint reading", "U2", "drawings", 0.40),
        _skill("O1", "forklift", "U3", "driving", 0.71),
    ]


def _detail(essential_ids):
    return SkillDetail(
        per_job_skill=_per_job_skill(), essential_ids=frozenset(essential_ids)
    )


def _preference():
    return PreferenceScore(
        u_hat=0.8,
        score=0.42,
        attributes=[
            AttributePreference(
                attribute="earnings_per_month",
                attr_label="Monthly Earnings",
                job_value="earn_70k",
                job_value_label="~70k",
                user_weight=0.8,
                beta=1.39,
                encoded_value=1.0,
                contribution=1.39,
                matched=True,
            )
        ],
        work_activities=WorkActivityUtility(
            wa_details=[
                WorkActivityContribution(
                    wa_code="4.A.1",
                    wa_label="Plan",
                    user_bws=2.0,
                    wa_importance=3.0,
                    wa_level=5.0,
                    norm_importance=0.6,
                    norm_level=0.71,
                    wa_contribution=0.5,
                )
            ],
            wa_score_sum=0.5,
            wa_aggregation="importance_weighted",
            n_work_activities=2,
            V_task=0.5,
            V_task_hat=0.25,
        ),
        bws_score_type="work_activity_id",
        scoring_model="unified_dce_bws_v1",
    )


def _scored(item):
    return ScoredCandidate(
        candidate=make_candidate(item),
        rank=1,
        preference=_preference(),
        p_hat=0.6,
        p_hat_source="concat_cosine_whitened",
        final_score=0.48,
        breakdown=FinalScoreBreakdown(
            final_score_combiner="product",
            u_hat=0.8,
            p_hat=0.6,
            p_hat_source="concat_cosine_whitened",
            final_score=0.48,
            preference_score_legacy=0.42,
        ),
    )


def _job_item(**overrides):
    fields = dict(
        uuid="job-1",
        opportunity_title="Welder",
        employer="Acme",
        location="Nairobi",
        url="http://x/job-1",
        essential_skills=make_skills(("E1", "welding"), ("E2", "blueprint reading")),
        optional_skills=make_skills(("O1", "forklift")),
    )
    fields.update(overrides)
    return make_job(**fields)


def _opportunity(item, essential_ids=("E1", "E2")):
    return fmt.build_opportunity_row(
        item,
        _detail(essential_ids),
        rank=1,
        final_score=0.48,
        scored=_scored(item),
        sim_threshold=0.6,
        min_ess_share=0.0,
    )


def _occupation(item, essential_ids=("E1",)):
    return fmt.build_occupation_row(
        item,
        _detail(essential_ids),
        rank=1,
        final_score=0.48,
        scored=_scored(item),
        sim_threshold=0.6,
        min_ess_share=0.0,
    )


def _ms(*matches: MatchedSkill) -> MatchedSkills:
    return MatchedSkills(essential_skill_matches=list(matches))


def _met(user_label, *, sim=0.0, job_label="x", meets=True):
    return MatchedSkill(
        job_skill_id=job_label,
        job_skill_label=job_label,
        best_user_skill_label=user_label,
        similarity=sim,
        meets_threshold=meets,
    )


def _pref(attribute, attr_label, value_label, matched):
    return AttributePreference(
        attribute=attribute,
        attr_label=attr_label,
        job_value=None,
        job_value_label=value_label,
        user_weight=0.5,
        beta=0.0,
        encoded_value=0.0,
        contribution=0.0,
        matched=matched,
    )


def test_build_matched_skills_splits_and_thresholds():
    # essential_ids are the matcher-resolved ids of the item's essential skills.
    ms = coverage.build_matched_skills(
        _per_job_skill(), frozenset({"E1", "E2"}), sim_threshold=0.6
    )
    ess = {m.job_skill_id: m for m in ms.essential_skill_matches}
    assert set(ess) == {"E1", "E2"}
    assert ess["E1"].meets_threshold is True  # 0.82 >= 0.6
    assert ess["E2"].meets_threshold is False  # 0.40 < 0.6
    # O1 not in essential set -> optional; cosine 0.71 >= 0.6 keeps it.
    assert [(o.skill_id, o.skill_label) for o in ms.optional_exact_matches] == [
        ("O1", "forklift")
    ]
    assert ms.skill_group_matches == []


def test_build_matched_skills_optional_below_threshold_dropped():
    pjs = [_skill("O9", "x", sim=0.3)]
    ms = coverage.build_matched_skills(pjs, frozenset(), sim_threshold=0.6)
    assert ms.essential_skill_matches == []
    assert ms.optional_exact_matches == []  # 0.3 < 0.6 dropped


def test_is_eligible_threshold():
    ess = [_met("a"), _met("b", meets=False)]
    assert (
        coverage.is_eligible_from_skills(ess, n_essential_total=2, min_ess_share=0.5)
        is True
    )  # 1/2 >= 0.5
    assert (
        coverage.is_eligible_from_skills(ess, n_essential_total=2, min_ess_share=0.75)
        is False
    )
    assert (
        coverage.is_eligible_from_skills([], n_essential_total=0, min_ess_share=0.9)
        is True
    )  # nothing to gate


def test_unparsed_ranking_coverage_uses_mean_not_free_pass():
    # Default (override < 0): an unparsed posting is treated as a TYPICAL one -> the mean of the
    # shortlist's parsed coverages, NOT the old 1.0 free pass. This is the core of the nail-tech fix.
    assert coverage.unparsed_ranking_coverage(
        [0.2, 0.4, 0.6], override=-1.0
    ) == pytest.approx(0.4)
    # Crucially below the best parsed coverage, so a well-covered job out-ranks an unparsed one under
    # the monotonic demotion (final *= coverage ** gamma).
    assert coverage.unparsed_ranking_coverage([0.2, 0.4, 0.6], override=-1.0) < 0.6


def test_unparsed_ranking_coverage_override_and_fallback():
    # A fixed override (>=0) wins; 1.0 restores the old demotion-free behaviour (rollback path).
    assert coverage.unparsed_ranking_coverage([0.2, 0.9], override=1.0) == 1.0
    assert coverage.unparsed_ranking_coverage(
        [0.2, 0.9], override=0.39
    ) == pytest.approx(0.39)
    # No parsed coverages in the shortlist -> neutral 0.5 fallback (not 1.0).
    assert coverage.unparsed_ranking_coverage([], override=-1.0) == 0.5


def test_preference_rows_map_to_response_models():
    prefs = fmt.matched_preferences(_preference())
    wa = fmt.matched_work_activities(_preference())
    assert [p.attribute for p in prefs] == ["earnings_per_month"]
    assert wa is not None and wa.wa_score_sum == 0.5 and wa.n_work_activities == 2
    assert (
        fmt.matched_preferences(None) == []
        and fmt.matched_work_activities(None) is None
    )


def test_opportunity_row_carries_scores():
    model = _opportunity(_job_item())
    assert model.final_score == pytest.approx(0.48)
    assert model.score_breakdown.u_hat == pytest.approx(0.8)
    assert model.score_breakdown.p_hat == pytest.approx(0.6)
    assert model.score_breakdown.p_hat_components is None  # Node2Vec-only -> null
    assert len(model.matched_skills.essential_skill_matches) == 2
    assert model.matched_preferences[0].attribute == "earnings_per_month"
    assert model.matched_work_activities.wa_score_sum == pytest.approx(0.5)
    assert model.is_eligible is True


def test_opportunity_row_without_ranker_scores_is_null_scored():
    item = _job_item()
    model = fmt.build_opportunity_row(
        item,
        _detail(("E1", "E2")),
        rank=1,
        final_score=0.3,
        scored=None,
        sim_threshold=0.6,
        min_ess_share=0.0,
    )
    assert model.final_score == pytest.approx(0.3)
    assert model.score_breakdown.u_hat is None and model.score_breakdown.p_hat is None
    assert model.matched_preferences == [] and model.matched_work_activities is None


def test_opportunity_row_passthrough_fields():
    item = _job_item(
        originUuid="fp-abc",
        posted_date="2026-05-20",
        related_occupation_id="2511",
        opportunity_isco_occupation_group_id="2511",
        attributes=make_attributes(expected_demand="High Expected Demand"),
    )
    model = _opportunity(item)
    assert model.originUuid == "fp-abc"
    assert model.posted_date == "2026-05-20"
    assert model.related_occupation_id == "2511"
    assert model.score_breakdown.demand_score == 0.75  # High Expected Demand
    # Approximated skill components present and in range.
    assert model.score_breakdown.skill_components.ess == pytest.approx(
        (0.82 + 0.40) / 2
    )


def test_occupation_row_carries_scores():
    item = make_occupation(
        uuid="1234_Nairobi",
        originUuid="1234",
        occupation_label="Welder",
        county="Nairobi",
        description="Joins metal.",
        essential_skills=make_skills(("E1", "welding")),
    )
    model = _occupation(item)
    assert model.occupation_label == "Welder"
    assert model.originUuid == "1234"
    assert model.province == "Nairobi"
    sb = model.score_breakdown
    # skill_components is an interpretable [0,1] approximation (not null).
    assert sb.skill_components is not None
    assert sb.skill_components.ess == pytest.approx(0.82)  # single essential E1
    assert sb.skill_components.opt == pytest.approx(
        (0.40 + 0.71) / 2
    )  # E2, O1 outside essential set
    assert sb.skill_components.loc is None and sb.skill_components.grp is None
    assert sb.total_skill_utility == pytest.approx(0.82)
    assert sb.skill_penalty_applied == pytest.approx(0.0)  # 0.82 >= 0.6, no gap
    # No attributes on this item -> demand + salary stay null; career path always empty.
    assert sb.demand_score is None and sb.demand_label is None
    assert model.salary_range is None
    assert model.career_path_next_steps == []


def test_occupation_row_demand_salary_and_tasks():
    item = make_occupation(
        uuid="55_Nairobi",
        originUuid="55",
        occupation_label="Data Analyst",
        county="Nairobi",
        description="Analyzes data.",
        attributes=make_attributes(
            earnings_per_month="earn_70k",
            expected_demand="Extremely High Expected Demand",
        ),
        onet_work_activities=make_work_activities(
            ("4.A.2.b.1", 4.6, 5.0, "Making Decisions"),
            ("4.A.3.a.1", 3.2, 4.0, "Studying Data"),
        ),
        included_tasks="Tasks include -\r\n(a) build dashboards; (b) clean datasets; (c) report findings",
        essential_skills=make_skills(("E1", "sql")),
    )
    model = _occupation(item)
    assert model.salary_range == "~70k"  # earn_70k -> label
    assert model.typical_tasks[:1] == ["build dashboards"]  # included_tasks split
    assert model.score_breakdown.demand_score == 1.0  # previously-unmapped label
    assert model.score_breakdown.demand_label == "Extremely High Expected Demand"


def test_typical_tasks_falls_back_to_onet_labels():
    item = make_occupation(
        onet_work_activities=make_work_activities(
            ("a", 4.0, 1.0, "Getting Information"), ("b", 4.9, 1.0, "Making Decisions")
        ),
    )
    # No included_tasks -> top WA labels by importance, highest first.
    assert fmt._typical_tasks(item)[:2] == ["Making Decisions", "Getting Information"]


def test_skill_components_clamped_to_unit_interval():
    pjs = [
        _skill("E1", None, sim=1.4),  # out-of-range high
        _skill("E2", None, sim=-0.3),  # negative
        _skill("O1", None, sim=0.5),
    ]
    sc, total, penalty = fmt._skill_components_from_cosine(
        pjs, frozenset({"E1", "E2"}), sim_threshold=0.6
    )
    assert 0.0 <= sc.ess <= 1.0 and 0.0 <= sc.opt <= 1.0
    assert sc.ess == pytest.approx((1.0 + 0.0) / 2)  # clamped to [0,1] before averaging
    assert total == sc.ess
    assert 0.0 <= penalty <= 1.0
    assert sc.loc is None and sc.grp is None


def test_justification_is_natural_second_person_and_enriched():
    ms = _ms(
        _met("hairdressing", job_label="cut hair"),
        _met("styling", job_label="style hair"),
        _met("customer care", job_label="customer service"),
        _met("scheduling", job_label="manage appointments"),
    )
    prefs = [
        _pref("earnings_per_month", "Monthly Earnings", "~70k", True),
        _pref("career_growth", "Career Growth", "Strong prospects", True),
        _pref("physical_demand", "Physical Demand", None, False),  # excluded
    ]
    item = make_job(
        attributes=make_attributes(expected_demand="Very High Expected Demand")
    )
    j = fmt._justification(ms, prefs, item)
    assert j.startswith("Strong match on your ")
    # ALL matched skills are listed (no cap, no "plus N more").
    for nm in ("hairdressing", "styling", "customer care", "scheduling"):
        assert nm in j
    assert "more" not in j.lower()
    assert "monthly earnings (~70k)" in j  # preference value included
    assert "Physical Demand" not in j  # unmatched preference excluded
    assert "in very high demand" in j  # demand humanized
    assert "match score" not in j.lower()  # no mechanical score line


def test_justification_caps_skills_at_top_ten_by_similarity():
    # 12 matched essential skills; only the top 10 by similarity should appear, no "more".
    ms = _ms(
        *[_met(f"skill{i:02d}", sim=i / 100.0) for i in range(12)]
    )  # skill00 (0.00) .. skill11 (0.11)
    j = fmt._justification(ms, [], make_job())
    assert "skill11" in j and "skill02" in j  # top by similarity present
    assert (
        "skill00" not in j and "skill01" not in j
    )  # two lowest dropped (beyond top 10)
    assert "more" not in j.lower()
    assert j.count(",") + 1 <= 10 or " and " in j  # at most 10 items listed


def test_justification_omits_absent_parts_with_fallback():
    ms = _ms(_met("SQL", job_label="sql"))
    assert fmt._justification(ms, [], make_job()) == "Strong match on your SQL skills."
    assert (
        fmt._justification(MatchedSkills(), [], make_job())
        == "Recommended role based on your overall profile."
    )


def test_full_match_response_assembles():
    resp = MatchResponse(
        user_id="u1",
        occupation_recommendations=[],
        opportunity_recommendations=[_opportunity(_job_item())],
        skill_gap_recommendations=[],
    )
    assert resp.opportunity_recommendations[0].uuid == "job-1"
