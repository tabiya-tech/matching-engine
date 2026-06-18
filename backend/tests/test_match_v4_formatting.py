"""Tests for the v4 full-response formatters (enriched Gemini rec -> MatchResponse rows)."""

import pytest

from app.schemas import MatchResponse, OccupationRecommendation, OpportunityRecommendation
from app.services import match_v4_formatting as fmt


def _per_job_skill():
    return [
        {"job_skill_id": "E1", "job_skill_label": "welding", "best_user_skill_id": "U1",
         "best_user_skill_label": "metalwork", "cosine_similarity": 0.82},
        {"job_skill_id": "E2", "job_skill_label": "blueprint reading", "best_user_skill_id": "U2",
         "best_user_skill_label": "drawings", "cosine_similarity": 0.40},
        {"job_skill_id": "O1", "job_skill_label": "forklift", "best_user_skill_id": "U3",
         "best_user_skill_label": "driving", "cosine_similarity": 0.71},
    ]


def _enriched_rec():
    return {
        "job_uuid": "job-1",
        "u_hat": 0.8,
        "p_hat": 0.6,
        "final_score": 0.48,
        "score_breakdown": {"preference_score_legacy": 0.42},
        "preference_details": [
            {"attribute": "earnings_per_month", "attr_label": "Monthly Earnings", "job_value": "earn_70k",
             "job_value_label": "~70k", "user_weight": 0.8, "beta": 1.39, "encoded_value": 1.0,
             "contribution": 1.39, "matched": True, "layer": "dce_attributes"},
            {"attribute": "work_activity_bws", "wa_score_sum": 0.5, "wa_aggregation": "importance_weighted",
             "n_work_activities": 2, "V_task": 0.5, "V_task_hat": 0.25,
             "wa_details": [{"wa_code": "4.A.1", "wa_label": "Plan", "user_bws": 2.0, "wa_importance": 3.0,
                             "wa_level": 5.0, "norm_importance": 0.6, "norm_level": 0.71, "wa_contribution": 0.5}]},
        ],
    }


def _job_item():
    return {
        "uuid": "job-1", "opportunity_title": "Welder", "employer": "Acme", "location": "Nairobi",
        "url": "http://x/job-1",
        "essential_skills": [{"id": "E1", "label": "welding"}, {"id": "E2", "label": "blueprint reading"}],
        "optional_skills": [{"id": "O1", "label": "forklift"}],
    }


def test_build_matched_skills_splits_and_thresholds():
    # essential_ids are the matcher-resolved ids of the item's essential skills.
    ms = fmt.build_matched_skills(_per_job_skill(), {"E1", "E2"}, sim_threshold=0.6)
    ess = {m["job_skill_id"]: m for m in ms["essential_skill_matches"]}
    assert set(ess) == {"E1", "E2"}
    assert ess["E1"]["meets_threshold"] is True   # 0.82 >= 0.6
    assert ess["E2"]["meets_threshold"] is False  # 0.40 < 0.6
    # O1 not in essential set -> optional; cosine 0.71 >= 0.6 keeps it.
    assert ms["optional_exact_matches"] == [{"skill_id": "O1", "skill_label": "forklift"}]
    assert ms["skill_group_matches"] == []


def test_build_matched_skills_optional_below_threshold_dropped():
    pjs = [{"job_skill_id": "O9", "job_skill_label": "x", "cosine_similarity": 0.3}]
    ms = fmt.build_matched_skills(pjs, set(), sim_threshold=0.6)
    assert ms["essential_skill_matches"] == []
    assert ms["optional_exact_matches"] == []  # 0.3 < 0.6 dropped


def test_is_eligible_threshold():
    ess = [{"meets_threshold": True}, {"meets_threshold": False}]
    assert fmt.is_eligible_from_skills(ess, n_essential_total=2, min_ess_share=0.5) is True   # 1/2 >= 0.5
    assert fmt.is_eligible_from_skills(ess, n_essential_total=2, min_ess_share=0.75) is False
    assert fmt.is_eligible_from_skills([], n_essential_total=0, min_ess_share=0.9) is True    # nothing to gate


def test_unparsed_ranking_coverage_uses_mean_not_free_pass():
    # Default (override < 0): an unparsed posting is treated as a TYPICAL one -> the mean of the
    # shortlist's parsed coverages, NOT the old 1.0 free pass. This is the core of the nail-tech fix.
    assert fmt.unparsed_ranking_coverage([0.2, 0.4, 0.6], override=-1.0) == pytest.approx(0.4)
    # Crucially below the best parsed coverage, so a well-covered job out-ranks an unparsed one under
    # the monotonic demotion (final *= coverage ** gamma).
    assert fmt.unparsed_ranking_coverage([0.2, 0.4, 0.6], override=-1.0) < 0.6


def test_unparsed_ranking_coverage_override_and_fallback():
    # A fixed override (>=0) wins; 1.0 restores the old demotion-free behaviour (rollback path).
    assert fmt.unparsed_ranking_coverage([0.2, 0.9], override=1.0) == 1.0
    assert fmt.unparsed_ranking_coverage([0.2, 0.9], override=0.39) == pytest.approx(0.39)
    # No parsed coverages in the shortlist -> neutral 0.5 fallback (not 1.0).
    assert fmt.unparsed_ranking_coverage([], override=-1.0) == 0.5


def test_split_pref_details():
    prefs, wa = fmt.split_pref_details(_enriched_rec()["preference_details"])
    assert [p["attribute"] for p in prefs] == ["earnings_per_month"]
    assert wa["wa_score_sum"] == 0.5 and wa["n_work_activities"] == 2


def test_opportunity_row_validates_as_schema():
    row = fmt.build_opportunity_row(
        _enriched_rec(), _job_item(), _per_job_skill(), {"E1", "E2"},
        rank=1, sim_threshold=0.6, min_ess_share=0.0,
    )
    model = OpportunityRecommendation(**row)  # must not raise
    assert model.final_score == pytest.approx(0.48)
    assert model.score_breakdown.u_hat == pytest.approx(0.8)
    assert model.score_breakdown.p_hat == pytest.approx(0.6)
    assert model.score_breakdown.p_hat_components is None  # Node2Vec-only -> null
    assert len(model.matched_skills.essential_skill_matches) == 2
    assert model.matched_preferences[0].attribute == "earnings_per_month"
    assert model.matched_work_activities.wa_score_sum == pytest.approx(0.5)
    assert model.is_eligible is True


def test_opportunity_row_passthrough_fields():
    item = dict(_job_item())
    item.update({
        "originUuid": "fp-abc", "posted_date": "2026-05-20", "related_occupation_id": "2511",
        "opportunity_isco_occupation_group_id": "2511",
        "attributes": {"expected_demand": "High Expected Demand"},
    })
    row = fmt.build_opportunity_row(
        _enriched_rec(), item, _per_job_skill(), {"E1", "E2"}, rank=1, sim_threshold=0.6, min_ess_share=0.0,
    )
    model = OpportunityRecommendation(**row)
    assert model.originUuid == "fp-abc"
    assert model.posted_date == "2026-05-20"
    assert model.related_occupation_id == "2511"
    assert model.score_breakdown.demand_score == 0.75  # High Expected Demand
    # Approximated skill components present and in range.
    assert model.score_breakdown.skill_components.ess == pytest.approx((0.82 + 0.40) / 2)


def test_occupation_row_validates_as_schema():
    item = {
        "uuid": "1234_Nairobi", "originUuid": "1234", "occupation_label": "Welder",
        "province": "Nairobi", "description": "Joins metal.",
        "essential_skills": [{"id": "E1", "label": "welding"}],
        "optional_skills": [],
    }
    rec = _enriched_rec()
    rec["job_uuid"] = "1234_Nairobi"
    row = fmt.build_occupation_row(rec, item, _per_job_skill(), {"E1"}, rank=1, sim_threshold=0.6, min_ess_share=0.0)
    model = OccupationRecommendation(**row)
    assert model.occupation_label == "Welder"
    assert model.originUuid == "1234"
    sb = model.score_breakdown
    # Fix 5: skill_components is now an interpretable [0,1] approximation (not null).
    assert sb.skill_components is not None
    assert sb.skill_components.ess == pytest.approx(0.82)            # single essential E1
    assert sb.skill_components.opt == pytest.approx((0.40 + 0.71) / 2)  # E2, O1 outside essential set
    assert sb.skill_components.loc is None and sb.skill_components.grp is None
    assert sb.total_skill_utility == pytest.approx(0.82)
    assert sb.skill_penalty_applied == pytest.approx(0.0)           # 0.82 >= 0.6, no gap
    # No attributes on this item -> demand + salary stay null; career path always empty.
    assert sb.demand_score is None and sb.demand_label is None
    assert model.salary_range is None
    assert model.career_path_next_steps == []


def test_occupation_row_demand_salary_and_tasks():
    item = {
        "uuid": "55_Nairobi", "originUuid": "55", "occupation_label": "Data Analyst",
        "province": "Nairobi", "description": "Analyzes data.",
        "attributes": {"earnings_per_month": "earn_70k", "expected_demand": "Extremely High Expected Demand"},
        "onet_work_activities": [
            {"WA_code": "4.A.2.b.1", "WA_label": "Making Decisions", "WA_Importance": 4.6, "WA_Level": 5.0},
            {"WA_code": "4.A.3.a.1", "WA_label": "Studying Data", "WA_Importance": 3.2, "WA_Level": 4.0},
        ],
        "included_tasks": "Tasks include -\r\n(a) build dashboards; (b) clean datasets; (c) report findings",
        "essential_skills": [{"id": "E1", "label": "sql"}],
    }
    row = fmt.build_occupation_row(_enriched_rec(), item, _per_job_skill(), {"E1"}, rank=1, sim_threshold=0.6, min_ess_share=0.0)
    model = OccupationRecommendation(**row)
    assert model.salary_range == "~70k"                                   # earn_70k -> label
    assert model.typical_tasks[:1] == ["build dashboards"]                # included_tasks split
    assert model.score_breakdown.demand_score == 1.0                      # previously-unmapped label
    assert model.score_breakdown.demand_label == "Extremely High Expected Demand"


def test_typical_tasks_falls_back_to_onet_labels():
    item = {
        "uuid": "x", "occupation_label": "Manager", "province": "Nairobi",
        "onet_work_activities": [
            {"WA_label": "Getting Information", "WA_Importance": 4.0},
            {"WA_label": "Making Decisions", "WA_Importance": 4.9},
        ],
    }
    # No included_tasks -> top WA labels by importance, highest first.
    assert fmt._typical_tasks(item)[:2] == ["Making Decisions", "Getting Information"]


def test_skill_components_clamped_to_unit_interval():
    pjs = [
        {"job_skill_id": "E1", "cosine_similarity": 1.4},   # out-of-range high
        {"job_skill_id": "E2", "cosine_similarity": -0.3},  # negative
        {"job_skill_id": "O1", "cosine_similarity": 0.5},
    ]
    out = fmt._skill_components_from_cosine(pjs, {"E1", "E2"}, sim_threshold=0.6)
    sc = out["skill_components"]
    assert 0.0 <= sc["ess"] <= 1.0 and 0.0 <= sc["opt"] <= 1.0
    assert sc["ess"] == pytest.approx((1.0 + 0.0) / 2)   # clamped to [0,1] before averaging
    assert 0.0 <= out["skill_penalty_applied"] <= 1.0
    assert sc["loc"] is None and sc["grp"] is None


def test_justification_is_natural_second_person_and_enriched():
    ms = {
        "essential_skill_matches": [
            {"job_skill_label": "cut hair", "best_user_skill_label": "hairdressing", "meets_threshold": True},
            {"job_skill_label": "style hair", "best_user_skill_label": "styling", "meets_threshold": True},
            {"job_skill_label": "customer service", "best_user_skill_label": "customer care", "meets_threshold": True},
            {"job_skill_label": "manage appointments", "best_user_skill_label": "scheduling", "meets_threshold": True},
        ],
        "optional_exact_matches": [], "skill_group_matches": [],
    }
    prefs = [
        {"attribute": "earnings_per_month", "attr_label": "Monthly Earnings", "job_value_label": "~70k", "matched": True},
        {"attribute": "career_growth", "attr_label": "Career Growth", "job_value_label": "Strong prospects", "matched": True},
        {"attribute": "physical_demand", "attr_label": "Physical Demand", "matched": False},  # excluded
    ]
    item = {"attributes": {"expected_demand": "Very High Expected Demand"}}
    j = fmt._justification(ms, prefs, item)
    assert j.startswith("Strong match on your ")
    # ALL matched skills are listed (no cap, no "plus N more").
    for nm in ("hairdressing", "styling", "customer care", "scheduling"):
        assert nm in j
    assert "more" not in j.lower()
    assert "monthly earnings (~70k)" in j                     # preference value included
    assert "Physical Demand" not in j                          # unmatched preference excluded
    assert "in very high demand" in j                          # demand humanized
    assert "match score" not in j.lower()                      # no mechanical score line


def test_justification_caps_skills_at_top_ten_by_similarity():
    # 12 matched essential skills; only the top 10 by similarity should appear, no "more".
    ess = [
        {"best_user_skill_label": f"skill{i:02d}", "meets_threshold": True, "similarity": i / 100.0}
        for i in range(12)  # skill00 (0.00) .. skill11 (0.11)
    ]
    ms = {"essential_skill_matches": ess, "optional_exact_matches": [], "skill_group_matches": []}
    j = fmt._justification(ms, [], {"attributes": {}})
    assert "skill11" in j and "skill02" in j        # top by similarity present
    assert "skill00" not in j and "skill01" not in j  # two lowest dropped (beyond top 10)
    assert "more" not in j.lower()
    assert j.count(",") + 1 <= 10 or " and " in j   # at most 10 items listed


def test_justification_omits_absent_parts_with_fallback():
    ms = {"essential_skill_matches": [{"job_skill_label": "sql", "best_user_skill_label": "SQL", "meets_threshold": True}],
          "optional_exact_matches": [], "skill_group_matches": []}
    assert fmt._justification(ms, [], {"attributes": {}}) == "Strong match on your SQL skills."
    assert (
        fmt._justification({"essential_skill_matches": []}, [], {"attributes": {}})
        == "Recommended role based on your overall profile."
    )


def test_full_match_response_assembles():
    opp = fmt.build_opportunity_row(_enriched_rec(), _job_item(), _per_job_skill(), {"E1", "E2"}, rank=1, sim_threshold=0.6, min_ess_share=0.0)
    resp = MatchResponse(
        user_id="u1",
        occupation_recommendations=[],
        opportunity_recommendations=[opp],
        skill_gap_recommendations=[],
    )
    assert resp.opportunity_recommendations[0].uuid == "job-1"
