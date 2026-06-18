"""Integration tests: unified scorer (DCE+BWS) + legacy scorer escape hatch + schema validity."""

import math

import pytest

from app.schemas import MatchedPreference, WorkActivityBWS
from app.services.preference_score import PreferenceScorer
from app.services.preference_score_v1.scorer import UnifiedPreferenceScorer


def _user(**overrides):
    pv = {
        "earnings_per_month": 0.8,
        "career_growth": 0.75,
        "physical_demand": 0.8,        # v>0.5 => prefers light/safe (dislikes heavy/risky)
        "social_interaction": 0.5,
        "bws_scores": {"4.A.1": 2.0, "4.A.2": -2.0},
        "top_10_bws": ["4.A.1"],
    }
    pv.update(overrides)
    return {"preference_vector": pv}


def _job(earn="earn_70k", growth="growth_high", phys="phys_light", soc="soc_peers"):
    return {
        "attributes": {
            "earnings_per_month": earn,
            "career_growth": growth,
            "physical_demand": phys,
            "social_interaction": soc,
        },
        "onet_work_activities": [
            {"WA_code": "4.A.1", "WA_Importance": 3, "WA_Level": 5, "WA_label": "Plan work"},
            {"WA_code": "4.A.2", "WA_Importance": 1, "WA_Level": 5, "WA_label": "Lift loads"},
        ],
    }


# --- Unified scorer ---------------------------------------------------------

def test_unified_scorer_keys_and_bounds():
    res = UnifiedPreferenceScorer().calculate_score(_user(), _job())
    for k in ("u_hat", "score", "details", "S_attrs", "S_wa", "V", "V_task",
              "V_dce", "V_dce_hat", "confidence_f", "alpha", "gamma", "scoring_model"):
        assert k in res, f"missing key {k}"
    assert 0.0 <= res["u_hat"] <= 1.0
    assert -1.0 <= res["S_attrs"] <= 1.0   # harmonised DCE utility
    assert -1.0 <= res["S_wa"] <= 1.0
    assert res["scoring_model"] == "unified_dce_bws_v1"


def test_unified_dce_rows_validate_as_matched_preference():
    res = UnifiedPreferenceScorer().calculate_score(_user(), _job())
    dce_rows = [d for d in res["details"] if d.get("layer") == "dce_attributes"]
    assert dce_rows, "expected DCE attribute rows in details"
    for row in dce_rows:
        MatchedPreference(**row)  # must not raise
    # work_activity_bws still present and schema-valid
    wa = next(d for d in res["details"] if d.get("attribute") == "work_activity_bws")
    WorkActivityBWS(wa_score_sum=wa["wa_score_sum"], details=wa["wa_details"])


def test_unified_include_work_activities_false_zeros_task():
    res = UnifiedPreferenceScorer().calculate_score(_user(), _job(), include_work_activities=False)
    assert res["V_task"] == pytest.approx(0.0)
    assert 0.0 <= res["u_hat"] <= 1.0


def test_unified_good_job_outranks_bad_job():
    s = UnifiedPreferenceScorer()
    good = s.calculate_score(_user(), _job("earn_70k", "growth_high", "phys_light"))
    bad = s.calculate_score(_user(), _job("earn_15k", "growth_low", "phys_heavy"))
    assert good["u_hat"] > bad["u_hat"]
    assert good["V_dce"] > 0  # all preferred levels (high pay/growth, light work)
    assert bad["V_dce"] == pytest.approx(0.0)  # all-reference levels => 0 baseline (RUM dummy-coding)


@pytest.mark.parametrize("attr,target,reference,pref_high", [
    ("earnings_per_month", "earn_70k", "earn_15k", 0.85),
    ("career_growth", "growth_high", "growth_low", 0.85),
    ("physical_demand", "phys_light", "phys_heavy", 0.85),   # v>0.5 => prefers LIGHT/safe
    ("social_interaction", "soc_customers", "soc_alone", 0.85),
])
def test_directional_target_outranks_reference(attr, target, reference, pref_high):
    """A user preferring an attribute's target level ranks target jobs above reference jobs."""
    s = UnifiedPreferenceScorer()
    user = {"preference_vector": {attr: pref_high}}
    job_t = {"attributes": {attr: target}}
    job_r = {"attributes": {attr: reference}}
    u_t = s.calculate_score(user, job_t, include_work_activities=False)["u_hat"]
    u_r = s.calculate_score(user, job_r, include_work_activities=False)["u_hat"]
    assert u_t > u_r, f"{attr}: target {target} should outrank reference {reference}"


def test_confidence_input_shrinks_dce():
    s = UnifiedPreferenceScorer()
    base = s.calculate_score(_user(), _job(), include_work_activities=False)
    low_conf = s.calculate_score(
        _user(preference_confidence=0.2), _job(), include_work_activities=False
    )
    # Lower confidence pulls |V_dce_hat| toward 0.
    assert abs(low_conf["V_dce_hat"]) < abs(base["V_dce_hat"])


# --- Legacy scorer escape hatch (unchanged behavior) ------------------------

def _legacy_user():
    return {"preference_vector": {"bws_scores": {"4.A.1": 2.0, "4.A.2": -2.0}}}


def test_legacy_mode_reproduces_old_uhat(monkeypatch):
    monkeypatch.setattr("app.services.preference_score.BWS_INTEGRATION_MODE", "legacy")
    scorer = PreferenceScorer()
    res = scorer.calculate_score(_legacy_user(), _job())
    wa_sum = 2.0 * (3 / 5) * (5 / 7) + (-2.0) * (1 / 5) * (5 / 7)
    expected = 1.0 / (1.0 + math.exp(-(wa_sum * scorer._sigmoid_factor)))
    assert res["u_hat"] == pytest.approx(round(expected, 4), abs=1e-3)
    assert "V" not in res  # legacy mode emits no additive-RUM diagnostics


def test_legacy_scorer_additive_mode_keys():
    res = PreferenceScorer().calculate_score(_legacy_user(), _job())  # default additive_rum
    for k in ("u_hat", "score", "details", "V", "V_task", "alpha", "gamma"):
        assert k in res
    assert 0.0 <= res["u_hat"] <= 1.0
