"""Integration tests: unified scorer (DCE+BWS) + legacy scorer escape hatch + schema validity."""

import math

import pytest

from app.artifacts.repository import get_artifacts_repository
from app.ranking.preference import PreferenceScorer, UnifiedPreferenceScorer
from app.schemas import MatchedPreference, WorkActivityBWS
from tests.domain_factories import (
    make_attributes,
    make_job,
    make_user,
    make_work_activities,
)


def _user(**overrides):
    pv = {
        "earnings_per_month": 0.8,
        "career_growth": 0.75,
        "physical_demand": 0.8,  # v>0.5 => prefers light/safe (dislikes heavy/risky)
        "social_interaction": 0.5,
        "bws_scores": {"4.A.1": 2.0, "4.A.2": -2.0},
        "top_10_bws": ["4.A.1"],
    }
    pv.update(overrides)
    return make_user(preference_vector=pv)


def _job(earn="earn_70k", growth="growth_high", phys="phys_light", soc="soc_peers"):
    return make_job(
        attributes=make_attributes(
            earnings_per_month=earn,
            career_growth=growth,
            physical_demand=phys,
            social_interaction=soc,
        ),
        onet_work_activities=make_work_activities(
            ("4.A.1", 3, 5, "Plan work"), ("4.A.2", 1, 5, "Lift loads")
        ),
    )


# --- Unified scorer ---------------------------------------------------------


def test_unified_scorer_keys_and_bounds():
    res = UnifiedPreferenceScorer(
        get_artifacts_repository().load_attribute_schema()
    ).calculate_score(_user(), _job())
    for k in (
        "S_attrs",
        "S_wa",
        "V",
        "V_dce",
        "V_dce_hat",
        "confidence_f",
        "alpha",
        "gamma",
    ):
        assert getattr(res, k) is not None, f"missing {k}"
    assert res.work_activities is not None and res.work_activities.V_task is not None
    assert 0.0 <= res.u_hat <= 1.0
    assert -1.0 <= res.S_attrs <= 1.0  # harmonised DCE utility
    assert -1.0 <= res.S_wa <= 1.0
    assert res.scoring_model == "unified_dce_bws_v1"


def test_unified_dce_rows_validate_as_matched_preference():
    res = UnifiedPreferenceScorer(
        get_artifacts_repository().load_attribute_schema()
    ).calculate_score(_user(), _job())
    assert res.attributes, "expected DCE attribute rows"
    for row in res.attributes:
        MatchedPreference.model_validate(row.model_dump())  # must not raise
    # the work-activity utility is present and schema-valid
    wa = res.work_activities
    assert wa is not None
    WorkActivityBWS(
        wa_score_sum=wa.wa_score_sum, details=[d.model_dump() for d in wa.wa_details]
    )


def test_unified_include_work_activities_false_zeros_task():
    res = UnifiedPreferenceScorer(
        get_artifacts_repository().load_attribute_schema()
    ).calculate_score(_user(), _job(), include_work_activities=False)
    assert res.work_activities is None
    assert res.S_wa == pytest.approx(0.0)
    assert 0.0 <= res.u_hat <= 1.0


def test_unified_good_job_outranks_bad_job():
    s = UnifiedPreferenceScorer(get_artifacts_repository().load_attribute_schema())
    good = s.calculate_score(_user(), _job("earn_70k", "growth_high", "phys_light"))
    bad = s.calculate_score(_user(), _job("earn_15k", "growth_low", "phys_heavy"))
    assert good.u_hat > bad.u_hat
    assert good.V_dce > 0  # all preferred levels (high pay/growth, light work)
    assert bad.V_dce == pytest.approx(
        0.0
    )  # all-reference levels => 0 baseline (RUM dummy-coding)


@pytest.mark.parametrize(
    "attr,target,reference,pref_high",
    [
        ("earnings_per_month", "earn_70k", "earn_15k", 0.85),
        ("career_growth", "growth_high", "growth_low", 0.85),
        (
            "physical_demand",
            "phys_light",
            "phys_heavy",
            0.85,
        ),  # v>0.5 => prefers LIGHT/safe
        ("social_interaction", "soc_customers", "soc_alone", 0.85),
    ],
)
def test_directional_target_outranks_reference(attr, target, reference, pref_high):
    """A user preferring an attribute's target level ranks target jobs above reference jobs."""
    s = UnifiedPreferenceScorer(get_artifacts_repository().load_attribute_schema())
    user = make_user(preference_vector={attr: pref_high})
    job_t = make_job(attributes=make_attributes(**{attr: target}))
    job_r = make_job(attributes=make_attributes(**{attr: reference}))
    u_t = s.calculate_score(user, job_t, include_work_activities=False).u_hat
    u_r = s.calculate_score(user, job_r, include_work_activities=False).u_hat
    assert u_t > u_r, f"{attr}: target {target} should outrank reference {reference}"


# --- Legacy scorer escape hatch (unchanged behavior) ------------------------


def _legacy_user():
    return make_user(preference_vector={"bws_scores": {"4.A.1": 2.0, "4.A.2": -2.0}})


def test_legacy_mode_reproduces_old_uhat(monkeypatch):
    monkeypatch.setattr("app.ranking.preference.BWS_INTEGRATION_MODE", "legacy")
    scorer = PreferenceScorer()
    # No attribute levels on the job, so only the BWS work-activity term moves u_hat (the user's
    # preference_vector defaults every attribute weight to 0.5).
    job = make_job(onet_work_activities=_job().onet_work_activities)
    res = scorer.calculate_score(_legacy_user(), job)
    wa_sum = 2.0 * (3 / 5) * (5 / 7) + (-2.0) * (1 / 5) * (5 / 7)
    expected = 1.0 / (1.0 + math.exp(-(wa_sum * scorer._sigmoid_factor)))
    assert res.u_hat == pytest.approx(round(expected, 4), abs=1e-3)
    assert res.V is None  # legacy mode emits no additive-RUM diagnostics


def test_legacy_scorer_additive_mode_keys():
    res = PreferenceScorer().calculate_score(
        _legacy_user(), _job()
    )  # default additive_rum
    for k in ("V", "alpha", "gamma"):
        assert getattr(res, k) is not None
    assert res.work_activities is not None and res.work_activities.V_task is not None
    assert 0.0 <= res.u_hat <= 1.0
