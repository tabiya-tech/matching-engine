"""Unit + directional tests for the DCE-attribute utility (compute_dce_utility)."""

import math

import pytest

from app.services.preference_score_v1.work_activities import compute_dce_utility

# Inline schema: levels ordered reference(0) -> target(1).
SCHEMA = {
    "attributes": [
        {"name": "career_growth", "levels": [{"id": "growth_low"}, {"id": "growth_med"}, {"id": "growth_high"}]},
        {"name": "physical_demand", "levels": [{"id": "phys_light"}, {"id": "phys_heavy"}]},
    ]
}

LN3 = math.log(3.0)  # logit(0.75)


def _user(**vals):
    return {"preference_vector": dict(vals)}


def test_logit_recovers_signed_beta():
    # v=0.75 on a target-level job -> beta_hat = logit(0.75)=ln3, ladder=1 -> contribution=ln3
    v_dce, v_dce_hat, detail = compute_dce_utility(
        _user(career_growth=0.75), {"attributes": {"career_growth": "growth_high"}}, SCHEMA
    )
    assert v_dce == pytest.approx(LN3, abs=1e-3)
    row = detail["dce_details"][0]
    assert row["beta"] == pytest.approx(LN3, abs=1e-3)
    assert row["encoded_value"] == pytest.approx(1.0)
    # harmonised: single attr, V/D = +1 -> 1.0
    assert v_dce_hat == pytest.approx(1.0)


def test_neutral_value_zero_contribution():
    v_dce, v_dce_hat, _ = compute_dce_utility(
        _user(career_growth=0.5), {"attributes": {"career_growth": "growth_high"}}, SCHEMA
    )
    assert v_dce == pytest.approx(0.0)
    assert v_dce_hat == 0.0  # denom 0 -> 0


def test_reference_level_job_contributes_zero():
    # growth_low is the reference (ladder 0) -> 0 contribution even with a strong preference.
    v_dce, _, detail = compute_dce_utility(
        _user(career_growth=0.9), {"attributes": {"career_growth": "growth_low"}}, SCHEMA
    )
    assert v_dce == pytest.approx(0.0)
    assert detail["dce_details"][0]["encoded_value"] == pytest.approx(0.0)


def test_dislike_on_target_job_is_negative():
    # v<0.5 (dislikes the target level) on a target-level job -> negative contribution.
    v_dce, v_dce_hat, _ = compute_dce_utility(
        _user(physical_demand=0.2), {"attributes": {"physical_demand": "phys_heavy"}}, SCHEMA
    )
    assert v_dce < 0
    assert v_dce_hat == pytest.approx(-1.0)  # single attr, V/D = -1


def test_confidence_shrinks_toward_neutral():
    job = {"attributes": {"career_growth": "growth_high"}}
    full = compute_dce_utility(_user(career_growth=0.75), job, SCHEMA, confidence=1.0)[1]
    half = compute_dce_utility(_user(career_growth=0.75), job, SCHEMA, confidence=0.5)[1]
    assert full == pytest.approx(1.0)
    assert half == pytest.approx(0.5)


def test_harmoniser_bounds_unit_interval():
    # Two attrs both strongly positive on target jobs -> harmonised in [-1,1] (== 1 here).
    v_dce, v_dce_hat, _ = compute_dce_utility(
        _user(career_growth=0.99, physical_demand=0.99),
        {"attributes": {"career_growth": "growth_high", "physical_demand": "phys_heavy"}},
        SCHEMA,
    )
    assert -1.0 <= v_dce_hat <= 1.0
    assert v_dce_hat == pytest.approx(1.0)


def test_attr_scale_amplifies_raw_beta():
    base = compute_dce_utility(
        _user(career_growth=0.75), {"attributes": {"career_growth": "growth_high"}}, SCHEMA
    )[0]
    scaled = compute_dce_utility(
        _user(career_growth=0.75),
        {"attributes": {"career_growth": "growth_high"}},
        SCHEMA,
        attr_scale={"career_growth": 2.0},
    )[0]
    assert scaled == pytest.approx(2.0 * base)


def test_missing_job_level_skipped():
    v_dce, v_dce_hat, detail = compute_dce_utility(
        _user(career_growth=0.8), {"attributes": {}}, SCHEMA
    )
    assert v_dce == 0.0 and v_dce_hat == 0.0
    assert detail["dce_details"] == []


def test_rows_are_matched_preference_shaped():
    from app.schemas import MatchedPreference

    _, _, detail = compute_dce_utility(
        _user(career_growth=0.75), {"attributes": {"career_growth": "growth_high"}}, SCHEMA
    )
    # Should construct without error (extra diagnostic keys ignored).
    mp = MatchedPreference(**detail["dce_details"][0])
    assert mp.attribute == "career_growth"
    assert mp.beta == pytest.approx(LN3, abs=1e-3)
