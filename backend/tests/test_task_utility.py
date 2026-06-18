"""Unit tests for the additive-RUM BWS task-utility + combination (shared module)."""

import math

import pytest

from app.services.preference_score_v1.work_activities import (
    combine_utilities,
    compute_task_utility,
)


def _user(bws):
    return {"preference_vector": {"bws_scores": bws}}


def _job(activities):
    # activities: list of (code, importance, level)
    return {
        "onet_work_activities": [
            {"WA_code": c, "WA_Importance": imp, "WA_Level": lvl}
            for c, imp, lvl in activities
        ]
    }


def test_v_task_is_importance_weighted_average():
    # beta={A:+2, B:-2}, Importance={3,1} -> w={0.75,0.25} -> V_task = 0.75*2 + 0.25*(-2) = 1.0
    user = _user({"4.A.1": 2.0, "4.A.2": -2.0})
    job = _job([("4.A.1", 3, 5), ("4.A.2", 1, 5)])
    v_task, detail = compute_task_utility(user, job)
    assert v_task == pytest.approx(1.0)
    weights = [d["weight"] for d in detail["wa_details"]]
    assert weights == pytest.approx([0.75, 0.25])
    assert detail["wa_aggregation"] == "importance_weighted"
    assert detail["n_work_activities"] == 2
    # Level must not influence the score (only display).
    assert detail["wa_details"][0]["wa_contribution"] == pytest.approx(0.75 * 2.0)


def test_missing_activity_defaults_to_neutral_zero():
    # User has no BWS for the job's activities -> beta=0 -> V_task=0.
    user = _user({"9.9.9": 5.0})
    job = _job([("4.A.1", 3, 5), ("4.A.2", 1, 5)])
    v_task, detail = compute_task_utility(user, job)
    assert v_task == pytest.approx(0.0)
    assert all(d["beta"] == 0.0 for d in detail["wa_details"])


def test_zero_total_importance_returns_empty():
    user = _user({"4.A.1": 2.0})
    job = _job([("4.A.1", 0, 5)])
    v_task, detail = compute_task_utility(user, job)
    assert v_task == 0.0
    assert detail == {}


def test_non_work_activity_bws_returns_empty():
    # Occupation-id-style keys (2-digit) are not work-activity ids.
    user = _user({"01": 1.0, "02": -1.0})
    job = _job([("4.A.1", 3, 5)])
    v_task, detail = compute_task_utility(user, job)
    assert v_task == 0.0
    assert detail == {}


def test_no_activities_returns_empty():
    user = _user({"4.A.1": 2.0})
    v_task, detail = compute_task_utility(user, {"onet_work_activities": []})
    assert v_task == 0.0
    assert detail == {}


def test_combine_perfect_match_saturates_near_098():
    # gamma=4, both components harmonised to 1 -> V=4 -> logistic(4) ~ 0.982
    c = combine_utilities(2.0, 1.0, 1.0, alpha=0.5, gamma=4.0, v_dce_already_harmonized=True)
    assert c["v_task_h"] == pytest.approx(1.0)
    assert c["V"] == pytest.approx(4.0)
    assert c["u_hat"] == pytest.approx(1.0 / (1.0 + math.exp(-4.0)), abs=1e-3)


def test_harmonization_clamps_to_unit_range():
    # v_task beyond [-2,2] still clamps to [-1,1]
    assert combine_utilities(10.0, 0.0, 1.0, alpha=1.0, gamma=4.0)["v_task_h"] == 1.0
    assert combine_utilities(-10.0, 0.0, 1.0, alpha=1.0, gamma=4.0)["v_task_h"] == -1.0
    # raw (non-harmonized) v_dce normalized by dce_normalizer then clamped
    c = combine_utilities(0.0, 3.0, 1.5, alpha=0.0, gamma=4.0)  # 3.0/1.5 = 2 -> clamp 1
    assert c["v_dce_h"] == 1.0


def test_alpha_selects_component():
    # alpha=1 -> task only; alpha=0 -> dce only; alpha=0.5 -> average
    only_task = combine_utilities(2.0, -1.0, 1.0, alpha=1.0, gamma=4.0, v_dce_already_harmonized=True)
    only_dce = combine_utilities(2.0, -1.0, 1.0, alpha=0.0, gamma=4.0, v_dce_already_harmonized=True)
    avg = combine_utilities(2.0, -1.0, 1.0, alpha=0.5, gamma=4.0, v_dce_already_harmonized=True)
    assert only_task["V"] == pytest.approx(4.0 * 1.0)
    assert only_dce["V"] == pytest.approx(4.0 * -1.0)
    assert avg["V"] == pytest.approx(4.0 * 0.0)
    assert avg["u_hat"] == pytest.approx(0.5)


def test_negative_beta_lowers_uhat_below_half():
    # Disliked tasks (negative V_task) with neutral DCE pull u_hat below 0.5.
    c = combine_utilities(-2.0, 0.0, 1.0, alpha=0.5, gamma=4.0, v_dce_already_harmonized=True)
    assert c["u_hat"] < 0.5


def test_nonpositive_dce_normalizer_is_safe():
    c = combine_utilities(0.0, 5.0, 0.0, alpha=0.5, gamma=4.0)
    assert c["v_dce_h"] == 0.0
    assert c["u_hat"] == pytest.approx(0.5)
