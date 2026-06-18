"""Occupation-only demand tilt in /match_v4's enrich_recommendations_with_preferences.

Demand must re-rank OCCUPATIONS (include_demand=True) but never OPPORTUNITIES (default False).
"""
import pytest

from app.services.preference_score_v1 import UnifiedPreferenceScorer
from app.services.gemini_ce_preference_matching.scoring import enrich_recommendations_with_preferences


def _items_and_recs():
    # Two items: identical skills + DCE attributes, differing ONLY in expected_demand.
    base_attrs = {"earnings_per_month": "earn_50k"}
    items = {
        "HI": {"uuid": "HI", "essential_skills": [{"id": "s1", "label": "sql"}],
               "attributes": {**base_attrs, "expected_demand": "Very High Expected Demand"}},  # M=1.0
        "LO": {"uuid": "LO", "essential_skills": [{"id": "s1", "label": "sql"}],
               "attributes": {**base_attrs, "expected_demand": "Low Expected Demand"}},          # M=0.25
    }
    # Same cosine -> same p_hat; same attrs/skills -> same u_hat -> same pre-demand final.
    recs = [
        {"job_uuid": "HI", "rank": 1, "concat_cosine_similarity": 0.80},
        {"job_uuid": "LO", "rank": 2, "concat_cosine_similarity": 0.80},
    ]
    return items, recs


def _user():
    return {"user_id": "u1", "preference_vector": {"earnings_per_month": 0.7}}


def test_demand_tilt_reranks_occupations():
    scorer = UnifiedPreferenceScorer()
    items, recs = _items_and_recs()
    occ = enrich_recommendations_with_preferences(
        _user(), recs, items, preference_scorer=scorer, include_demand=True, demand_gamma=0.3,
    )
    by = {r["job_uuid"]: r for r in occ}
    assert occ[0]["job_uuid"] == "HI"                       # very-high demand ranks first
    assert by["HI"]["final_score"] > by["LO"]["final_score"]
    # HI: M=1.0 -> factor 1.0 (unchanged); LO: M=0.25 -> factor 0.25**0.3.
    assert by["LO"]["final_score"] == pytest.approx(by["HI"]["final_score"] * (0.25 ** 0.3), abs=1e-3)
    assert by["HI"]["score_breakdown"]["demand_score"] == 1.0


def test_opportunities_are_not_demand_tilted():
    scorer = UnifiedPreferenceScorer()
    items, recs = _items_and_recs()
    opp = enrich_recommendations_with_preferences(  # include_demand defaults False (opportunity path)
        _user(), recs, items, preference_scorer=scorer,
    )
    by = {r["job_uuid"]: r for r in opp}
    assert by["HI"]["final_score"] == pytest.approx(by["LO"]["final_score"])  # demand ignored
