"""Occupation-only demand tilt in /match_v4's enrich_recommendations_with_preferences.

Demand must re-rank OCCUPATIONS (include_demand=True) but never OPPORTUNITIES (default False).
"""

import pytest

from app.artifacts.repository import get_artifacts_repository
from app.ranking.preference import UnifiedPreferenceScorer
from app.ranking.scoring import enrich_recommendations_with_preferences
from tests.domain_factories import (
    make_attributes,
    make_candidate,
    make_job,
    make_skills,
    make_user,
)


def _recs():
    # Two items: identical skills + DCE attributes, differing ONLY in expected_demand.
    def _item(uuid, demand):
        return make_job(
            uuid=uuid,
            essential_skills=make_skills(("s1", "sql")),
            attributes=make_attributes(
                earnings_per_month="earn_50k", expected_demand=demand
            ),
        )

    # Same cosine -> same p_hat; same attrs/skills -> same u_hat -> same pre-demand final.
    return [
        make_candidate(
            _item("HI", "Very High Expected Demand"), rank=1, cosine=0.80
        ),  # M=1.0
        make_candidate(
            _item("LO", "Low Expected Demand"), rank=2, cosine=0.80
        ),  # M=0.25
    ]


def _user():
    return make_user(user_id="u1", preference_vector={"earnings_per_month": 0.7})


def test_demand_tilt_reranks_occupations():
    scorer = UnifiedPreferenceScorer(get_artifacts_repository().load_attribute_schema())
    occ = enrich_recommendations_with_preferences(
        _user(),
        _recs(),
        preference_scorer=scorer,
        include_demand=True,
        demand_gamma=0.3,
    )
    by = {r.uuid: r for r in occ}
    assert occ[0].uuid == "HI"  # very-high demand ranks first
    assert by["HI"].final_score > by["LO"].final_score
    # HI: M=1.0 -> factor 1.0 (unchanged); LO: M=0.25 -> factor 0.25**0.3.
    assert by["LO"].final_score == pytest.approx(
        by["HI"].final_score * (0.25**0.3), abs=1e-3
    )
    assert by["HI"].breakdown.demand_score == 1.0


def test_opportunities_are_not_demand_tilted():
    scorer = UnifiedPreferenceScorer(get_artifacts_repository().load_attribute_schema())
    opp = enrich_recommendations_with_preferences(  # include_demand defaults False (opportunity path)
        _user(),
        _recs(),
        preference_scorer=scorer,
    )
    by = {r.uuid: r for r in opp}
    assert by["HI"].final_score == pytest.approx(by["LO"].final_score)  # demand ignored
