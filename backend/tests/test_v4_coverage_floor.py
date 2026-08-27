import pytest

from app.services.gemini_ce_preference_matching import scoring
from app.services.gemini_ce_preference_matching.scoring import (
    enrich_recommendations_with_preferences,
)
from app.services.preference_score_v1 import UnifiedPreferenceScorer


def _items():
    # Same DCE attributes -> same u_hat; both postings carry essential skills (so they are
    # "parsed" and get a real ranking coverage).
    attrs = {"earnings_per_month": "earn_50k"}
    return {
        "HI": {
            "uuid": "HI",
            "essential_skills": [{"id": "s1", "label": "sql"}],
            "attributes": attrs,
        },
        "LO": {
            "uuid": "LO",
            "essential_skills": [{"id": "s2", "label": "welding"}],
            "attributes": attrs,
        },
    }


def _recs():
    # Different stage-1 cosines -> different p_hat -> a real ordering to preserve.
    return [
        {"job_uuid": "HI", "rank": 1, "concat_cosine_similarity": 0.80},
        {"job_uuid": "LO", "rank": 2, "concat_cosine_similarity": 0.40},
    ]


def _user():
    return {"user_id": "u1", "preference_vector": {"earnings_per_month": 0.7}}


def _enrich(coverage, **kwargs):
    return enrich_recommendations_with_preferences(
        _user(),
        _recs(),
        _items(),
        preference_scorer=UnifiedPreferenceScorer(),
        coverage_by_uuid=coverage,
        coverage_gamma=1.0,
        **kwargs,
    )


class TestZeroCoverage:
    def test_final_score_stays_non_zero(self):
        out = _enrich({"HI": 0.0, "LO": 0.0})
        assert len(out) == 2
        for row in out:
            assert row["final_score"] > 0.0, row["score_breakdown"]
            assert row["u_hat"] > 0.0 and row["p_hat"] > 0.0

    def test_u_hat_x_p_hat_ordering_survives(self):
        # Same u_hat, higher p_hat -> HI must rank first. Un-floored, both scored 0.0 and the
        # order came from the p_hat tie-break rather than the score.
        out = _enrich({"HI": 0.0, "LO": 0.0})
        assert [r["job_uuid"] for r in out] == ["HI", "LO"]
        assert out[0]["final_score"] > out[1]["final_score"]

    def test_the_floor_is_the_only_surviving_factor(self):
        out = _enrich({"HI": 0.0, "LO": 0.0})
        row = next(r for r in out if r["job_uuid"] == "HI")
        floor = scoring.V4_FULL_COVERAGE_FLOOR
        assert row["score_breakdown"]["coverage_factor"] == pytest.approx(
            floor, abs=1e-4
        )
        assert row["final_score"] == pytest.approx(
            row["u_hat"] * row["p_hat"] * floor, abs=1e-3
        )

    def test_floor_zero_restores_the_un_floored_behaviour(self, monkeypatch):
        # V4_FULL_COVERAGE_FLOOR=0.0 is the documented rollback.
        monkeypatch.setattr(scoring, "V4_FULL_COVERAGE_FLOOR", 0.0)
        out = _enrich({"HI": 0.0, "LO": 0.0})
        assert all(r["final_score"] == 0.0 for r in out)


class TestCoverageStillRanks:
    def test_full_coverage_is_not_penalised(self):
        out = _enrich({"HI": 1.0, "LO": 1.0})
        row = next(r for r in out if r["job_uuid"] == "HI")
        assert row["score_breakdown"]["coverage_factor"] == pytest.approx(1.0)
        assert row["final_score"] == pytest.approx(
            row["u_hat"] * row["p_hat"], abs=1e-3
        )

    def test_low_coverage_is_still_demoted_below_full_coverage(self):
        # LO has the better coverage but the worse cosine; the demotion has to be strong enough
        # to keep achievability meaningful — a 0-coverage item stays well below a 1.0-coverage one.
        out = _enrich({"HI": 0.0, "LO": 1.0})
        by = {r["job_uuid"]: r for r in out}
        assert by["LO"]["final_score"] > by["HI"]["final_score"]
        assert out[0]["job_uuid"] == "LO"

    def test_factor_is_monotone_in_coverage(self):
        def factor(cov):
            out = _enrich({"HI": cov, "LO": cov})
            return next(r for r in out if r["job_uuid"] == "HI")["score_breakdown"][
                "coverage_factor"
            ]

        factors = [factor(c) for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert factors == sorted(factors)
        assert factors[0] > 0.0 and factors[-1] == pytest.approx(1.0)


class TestLocationTier:
    def test_off_chain_candidates_are_dropped_not_scored_zero(self):
        out = _enrich(
            {"HI": 1.0, "LO": 1.0}, location_tier_by_uuid={"HI": 0.0, "LO": 1.0}
        )
        assert [r["job_uuid"] for r in out] == ["LO"]

    def test_in_chain_tier_scales_the_score(self):
        out = _enrich(
            {"HI": 1.0, "LO": 1.0}, location_tier_by_uuid={"HI": 0.5, "LO": 1.0}
        )
        row = next(r for r in out if r["job_uuid"] == "HI")
        assert row["score_breakdown"]["location_tier_factor"] == pytest.approx(0.5)
        assert row["final_score"] == pytest.approx(
            row["u_hat"] * row["p_hat"] * 0.5, abs=1e-3
        )
