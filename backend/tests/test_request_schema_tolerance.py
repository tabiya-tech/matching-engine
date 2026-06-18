"""MatchRequest / PreferenceVector accept partial payloads (future consumers may send fewer fields)."""

import pytest

from app.schemas import MatchRequest, PreferenceVector


def test_empty_request_validates_with_neutral_defaults():
    req = MatchRequest()  # no fields at all
    assert req.city == "" and req.province == ""
    assert req.user_id is None
    # neutral preference vector (0.5 == sigmoid(0) == neutral in the DCE contract)
    assert req.preference_vector.earnings_per_month == 0.5
    assert req.preference_vector.physical_demand == 0.5
    assert req.preference_vector.social_interaction == 0.5
    assert req.preference_vector.career_growth == 0.5
    assert req.preference_vector.bws_scores is None


def test_partial_preference_vector_fills_rest_neutral():
    req = MatchRequest(preference_vector={"earnings_per_month": 0.9})
    assert req.preference_vector.earnings_per_month == 0.9
    # everything else defaults neutral
    assert req.preference_vector.career_growth == 0.5
    assert req.preference_vector.physical_demand == 0.5


def test_location_only_request_validates():
    req = MatchRequest(city="Nairobi", province="Nairobi")
    assert req.city == "Nairobi"
    assert req.preference_vector.social_interaction == 0.5  # neutral default


def test_full_payload_still_validates_unchanged():
    req = MatchRequest(
        user_id="u1",
        city="Nairobi",
        province="Nairobi",
        skills_vector={"top_skills": [{"originUUID": "x", "preferredLabel": "welding"}]},
        preference_vector={
            "earnings_per_month": 0.8, "physical_demand": 0.2, "social_interaction": 0.6,
            "career_growth": 0.75, "bws_scores": {"4.A.1": 2.0},
        },
        any_post_secondary_educ=1,
    )
    assert req.user_id == "u1"
    assert req.preference_vector.bws_scores == {"4.A.1": 2.0}
    assert req.any_post_secondary_educ == 1


def test_preference_vector_alone_partial():
    pv = PreferenceVector(career_growth=0.3)
    assert pv.career_growth == 0.3
    assert pv.earnings_per_month == 0.5


def test_county_suffix_stripped_from_location():
    # "Nairobi County" -> "Nairobi" so the Mongo substring prefilter matches "Nairobi" jobs.
    req = MatchRequest(city="Nairobi County", province="Nairobi County")
    assert req.city == "Nairobi"
    assert req.province == "Nairobi"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Nairobi County", "Nairobi"),
        ("nairobi county", "nairobi"),   # case-insensitive suffix, casing otherwise preserved
        ("Trans Nzoia County", "Trans Nzoia"),
        ("Nairobi", "Nairobi"),          # no suffix -> unchanged
        ("", ""),                         # empty stays empty (relaxes prefilter)
        ("Countyside", "Countyside"),    # only a trailing " county" token is stripped
    ],
)
def test_county_normalization_cases(raw, expected):
    assert MatchRequest(province=raw).province == expected
