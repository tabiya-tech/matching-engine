"""Data-validation tests for request Pydantic models.

These import schemas directly -- no app startup, no mocking.
"""

from app.schemas import MatchRequest, MatchRequestV5


class TestCountySuffixStrip:
    """The _strip_county_suffix validator on city/province must normalise
    'Nairobi County' -> 'Nairobi' so location matching works against
    jobs/occupations stored with the bare county name.
    """

    def test_city_county_stripped(self):
        req = MatchRequest(city="Nairobi County", province="Gauteng")
        assert req.city == "Nairobi"

    def test_province_county_stripped(self):
        req = MatchRequest(city="Johannesburg", province="Nairobi County")
        assert req.province == "Nairobi"

    def test_case_insensitive(self):
        req = MatchRequest(city="mombasa COUNTY")
        assert req.city == "mombasa"

    def test_no_suffix_unchanged(self):
        req = MatchRequest(city="Johannesburg")
        assert req.city == "Johannesburg"


class TestMatchRequestV5Inheritance:
    """MatchRequestV5 extends MatchRequest with zqf_level. If inheritance
    breaks, the v5 endpoint silently loses every parent field.
    """

    def test_has_parent_fields(self):
        v5 = MatchRequestV5(
            user_id="u1",
            city="Lusaka",
            skills_vector={"top_skills": [{"originUUID": "abc"}]},
            zqf_level=5,
        )
        assert v5.user_id == "u1"
        assert v5.city == "Lusaka"
        assert v5.zqf_level == 5
        assert len(v5.skills_vector.top_skills) == 1

    def test_zqf_level_defaults_none(self):
        v5 = MatchRequestV5()
        assert v5.zqf_level is None
