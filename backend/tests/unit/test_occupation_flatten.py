"""Tests for flattening the occupation file into per-(occupation, county) ``Occupation`` rows."""

import pytest
from pydantic import ValidationError

from app.domain.attributes import JobAttributes
from app.domain.skills import SkillReference
from app.domain.work_activities import WorkActivity
from app.occupations.flatten import flatten_occupations
from app.occupations.occupation_file import OCCUPATION_FILE


def _flatten(*entries: dict):
    return flatten_occupations(OCCUPATION_FILE.validate_python(list(entries)))


def _entry(**overrides) -> dict:
    entry = {
        "occupation": {
            "code": "2221",
            "preferred_label": "Nurse",
            "description": "Cares for patients",
        },
        "skills": {
            "essential": {"uuids": ["s1", "s2"], "labels": ["triage"]},
            "optional": {"uuids": ["s3"], "labels": ["driving"]},
        },
        "onet_work_activities": [
            {
                "WA_code": "4.A.1",
                "WA_label": "Getting info",
                "WA_Importance": "3.5",
                "WA_Level": "4",
            },
            {
                "WA_code": "4.A.2",
                "WA_label": "Missing level",
                "WA_Importance": "3",
                "WA_Level": "",
            },
        ],
        "counties_data": [
            {
                "county": "Nairobi",
                "job_attributes": {
                    "attributes": [
                        {
                            "attribute_name": "earnings_per_month",
                            "selected_level_id": "earn_70k",
                        }
                    ]
                },
                "labor_demand": {"expected_demand": "rising"},
            },
            {
                "county": "Mombasa",
                "job_attributes": {"attributes": {"earnings_per_month": "earn_15k"}},
            },
        ],
    }
    entry.update(overrides)
    return entry


def test_one_row_per_county():
    actual = _flatten(_entry())

    assert [r.uuid for r in actual] == ["2221_Nairobi", "2221_Mombasa"]
    assert all(r.originUuid == "2221" for r in actual)
    assert [r.county for r in actual] == ["Nairobi", "Mombasa"]


def test_entry_without_counties_yields_one_county_less_row():
    actual = _flatten(_entry(counties_data=[]))

    assert len(actual) == 1
    assert actual[0].uuid == "2221"
    assert actual[0].county == ""
    assert actual[0].attributes == JobAttributes()


def test_skills_are_id_label_pairs_with_empty_labels_when_missing():
    row = _flatten(_entry())[0]

    assert row.essential_skills == [
        SkillReference(id="s1", label="triage"),
        SkillReference(id="s2", label=""),
    ]
    assert row.optional_skills == [SkillReference(id="s3", label="driving")]


def test_attributes_list_and_dict_forms_and_expected_demand():
    nairobi, mombasa = _flatten(_entry())

    assert nairobi.attributes == JobAttributes(
        earnings_per_month="earn_70k", expected_demand="rising"
    )
    assert mombasa.attributes == JobAttributes(earnings_per_month="earn_15k")


def test_work_activities_without_importance_or_level_are_dropped():
    row = _flatten(_entry())[0]

    assert row.onet_work_activities == [
        WorkActivity(code="4.A.1", label="Getting info", importance=3.5, level=4.0)
    ]


def test_attribute_names_outside_the_schema_are_ignored():
    entry = _entry()
    entry["counties_data"][0]["job_attributes"]["attributes"] = [
        {"attribute_name": "Monthly Earnings", "selected_level_id": "earn_70k"}
    ]
    row = _flatten(entry)[0]

    assert row.attributes == JobAttributes(expected_demand="rising")


def test_rows_expose_the_county_under_every_location_name():
    row = _flatten(_entry())[0]

    assert (row.city, row.province, row.location) == ("Nairobi",) * 3
    assert row.title == "Nurse"
    assert row.skill_groups_origin_uuids == []
    assert row.embedding is None


def test_malformed_file_is_rejected():
    with pytest.raises(ValidationError):
        OCCUPATION_FILE.validate_python([{"counties_data": "Nairobi"}])
