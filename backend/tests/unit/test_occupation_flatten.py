"""Tests for flattening the occupation taxonomy JSON into per-(occupation, county) rows."""

from app.occupations.flatten import flatten_occupations


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
                        {"attribute_name": "pay", "selected_level_id": "high"}
                    ]
                },
                "labor_demand": {"expected_demand": "rising"},
            },
            {"county": "Mombasa", "job_attributes": {"attributes": {"pay": "low"}}},
        ],
    }
    entry.update(overrides)
    return entry


def test_one_row_per_county():
    actual = flatten_occupations([_entry()])

    assert [r["uuid"] for r in actual] == ["2221_Nairobi", "2221_Mombasa"]
    assert all(r["originUuid"] == "2221" for r in actual)
    assert [r["city"] for r in actual] == ["Nairobi", "Mombasa"]


def test_entry_without_counties_yields_one_county_less_row():
    actual = flatten_occupations([_entry(counties_data=[])])

    assert len(actual) == 1
    assert actual[0]["uuid"] == "2221"
    assert actual[0]["location"] == ""
    assert actual[0]["attributes"] == {}


def test_skills_are_id_label_pairs_with_empty_labels_when_missing():
    row = flatten_occupations([_entry()])[0]

    assert row["essential_skills"] == [
        {"id": "s1", "label": "triage"},
        {"id": "s2", "label": ""},
    ]
    assert row["optional_skills"] == [{"id": "s3", "label": "driving"}]


def test_attributes_list_and_dict_forms_and_expected_demand():
    nairobi, mombasa = flatten_occupations([_entry()])

    assert nairobi["attributes"] == {"pay": "high", "expected_demand": "rising"}
    assert mombasa["attributes"] == {"pay": "low"}


def test_work_activities_without_importance_or_level_are_dropped():
    row = flatten_occupations([_entry()])[0]

    assert row["onet_work_activities"] == [
        {
            "WA_code": "4.A.1",
            "WA_label": "Getting info",
            "WA_Importance": 3.5,
            "WA_Level": 4.0,
        }
    ]
