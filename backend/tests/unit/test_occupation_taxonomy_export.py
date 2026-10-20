"""Tests for building the occupation file and group documents from a taxonomy export (CSV fixture)."""

import csv
from pathlib import Path

import pytest

from app.occupations.ingest import parse_occupation_file
from app.occupations.taxonomy_export import (
    build_group_documents,
    build_occupation_entries,
    group_ancestry,
    normalise_occupation_code,
    occupation_group_ancestry,
    read_taxonomy_export,
)


def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture()
def export_dir(tmp_path: Path) -> Path:
    """Major group 0 > minor 011 > unit 0110 > occupation o1 > sub-occupation o2; unit 2512 > o3.

    Skills: o1 essential s1, optional s2; o2 essential s3 (and s1 again); o3 has no relations.
    """
    _write(
        tmp_path / "model_info.csv",
        ["UUIDHISTORY", "NAME", "LOCALE", "VERSION"],
        [["new-model\nold-model", "Tabiya", "AR-es", "2.0"]],
    )
    groups = ["ID", "ORIGINURI", "UUIDHISTORY", "CODE", "GROUPTYPE", "PREFERREDLABEL"]
    _write(
        tmp_path / "occupation_groups.csv",
        [*groups, "ALTLABELS", "DESCRIPTION"],
        [
            ["g0", "uri:0", "u-g0", "0", "iscogroup", "Fuerzas armadas", "", ""],
            ["g01", "uri:01", "u-g01", "01", "iscogroup", "Oficiales", "", ""],
            [
                "g0110",
                "uri:0110",
                "u-g0110-new\nu-g0110-old",
                "0110",
                "iscogroup",
                "Oficiales de las fuerzas armadas",
                "Oficial\nOficiala",
                "Dirigen unidades",
            ],
            [
                "g2512",
                "uri:2512",
                "u-g2512",
                "2512",
                "iscogroup",
                "Desarrolladores",
                "",
                "",
            ],
            ["gI3", "uri:I3", "u-gI3", "I3", "localgroup", "Grupo local", "", ""],
        ],
    )
    _write(
        tmp_path / "occupations.csv",
        ["ID", "CODE", "OCCUPATIONTYPE"],
        [
            ["o1", "0110.1", "escooccupation"],
            ["o2", "0110.1.1", "escooccupation"],
            ["o3", "2512.1", "escooccupation"],
            ["o4", "I3_0", "localoccupation"],
        ],
    )
    _write(
        tmp_path / "occupation_hierarchy.csv",
        ["PARENTOBJECTTYPE", "PARENTID", "CHILDID", "CHILDOBJECTTYPE"],
        [
            ["iscogroup", "g0", "g01", "iscogroup"],
            ["iscogroup", "g01", "g0110", "iscogroup"],
            ["iscogroup", "g0110", "o1", "escooccupation"],
            ["escooccupation", "o1", "o2", "escooccupation"],
            ["iscogroup", "g2512", "o3", "escooccupation"],
            ["localgroup", "gI3", "o4", "localoccupation"],
        ],
    )
    _write(
        tmp_path / "skills.csv",
        ["ID", "UUIDHISTORY", "PREFERREDLABEL"],
        [
            ["s1", "s1-new\ns1-old", "liderar"],
            ["s2", "s2-old", "nadar"],
            ["s3", "s3-old", "planificar"],
            ["s9", "s9-old", "programar"],
        ],
    )
    _write(
        tmp_path / "occupation_to_skill_relations.csv",
        ["OCCUPATIONTYPE", "OCCUPATIONID", "RELATIONTYPE", "SKILLID"],
        [
            ["escooccupation", "o1", "essential", "s1"],
            ["escooccupation", "o1", "optional", "s2"],
            ["escooccupation", "o2", "essential", "s3"],
            ["escooccupation", "o2", "essential", "s1"],
            ["localoccupation", "o4", "", "s9"],
        ],
    )
    return tmp_path


def _enrichment(**occupation) -> list:
    return parse_occupation_file(
        [
            {
                "occupation": {
                    "code": "110",
                    "preferred_label": "Armed forces officer",
                    "included_tasks": "(a) lead",
                    "requires_post_secondary": True,
                    **occupation,
                },
                "skills": {
                    "essential": {
                        "uuids": ["s1-old", "curated"],
                        "labels": ["lead", "c"],
                    },
                    "optional": {"uuids": ["s2-old"], "labels": ["swim"]},
                },
                "counties_data": [
                    {"county": "Nairobi", "labor_demand": {"expected_demand": "High"}}
                ],
                "onet_work_activities": [
                    {"WA_code": "4.A.1", "WA_Importance": "3", "WA_Level": "4"}
                ],
            }
        ]
    )


def _by_code(entries: list[dict]) -> dict[str, dict]:
    return {e["occupation"]["code"]: e for e in entries}


def test_reads_model_info_with_oldest_uuid(export_dir):
    export = read_taxonomy_export(export_dir)

    assert export.model_info is not None
    assert (export.model_info.name, export.model_info.locale) == ("Tabiya", "AR-es")
    assert export.model_info.uuid == "old-model"


def test_one_entry_per_unit_group_in_code_order(export_dir):
    entries, report = build_occupation_entries(read_taxonomy_export(export_dir))

    assert [e["occupation"]["code"] for e in entries] == ["0110", "2512"]
    assert report.n_unit_groups == 2
    assert report.n_occupations_outside_unit_groups == 1  # the local occupation


def test_skills_are_the_union_over_all_descendant_occupations(export_dir):
    entries, _ = build_occupation_entries(read_taxonomy_export(export_dir))
    unit = _by_code(entries)["0110"]

    # o1 and its sub-occupation o2; s1 is listed once; uuid is the oldest history entry.
    assert unit["skills"]["essential"] == {
        "uuids": ["s1-old", "s3-old"],
        "labels": ["liderar", "planificar"],
    }
    assert unit["skills"]["optional"] == {"uuids": ["s2-old"], "labels": ["nadar"]}


def test_labels_and_description_come_from_the_export(export_dir):
    entries, report = build_occupation_entries(read_taxonomy_export(export_dir))
    occupation = _by_code(entries)["0110"]["occupation"]

    assert occupation["preferred_label"] == "Oficiales de las fuerzas armadas"
    assert occupation["description"] == "Dirigen unidades"
    assert occupation["alt_labels"] == ["Oficial", "Oficiala"]
    assert occupation["uuid"] == "u-g0110-old"
    assert report.codes_without_skills == ["2512"]
    assert report.codes_without_description == ["2512"]


def test_counties_become_empty_county_rows(export_dir):
    entries, _ = build_occupation_entries(
        read_taxonomy_export(export_dir), counties=["Buenos Aires", "Córdoba"]
    )

    assert _by_code(entries)["0110"]["counties_data"] == [
        {"county": "Buenos Aires"},
        {"county": "Córdoba"},
    ]


def test_enrichment_copies_language_neutral_fields_by_normalised_code(export_dir):
    entries, report = build_occupation_entries(
        read_taxonomy_export(export_dir), enrich_from=_enrichment()
    )
    unit = _by_code(entries)["0110"]

    assert unit["occupation"]["included_tasks"] == "(a) lead"
    assert unit["occupation"]["requires_post_secondary"] is True
    assert unit["onet_work_activities"][0]["WA_code"] == "4.A.1"
    assert unit["counties_data"] == []  # counties are not copied by default
    assert unit["occupation"]["preferred_label"] == "Oficiales de las fuerzas armadas"
    assert report.codes_not_enriched == ["2512"]


def test_copy_counties_takes_the_enrichment_counties(export_dir):
    entries, _ = build_occupation_entries(
        read_taxonomy_export(export_dir),
        enrich_from=_enrichment(),
        counties=["ignored"],
        copy_counties=True,
    )

    assert _by_code(entries)["0110"]["counties_data"][0]["county"] == "Nairobi"
    assert _by_code(entries)["2512"]["counties_data"] == [{"county": "ignored"}]


def test_skill_differences_are_reported(export_dir):
    _, report = build_occupation_entries(
        read_taxonomy_export(export_dir), enrich_from=_enrichment()
    )

    [diff] = report.skill_diffs
    assert diff.code == "0110"
    assert (diff.missing_essential, diff.extra_essential) == (1, 1)  # curated / s3-old
    assert (diff.missing_optional, diff.extra_optional) == (0, 0)


def test_merge_skills_adds_curated_skills_labelled_from_the_export(export_dir):
    entries, report = build_occupation_entries(
        read_taxonomy_export(export_dir), enrich_from=_enrichment(), merge_skills=True
    )
    essential = _by_code(entries)["0110"]["skills"]["essential"]

    assert essential["uuids"] == ["s1-old", "s3-old", "curated"]
    # s1-old keeps the export's label; "curated" is not in the export, so it keeps its own.
    assert essential["labels"] == ["liderar", "planificar", "c"]
    assert len(report.skill_diffs) == 1  # reported before the merge


def test_built_entries_are_a_valid_occupation_file(export_dir):
    entries, _ = build_occupation_entries(
        read_taxonomy_export(export_dir), enrich_from=_enrichment(), counties=["X"]
    )

    parsed = parse_occupation_file(entries)
    assert [p.code for p in parsed] == ["0110", "2512"]


def test_group_ancestry_is_broadest_first(export_dir):
    ancestry = group_ancestry(read_taxonomy_export(export_dir))

    assert ancestry["0110"] == ["0", "01", "0110"]
    assert ancestry["I3"] == ["I3"]


def test_occupation_group_ancestry_is_keyed_by_normalised_unit_group_code(export_dir):
    ancestry = occupation_group_ancestry(read_taxonomy_export(export_dir))

    assert set(ancestry) == {"110", "2512", "I3"}
    assert ancestry["110"] == ["0", "01", "0110"]


def _append(path: Path, rows: list[list[str]]) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        header, *existing = list(csv.reader(f))
    _write(path, header, [*existing, *rows])


def test_group_codes_that_normalise_alike_keep_their_own_ancestry(export_dir):
    # Major "1" > sub-major "11" next to the fixture's "0" > "01": "01" and "1" normalise alike.
    _append(
        export_dir / "occupation_groups.csv",
        [
            ["g1", "uri:1", "u-g1", "1", "iscogroup", "Directores", "", ""],
            [
                "g11",
                "uri:11",
                "u-g11",
                "11",
                "iscogroup",
                "Directores ejecutivos",
                "",
                "",
            ],
        ],
    )
    _append(
        export_dir / "occupation_hierarchy.csv",
        [["iscogroup", "g1", "g11", "iscogroup"]],
    )
    export = read_taxonomy_export(export_dir)

    ancestry = group_ancestry(export)
    assert ancestry["01"] == ["0", "01"]
    assert ancestry["1"] == ["1"]
    assert ancestry["11"] == ["1", "11"]
    docs = build_group_documents(export)
    assert docs["01"]["ancestor_codes"] == ["0"]
    assert docs["1"]["ancestor_codes"] == []
    assert docs["11"]["ancestor_codes"] == ["1"]


def test_group_documents_are_keyed_by_code_with_parents(export_dir):
    docs = build_group_documents(read_taxonomy_export(export_dir))

    assert set(docs) == {"0", "01", "0110", "2512", "I3"}
    unit = docs["0110"]
    assert unit["_id"] == "0110"
    assert unit["parent_code"] == "01"
    assert unit["ancestor_codes"] == ["0", "01"]
    assert unit["is_unit_group"] is True
    assert docs["0"]["parent_code"] is None
    assert docs["I3"]["is_unit_group"] is False
    assert unit["content_sha256"]


@pytest.mark.parametrize(
    "code, expected", [("0110", "110"), ("110", "110"), ("0", "0"), ("I31", "I31")]
)
def test_normalise_occupation_code(code, expected):
    assert normalise_occupation_code(code) == expected
