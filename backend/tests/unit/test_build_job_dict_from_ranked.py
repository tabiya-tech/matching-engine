"""Regression tests for ``build_job_dict_from_ranked`` (Mongo ranked job → flat job dict).

Covers every field that has **mapping logic** (fallbacks, precedence, coercion, filtering).
Simple 1:1 passthrough fields from ``classifier_metadata`` are asserted together in one
happy-path test — not duplicated per field.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from app.database import build_job_dict_from_ranked
from app.services.cross_encoder.gemini_embeddings import EMBEDDING_DIM


def _base_doc() -> Dict[str, Any]:
    return {
        "job_id": "test-job-001",
        "job_fingerprint": "fp-abc123",
        "is_active": True,
        "classifier_metadata": {
            "title": "Software Engineer",
            "employer": "TechCorp",
            "city": "Lusaka",
            "county": "Lusaka",
            "employment_type": "full_time",
            "salary": "K8000/month",
            "closing_date": "2026-07-01",
            "application_url": "https://example.com/apply",
            "job_description": "Build things.",
            "isco_occupation_group": "Software developers",
            "isco_occupation_group_id": "2512",
        },
        "llm_classified_skills": {
            "essential": [{"tabiya_skill_id": "s1", "label": "Python"}],
            "optional": [{"tabiya_skill_id": "s2", "label": "SQL"}],
        },
        "llm_job_attributes": {"attributes": {"requires_post_secondary": True}},
        "onet_work_activities": ["4.A.1.a.1"],
        "skill_groups_origin_uuids": ["grp-1"],
    }


class TestInactiveJobs:
    def test_inactive_document_returns_none(self):
        doc = _base_doc()
        doc["is_active"] = False
        assert build_job_dict_from_ranked(doc) is None


class TestZqfMapping:
    def test_testautomated_demandside_min_zqf_level_format(self):
        doc = _base_doc()
        doc["classifier_metadata"].update(
            {
                "province": "Copperbelt",
                "min_zqf_level": 5,
                "max_zqf_level": 7,
                "min_zqf_label": "Diploma / Technician",
                "max_zqf_label": "Bachelor's Degree",
            }
        )
        job = build_job_dict_from_ranked(doc)
        assert job["province"] == "Copperbelt"
        assert job["zqf_min"] == 5
        assert job["zqf_max"] == 7
        assert job["zqf_min_label"] == "Diploma / Technician"
        assert job["zqf_max_label"] == "Bachelor's Degree"

    def test_zambia_zqf_min_format_in_classifier_metadata(self):
        doc = _base_doc()
        doc["classifier_metadata"].update(
            {
                "zqf_min": 4,
                "zqf_max": 4,
                "zqf_min_label": "Craft Certificate / Grade 12",
                "zqf_max_label": "Craft Certificate / Grade 12",
            }
        )
        job = build_job_dict_from_ranked(doc)
        assert job["zqf_min"] == 4
        assert job["zqf_max"] == 4
        assert job["zqf_min_label"] == "Craft Certificate / Grade 12"
        assert job["zqf_max_label"] == "Craft Certificate / Grade 12"

    def test_classifier_metadata_overrides_root_level_zqf(self):
        doc = _base_doc()
        doc["zqf_min"] = 3
        doc["zqf_max"] = 6
        doc["classifier_metadata"]["min_zqf_level"] = 5
        doc["classifier_metadata"]["max_zqf_level"] = 8
        job = build_job_dict_from_ranked(doc)
        assert job["zqf_min"] == 5
        assert job["zqf_max"] == 8

    def test_legacy_root_level_zqf_when_metadata_absent(self):
        doc = _base_doc()
        doc["zqf_min"] = 4
        doc["zqf_max"] = 9
        job = build_job_dict_from_ranked(doc)
        assert job["zqf_min"] == 4
        assert job["zqf_max"] == 9
        assert job["zqf_min_label"] is None
        assert job["zqf_max_label"] is None

    def test_missing_zqf_fields(self):
        doc = _base_doc()
        doc["classifier_metadata"].pop("county", None)
        job = build_job_dict_from_ranked(doc)
        assert job["zqf_min"] is None
        assert job["zqf_max"] is None


class TestLocationMapping:
    def test_province_falls_back_to_county(self):
        doc = _base_doc()
        doc["classifier_metadata"].pop("province", None)
        job = build_job_dict_from_ranked(doc)
        assert job["province"] == "Lusaka"

    def test_empty_province_uses_county(self):
        doc = _base_doc()
        doc["classifier_metadata"]["province"] = ""
        doc["classifier_metadata"]["county"] = "Central"
        job = build_job_dict_from_ranked(doc)
        assert job["province"] == "Central"

    def test_location_joins_city_and_province(self):
        doc = _base_doc()
        doc["classifier_metadata"]["province"] = "Copperbelt"
        job = build_job_dict_from_ranked(doc)
        assert job["location"] == "Lusaka Copperbelt"


class TestIdentifierAndRelationMapping:
    def test_origin_uuid_precedence(self):
        doc = _base_doc()
        doc["origin_uuid"] = "origin-a"
        doc["originUuid"] = "origin-b"
        job = build_job_dict_from_ranked(doc)
        assert job["uuid"] == "test-job-001"
        assert job["originUuid"] == "origin-a"
        assert job["job_fingerprint"] == "fp-abc123"

    def test_related_occupation_id_falls_back_to_isco_group_id(self):
        doc = _base_doc()
        job = build_job_dict_from_ranked(doc)
        assert job["related_occupation_id"] == "2512"


class TestDateAndDescriptionMapping:
    @pytest.mark.parametrize(
        "meta_key,meta_value",
        [
            ("posted_date", "2026-01-15"),
            ("date_posted", "2026-02-01"),
            ("posted_at", "2026-03-10"),
        ],
    )
    def test_posted_date_fallback_chain(self, meta_key, meta_value):
        doc = _base_doc()
        doc["classifier_metadata"][meta_key] = meta_value
        job = build_job_dict_from_ranked(doc)
        assert job["posted_date"] == meta_value

    def test_opportunity_description_falls_back_to_description(self):
        doc = _base_doc()
        doc["classifier_metadata"].pop("job_description", None)
        doc["classifier_metadata"]["description"] = "Alt description"
        job = build_job_dict_from_ranked(doc)
        assert job["opportunity_description"] == "Alt description"

    def test_closing_date_none_becomes_empty_string(self):
        doc = _base_doc()
        doc["classifier_metadata"]["closing_date"] = None
        job = build_job_dict_from_ranked(doc)
        assert job["closing_date"] == ""


class TestEducationAndSkillsMapping:
    def test_requires_post_secondary_from_llm_attributes(self):
        job = build_job_dict_from_ranked(_base_doc())
        assert job["requires_post_secondary"] is True

    def test_essential_skills_skip_rows_without_tabiya_skill_id(self):
        doc = _base_doc()
        doc["llm_classified_skills"]["essential"] = [
            {"tabiya_skill_id": "s1", "label": "Python"},
            {"label": "no-id"},
        ]
        job = build_job_dict_from_ranked(doc)
        assert len(job["essential_skills"]) == 1
        assert job["essential_skills"][0]["id"] == "s1"

    def test_skill_groups_scalar_coerced_to_single_element_list(self):
        doc = _base_doc()
        doc["skill_groups_origin_uuids"] = "grp-solo"
        job = build_job_dict_from_ranked(doc)
        assert job["skill_groups_origin_uuids"] == ["grp-solo"]

    def test_skill_groups_none_becomes_empty_list(self):
        doc = _base_doc()
        doc["skill_groups_origin_uuids"] = None
        job = build_job_dict_from_ranked(doc)
        assert job["skill_groups_origin_uuids"] == []


class TestEmbeddingPassthrough:
    def test_job_embedding_only_when_gemini_dim(self):
        doc = _base_doc()
        doc["job_embedding"] = [0.1] * EMBEDDING_DIM
        job = build_job_dict_from_ranked(doc)
        assert len(job["job_embedding"]) == EMBEDDING_DIM

    def test_job_embedding_wrong_dim_omitted(self):
        doc = _base_doc()
        doc["job_embedding"] = [0.1, 0.2, 0.3]
        job = build_job_dict_from_ranked(doc)
        assert "job_embedding" not in job

    def test_concat_gemini_embedding_passthrough(self):
        doc = _base_doc()
        gem = {"vector_bin": b"\x00\x00\x00\x00"}
        doc["concat_skill_embedding_gemini"] = gem
        job = build_job_dict_from_ranked(doc)
        assert job["concat_skill_embedding_gemini"] is gem


class TestClassifierMetadataHappyPath:
    """1:1 passthrough fields — one test avoids per-field duplication."""

    def test_core_listing_fields_from_classifier_metadata(self):
        job = build_job_dict_from_ranked(_base_doc())
        assert job["opportunity_title"] == "Software Engineer"
        assert job["employer"] == "TechCorp"
        assert job["city"] == "Lusaka"
        assert job["employment_type"] == "full_time"
        assert job["salary_text"] == "K8000/month"
        assert job["url"] == "https://example.com/apply"
        assert job["contract_type"] == "full_time"
        assert job["opportunity_isco_occupation_group"] == "Software developers"
        assert job["opportunity_isco_occupation_group_id"] == "2512"
        assert job["opportunity_description"] == "Build things."
        assert job["onet_work_activities"] == ["4.A.1.a.1"]
        assert job["attributes"] == {"requires_post_secondary": True}

    def test_default_opportunity_title_when_missing(self):
        doc = _base_doc()
        doc["classifier_metadata"].pop("title", None)
        job = build_job_dict_from_ranked(doc)
        assert job["opportunity_title"] == "Unknown"

    def test_default_employment_type_when_missing(self):
        doc = _base_doc()
        doc["classifier_metadata"].pop("employment_type", None)
        job = build_job_dict_from_ranked(doc)
        assert job["contract_type"] == "full_time"
