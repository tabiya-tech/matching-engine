"""Regression tests for ``build_job_from_ranked`` (Mongo ranked job document → ``Job``).

Covers every field that has **mapping logic** (fallbacks, precedence, coercion, filtering).
Simple 1:1 passthrough fields from ``classifier_metadata`` are asserted together in one
happy-path test — not duplicated per field.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
from pydantic import ValidationError

from app.clients.gemini_embedding_client import EMBEDDING_DIM
from app.domain.attributes import JobAttributes
from app.domain.skills import SkillReference
from app.domain.work_activities import WorkActivity
from app.jobs.job_document import build_job_from_ranked


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
        "onet_work_activities": [
            {
                "WA_code": "4.A.1.a.1",
                "WA_label": "Getting Information",
                "WA_Importance": 4.2,
                "WA_Level": 3.9,
            }
        ],
        "skill_groups_origin_uuids": ["grp-1"],
    }


class TestInactiveJobs:
    def test_inactive_document_returns_none(self):
        doc = _base_doc()
        doc["is_active"] = False
        assert build_job_from_ranked(doc) is None


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
        job = build_job_from_ranked(doc)
        assert job.province == "Copperbelt"
        assert job.zqf_min == 5
        assert job.zqf_max == 7
        assert job.zqf_min_label == "Diploma / Technician"
        assert job.zqf_max_label == "Bachelor's Degree"

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
        job = build_job_from_ranked(doc)
        assert job.zqf_min == 4
        assert job.zqf_max == 4
        assert job.zqf_min_label == "Craft Certificate / Grade 12"
        assert job.zqf_max_label == "Craft Certificate / Grade 12"

    def test_classifier_metadata_overrides_root_level_zqf(self):
        doc = _base_doc()
        doc["zqf_min"] = 3
        doc["zqf_max"] = 6
        doc["classifier_metadata"]["min_zqf_level"] = 5
        doc["classifier_metadata"]["max_zqf_level"] = 8
        job = build_job_from_ranked(doc)
        assert job.zqf_min == 5
        assert job.zqf_max == 8

    def test_legacy_root_level_zqf_when_metadata_absent(self):
        doc = _base_doc()
        doc["zqf_min"] = 4
        doc["zqf_max"] = 9
        job = build_job_from_ranked(doc)
        assert job.zqf_min == 4
        assert job.zqf_max == 9
        assert job.zqf_min_label is None
        assert job.zqf_max_label is None

    def test_missing_zqf_fields(self):
        doc = _base_doc()
        doc["classifier_metadata"].pop("county", None)
        job = build_job_from_ranked(doc)
        assert job.zqf_min is None
        assert job.zqf_max is None


class TestLocationMapping:
    def test_province_falls_back_to_county(self):
        doc = _base_doc()
        doc["classifier_metadata"].pop("province", None)
        job = build_job_from_ranked(doc)
        assert job.province == "Lusaka"

    def test_empty_province_uses_county(self):
        doc = _base_doc()
        doc["classifier_metadata"]["province"] = ""
        doc["classifier_metadata"]["county"] = "Central"
        job = build_job_from_ranked(doc)
        assert job.province == "Central"

    def test_location_joins_city_and_province(self):
        doc = _base_doc()
        doc["classifier_metadata"]["province"] = "Copperbelt"
        job = build_job_from_ranked(doc)
        assert job.location == "Lusaka Copperbelt"


class TestIdentifierAndRelationMapping:
    def test_origin_uuid_precedence(self):
        doc = _base_doc()
        doc["origin_uuid"] = "origin-a"
        doc["originUuid"] = "origin-b"
        job = build_job_from_ranked(doc)
        assert job.uuid == "test-job-001"
        assert job.originUuid == "origin-a"
        assert job.job_fingerprint == "fp-abc123"

    def test_related_occupation_id_falls_back_to_isco_group_id(self):
        doc = _base_doc()
        job = build_job_from_ranked(doc)
        assert job.related_occupation_id == "2512"


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
        job = build_job_from_ranked(doc)
        assert job.posted_date == meta_value

    def test_opportunity_description_falls_back_to_description(self):
        doc = _base_doc()
        doc["classifier_metadata"].pop("job_description", None)
        doc["classifier_metadata"]["description"] = "Alt description"
        job = build_job_from_ranked(doc)
        assert job.opportunity_description == "Alt description"

    def test_closing_date_none_becomes_empty_string(self):
        doc = _base_doc()
        doc["classifier_metadata"]["closing_date"] = None
        job = build_job_from_ranked(doc)
        assert job.closing_date == ""


class TestEducationAndSkillsMapping:
    def test_requires_post_secondary_from_llm_attributes(self):
        job = build_job_from_ranked(_base_doc())
        assert job.requires_post_secondary is True

    def test_essential_skills_skip_rows_without_tabiya_skill_id(self):
        doc = _base_doc()
        doc["llm_classified_skills"]["essential"] = [
            {"tabiya_skill_id": "s1", "label": "Python"},
            {"label": "no-id"},
        ]
        job = build_job_from_ranked(doc)
        assert len(job.essential_skills) == 1
        assert job.essential_skills[0] == SkillReference(id="s1", label="Python")

    def test_skill_groups_scalar_coerced_to_single_element_list(self):
        doc = _base_doc()
        doc["skill_groups_origin_uuids"] = "grp-solo"
        job = build_job_from_ranked(doc)
        assert job.skill_groups_origin_uuids == ["grp-solo"]

    def test_skill_groups_none_becomes_empty_list(self):
        doc = _base_doc()
        doc["skill_groups_origin_uuids"] = None
        job = build_job_from_ranked(doc)
        assert job.skill_groups_origin_uuids == []


class TestEmbeddingPassthrough:
    def test_job_embedding_only_when_gemini_dim(self):
        doc = _base_doc()
        doc["job_embedding"] = [0.1] * EMBEDDING_DIM
        job = build_job_from_ranked(doc)
        assert job.embedding.dtype == np.float32
        assert job.embedding.shape == (EMBEDDING_DIM,)

    def test_job_embedding_wrong_dim_omitted(self):
        doc = _base_doc()
        doc["job_embedding"] = [0.1, 0.2, 0.3]
        job = build_job_from_ranked(doc)
        assert job.embedding is None

    def test_concat_gemini_vector_bin_preferred_over_job_embedding(self):
        doc = _base_doc()
        synced = np.full(EMBEDDING_DIM, 0.5, dtype=np.float32)
        doc["concat_skill_embedding_gemini"] = {"vector_bin": synced.tobytes()}
        doc["job_embedding"] = [0.1] * EMBEDDING_DIM
        job = build_job_from_ranked(doc)
        np.testing.assert_array_equal(job.embedding, synced)

    def test_concat_gemini_vector_bin_wrong_size_falls_back_to_job_embedding(self):
        doc = _base_doc()
        doc["concat_skill_embedding_gemini"] = {"vector_bin": b"\x00\x00\x00\x00"}
        doc["job_embedding"] = [0.1] * EMBEDDING_DIM
        job = build_job_from_ranked(doc)
        assert job.embedding[0] == pytest.approx(0.1)

    def test_job_embedding_whitened_flag_true(self):
        # When the DB has whitened job_embedding (whitening.enabled True), the flag is set so the
        # matcher consumes it directly instead of re-whitening in-process.
        doc = _base_doc()
        doc["llm_reranker_meta"] = {"embedding": {"whitening": {"enabled": True}}}
        assert build_job_from_ranked(doc).embedding_whitened is True

    def test_job_embedding_whitened_flag_defaults_false(self):
        # No metadata (offline corpus / not-yet-whitened jobs) -> raw -> whitened in-process.
        assert build_job_from_ranked(_base_doc()).embedding_whitened is False
        doc = _base_doc()
        doc["llm_reranker_meta"] = {"embedding": {"whitening": {"enabled": False}}}
        assert build_job_from_ranked(doc).embedding_whitened is False


class TestCompassConsumerContractFields:
    """Fields added for the Compass jobs board: category, source_platform, skills."""

    def test_category_prefers_explicit_field(self):
        doc = _base_doc()
        doc["classifier_metadata"]["category"] = "Healthcare"
        job = build_job_from_ranked(doc)
        assert job.category == "Healthcare"

    def test_category_falls_back_to_sector_then_isco_group(self):
        doc = _base_doc()
        doc["classifier_metadata"]["sector"] = "Tech sector"
        job = build_job_from_ranked(doc)
        assert job.category == "Tech sector"

        doc = _base_doc()  # no category, no sector → ISCO group label
        job = build_job_from_ranked(doc)
        assert job.category == "Software developers"

    def test_source_platform_candidate_fields(self):
        doc = _base_doc()
        doc["classifier_metadata"]["source"] = "BrighterMonday"
        job = build_job_from_ranked(doc)
        assert job.source_platform == "BrighterMonday"

    def test_skills_flattens_essential_then_optional_deduped(self):
        doc = _base_doc()
        doc["llm_classified_skills"] = {
            "essential": [
                {"tabiya_skill_id": "s1", "label": "Python"},
                {"tabiya_skill_id": "s2", "label": "SQL"},
            ],
            "optional": [
                {"tabiya_skill_id": "s3", "label": "SQL"},  # duplicate label dropped
                {"tabiya_skill_id": "s4", "label": "Docker"},
            ],
        }
        job = build_job_from_ranked(doc)
        assert job.skills == ["Python", "SQL", "Docker"]


class TestClassifierMetadataHappyPath:
    """1:1 passthrough fields — one test avoids per-field duplication."""

    def test_core_listing_fields_from_classifier_metadata(self):
        job = build_job_from_ranked(_base_doc())
        assert job.opportunity_title == "Software Engineer"
        assert job.employer == "TechCorp"
        assert job.city == "Lusaka"
        assert job.employment_type == "full_time"
        assert job.salary_text == "K8000/month"
        assert job.url == "https://example.com/apply"
        assert job.contract_type == "full_time"
        assert job.opportunity_isco_occupation_group == "Software developers"
        assert job.opportunity_isco_occupation_group_id == "2512"
        assert job.opportunity_description == "Build things."
        assert job.onet_work_activities == [
            WorkActivity(
                code="4.A.1.a.1", label="Getting Information", importance=4.2, level=3.9
            )
        ]
        assert job.attributes == JobAttributes()

    def test_default_opportunity_title_when_missing(self):
        doc = _base_doc()
        doc["classifier_metadata"].pop("title", None)
        job = build_job_from_ranked(doc)
        assert job.opportunity_title == "Unknown"

    def test_default_employment_type_when_missing(self):
        doc = _base_doc()
        doc["classifier_metadata"].pop("employment_type", None)
        job = build_job_from_ranked(doc)
        assert job.contract_type == "full_time"


class TestBoundaryNormalisation:
    """Upstream values are normalised into the strict ``Job`` types, or the document is rejected."""

    def test_numeric_ids_and_dates_are_stringified(self):
        doc = _base_doc()
        doc["classifier_metadata"]["isco_occupation_group_id"] = 2512
        doc["job_id"] = 42
        job = build_job_from_ranked(doc)
        assert job.opportunity_isco_occupation_group_id == "2512"
        assert job.uuid == "42"

    def test_requires_post_secondary_accepts_0_1_and_strings(self):
        for raw, expected in ((1, True), (0, False), ("false", False), (None, None)):
            doc = _base_doc()
            doc["llm_job_attributes"]["attributes"]["requires_post_secondary"] = raw
            assert build_job_from_ranked(doc).requires_post_secondary is expected

    def test_attribute_levels_keep_known_attributes_only(self):
        doc = _base_doc()
        doc["llm_job_attributes"]["attributes"].update(
            {"earnings_per_month": "earn_50_80k", "not_a_schema_attribute": "x"}
        )
        job = build_job_from_ranked(doc)
        assert job.attributes == JobAttributes(earnings_per_month="earn_50_80k")

    def test_unrated_work_activities_are_dropped(self):
        doc = _base_doc()
        doc["onet_work_activities"] = [
            {"WA_code": "4.A.1", "WA_Importance": "3.5", "WA_Level": "4"},
            {"WA_code": "4.A.2", "WA_Importance": "", "WA_Level": ""},
            {
                "WA_code": "4.A.3",
                "WA_Importance": "code is not available",
                "WA_Level": "code is not available",
            },
        ]
        job = build_job_from_ranked(doc)
        assert job.onet_work_activities == [
            WorkActivity(code="4.A.1", label="", importance=3.5, level=4.0)
        ]

    def test_zqf_levels_are_parsed_from_numeric_strings(self):
        doc = _base_doc()
        doc["classifier_metadata"]["zqf_min"] = "4"
        assert build_job_from_ranked(doc).zqf_min == 4

    def test_malformed_document_is_rejected(self):
        doc = _base_doc()
        doc["classifier_metadata"]["title"] = {"en": "nested"}
        with pytest.raises(ValidationError):
            build_job_from_ranked(doc)
