"""Shape of a stored ranked-job document and its mapping to the ``Job`` domain model.

``RankedJobDocument`` validates the raw Mongo document (the llm-reranker's output contract) and
``build_job_from_ranked`` maps it to a ``Job``. ``RANKED_JOB_FIND_PROJECTION`` must stay aligned with
both: every field the mapper reads has to be projected. Field sources are documented in
``app/domain/README.md``.
"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import ConfigDict, Field

from app.domain.attributes import JobAttributes
from app.domain.base import SourceRecord
from app.domain.coercion import (
    OptionalFlag,
    OptionalLevel,
    OptionalText,
    OptionalVector,
    TextList,
    parse_flag,
)
from app.domain.job import Job
from app.domain.skills import SkillReference
from app.domain.work_activities import WorkActivityRecord, work_activities_from_records

logger = logging.getLogger(__name__)

# Only jobs intended to be shown / matched; keeps Mongo transfers and Python work small.
# Every jobs query leads with this; see JOBS_INDEX_MODELS in app.server_dependencies.db_dependencies.
RANKED_JOBS_ACTIVE_FILTER: dict[str, Any] = {"is_active": True}

# Ranked / enriched job docs: listing fields on classifier_metadata (see build_job_from_ranked).
CITY_FIELD = "classifier_metadata.city"
COUNTY_FIELD = "classifier_metadata.county"
PROVINCE_FIELD = "classifier_metadata.province"


# Inclusion projection for job find (must stay aligned with RankedJobDocument / build_job_from_ranked).
RANKED_JOB_FIND_PROJECTION: dict[str, int] = {
    "job_id": 1,
    "job_fingerprint": 1,
    "is_active": 1,
    "classifier_metadata.city": 1,
    "classifier_metadata.county": 1,
    "classifier_metadata.title": 1,
    "classifier_metadata.employer": 1,
    "classifier_metadata.employment_type": 1,
    "classifier_metadata.salary": 1,
    "classifier_metadata.closing_date": 1,
    "classifier_metadata.application_url": 1,
    "classifier_metadata.job_description": 1,
    "classifier_metadata.description": 1,
    # Opportunity passthrough (consumer contract). Best-effort candidate names — absent fields
    # are simply not returned by Mongo; confirm exact names against the live collection.
    "classifier_metadata.posted_date": 1,
    "classifier_metadata.date_posted": 1,
    "classifier_metadata.isco_occupation_group": 1,
    "classifier_metadata.isco_occupation_group_id": 1,
    # Compass jobs-board consumer contract: sector/category and source platform.
    # Best-effort candidate names — absent fields are simply not returned by Mongo.
    "classifier_metadata.category": 1,
    "classifier_metadata.sector": 1,
    "classifier_metadata.source_platform": 1,
    "classifier_metadata.source": 1,
    "classifier_metadata.platform": 1,
    "llm_classified_skills": 1,
    "llm_job_attributes": 1,
    "onet_work_activities": 1,
    "skill_groups_origin_uuids": 1,
    # Gemini concat NPZ sync (see gemini_vs_minilm.sync_gemini_embeddings_to_mongo)
    "concat_skill_embedding_gemini": 1,
    # Float array on ranked job docs (e.g. SouthAfricaJobs_V2.ranked_jobs); /match_v3 fallback if no vector_bin.
    "job_embedding": 1,
    # Whether job_embedding is already whitened on the DB side (colleague's enrichment). When True the
    # matching code consumes it directly instead of whitening in-process. Nested scalar only — keeps the
    # doc small (does not pull the 3072-float raw_embedding).
    "llm_reranker_meta.embedding.whitening.enabled": 1,
    # ZQF education annotation (Zambia); root-level fields set by scrape-time enrichment / backfill.
    "zqf_min": 1,
    "zqf_max": 1,
    # Zambia: ZQF and province under classifier_metadata.
    # TestZambiaJobs uses zqf_min/zqf_max; TestAutomatedDemandside uses min_zqf_level/max_zqf_level.
    "classifier_metadata.province": 1,
    "classifier_metadata.zqf_min": 1,
    "classifier_metadata.zqf_max": 1,
    "classifier_metadata.zqf_min_label": 1,
    "classifier_metadata.zqf_max_label": 1,
    "classifier_metadata.min_zqf_level": 1,
    "classifier_metadata.max_zqf_level": 1,
    "classifier_metadata.min_zqf_label": 1,
    "classifier_metadata.max_zqf_label": 1,
}


class ClassifierMetadata(SourceRecord):
    """Listing fields of a ranked job (``classifier_metadata``), copied from the scraped ad.

    Several fields have alternative names because producers differ by country/pipeline version;
    ``build_job_from_ranked`` documents the precedence.
    """

    title: OptionalText = Field(default=None, description="Job title.")
    employer: OptionalText = Field(default=None, description="Employer name.")
    city: OptionalText = Field(default=None, description="City/town.")
    county: OptionalText = Field(
        default=None, description="County (older documents; replaced by ``province``)."
    )
    province: OptionalText = Field(default=None, description="Province / admin region.")
    employment_type: OptionalText = Field(
        default=None, description="Employment type as stated in the ad."
    )
    salary: OptionalText = Field(
        default=None, description="Salary as stated in the ad."
    )
    closing_date: OptionalText = Field(
        default=None, description="Application closing date."
    )
    application_url: OptionalText = Field(
        default=None, description="Link to the posting / application."
    )
    job_description: OptionalText = Field(
        default=None, description="Full job description."
    )
    description: OptionalText = Field(
        default=None, description="Job description (older documents)."
    )
    posted_date: OptionalText = Field(default=None, description="Date posted.")
    date_posted: OptionalText = Field(
        default=None, description="Date posted (alternative name)."
    )
    posted_at: OptionalText = Field(
        default=None, description="Date posted (alternative name)."
    )
    isco_occupation_group: OptionalText = Field(
        default=None, description="ISCO occupation group label."
    )
    isco_occupation_group_id: OptionalText = Field(
        default=None, description="ISCO occupation group code."
    )
    category: OptionalText = Field(default=None, description="Sector/category.")
    sector: OptionalText = Field(default=None, description="Sector (alternative name).")
    source_platform: OptionalText = Field(
        default=None, description="Platform the ad was scraped from."
    )
    source: OptionalText = Field(
        default=None, description="Source platform (alternative name)."
    )
    platform: OptionalText = Field(
        default=None, description="Source platform (alternative name)."
    )
    zqf_min: OptionalLevel = Field(
        default=None, description="Minimum ZQF level (TestZambiaJobs naming)."
    )
    zqf_max: OptionalLevel = Field(
        default=None, description="Maximum ZQF level (TestZambiaJobs naming)."
    )
    zqf_min_label: OptionalText = Field(
        default=None, description="Label of ``zqf_min``."
    )
    zqf_max_label: OptionalText = Field(
        default=None, description="Label of ``zqf_max``."
    )
    min_zqf_level: OptionalLevel = Field(
        default=None, description="Minimum ZQF level (demand-side naming)."
    )
    max_zqf_level: OptionalLevel = Field(
        default=None, description="Maximum ZQF level (demand-side naming)."
    )
    min_zqf_label: OptionalText = Field(
        default=None, description="Label of ``min_zqf_level``."
    )
    max_zqf_label: OptionalText = Field(
        default=None, description="Label of ``max_zqf_level``."
    )


class ClassifiedSkill(SourceRecord):
    """One skill ranked essential/optional by the llm-reranker (``llm_classified_skills.*[]``)."""

    tabiya_skill_id: OptionalText = Field(
        default=None, description="Tabiya taxonomy skill id; None when unmapped."
    )
    label: OptionalText = Field(
        default=None, description="Skill label; may be empty upstream."
    )


class LlmClassifiedSkills(SourceRecord):
    """Skills the llm-reranker kept for the job (``llm_classified_skills``)."""

    essential: list[ClassifiedSkill] | None = Field(
        default=None, description="Essential skills."
    )
    optional: list[ClassifiedSkill] | None = Field(
        default=None, description="Optional skills."
    )


class LlmJobAttributes(SourceRecord):
    """Job attributes estimated by the llm-reranker (``llm_job_attributes``)."""

    attributes: dict[str, str | bool | int | float | None] | None = Field(
        default=None,
        description="Attribute name to level id, plus the ``requires_post_secondary`` flag.",
    )


class ConcatSkillEmbedding(SourceRecord):
    """Concat skill embedding synced from an NPZ (``concat_skill_embedding_gemini``)."""

    vector_bin: bytes | None = Field(
        default=None, description="float32 vector as BSON binary."
    )


class EmbeddingWhitening(SourceRecord):
    """``llm_reranker_meta.embedding.whitening``."""

    enabled: bool | int | str | None = Field(
        default=None,
        description="True when ``job_embedding`` was whitened by the llm-reranker.",
    )


class EmbeddingMeta(SourceRecord):
    """``llm_reranker_meta.embedding``: provenance of ``job_embedding``."""

    whitening: EmbeddingWhitening | None = Field(
        default=None, description="Whitening provenance."
    )


class LlmRerankerMeta(SourceRecord):
    """``llm_reranker_meta``: pipeline metadata written by the llm-reranker."""

    embedding: EmbeddingMeta | None = Field(
        default=None, description="Embedding provenance."
    )


class RankedJobDocument(SourceRecord):
    """A ranked job document as read from MongoDB (``MONGO_JOBS_COLLECTION``).

    Only the fields the matching service reads are declared; everything else is ignored.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    job_id: OptionalText = Field(
        default=None, description="Job id assigned by the scraper."
    )
    job_fingerprint: OptionalText = Field(
        default=None, description="Content fingerprint of the ad."
    )
    origin_uuid: OptionalText = Field(
        default=None, description="Source-system id, when the producer sets one."
    )
    originUuid: OptionalText = Field(
        default=None, description="Source-system id (alternative name)."
    )
    related_occupation_id: OptionalText = Field(
        default=None, description="Occupation the job maps to."
    )
    is_active: OptionalFlag = Field(
        default=None, description="False for expired / withdrawn ads."
    )
    classifier_metadata: ClassifierMetadata | None = Field(
        default=None, description="Listing fields."
    )
    llm_classified_skills: LlmClassifiedSkills | None = Field(
        default=None, description="Ranked skills."
    )
    llm_job_attributes: LlmJobAttributes | None = Field(
        default=None, description="Estimated attributes."
    )
    onet_work_activities: list[WorkActivityRecord] | None = Field(
        default=None, description="O*NET work activities with importance and level."
    )
    skill_groups_origin_uuids: TextList = Field(
        default_factory=list, description="Skill-group origin UUIDs."
    )
    concat_skill_embedding_gemini: ConcatSkillEmbedding | None = Field(
        default=None,
        description="NPZ-synced concat embedding (preferred over ``job_embedding``).",
    )
    job_embedding: OptionalVector = Field(
        default=None, description="Concat skill embedding as a float list."
    )
    llm_reranker_meta: LlmRerankerMeta | None = Field(
        default=None, description="Pipeline metadata."
    )
    zqf_min: OptionalLevel = Field(
        default=None, description="Minimum ZQF level (root-level backfill)."
    )
    zqf_max: OptionalLevel = Field(
        default=None, description="Maximum ZQF level (root-level backfill)."
    )


def _skill_references(skills: list[ClassifiedSkill] | None) -> list[SkillReference]:
    """Skills that are mapped to a taxonomy id, as ``SkillReference``; unmapped ones are dropped."""
    return [
        SkillReference(id=s.tabiya_skill_id, label=s.label or "")
        for s in skills or []
        if s.tabiya_skill_id
    ]


def _stage1_embedding(doc: RankedJobDocument) -> np.ndarray | None:
    """The job's concat embedding: the NPZ-synced ``vector_bin`` if it has the right size, else
    ``job_embedding`` if it has the right size, else None."""
    from app.clients.gemini_embedding_client import EMBEDDING_DIM

    synced = doc.concat_skill_embedding_gemini
    if synced is not None and synced.vector_bin is not None:
        vector = np.frombuffer(bytes(synced.vector_bin), dtype=np.float32)
        if vector.size == EMBEDDING_DIM:
            return vector
    if doc.job_embedding is not None and doc.job_embedding.size == EMBEDDING_DIM:
        return doc.job_embedding
    return None


def _embedding_is_whitened(doc: RankedJobDocument) -> bool:
    meta = doc.llm_reranker_meta
    whitening = meta.embedding.whitening if meta and meta.embedding else None
    return whitening is not None and whitening.enabled is True


def _strip(text: str | None) -> str:
    return text.strip() if text else ""


def job_from_document(doc: RankedJobDocument) -> Job | None:
    """Map a validated ranked-job document to a ``Job``.

    Listing metadata (title, employer, location, …) comes from ``classifier_metadata``; skills and
    preference attributes come from ``llm_classified_skills`` and ``llm_job_attributes``.
    ``onet_work_activities`` and ``skill_groups_origin_uuids`` must already be on the document
    (written by the llm-reranker's enrichment); they are not computed at request time.

    :param doc: The validated document
    :return: The job, or None when the document is inactive (``is_active`` is False)
    """
    if doc.is_active is False:
        return None

    meta = doc.classifier_metadata or ClassifierMetadata()
    job_id = doc.job_id or ""

    llm_skills = doc.llm_classified_skills or LlmClassifiedSkills()
    essential_skills = _skill_references(llm_skills.essential)
    optional_skills = _skill_references(llm_skills.optional)
    # Label-primary resolver requires non-empty labels. Surface jobs that arrived
    # without them so the upstream pipeline gap (NEL/llm-reranker emitting empty
    # label when a URI isn't in ranker_candidates) is visible at consumer side.
    n_missing_ess = sum(1 for s in essential_skills if not s.label)
    n_missing_opt = sum(1 for s in optional_skills if not s.label)
    if n_missing_ess or n_missing_opt:
        logger.warning(
            "build_job_from_ranked: job_id=%s job_fingerprint=%s arrived with "
            "empty labels: %d/%d essential, %d/%d optional",
            job_id or "?",
            (doc.job_fingerprint or "")[:16] or "?",
            n_missing_ess,
            len(essential_skills),
            n_missing_opt,
            len(optional_skills),
        )

    raw_attributes = (
        doc.llm_job_attributes.attributes if doc.llm_job_attributes else None
    ) or {}

    city = _strip(meta.city)
    province = _strip(meta.province) or _strip(meta.county)
    location = " ".join(p for p in (city, province) if p)

    job_fingerprint = _strip(doc.job_fingerprint)
    isco_group = meta.isco_occupation_group
    isco_group_id = meta.isco_occupation_group_id

    # Flat list of skill labels for the jobs board: essential first, then optional, de-duplicated.
    skill_labels: list[str] = []
    for s in essential_skills + optional_skills:
        if s.label and s.label not in skill_labels:
            skill_labels.append(s.label)

    return Job(
        uuid=job_id,
        # Stable content fingerprint, falling back to job_id.
        originUuid=(doc.origin_uuid or doc.originUuid or job_fingerprint or job_id)
        or None,
        job_fingerprint=job_fingerprint or None,
        opportunity_title=meta.title or "Unknown",
        location=location,
        city=city,
        province=province,
        employer=meta.employer,
        employment_type=meta.employment_type,
        contract_type=meta.employment_type or "full_time",
        salary_text=meta.salary,
        closing_date=meta.closing_date or "",
        posted_date=_strip(meta.posted_date or meta.date_posted or meta.posted_at)
        or None,
        url=meta.application_url,
        opportunity_description=meta.job_description or meta.description or "",
        opportunity_isco_occupation_group=isco_group,
        opportunity_isco_occupation_group_id=isco_group_id,
        related_occupation_id=(doc.related_occupation_id or isco_group_id) or None,
        # Sector/category: explicit field, else the ISCO occupation group label.
        category=meta.category or meta.sector or isco_group,
        source_platform=meta.source_platform or meta.source or meta.platform,
        essential_skills=essential_skills,
        optional_skills=optional_skills,
        skills=skill_labels,
        skill_groups_origin_uuids=doc.skill_groups_origin_uuids,
        attributes=JobAttributes.from_levels(raw_attributes),
        onet_work_activities=work_activities_from_records(
            doc.onet_work_activities or [], source_id=f"job {job_id or '?'}"
        ),
        # Post-secondary education gate (see app.ranking.retrieval).
        requires_post_secondary=parse_flag(
            raw_attributes.get("requires_post_secondary")
        ),
        # ZQF education annotation (Zambia): classifier_metadata (two naming conventions) or root.
        zqf_min=meta.min_zqf_level or meta.zqf_min or doc.zqf_min,
        zqf_max=meta.max_zqf_level or meta.zqf_max or doc.zqf_max,
        zqf_min_label=meta.min_zqf_label or meta.zqf_min_label,
        zqf_max_label=meta.max_zqf_label or meta.zqf_max_label,
        embedding=_stage1_embedding(doc),
        # True iff job_embedding has been whitened on the DB side (same concat artifact the code
        # ships); the engine then consumes it directly. Absent/False => raw (safe default).
        embedding_whitened=_embedding_is_whitened(doc),
    )


def build_job_from_ranked(raw: Mapping[str, Any]) -> Job | None:
    """Validate one stored ranked-job document and map it to a ``Job``.

    :param raw: The document as read from MongoDB
    :return: The job, or None when the document is inactive
    :raises pydantic.ValidationError: If the document does not have the ranked-job shape
    """
    return job_from_document(RankedJobDocument.model_validate(raw))
