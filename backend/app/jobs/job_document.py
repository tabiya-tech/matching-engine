"""Shape of a stored job document and its mapping to the flat job dict used by matching.

``RANKED_JOB_FIND_PROJECTION`` and ``build_job_dict_from_ranked`` must stay aligned: every field the
mapper reads has to be projected. This is the contract with the llm-reranker's output.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Only jobs intended to be shown / matched; keeps Mongo transfers and Python work small.
# Every jobs query leads with this; see JOBS_INDEX_MODELS in app.server_dependencies.db_dependencies.
RANKED_JOBS_ACTIVE_FILTER: dict[str, Any] = {"is_active": True}

# Ranked / enriched job docs: listing fields on classifier_metadata (see build_job_dict_from_ranked).
CITY_FIELD = "classifier_metadata.city"
COUNTY_FIELD = "classifier_metadata.county"
PROVINCE_FIELD = "classifier_metadata.province"


# Inclusion projection for job find (must stay aligned with build_job_dict_from_ranked).
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


def _str_or_empty(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip() if isinstance(v, str) else str(v)


def build_job_dict_from_ranked(rd: dict[str, Any]) -> dict[str, Any] | None:
    """Build the flat job dict used by matching from one stored job document.

    Listing metadata (title, employer, location, …) comes from ``classifier_metadata``.
    Skills and preference attributes come from ``llm_classified_skills`` and ``llm_job_attributes``.

    ``onet_work_activities`` and ``skill_groups_origin_uuids`` must be present on the document
    (e.g. ``RankedJobsEnriched`` produced by the enrichment script / reranker). They are not
    computed at request time.

    Returns ``None`` if the job should be skipped (document ``is_active`` is False).
    """
    if rd.get("is_active") is False:
        return None

    meta = rd.get("classifier_metadata") or {}
    job_id = str(rd.get("job_id", ""))

    lcs = rd.get("llm_classified_skills", {})
    essential_skills = [
        {"id": s["tabiya_skill_id"], "label": s.get("label", "")}
        for s in lcs.get("essential", [])
        if s.get("tabiya_skill_id")
    ]
    optional_skills = [
        {"id": s["tabiya_skill_id"], "label": s.get("label", "")}
        for s in lcs.get("optional", [])
        if s.get("tabiya_skill_id")
    ]
    # Label-primary resolver requires non-empty labels. Surface jobs that arrived
    # without them so the upstream pipeline gap (NEL/llm-reranker emitting empty
    # label when a URI isn't in ranker_candidates) is visible at consumer side.
    n_missing_ess = sum(1 for s in essential_skills if not s.get("label"))
    n_missing_opt = sum(1 for s in optional_skills if not s.get("label"))
    if n_missing_ess or n_missing_opt:
        logger.warning(
            "build_job_dict_from_ranked: job_id=%s job_fingerprint=%s arrived with "
            "empty labels: %d/%d essential, %d/%d optional",
            job_id or "?",
            (rd.get("job_fingerprint") or "")[:16] or "?",
            n_missing_ess,
            len(essential_skills),
            n_missing_opt,
            len(optional_skills),
        )

    llm_attrs = rd.get("llm_job_attributes", {})
    attributes = llm_attrs.get("attributes", {})

    city = _str_or_empty(meta.get("city"))
    province = _str_or_empty(meta.get("province")) or _str_or_empty(meta.get("county"))
    loc_parts = [p for p in (city, province) if p]
    location = " ".join(loc_parts) if loc_parts else ""

    onet_wa = list(rd.get("onet_work_activities") or [])
    raw_sgu = rd.get("skill_groups_origin_uuids")
    if raw_sgu is None:
        skill_groups: list[str] = []
    elif isinstance(raw_sgu, list):
        skill_groups = [str(x) for x in raw_sgu]
    else:
        skill_groups = [str(raw_sgu)]

    raw_closing = meta.get("closing_date")
    closing_s = "" if raw_closing is None else str(raw_closing)
    et = meta.get("employment_type") or "full_time"

    job_fp = rd.get("job_fingerprint")
    job_fp_s = str(job_fp).strip() if job_fp is not None else ""

    # Opportunity passthrough for the consumer contract. originUuid uses the stable
    # content fingerprint (falls back to job_id); posted_date / occupation classification are
    # best-effort from candidate Mongo fields and stay None when the document lacks them.
    posted_date = (
        _str_or_empty(
            meta.get("posted_date") or meta.get("date_posted") or meta.get("posted_at")
        )
        or None
    )
    isco_group = meta.get("isco_occupation_group")
    isco_group_id = meta.get("isco_occupation_group_id")
    # Compass jobs-board consumer contract: sector/category (explicit field, else the ISCO
    # occupation group label as a sensible fallback), the source platform the posting was
    # scraped from, and the flat list of skill labels (essential first, then optional).
    category = meta.get("category") or meta.get("sector") or isco_group
    source_platform = (
        meta.get("source_platform") or meta.get("source") or meta.get("platform")
    )
    skill_labels: list[str] = []
    _seen_labels: set = set()
    for s in essential_skills + optional_skills:
        lbl = s.get("label")
        if lbl and lbl not in _seen_labels:
            _seen_labels.add(lbl)
            skill_labels.append(lbl)
    out: dict[str, Any] = {
        "uuid": job_id,
        "originUuid": (
            rd.get("origin_uuid") or rd.get("originUuid") or job_fp_s or job_id
        )
        or None,
        "opportunity_title": meta.get("title") or "Unknown",
        "location": location,
        "city": city,
        "province": province,
        "employer": meta.get("employer"),
        "employment_type": meta.get("employment_type"),
        "salary_text": meta.get("salary"),
        "closing_date": closing_s,
        "posted_date": posted_date,
        "opportunity_isco_occupation_group": isco_group,
        "opportunity_isco_occupation_group_id": isco_group_id,
        "related_occupation_id": (rd.get("related_occupation_id") or isco_group_id)
        or None,
        "contract_type": et,
        "url": meta.get("application_url"),
        "essential_skills": essential_skills,
        "optional_skills": optional_skills,
        "skill_groups_origin_uuids": skill_groups,
        "attributes": attributes,
        # Post-secondary education gate (see app.ranking.retrieval).
        # llm_job_attributes is fully projected, so this subfield is already loaded.
        "requires_post_secondary": attributes.get("requires_post_secondary"),
        # ZQF education annotation (Zambia): classifier_metadata (two naming conventions) or root.
        "zqf_min": meta.get("min_zqf_level")
        or meta.get("zqf_min")
        or rd.get("zqf_min"),
        "zqf_max": meta.get("max_zqf_level")
        or meta.get("zqf_max")
        or rd.get("zqf_max"),
        "zqf_min_label": meta.get("min_zqf_label") or meta.get("zqf_min_label"),
        "zqf_max_label": meta.get("max_zqf_label") or meta.get("zqf_max_label"),
        "opportunity_description": meta.get("job_description")
        or meta.get("description")
        or "",
        "category": category,
        "source_platform": source_platform,
        "skills": skill_labels,
        "onet_work_activities": onet_wa,
    }
    if job_fp_s:
        out["job_fingerprint"] = job_fp_s

    # Passthrough for concat-Gemini cosine (see POST /match_v3); not returned on HTTP envelopes.
    gem_sub = rd.get("concat_skill_embedding_gemini")
    if isinstance(gem_sub, dict) and gem_sub.get("vector_bin") is not None:
        out["concat_skill_embedding_gemini"] = gem_sub
    raw_je = rd.get("job_embedding")
    if isinstance(raw_je, list) and raw_je:
        from app.services.cross_encoder.gemini_embeddings import (
            EMBEDDING_DIM as _gem_concat_dim,
        )

        if len(raw_je) == _gem_concat_dim:
            out["job_embedding"] = raw_je
    # True iff job_embedding has been whitened on the DB side (same concat artifact the code ships).
    # The matching engine consumes such vectors directly; raw vectors (offline, occupations, not-yet-
    # whitened jobs) are whitened in-process. Absent/False => raw (safe default).
    out["job_embedding_whitened"] = (
        ((rd.get("llm_reranker_meta") or {}).get("embedding") or {}).get("whitening")
        or {}
    ).get("enabled") is True
    return out
