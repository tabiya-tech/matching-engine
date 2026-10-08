"""Mongo job documents: filters, projection, cursor codec, index models and the flat job dict.

Everything here is pure (no database access); ``JobsRepository`` composes it into queries.
"""

import base64
import binascii
import logging
import re
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ASCENDING, DESCENDING, IndexModel

from app.config import LOCATION_HUB_CHAINS_PATH, LOCATION_TIER_ENABLED
from app.jobs.errors import InvalidCursor

logger = logging.getLogger(__name__)

# Only jobs intended to be shown / matched; keeps Mongo transfers and Python work small.
# Recommended index: { "is_active": 1 } (plus compounds if you add more filters)
RANKED_JOBS_ACTIVE_FILTER: dict[str, Any] = {"is_active": True}

# Ranked / enriched job docs: listing fields on classifier_metadata (see build_job_dict_from_ranked).
_M_CITY = "classifier_metadata.city"
_M_COUNTY = "classifier_metadata.county"
_M_PROVINCE = "classifier_metadata.province"

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


def _norm_loc_value(v: Any) -> str:
    """Casefold + strip, aligned with matching_service._norm for city/province."""
    if v is None:
        return ""
    s = str(v).strip()
    return s.casefold() if s else ""


def _remote_substring_ors() -> list[dict[str, Any]]:
    r = "remote"
    return [
        {_M_CITY: {"$regex": r, "$options": "i"}},
        {_M_COUNTY: {"$regex": r, "$options": "i"}},
    ]


def _field_contains_substr_regex(field: str, needle_cf: str) -> dict[str, Any] | None:
    if not needle_cf:
        return None
    return {field: {"$regex": re.escape(needle_cf), "$options": "i"}}


def _expr_haystack_contains_mongo_subfield(
    haystack_casefold: str, dollar_field: str
) -> dict[str, Any] | None:
    """True when haystack (user string) contains the job’s city/county (Python: job in user).

    Requires a non-empty job field: MongoDB matches an empty substring at index 0 for
    ``$indexOfCP``, which would incorrectly match every document if city/county were missing.
    """
    if not haystack_casefold:
        return None
    needle = {"$ifNull": [{"$toLower": dollar_field}, ""]}
    return {
        "$expr": {
            "$and": [
                {"$gt": [{"$strLenCP": needle}, 0]},
                {"$gte": [{"$indexOfCP": [haystack_casefold, needle]}, 0]},
            ]
        }
    }


def _location_or_clauses_for_one_user(user: dict) -> list[dict[str, Any]]:
    """Superset of matching_service._job_matches_user_location, on classifier_metadata fields.

    By default the needle set is the user's {city, province}. When LOCATION_TIER_ENABLED (urban-pull
    Part A), it is widened to also include the user's hub-chain regions (regional + national hub) so
    hub jobs become candidates in the pool; the per-user soft tier re-rank (MatchingService)
    then keeps local jobs preferred. Strictly additive — remote clauses always come first.
    """
    uc = _norm_loc_value(user.get("city"))
    up = _norm_loc_value(user.get("province"))
    ors: list[dict[str, Any]] = list(_remote_substring_ors())
    if not uc or not up:
        return ors
    needles = {uc, up}
    if LOCATION_TIER_ENABLED:
        from app.ranking.location import load_hub_chains

        hc = load_hub_chains(LOCATION_HUB_CHAINS_PATH)
        if hc is not None:
            needles.update(hc.chain_for(up))  # local + regional + national hub regions
    for needle in sorted(n for n in needles if n):
        for field in (_M_CITY, _M_COUNTY, _M_PROVINCE):
            f = _field_contains_substr_regex(field, needle)
            if f is not None:
                ors.append(f)
        for fpath in (
            "$classifier_metadata.city",
            "$classifier_metadata.county",
            "$classifier_metadata.province",
        ):
            ex = _expr_haystack_contains_mongo_subfield(needle, fpath)
            if ex is not None:
                ors.append(ex)
    return ors


def build_mongo_filter_active_and_location(
    users: Sequence[dict],
) -> dict[str, Any] | None:
    """
    is_active and (OR of all per-user location clauses). None if the caller should
    use active-only (no user context or empty list).
    """
    if not users:
        return None
    parts: list[dict[str, Any]] = []
    for u in users:
        parts.extend(_location_or_clauses_for_one_user(u))
    if not parts:
        return RANKED_JOBS_ACTIVE_FILTER
    return {"$and": [RANKED_JOBS_ACTIVE_FILTER, {"$or": parts}]}


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
        # Post-secondary education gate (see app.ranking.education).
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
        from app.clients.gemini_embedding_client import (
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


# GET /jobs is sorted by ``_id`` descending (newest first). ``_id`` is always indexed, so the
# keyset seek stays fast without a dedicated sort index.
JOBS_PAGE_SORT = [("_id", -1)]


def _encode_jobs_cursor(object_id: ObjectId) -> str:
    """Opaque, URL-safe cursor wrapping a Mongo ``_id`` (the last item on the page)."""
    return base64.urlsafe_b64encode(str(object_id).encode("ascii")).decode("ascii")


def _decode_jobs_cursor(cursor: str) -> ObjectId:
    """Inverse of _encode_jobs_cursor. Raises InvalidCursor on any malformed input."""
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("ascii")
        return ObjectId(raw)
    except (ValueError, InvalidId, binascii.Error, UnicodeDecodeError) as e:
        raise InvalidCursor(f"invalid cursor: {cursor!r}") from e


def build_jobs_browse_filter(
    *,
    search: str | None = None,
    category: str | None = None,
    employment_type: str | None = None,
    location: str | None = None,
    skills: str | None = None,
    days: int | None = None,
) -> dict[str, Any]:
    """Mongo filter for the /jobs browse endpoint, composed with ``is_active``.

    Every clause is optional; all supplied clauses are AND-ed together (a job must match
    all of them). Field paths target the raw ``classifier_metadata`` / ``llm_classified_skills``
    document so the filter is applied by Mongo before shaping. ``category`` and ``location``
    span several candidate field names because the stored data is not uniform.
    """
    clauses: list[dict[str, Any]] = [RANKED_JOBS_ACTIVE_FILTER]

    if search and search.strip():
        clauses.append(
            {
                "classifier_metadata.title": {
                    "$regex": re.escape(search.strip()),
                    "$options": "i",
                }
            }
        )
    if category and category.strip():
        rx = {"$regex": re.escape(category.strip()), "$options": "i"}
        clauses.append(
            {
                "$or": [
                    {"classifier_metadata.category": rx},
                    {"classifier_metadata.sector": rx},
                    {"classifier_metadata.isco_occupation_group": rx},
                ]
            }
        )
    if employment_type and employment_type.strip():
        clauses.append({"classifier_metadata.employment_type": employment_type.strip()})
    if location and location.strip():
        rx = {"$regex": re.escape(location.strip()), "$options": "i"}
        clauses.append({"$or": [{_M_CITY: rx}, {_M_COUNTY: rx}, {_M_PROVINCE: rx}]})
    if skills and skills.strip():
        rx = {"$regex": re.escape(skills.strip()), "$options": "i"}
        clauses.append(
            {
                "$or": [
                    {"llm_classified_skills.essential.label": rx},
                    {"llm_classified_skills.optional.label": rx},
                ]
            }
        )
    if days is not None:
        cutoff = (
            (datetime.now(timezone.utc) - timedelta(days=int(days))).date().isoformat()
        )
        gte = {"$gte": cutoff}
        clauses.append(
            {
                "$or": [
                    {"classifier_metadata.posted_date": gte},
                    {"classifier_metadata.date_posted": gte},
                ]
            }
        )

    if len(clauses) == 1:
        return dict(RANKED_JOBS_ACTIVE_FILTER)
    return {"$and": clauses}


# Indexes on MONGO_JOBS_COLLECTION that the /jobs (browse, stats) and /match queries rely on.
#
# MongoDB uses ONE index per query (index intersection is rarely chosen by the planner, and never
# when a sort is present), so these are not combined at query time — each is sized for a specific
# access pattern:
#
#   * {is_active: 1, _id: -1}  — the workhorse. The browse endpoint always filters is_active, sorts
#     by _id desc, and seeks with a keyset (_id < cursor); this one index serves the equality, the
#     sort, AND the cursor range together (also covers count_documents(is_active...) and the
#     active-jobs load behind /match). Without it every such call is a full scan + in-memory sort.
#   * {is_active: 1, employment_type: 1, _id: -1}  — the only exact-match browse filter. The trailing
#     _id key lets an employment_type-filtered browse use the index for the equality AND the sort.
#   * {is_active: 1, <category|isco_group|source_platform>: 1}  — back the distinct() calls in
#     /jobs/stats (used by those, not by the sorted browse query).
#
# Substring/regex filters (title, category, location, skills) cannot use a B-tree index in any
# combination — they are always applied as residual filters. Speeding those up further would need
# an Atlas Search / ``$text`` index plus a query change.
JOBS_INDEX_MODELS = [
    IndexModel([("is_active", ASCENDING), ("_id", DESCENDING)], name="is_active_-_id"),
    IndexModel(
        [
            ("is_active", ASCENDING),
            ("classifier_metadata.employment_type", ASCENDING),
            ("_id", DESCENDING),
        ],
        name="is_active_employment_type_-_id",
    ),
    IndexModel(
        [("is_active", ASCENDING), ("classifier_metadata.category", ASCENDING)],
        name="is_active_category",
    ),
    IndexModel(
        [
            ("is_active", ASCENDING),
            ("classifier_metadata.isco_occupation_group", ASCENDING),
        ],
        name="is_active_isco_group",
    ),
    IndexModel(
        [("is_active", ASCENDING), ("classifier_metadata.source_platform", ASCENDING)],
        name="is_active_source_platform",
    ),
]
