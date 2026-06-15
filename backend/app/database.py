import asyncio
import base64
import binascii
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING, IndexModel
from dotenv import load_dotenv

from app.config import (
    JOBS_FIND_USE_PROJECTION,
    JOBS_RETRIEVAL_FILTER,
    JOBS_RETRIEVAL_LIMIT,
    MONGO_JOBS_COLLECTION,
    OCCUPATION_CONCAT_EMBEDDINGS_PATH,
    OCCUPATION_JSON_PATH,
)
from app.schemas import JobsStats, JobListItem

load_dotenv()

# Load from environment
MONGO_URL = os.getenv("MONGO_URL")
DATABASE_NAME = os.getenv("MONGO_DB_NAME")

if not MONGO_URL:
    raise ValueError("MONGO_URL environment variable is not set")

_mongo_sel_ms = int(
    (os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS") or "30000").strip() or "30000"
)
_mongo_max_pool = int((os.getenv("MONGO_MAX_POOL_SIZE") or "50").strip() or "50")
_mongo_min_pool = int((os.getenv("MONGO_MIN_POOL_SIZE") or "0").strip() or "0")
_mongo_client_kwargs: Dict[str, Any] = {
    "serverSelectionTimeoutMS": _mongo_sel_ms,
    "maxPoolSize": max(1, _mongo_max_pool),
}
if _mongo_min_pool > 0:
    _mongo_client_kwargs["minPoolSize"] = _mongo_min_pool


def _looks_like_tls_mongodb(uri: str) -> bool:
    u = uri.lower().strip()
    return (
        "mongodb+srv://" in u
        or "tls=true" in u
        or "tls = true" in u
        or "ssl=true" in u
        or "ssl = true" in u
    )


def _configure_mongodb_tls(kwargs: Dict[str, Any]) -> None:
    """Atlas and other TLS backends need a CA bundle. macOS/Python.org installs often lack one.

    * ``MONGO_TLS_ALLOW_INVALID_CERTIFICATES=1`` — dev-only; skips verification (unsafe).
    * ``MONGO_TLS_CA_FILE=/path.pem`` — explicit CA bundle path.
    * Otherwise, if URI looks TLS and ``certifi`` is installed, use ``certifi.where()``.
      (Typically present as a transitive dep of ``requests`` / ``httpx``.)
    """
    if not MONGO_URL or not _looks_like_tls_mongodb(MONGO_URL):
        return

    allow_invalid = os.getenv(
        "MONGO_TLS_ALLOW_INVALID_CERTIFICATES", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    if allow_invalid:
        kwargs["tlsAllowInvalidCertificates"] = True
        logger.warning(
            "MONGO_TLS_ALLOW_INVALID_CERTIFICATES is set — TLS verification disabled (not for production)."
        )
        return

    ca_explicit = os.getenv("MONGO_TLS_CA_FILE", "").strip()
    if ca_explicit:
        kwargs["tlsCAFile"] = ca_explicit
        return

    try:
        import certifi

        kwargs["tlsCAFile"] = certifi.where()
    except ImportError:
        logger.warning(
            "TLS MongoDB URL detected but certifi not installed — install certifi or set "
            "MONGO_TLS_CA_FILE for certificate verification "
            "(or use MONGO_TLS_ALLOW_INVALID_CERTIFICATES=1 for local dev only)."
        )


_configure_mongodb_tls(_mongo_client_kwargs)


def _mongo_tls_client_options() -> Dict[str, Any]:
    """Extra Motor/PyMongo TLS options from env.

    Atlas (mongodb+srv) uses TLS. On some macOS/Python installs the default CA
    store is empty or incomplete and you get::

        SSL: CERTIFICATE_VERIFY_FAILED / unable to get local issuer certificate

    """
    extra: Dict[str, Any] = {}
    insecure = (os.getenv("MONGO_TLS_INSECURE") or "").strip().lower()
    if insecure in ("1", "true", "yes", "on"):
        extra["tlsAllowInvalidCertificates"] = True
        return extra

    ca_raw = (os.getenv("MONGO_TLS_CA_FILE") or "").strip()
    if not ca_raw:
        return extra
    if ca_raw.lower() == "certifi":
        import certifi

        extra["tlsCAFile"] = certifi.where()
        return extra
    extra["tlsCAFile"] = ca_raw
    return extra


_mongo_client_kwargs.update(_mongo_tls_client_options())

client = AsyncIOMotorClient(MONGO_URL, **_mongo_client_kwargs)
db = client[DATABASE_NAME]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Work-activity enrichment: look up importance/level from occupation taxonomy
# ---------------------------------------------------------------------------
_wa_lookup = None  # {occupation_label_lower: {WA_code: {importance, level}}}
_wa_averages = None  # {WA_code: {importance, level}} — fallback for unmatched


def _load_wa_lookup():
    """Build WA importance/level lookup from the occupation taxonomy JSON.

    Returns (per_occupation_lookup, cross_occupation_averages).
    """
    global _wa_lookup, _wa_averages
    if _wa_lookup is not None:
        return _wa_lookup, _wa_averages

    with open(OCCUPATION_JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    per_occ = {}
    from collections import defaultdict

    sums = defaultdict(lambda: {"imp": 0.0, "lvl": 0.0, "n": 0})

    for entry in raw:
        label = entry.get("occupation", {}).get("preferred_label", "").lower().strip()
        wa_dict = {}
        for w in entry.get("onet_work_activities", []):
            code = w.get("WA_code")
            imp = w.get("WA_Importance", "")
            lvl = w.get("WA_Level", "")
            if code and imp and lvl and imp != "" and lvl != "":
                imp_f, lvl_f = float(imp), float(lvl)
                wa_dict[code] = {"importance": imp_f, "level": lvl_f}
                sums[code]["imp"] += imp_f
                sums[code]["lvl"] += lvl_f
                sums[code]["n"] += 1
        if wa_dict:
            per_occ[label] = wa_dict

    averages = {}
    for code, s in sums.items():
        averages[code] = {
            "importance": round(s["imp"] / s["n"], 2),
            "level": round(s["lvl"] / s["n"], 2),
        }

    _wa_lookup = per_occ
    _wa_averages = averages
    logger.info(
        "Built WA lookup: %d occupations, %d WA codes", len(per_occ), len(averages)
    )
    return per_occ, averages


def _enrich_work_activities(wa_items: list, classified_occupations: list) -> list:
    """Attach importance/level to a job's work activity items.

    Strategy:
      1. If the job has a classified occupation that matches the taxonomy → use
         that occupation's importance/level per WA code.
      2. Otherwise → use the cross-occupation average for each WA code.
    """
    per_occ, averages = _load_wa_lookup()

    # Try to find a matching occupation
    occ_wa = None
    for co in classified_occupations:
        label = (co.get("label") or "").lower().strip()
        if label in per_occ:
            occ_wa = per_occ[label]
            break

    enriched = []
    for item in wa_items:
        code = item.get("id")
        if not code:
            continue
        if occ_wa and code in occ_wa:
            vals = occ_wa[code]
        elif code in averages:
            vals = averages[code]
        else:
            vals = {"importance": 3.5, "level": 3.5}

        enriched.append(
            {
                "WA_code": code,
                "WA_label": item.get("name", ""),
                "WA_Importance": vals["importance"],
                "WA_Level": vals["level"],
            }
        )

    return enriched


def get_database():
    return db


# Only jobs intended to be shown / matched; keeps Mongo transfers and Python work small.
# Recommended index: { "is_active": 1 } (plus compounds if you add more filters)
RANKED_JOBS_ACTIVE_FILTER: Dict[str, Any] = {"is_active": True}

# Ranked / enriched job docs: listing fields on classifier_metadata (see build_job_dict_from_ranked).
_M_CITY = "classifier_metadata.city"
_M_COUNTY = "classifier_metadata.county"
_M_PROVINCE = "classifier_metadata.province"

# Inclusion projection for job find (must stay aligned with build_job_dict_from_ranked).
RANKED_JOB_FIND_PROJECTION: Dict[str, int] = {
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


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


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


def _remote_substring_ors() -> List[Dict[str, Any]]:
    r = "remote"
    return [
        {_M_CITY: {"$regex": r, "$options": "i"}},
        {_M_COUNTY: {"$regex": r, "$options": "i"}},
    ]


def _field_contains_substr_regex(
    field: str, needle_cf: str
) -> Optional[Dict[str, Any]]:
    if not needle_cf:
        return None
    return {field: {"$regex": re.escape(needle_cf), "$options": "i"}}


def _expr_haystack_contains_mongo_subfield(
    haystack_casefold: str, dollar_field: str
) -> Optional[Dict[str, Any]]:
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


def _location_or_clauses_for_one_user(user: dict) -> List[Dict[str, Any]]:
    """Superset of matching_service._job_matches_user_location, on classifier_metadata fields."""
    uc = _norm_loc_value(user.get("city"))
    up = _norm_loc_value(user.get("province"))
    ors: List[Dict[str, Any]] = list(_remote_substring_ors())
    if not uc or not up:
        return ors
    for field in (_M_CITY, _M_COUNTY, _M_PROVINCE):
        f_c = _field_contains_substr_regex(field, uc)
        if f_c is not None:
            ors.append(f_c)
        f_p = _field_contains_substr_regex(field, up)
        if f_p is not None:
            ors.append(f_p)
    for hay, fpath in (
        (uc, "$classifier_metadata.city"),
        (uc, "$classifier_metadata.county"),
        (uc, "$classifier_metadata.province"),
        (up, "$classifier_metadata.city"),
        (up, "$classifier_metadata.county"),
        (up, "$classifier_metadata.province"),
    ):
        ex = _expr_haystack_contains_mongo_subfield(hay, fpath)
        if ex is not None:
            ors.append(ex)
    return ors


def build_mongo_filter_active_and_location(
    users: Sequence[dict],
) -> Optional[Dict[str, Any]]:
    """
    is_active and (OR of all per-user location clauses). None if the caller should
    use active-only (no user context or empty list).
    """
    if not users:
        return None
    parts: List[Dict[str, Any]] = []
    for u in users:
        parts.extend(_location_or_clauses_for_one_user(u))
    if not parts:
        return RANKED_JOBS_ACTIVE_FILTER
    return {"$and": [RANKED_JOBS_ACTIVE_FILTER, {"$or": parts}]}


def build_job_dict_from_ranked(rd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        skill_groups: List[str] = []
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
    skill_labels: List[str] = []
    _seen_labels: set = set()
    for s in essential_skills + optional_skills:
        lbl = s.get("label")
        if lbl and lbl not in _seen_labels:
            _seen_labels.add(lbl)
            skill_labels.append(lbl)
    out: Dict[str, Any] = {
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
        # Post-secondary education gate (see app.services.education_eligibility).
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
    return out


async def get_all_jobs(users: Optional[Sequence[dict]] = None):
    """Load jobs from ``MONGO_JOBS_COLLECTION`` (enriched rows: onet WA + skill groups on document)."""
    out, _ = await get_all_jobs_with_timing(users=users)
    return out


async def get_all_jobs_with_timing(users: Optional[Sequence[dict]] = None):
    """Same as get_all_jobs; returns (jobs, timing_dict) for observability.

    Reads from ``MONGO_JOBS_COLLECTION`` (default ``RankedJobsEnriched``). Only documents with
    top-level ``is_active`` equal to true are loaded.

    If ``JOBS_RETRIEVAL_FILTER`` is true and ``users`` is non-empty, the query also ORs
    per-user location clauses (superset of ``_job_matches_user_location`` on
    ``classifier_metadata``), sorts by ``_id`` descending, and applies ``JOBS_RETRIEVAL_LIMIT``.
    With no ``users`` (or filter off), behavior matches the previous active-only ``find`` with no
    sort or cap.

    timing_dict keys:
      mongo_ranked_find_ms, python_build_jobs_ms, n_ranked_raw, n_jobs, n_skipped_inactive, total_ms,
      jobs_retrieval_filter_applied, jobs_find_use_projection
    """
    t_total = time.perf_counter()
    t0 = time.perf_counter()
    filt: Dict[str, Any] = RANKED_JOBS_ACTIVE_FILTER
    retrieval_applied = False
    if JOBS_RETRIEVAL_FILTER and users:
        built = build_mongo_filter_active_and_location(users)
        if built is not None and built != RANKED_JOBS_ACTIVE_FILTER:
            filt = built
            retrieval_applied = True
    col = db[MONGO_JOBS_COLLECTION]
    if JOBS_FIND_USE_PROJECTION:
        cursor = col.find(filt, RANKED_JOB_FIND_PROJECTION)
    else:
        cursor = col.find(filt)
    if retrieval_applied:
        cursor = cursor.sort([("_id", -1)])
        if JOBS_RETRIEVAL_LIMIT > 0:
            cursor = cursor.limit(JOBS_RETRIEVAL_LIMIT)
    ranked_docs = [d async for d in cursor]
    mongo_ranked_find_ms = _ms(t0)

    t0 = time.perf_counter()
    jobs: List[dict] = []
    skipped = 0
    for rd in ranked_docs:
        built = build_job_dict_from_ranked(rd)
        if built is None:
            skipped += 1
            continue
        jobs.append(built)

    python_build_jobs_ms = _ms(t0)
    total_ms = _ms(t_total)
    logger.info(
        "Loaded %d active jobs from %s (matched=%d, skipped_in_build=%d)",
        len(jobs),
        MONGO_JOBS_COLLECTION,
        len(ranked_docs),
        skipped,
    )
    return jobs, {
        "mongo_ranked_find_ms": mongo_ranked_find_ms,
        "python_build_jobs_ms": python_build_jobs_ms,
        "n_ranked_raw": len(ranked_docs),
        "n_jobs": len(jobs),
        "n_skipped_inactive": skipped,
        "get_all_jobs_total_ms": total_ms,
        "jobs_retrieval_filter_applied": retrieval_applied,
        "jobs_find_use_projection": JOBS_FIND_USE_PROJECTION,
    }


class InvalidCursor(ValueError):
    """Raised when a /jobs pagination cursor cannot be decoded into a Mongo _id."""


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
    search: Optional[str] = None,
    category: Optional[str] = None,
    employment_type: Optional[str] = None,
    location: Optional[str] = None,
    skills: Optional[str] = None,
    days: Optional[int] = None,
) -> Dict[str, Any]:
    """Mongo filter for the /jobs browse endpoint, composed with ``is_active``.

    Every clause is optional; all supplied clauses are AND-ed together (a job must match
    all of them). Field paths target the raw ``classifier_metadata`` / ``llm_classified_skills``
    document so the filter is applied by Mongo before shaping. ``category`` and ``location``
    span several candidate field names because the stored data is not uniform.
    """
    clauses: List[Dict[str, Any]] = [RANKED_JOBS_ACTIVE_FILTER]

    if search and search.strip():
        clauses.append(
            {"classifier_metadata.title": {"$regex": re.escape(search.strip()), "$options": "i"}}
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
            datetime.now(timezone.utc) - timedelta(days=int(days))
        ).date().isoformat()
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
        [("is_active", ASCENDING), ("classifier_metadata.isco_occupation_group", ASCENDING)],
        name="is_active_isco_group",
    ),
    IndexModel(
        [("is_active", ASCENDING), ("classifier_metadata.source_platform", ASCENDING)],
        name="is_active_source_platform",
    ),
]


async def ensure_jobs_indexes() -> List[str]:
    """Create (idempotently) the indexes the jobs queries need, on ``MONGO_JOBS_COLLECTION``.

    ``create_indexes`` is a no-op for indexes that already exist with the same spec, so this is
    safe to run on every startup. Returns the list of ensured index names.
    """
    t0 = time.perf_counter()
    col = db[MONGO_JOBS_COLLECTION]
    created = await col.create_indexes(JOBS_INDEX_MODELS)
    logger.info(
        "Ensured %d indexes on %s in %.2f ms: %s",
        len(created),
        MONGO_JOBS_COLLECTION,
        _ms(t0),
        created,
    )
    return created


async def get_jobs_stats() -> JobsStats:
    """Aggregate counts over the active jobs catalog for the /jobs/stats endpoint.

    ``sectors`` counts distinct, case-insensitively-deduplicated categories (falling back
    to the ISCO occupation group label, matching ``build_job_dict_from_ranked``);
    ``platforms`` counts distinct source platforms.
    """
    col = db[MONGO_JOBS_COLLECTION]
    total = await col.count_documents(RANKED_JOBS_ACTIVE_FILTER)

    raw_categories = await col.distinct("classifier_metadata.category", RANKED_JOBS_ACTIVE_FILTER)
    if not raw_categories:
        raw_categories = await col.distinct(
            "classifier_metadata.isco_occupation_group", RANKED_JOBS_ACTIVE_FILTER
        )
    sectors = len({str(c).strip().lower() for c in raw_categories if str(c).strip()})

    platforms_set: set = set()
    for field in (
        "classifier_metadata.source_platform",
        "classifier_metadata.source",
        "classifier_metadata.platform",
    ):
        for p in await col.distinct(field, RANKED_JOBS_ACTIVE_FILTER):
            if str(p).strip():
                platforms_set.add(str(p).strip().lower())

    return JobsStats(total=total, sectors=sectors, platforms=len(platforms_set))


async def get_jobs_page_with_timing(
    cursor: Optional[str] = None,
    limit: int = 20,
    *,
    search: Optional[str] = None,
    category: Optional[str] = None,
    employment_type: Optional[str] = None,
    location: Optional[str] = None,
    skills: Optional[str] = None,
    days: Optional[int] = None,
    include_total: bool = False,
) -> Tuple[List[JobListItem], Optional[str], Optional[int], Dict[str, Any]]:
    """Cursor-paginated, filterable browse over active jobs in ``MONGO_JOBS_COLLECTION``.

    Reads from the same collection and through the same ``build_job_dict_from_ranked``
    shaping as ``get_all_jobs_with_timing`` (the matched-jobs data source), so browse
    and match return identical job objects.

    Pagination is keyset-based on ``_id`` descending (newest first, stable under inserts):
    ``cursor`` is an opaque token wrapping the last ``_id`` of the previous page, and the
    next page is ``_id < cursor_id``. One extra document is fetched to compute ``has_more``
    and the next cursor without a second round-trip. Optional ``search``/``category``/
    ``employment_type``/``location``/``skills``/``days`` filters narrow the catalog
    (see ``build_jobs_browse_filter``); when ``include_total`` is set, the total count of
    the filtered catalog (ignoring the cursor) is returned for client-side pagination UIs.

    Returns ``(jobs, next_cursor, total, timing)``. ``next_cursor`` is ``None`` on the last
    page; ``total`` is ``None`` unless ``include_total`` is set. Raises ``InvalidCursor`` if
    ``cursor`` is malformed.
    """
    t_total = time.perf_counter()
    limit = max(1, int(limit))

    base_filt = build_jobs_browse_filter(
        search=search,
        category=category,
        employment_type=employment_type,
        location=location,
        skills=skills,
        days=days,
    )
    filt: Dict[str, Any] = dict(base_filt)
    if cursor:
        # Compose the keyset seek with the (possibly compound) filter without clobbering it.
        filt = {"$and": [base_filt, {"_id": {"$lt": _decode_jobs_cursor(cursor)}}]}

    col = db[MONGO_JOBS_COLLECTION]
    projection = RANKED_JOB_FIND_PROJECTION if JOBS_FIND_USE_PROJECTION else None
    # Fetch limit+1 so we can tell whether another page exists.
    t0 = time.perf_counter()
    query = col.find(filt, projection) if projection else col.find(filt)
    query = query.sort(JOBS_PAGE_SORT).limit(limit + 1)
    raw_docs = [d async for d in query]
    mongo_find_ms = _ms(t0)

    total: Optional[int] = None
    if include_total:
        total = await col.count_documents(base_filt)

    has_more = len(raw_docs) > limit
    page_docs = raw_docs[:limit]

    t0 = time.perf_counter()
    jobs: List[JobListItem] = []
    skipped = 0
    for rd in page_docs:
        built = build_job_dict_from_ranked(rd)
        if built is None:
            skipped += 1
            continue
        jobs.append(JobListItem(
            uuid=built.get("uuid"),
            originUuid=built.get("originUuid"),
            url=built.get("url"),
            opportunity_title=built.get("opportunity_title", "No title"),
            opportunity_isco_occupation_group=built.get("opportunity_isco_occupation_group"),
            opportunity_isco_occupation_group_id=built.get("opportunity_isco_occupation_group_id"),
            related_occupation_id=built.get("related_occupation_id"),
            location=built.get("location"),
            city=built.get("city"),
            province=built.get("province"),
            employer=built.get("employer"),
            employment_type=built.get("employment_type"),
            contract_type=built.get("contract_type"),
            salary_text=built.get("salary_text"),
            closing_date=built.get("closing_date"),
            posted_date=built.get("posted_date"),
            opportunity_description=built.get("opportunity_description"),
            # Consumer-contract fields (Compass jobs board)=built.get("# Consumer-contract fields (Compass jobs board),
            # posting was scraped from, and the flat list of skill labels for this opportunity.
            category=built.get("category"),
            source_platform=built.get("source_platform"),
            skills=built.get("skills", [])
        ))
    python_build_ms = _ms(t0)

    next_cursor = (
        _encode_jobs_cursor(page_docs[-1]["_id"]) if has_more and page_docs else None
    )

    return (
        jobs,
        next_cursor,
        total,
        {
            "mongo_find_ms": mongo_find_ms,
            "python_build_jobs_ms": python_build_ms,
            "n_page_raw": len(page_docs),
            "n_jobs": len(jobs),
            "n_skipped_inactive": skipped,
            "has_more": has_more,
            "limit": limit,
            "total": total,
            "get_jobs_page_total_ms": _ms(t_total),
        },
    )


_cached_occupations = None
_cached_occ_embeddings = None  # {occupation_code: np.ndarray(float32, EMBEDDING_DIM)}


def _occ_skill_pairs(uuids: list, labels: list) -> List[Dict[str, str]]:
    """Zip occupation skill uuids with their labels (from the occupation JSON; '' if absent)."""
    pairs: List[Dict[str, str]] = []
    for i, u in enumerate(uuids):
        lab = labels[i] if i < len(labels) else ""
        pairs.append({"id": str(u), "label": str(lab) if lab else ""})
    return pairs


def _load_occupation_embeddings() -> Dict[str, Any]:
    """Lazy/cached load of the committed occupation concat-embeddings NPZ (code -> vector).

    Returns {} (with a warning) if the artifact is missing/unreadable, so occupations are
    simply skipped by the /match_v4 retrieval rather than crashing the request.
    """
    global _cached_occ_embeddings
    if _cached_occ_embeddings is not None:
        return _cached_occ_embeddings
    out: Dict[str, Any] = {}
    try:
        import numpy as np

        with np.load(OCCUPATION_CONCAT_EMBEDDINGS_PATH, allow_pickle=True) as data:
            codes = [str(c) for c in data["codes"].tolist()]
            vectors = np.asarray(data["vectors"], dtype=np.float32)
        for code, vec in zip(codes, vectors):
            out[code] = np.ascontiguousarray(vec, dtype=np.float32)
        logger.info(
            "Loaded %d occupation concat embeddings from %s",
            len(out),
            OCCUPATION_CONCAT_EMBEDDINGS_PATH,
        )
    except FileNotFoundError:
        logger.warning(
            "Occupation embeddings NPZ not found at %s; /match_v4 occupations will be skipped. "
            "Build it via `python -m app.services.cross_encoder.embed_occupations`.",
            OCCUPATION_CONCAT_EMBEDDINGS_PATH,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "Failed to load occupation embeddings (%s): %s; occupations skipped.",
            OCCUPATION_CONCAT_EMBEDDINGS_PATH,
            e,
        )
    _cached_occ_embeddings = out
    return out


def attach_occupation_embeddings(occupations: Sequence[dict]) -> List[dict]:
    """Return occupation dicts with ``job_embedding`` (shared np.ndarray) attached by code.

    Vector is shared across all county-rows of the same occupation code (skills are identical),
    so memory stays at one array per code. Rows with no matching embedding are returned
    unchanged (the v4 engine skips items without a stage-1 vector).
    """
    emb = _load_occupation_embeddings()
    if not emb:
        return list(occupations)
    out: List[dict] = []
    for occ in occupations:
        vec = emb.get(str(occ.get("originUuid") or ""))
        if vec is None:
            out.append(occ)
        else:
            o = dict(occ)
            o["job_embedding"] = vec
            out.append(o)
    return out


async def get_all_occupations():
    out, _ = await get_all_occupations_with_timing()
    return out


async def get_all_occupations_with_timing():
    """Load occupations; returns (flat_list, timing_dict).

    On cache hit, occupation_file_read_ms is 0 and occupation_cache_hit is True.
    """
    global _cached_occupations
    t_total = time.perf_counter()

    if _cached_occupations is not None:
        total_ms = _ms(t_total)
        return _cached_occupations, {
            "occupation_cache_hit": True,
            "occupation_file_read_ms": 0.0,
            "occupation_json_parse_and_flatten_ms": 0.0,
            "n_occupation_rows": len(_cached_occupations),
            "get_all_occupations_total_ms": total_ms,
        }

    try:
        t0 = time.perf_counter()
        with open(OCCUPATION_JSON_PATH, "r", encoding="utf-8") as f:
            raw_occupations = json.load(f)
        file_read_and_json_ms = _ms(t0)

        t1 = time.perf_counter()
        flattened = []
        for entry in raw_occupations:
            occ = entry.get("occupation", {})
            skills = entry.get("skills", {})
            ess_block = (
                skills.get("essential", {})
                if isinstance(skills.get("essential"), dict)
                else {}
            )
            opt_block = (
                skills.get("optional", {})
                if isinstance(skills.get("optional"), dict)
                else {}
            )
            ess_uuids = ess_block.get("uuids", []) or []
            opt_uuids = opt_block.get("uuids", []) or []
            ess_labels = ess_block.get("labels", []) or []
            opt_labels = opt_block.get("labels", []) or []
            counties = entry.get("counties_data", [])

            code = occ.get("code", "")
            label = occ.get("preferred_label", "Unknown")
            description = occ.get("description", "")

            # Post-secondary education gate (see app.services.education_eligibility):
            # occupation-level flag, applied to all of this occupation's county rows.
            requires_post_secondary = occ.get("requires_post_secondary")
            if requires_post_secondary is None:
                requires_post_secondary = entry.get("requires_post_secondary")

            raw_wa = entry.get("onet_work_activities", [])
            onet_wa = []
            for w in raw_wa:
                wc = w.get("WA_code")
                imp = w.get("WA_Importance", "")
                lvl = w.get("WA_Level", "")
                if wc and imp != "" and lvl != "":
                    onet_wa.append(
                        {
                            "WA_code": wc,
                            "WA_label": w.get("WA_label", ""),
                            "WA_Importance": float(imp),
                            "WA_Level": float(lvl),
                        }
                    )

            if not counties:
                counties = [{"county": "", "job_attributes": {}}]

            for cd in counties:
                county = cd.get("county", "")
                job_attrs = cd.get("job_attributes", {})
                attrs_raw = job_attrs.get("attributes", [])
                attributes = {}
                if isinstance(attrs_raw, list):
                    for a in attrs_raw:
                        name = a.get("attribute_name")
                        val = a.get("selected_level_id")
                        if name and val:
                            attributes[name] = val
                elif isinstance(attrs_raw, dict):
                    attributes = attrs_raw

                # Demand label so DemandScorer can read attributes["expected_demand"]
                # (engine-agnostic; powers score_breakdown.demand_* on /match_v4).
                expected_demand = (cd.get("labor_demand") or {}).get("expected_demand")
                if expected_demand:
                    attributes = {**attributes, "expected_demand": expected_demand}

                # Wrap skills in the same {id, label} shape used by job dicts. Labels come
                # from the occupation JSON (skills.*.labels) when present, else empty; gap
                # analysis reads id directly without going through label resolution.
                flattened.append(
                    {
                        "uuid": f"{code}_{county}" if county else code,
                        "originUuid": code,
                        "occupation_label": label,
                        "preferredLabel": label,
                        "description": description,
                        "location": county,
                        "city": county,
                        "province": county,
                        "essential_skills": _occ_skill_pairs(ess_uuids, ess_labels),
                        "optional_skills": _occ_skill_pairs(opt_uuids, opt_labels),
                        "skill_groups_origin_uuids": [],
                        "attributes": attributes,
                        "requires_post_secondary": requires_post_secondary,
                        "onet_work_activities": onet_wa,
                        # Occupation-specific tasks (sparse in source); formatter falls back to
                        # O*NET WA labels when absent. See match_v4_formatting._typical_tasks.
                        "included_tasks": occ.get("included_tasks") or "",
                    }
                )

        flatten_ms = _ms(t1)
        _cached_occupations = flattened
        total_ms = _ms(t_total)
        logger.info(
            "Loaded %d occupation-county items from %d raw occupations",
            len(flattened),
            len(raw_occupations),
        )
        return _cached_occupations, {
            "occupation_cache_hit": False,
            "occupation_file_read_ms": file_read_and_json_ms,
            "occupation_json_parse_and_flatten_ms": flatten_ms,
            "n_occupation_rows": len(flattened),
            "n_raw_occupation_entries": len(raw_occupations),
            "get_all_occupations_total_ms": total_ms,
        }
    except Exception as e:
        logger.exception(e)
        raise RuntimeError(f"Failed to load occupations: {e}")


async def close_mongo_connection():
    client.close()


def _env_warmup_flag(name: str, default: bool = True) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


async def warmup_on_startup() -> None:
    """Ping Mongo and preload heavy one-time caches. Called from FastAPI lifespan (not per /match).

    Toggle with ``MONGO_WARMUP_ON_STARTUP``, ``WARMUP_OCCUPATIONS_CACHE``, ``WARMUP_WA_LOOKUP``
    (see ``.env.example``). WA lookup defaults to off — jobs use pre-enriched ``onet_work_activities``.
    """
    if _env_warmup_flag("MONGO_WARMUP_ON_STARTUP", True):
        t0 = time.perf_counter()
        try:
            await db.command("ping")
            logger.info("Mongo warmup: ping ok (%.2f ms)", _ms(t0))
        except Exception:
            logger.exception("Mongo warmup: ping failed")
    else:
        logger.info("Mongo warmup skipped (MONGO_WARMUP_ON_STARTUP=0)")

    if _env_warmup_flag("ENSURE_INDEXES_ON_STARTUP", True):
        try:
            await ensure_jobs_indexes()
        except Exception:
            logger.exception("Ensuring jobs indexes failed")
    else:
        logger.info("Index creation skipped (ENSURE_INDEXES_ON_STARTUP=0)")

    if _env_warmup_flag("WARMUP_OCCUPATIONS_CACHE", True):
        t0 = time.perf_counter()
        try:
            await get_all_occupations()
            logger.info("Occupation cache warmup: ok (%.2f ms)", _ms(t0))
        except Exception:
            logger.exception("Occupation cache warmup failed")
    else:
        logger.info("Occupation cache warmup skipped (WARMUP_OCCUPATIONS_CACHE=0)")

    if _env_warmup_flag("WARMUP_WA_LOOKUP", False):
        try:
            _load_wa_lookup()
            logger.info("WA taxonomy lookup: built at startup")
        except Exception:
            logger.exception("WA lookup warmup failed")
    else:
        logger.info(
            "WA lookup warmup skipped (enriched jobs carry onet_work_activities; set WARMUP_WA_LOOKUP=1 to force)"
        )

    if _env_warmup_flag("WARMUP_MATCH_V3_MODELS", False):
        try:
            from app.services.match_concat_gemini_ce_service import (
                preload_match_v3_models,
            )

            t0 = time.perf_counter()
            timings = await asyncio.to_thread(preload_match_v3_models)
            logger.info(
                "/match_v3 model warmup: ok (total %.2f ms; matcher %.2f ms, cross-encoder %.2f ms)",
                _ms(t0),
                timings.get("cosine_skill_matcher_ms", 0.0),
                timings.get("cross_encoder_ms", 0.0),
            )
        except Exception:
            logger.exception("/match_v3 model warmup failed")
    else:
        logger.info(
            "/match_v3 model warmup skipped (set WARMUP_MATCH_V3_MODELS=1 to preload CosineSkillMatcher + CrossEncoder at startup)"
        )
