import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

from app.config import (
    JOBS_FIND_USE_PROJECTION,
    JOBS_RETRIEVAL_FILTER,
    JOBS_RETRIEVAL_LIMIT,
    MONGO_JOBS_COLLECTION,
    OCCUPATION_JSON_PATH,
)

load_dotenv()

# Load from environment
MONGO_URL = os.getenv("MONGO_URL")
DATABASE_NAME = os.getenv("MONGO_DB_NAME")

if not MONGO_URL:
    raise ValueError("MONGO_URL environment variable is not set")

_mongo_sel_ms = int((os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS") or "30000").strip() or "30000")
_mongo_max_pool = int((os.getenv("MONGO_MAX_POOL_SIZE") or "50").strip() or "50")
_mongo_min_pool = int((os.getenv("MONGO_MIN_POOL_SIZE") or "0").strip() or "0")
_mongo_client_kwargs: Dict[str, Any] = {
    "serverSelectionTimeoutMS": _mongo_sel_ms,
    "maxPoolSize": max(1, _mongo_max_pool),
}
if _mongo_min_pool > 0:
    _mongo_client_kwargs["minPoolSize"] = _mongo_min_pool


def _mongo_tls_client_options() -> Dict[str, Any]:
    """Extra Motor/PyMongo TLS options from env.

    Atlas (mongodb+srv) uses TLS. On some macOS/Python installs the default CA
    store is empty or incomplete and you get::

        SSL: CERTIFICATE_VERIFY_FAILED / unable to get local issuer certificate

    Fix (recommended): set ``MONGO_TLS_CA_FILE=certifi`` to use Mozilla's CA bundle via certifi.

    Escape hatch (local debug only): ``MONGO_TLS_INSECURE=1`` skips certificate verification —
    never use in production.
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
    logger.info("Built WA lookup: %d occupations, %d WA codes", len(per_occ), len(averages))
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

        enriched.append({
            "WA_code": code,
            "WA_label": item.get("name", ""),
            "WA_Importance": vals["importance"],
            "WA_Level": vals["level"],
        })

    return enriched


def get_database():
    return db


# Only jobs intended to be shown / matched; keeps Mongo transfers and Python work small.
# Recommended index: { "is_active": 1 } (plus compounds if you add more filters)
RANKED_JOBS_ACTIVE_FILTER: Dict[str, Any] = {"is_active": True}

# Ranked / enriched job docs: listing fields on classifier_metadata (see build_job_dict_from_ranked).
_M_CITY = "classifier_metadata.city"
_M_COUNTY = "classifier_metadata.county"

# Inclusion projection for job find (must stay aligned with build_job_dict_from_ranked).
RANKED_JOB_FIND_PROJECTION: Dict[str, int] = {
    "job_id": 1,
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
    "llm_classified_skills": 1,
    "llm_job_attributes": 1,
    "onet_work_activities": 1,
    "skill_groups_origin_uuids": 1,
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


def _field_contains_substr_regex(field: str, needle_cf: str) -> Optional[Dict[str, Any]]:
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
    for field in (_M_CITY, _M_COUNTY):
        f_c = _field_contains_substr_regex(field, uc)
        if f_c is not None:
            ors.append(f_c)
        f_p = _field_contains_substr_regex(field, up)
        if f_p is not None:
            ors.append(f_p)
    for hay, fpath in (
        (uc, "$classifier_metadata.city"),
        (uc, "$classifier_metadata.county"),
        (up, "$classifier_metadata.city"),
        (up, "$classifier_metadata.county"),
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
            n_missing_ess, len(essential_skills),
            n_missing_opt, len(optional_skills),
        )

    llm_attrs = rd.get("llm_job_attributes", {})
    attributes = llm_attrs.get("attributes", {})

    city = _str_or_empty(meta.get("city"))
    county = _str_or_empty(meta.get("county"))
    loc_parts = [p for p in (city, county) if p]
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

    return {
        "uuid": job_id,
        "opportunity_title": meta.get("title") or "Unknown",
        "location": location,
        "city": city,
        "province": county,
        "employer": meta.get("employer"),
        "employment_type": meta.get("employment_type"),
        "salary_text": meta.get("salary"),
        "closing_date": closing_s,
        "contract_type": et,
        "url": meta.get("application_url"),
        "essential_skills": essential_skills,
        "optional_skills": optional_skills,
        "skill_groups_origin_uuids": skill_groups,
        "attributes": attributes,
        "opportunity_description": meta.get("job_description") or meta.get("description") or "",
        "onet_work_activities": onet_wa,
    }


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


_cached_occupations = None


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
                ess_uuids = skills.get("essential", {}).get("uuids", []) if isinstance(skills.get("essential"), dict) else []
                opt_uuids = skills.get("optional", {}).get("uuids", []) if isinstance(skills.get("optional"), dict) else []
                counties = entry.get("counties_data", [])

                code = occ.get("code", "")
                label = occ.get("preferred_label", "Unknown")
                description = occ.get("description", "")

                raw_wa = entry.get("onet_work_activities", [])
                onet_wa = []
                for w in raw_wa:
                    wc = w.get("WA_code")
                    imp = w.get("WA_Importance", "")
                    lvl = w.get("WA_Level", "")
                    if wc and imp != "" and lvl != "":
                        onet_wa.append({
                            "WA_code": wc,
                            "WA_label": w.get("WA_label", ""),
                            "WA_Importance": float(imp),
                            "WA_Level": float(lvl),
                        })

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

                    # Occupation JSON only carries bare skill UUIDs (no labels).
                    # Wrap in the same {id, label} shape used by job dicts so
                    # downstream consumers see one uniform contract; gap analysis
                    # reads id directly without going through label resolution.
                    flattened.append({
                        "uuid": f"{code}_{county}" if county else code,
                        "originUuid": code,
                        "occupation_label": label,
                        "preferredLabel": label,
                        "description": description,
                        "location": county,
                        "city": county,
                        "province": county,
                        "essential_skills": [{"id": str(u), "label": ""} for u in ess_uuids],
                        "optional_skills": [{"id": str(u), "label": ""} for u in opt_uuids],
                        "skill_groups_origin_uuids": [],
                        "attributes": attributes,
                        "onet_work_activities": onet_wa,
                })

        flatten_ms = _ms(t1)
        _cached_occupations = flattened
        total_ms = _ms(t_total)
        logger.info("Loaded %d occupation-county items from %d raw occupations", len(flattened), len(raw_occupations))
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
        logger.info("WA lookup warmup skipped (enriched jobs carry onet_work_activities; set WARMUP_WA_LOOKUP=1 to force)")
