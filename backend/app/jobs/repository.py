"""Repository for the jobs collection (``MONGO_JOBS_COLLECTION``): the queries, the filters and
projection they use, and the shaping of stored documents into flat job dicts."""

import base64
import binascii
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel

from app.config import (
    JOBS_FIND_USE_PROJECTION,
    JOBS_RETRIEVAL_FILTER,
    JOBS_RETRIEVAL_LIMIT,
    LOCATION_HUB_CHAINS_PATH,
    LOCATION_TIER_ENABLED,
    MONGO_JOBS_COLLECTION,
)
from app.jobs.errors import InvalidCursor
from app.schemas import JobListItem, JobsStats

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


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class IJobsRepository(ABC):
    """Interface for reading the jobs collection."""

    @abstractmethod
    async def find_active(
        self, users: Sequence[dict] | None = None
    ) -> tuple[list[dict], dict[str, Any]]:
        """
        Loads active jobs as flat job dicts (see ``build_job_dict_from_ranked``).

        If ``JOBS_RETRIEVAL_FILTER`` is true and ``users`` is non-empty, the query also ORs
        per-user location clauses, sorts by ``_id`` descending and applies ``JOBS_RETRIEVAL_LIMIT``.

        :param users: Users whose locations scope the query, or None for every active job
        :return: ``(jobs, timing)``; timing keys: mongo_ranked_find_ms, python_build_jobs_ms,
            n_ranked_raw, n_jobs, n_skipped_inactive, get_all_jobs_total_ms,
            jobs_retrieval_filter_applied, jobs_find_use_projection
        :raises Exception: If any database error occurs
        """
        raise NotImplementedError()

    @abstractmethod
    async def browse_page(
        self,
        cursor: str | None = None,
        limit: int = 20,
        *,
        search: str | None = None,
        category: str | None = None,
        employment_type: str | None = None,
        location: str | None = None,
        skills: str | None = None,
        days: int | None = None,
        include_total: bool = False,
    ) -> tuple[list[JobListItem], str | None, int | None, dict[str, Any]]:
        """
        Cursor-paginated, filterable browse over active jobs, newest first.

        Pagination is keyset-based on ``_id`` descending; ``cursor`` wraps the last ``_id`` of the
        previous page. Supplied filters are AND-ed with ``is_active``.

        :return: ``(jobs, next_cursor, total, timing)``; ``next_cursor`` is None on the last page,
            ``total`` is None unless ``include_total`` is set
        :raises InvalidCursor: If ``cursor`` is malformed
        :raises Exception: If any database error occurs
        """
        raise NotImplementedError()

    @abstractmethod
    async def stats(self) -> JobsStats:
        """
        Aggregate counts over the active jobs catalog.

        :return: Total jobs, distinct sectors and distinct source platforms
        :raises Exception: If any database error occurs
        """
        raise NotImplementedError()

    @abstractmethod
    async def ensure_indexes(self) -> list[str]:
        """
        Creates (idempotently) the indexes the jobs queries need.

        :return: The ensured index names
        :raises Exception: If any database error occurs
        """
        raise NotImplementedError()

    @abstractmethod
    async def ping(self) -> None:
        """
        Pings the database.

        :raises Exception: If the database cannot be reached
        """
        raise NotImplementedError()


class JobsRepository(IJobsRepository):
    def __init__(self, *, db: AsyncIOMotorDatabase):
        self._db = db
        self._logger = logging.getLogger(self.__class__.__name__)

    async def find_active(
        self, users: Sequence[dict] | None = None
    ) -> tuple[list[dict], dict[str, Any]]:
        t_total = time.perf_counter()
        t0 = time.perf_counter()
        filt: dict[str, Any] = RANKED_JOBS_ACTIVE_FILTER
        retrieval_applied = False
        if JOBS_RETRIEVAL_FILTER and users:
            # is_active AND (OR of every user's location clauses): a superset of
            # job_matches_user_location on the classifier_metadata fields. Remote jobs always match.
            # With LOCATION_TIER_ENABLED (urban-pull Part A) each user's needles widen to their
            # hub-chain regions (regional + national hub) so hub jobs reach the pool; the per-user
            # soft tier re-rank in MatchingService then keeps local jobs preferred.
            hub_chains = None
            if LOCATION_TIER_ENABLED:
                from app.artifacts.repository import get_artifacts_repository

                hub_chains = get_artifacts_repository().load_hub_chains(
                    LOCATION_HUB_CHAINS_PATH
                )
            location_clauses: list[dict[str, Any]] = []
            for user in users:
                location_clauses.extend(
                    {field: {"$regex": "remote", "$options": "i"}}
                    for field in (_M_CITY, _M_COUNTY)
                )
                user_city = _norm_loc_value(user.get("city"))
                user_province = _norm_loc_value(user.get("province"))
                if not user_city or not user_province:
                    continue
                needles = {user_city, user_province}
                if hub_chains is not None:
                    # local + regional + national hub regions
                    needles.update(hub_chains.chain_for(user_province))
                for needle in sorted(n for n in needles if n):
                    # job field contains the user's string
                    for field in (_M_CITY, _M_COUNTY, _M_PROVINCE):
                        location_clauses.append(
                            {field: {"$regex": re.escape(needle), "$options": "i"}}
                        )
                    # user's string contains the job field. The non-empty check matters: Mongo
                    # finds an empty substring at index 0, which would match every document
                    # missing the field.
                    for field_path in (
                        "$classifier_metadata.city",
                        "$classifier_metadata.county",
                        "$classifier_metadata.province",
                    ):
                        job_value = {"$ifNull": [{"$toLower": field_path}, ""]}
                        location_clauses.append(
                            {
                                "$expr": {
                                    "$and": [
                                        {"$gt": [{"$strLenCP": job_value}, 0]},
                                        {
                                            "$gte": [
                                                {"$indexOfCP": [needle, job_value]},
                                                0,
                                            ]
                                        },
                                    ]
                                }
                            }
                        )
            filt = {"$and": [RANKED_JOBS_ACTIVE_FILTER, {"$or": location_clauses}]}
            retrieval_applied = True
        col = self._db[MONGO_JOBS_COLLECTION]
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
        jobs: list[dict] = []
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

    async def browse_page(
        self,
        cursor: str | None = None,
        limit: int = 20,
        *,
        search: str | None = None,
        category: str | None = None,
        employment_type: str | None = None,
        location: str | None = None,
        skills: str | None = None,
        days: int | None = None,
        include_total: bool = False,
    ) -> tuple[list[JobListItem], str | None, int | None, dict[str, Any]]:
        t_total = time.perf_counter()
        limit = max(1, int(limit))

        # Every filter is optional; supplied ones are AND-ed with ``is_active``. Field paths target the
        # raw ``classifier_metadata`` / ``llm_classified_skills`` document so Mongo filters before
        # shaping; ``category`` and ``location`` span several candidate field names because the stored
        # data is not uniform.
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
            clauses.append(
                {"classifier_metadata.employment_type": employment_type.strip()}
            )
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
                (datetime.now(timezone.utc) - timedelta(days=int(days)))
                .date()
                .isoformat()
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

        base_filt = (
            dict(RANKED_JOBS_ACTIVE_FILTER) if len(clauses) == 1 else {"$and": clauses}
        )
        filt: dict[str, Any] = dict(base_filt)
        if cursor:
            # Compose the keyset seek with the (possibly compound) filter without clobbering it.
            filt = {"$and": [base_filt, {"_id": {"$lt": _decode_jobs_cursor(cursor)}}]}

        col = self._db[MONGO_JOBS_COLLECTION]
        projection = RANKED_JOB_FIND_PROJECTION if JOBS_FIND_USE_PROJECTION else None
        # Fetch limit+1 so we can tell whether another page exists.
        t0 = time.perf_counter()
        query = col.find(filt, projection) if projection else col.find(filt)
        query = query.sort(JOBS_PAGE_SORT).limit(limit + 1)
        raw_docs = [d async for d in query]
        mongo_find_ms = _ms(t0)

        total: int | None = None
        if include_total:
            total = await col.count_documents(base_filt)

        has_more = len(raw_docs) > limit
        page_docs = raw_docs[:limit]

        t0 = time.perf_counter()
        jobs: list[JobListItem] = []
        skipped = 0
        for rd in page_docs:
            built = build_job_dict_from_ranked(rd)
            if built is None:
                skipped += 1
                continue
            jobs.append(
                JobListItem(
                    uuid=built.get("uuid"),
                    originUuid=built.get("originUuid"),
                    url=built.get("url"),
                    opportunity_title=built.get("opportunity_title", "No title"),
                    opportunity_isco_occupation_group=built.get(
                        "opportunity_isco_occupation_group"
                    ),
                    opportunity_isco_occupation_group_id=built.get(
                        "opportunity_isco_occupation_group_id"
                    ),
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
                    skills=built.get("skills", []),
                )
            )
        python_build_ms = _ms(t0)

        next_cursor = (
            _encode_jobs_cursor(page_docs[-1]["_id"])
            if has_more and page_docs
            else None
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

    async def stats(self) -> JobsStats:
        col = self._db[MONGO_JOBS_COLLECTION]
        total = await col.count_documents(RANKED_JOBS_ACTIVE_FILTER)

        raw_categories = await col.distinct(
            "classifier_metadata.category", RANKED_JOBS_ACTIVE_FILTER
        )
        if not raw_categories:
            raw_categories = await col.distinct(
                "classifier_metadata.isco_occupation_group", RANKED_JOBS_ACTIVE_FILTER
            )
        sectors = len(
            {str(c).strip().lower() for c in raw_categories if str(c).strip()}
        )

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

    async def ensure_indexes(self) -> list[str]:
        t0 = time.perf_counter()
        col = self._db[MONGO_JOBS_COLLECTION]
        created = await col.create_indexes(JOBS_INDEX_MODELS)
        logger.info(
            "Ensured %d indexes on %s in %.2f ms: %s",
            len(created),
            MONGO_JOBS_COLLECTION,
            _ms(t0),
            created,
        )
        return created

    async def ping(self) -> None:
        await self._db.command("ping")
