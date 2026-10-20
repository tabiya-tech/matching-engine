"""Occupation corpus for matching: read from the occupations repository, flattened, and cached in-process.

The source is ``OCCUPATIONS_SOURCE`` (``app.occupations.get_occupations_repository``): the per-deployment
Mongo collection written by ``scripts/ingest_occupations.py``, or the committed JSON + NPZ. The
catalogue changes rarely, so it is cached for ``OCCUPATIONS_CACHE_TTL_S``. An empty or unreachable
source is logged and yields no occupations; job matching is unaffected.
"""

import asyncio
import logging
import math
import time
from collections.abc import Sequence

import numpy as np
from pydantic import ConfigDict, Field

from app.config import OCCUPATIONS_CACHE_TTL_S, V4_FULL_CONCAT_WHITENING_PATH
from app.domain.base import DomainModel
from app.domain.occupation import Occupation
from app.occupations.flatten import flatten_occupations
from app.occupations.occupation_file import OccupationFileEntry
from app.occupations.repository import IOccupationsRepository, OccupationRecord

logger = logging.getLogger(__name__)

# After a failed or empty load, how long to serve that result before trying the source again, so an
# outage does not cost every request a server-selection timeout.
_RETRY_AFTER_FAILURE_S = 60.0


class OccupationsLoadTiming(DomainModel):
    """Durations and counts of one ``get_all_occupations_with_timing`` call (logged and traced)."""

    occupation_cache_hit: bool = Field(
        description="True when the in-process cache served the corpus."
    )
    occupation_source: str = Field(
        default="",
        description="Repository the corpus was read from (``mongo:<collection>``, ``file:<path>``).",
    )
    occupation_source_read_ms: float = Field(
        description="Time reading and validating the occupations; 0 on a cache hit."
    )
    occupation_flatten_ms: float = Field(
        description="Time flattening and preparing embeddings; 0 on a cache hit."
    )
    n_occupation_rows: int = Field(description="Occupation × county rows returned.")
    n_occupation_entries: int | None = Field(
        default=None, description="Occupations read; None on a cache hit."
    )
    get_all_occupations_total_ms: float = Field(description="Total time of the call.")


class _OccupationCache(DomainModel):
    """One load of the catalogue: entries, flattened rows and stage-1 vectors by code."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: str
    entries: list[OccupationFileEntry]
    rows: list[Occupation]
    # occupation code -> (stage-1 vector, already whitened)
    embeddings: dict[str, tuple[np.ndarray, bool]]
    expires_at: float


_repository_override: IOccupationsRepository | None = None
_cache: _OccupationCache | None = None
# The cached rows with embeddings attached; reused across requests so the ~1.7k rows are copied
# with their embedding once per load, not per request.
_cached_attached: list[Occupation] | None = None
_load_lock: asyncio.Lock | None = None


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def set_occupations_repository(repository: IOccupationsRepository | None) -> None:
    """Pin the occupation source (tests, offline tools); None restores ``OCCUPATIONS_SOURCE``.

    Clears the cache so the next call reads from the new source.
    """
    global _repository_override
    _repository_override = repository
    reset_occupations_cache()


def reset_occupations_cache() -> None:
    """Forget the cached corpus; the next call reloads it."""
    global _cache, _cached_attached
    _cache = None
    _cached_attached = None


async def _repository() -> IOccupationsRepository:
    if _repository_override is not None:
        return _repository_override
    from app.occupations.get_occupations_repository import get_occupations_repository

    return await get_occupations_repository()


def _current_whitening_sha256() -> str | None:
    from app.artifacts.repository import get_artifacts_repository

    try:
        return get_artifacts_repository().file_sha256(V4_FULL_CONCAT_WHITENING_PATH)
    except OSError:
        return None


def _stage1_embeddings(
    records: Sequence[OccupationRecord],
) -> dict[str, tuple[np.ndarray, bool]]:
    """Stage-1 vector per occupation code, in the whitened concat space the matcher uses.

    A vector whitened at ingest is used as-is when it was made with the deployed whitening
    artifact; otherwise the raw vector is whitened here, once per load, so the engine never
    re-whitens ~1.7k rows per request. Raw vectors are kept when the artifact is unavailable.
    """
    out: dict[str, tuple[np.ndarray, bool]] = {}
    raw: dict[str, np.ndarray] = {}
    for rec in records:
        code = rec.entry.occupation.code or ""
        if code and rec.embedding is not None:
            raw[code] = rec.embedding
    if not raw and not any(r.whitened_embedding is not None for r in records):
        return out

    try:
        from app.server_dependencies.model_dependencies import get_concat_whitener

        whitener = get_concat_whitener()
        whitening_active = whitener.rescale_target() > 0
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Concat whitener unavailable (%s); occupations stay raw.", e)
        whitener, whitening_active = None, False

    n_stored = n_stale = 0
    if whitening_active:
        deployed_sha = _current_whitening_sha256()
        for rec in records:
            code = rec.entry.occupation.code or ""
            if not code or rec.whitened_embedding is None:
                continue
            if deployed_sha and rec.whitening_artifact_sha256 == deployed_sha:
                out[code] = (rec.whitened_embedding, True)
                raw.pop(code, None)
                n_stored += 1
            else:
                n_stale += 1
        if n_stale:
            logger.warning(
                "%d occupations were whitened at ingest with another artifact than %s; "
                "whitening them in-process. Re-run scripts/ingest_occupations.py.",
                n_stale,
                V4_FULL_CONCAT_WHITENING_PATH,
            )

    if raw and whitener is not None and whitening_active:
        codes = list(raw)
        try:
            wmat = whitener.whiten_rows(np.stack([raw[c] for c in codes])).astype(
                np.float32
            )
            for code, vec in zip(codes, wmat, strict=True):
                out[code] = (np.ascontiguousarray(vec, dtype=np.float32), True)
            raw = {}
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "Could not whiten occupation embeddings (%s); whitening in-process per request.",
                e,
            )
    for code, vec in raw.items():
        out[code] = (vec, False)

    logger.info(
        "Occupation embeddings: %d whitened at ingest, %d whitened at load, %d raw",
        n_stored,
        sum(1 for _, w in out.values() if w) - n_stored,
        sum(1 for _, w in out.values() if not w),
    )
    return out


def _expiry(ttl_s: float) -> float:
    return math.inf if ttl_s <= 0 else time.monotonic() + ttl_s


async def _load(
    repository: IOccupationsRepository, stale: _OccupationCache | None
) -> tuple[_OccupationCache, int | None, float, float]:
    """Read and prepare the catalogue; never raises.

    :return: ``(cache, n_entries, read_ms, flatten_ms)``; on failure the stale cache (or an empty
        one), kept for ``_RETRY_AFTER_FAILURE_S``
    """
    t0 = time.perf_counter()
    try:
        records = await repository.load_occupations()
    except Exception as e:
        logger.error(
            "Occupations unavailable from %s (%s: %s); %s. Job matching is unaffected.",
            repository.source,
            type(e).__name__,
            e,
            "serving the previously loaded occupations"
            if stale and stale.rows
            else "occupation recommendations will be empty",
        )
        fallback = stale or _OccupationCache(
            source=repository.source, entries=[], rows=[], embeddings={}, expires_at=0
        )
        return (
            fallback.model_copy(update={"expires_at": _expiry(_RETRY_AFTER_FAILURE_S)}),
            None,
            _ms(t0),
            0.0,
        )
    read_ms = _ms(t0)

    t1 = time.perf_counter()
    entries = [r.entry for r in records]
    rows = flatten_occupations(entries)
    embeddings = _stage1_embeddings(records)
    flatten_ms = _ms(t1)

    if not records:
        logger.error(
            "No occupations in %s; occupation recommendations will be empty until they are "
            "loaded with scripts/ingest_occupations.py. Job matching is unaffected.",
            repository.source,
        )
        ttl = _RETRY_AFTER_FAILURE_S
    else:
        logger.info(
            "Loaded %d occupation-county items from %d occupations (%s)",
            len(rows),
            len(records),
            repository.source,
        )
        ttl = OCCUPATIONS_CACHE_TTL_S
    cache = _OccupationCache(
        source=repository.source,
        entries=entries,
        rows=rows,
        embeddings=embeddings,
        expires_at=_expiry(ttl),
    )
    return cache, len(records), read_ms, flatten_ms


def _hit(cache: _OccupationCache, t_total: float) -> OccupationsLoadTiming:
    return OccupationsLoadTiming(
        occupation_cache_hit=True,
        occupation_source=cache.source,
        occupation_source_read_ms=0.0,
        occupation_flatten_ms=0.0,
        n_occupation_rows=len(cache.rows),
        get_all_occupations_total_ms=_ms(t_total),
    )


async def get_all_occupations_with_timing() -> tuple[
    list[Occupation], OccupationsLoadTiming
]:
    """Load the occupation corpus, validated and flattened, from the in-process cache or the source.

    Never raises: an empty or unreachable source is logged and returns no occupations (or the
    previously loaded ones, while they are still cached).

    :return: ``(occupations, timing)``; on a cache hit the read/flatten times are 0
    """
    global _cache, _cached_attached, _load_lock
    t_total = time.perf_counter()

    cache = _cache
    if cache is not None and time.monotonic() < cache.expires_at:
        return cache.rows, _hit(cache, t_total)

    if _load_lock is None:
        _load_lock = asyncio.Lock()
    async with _load_lock:
        # Another request may have reloaded while this one waited.
        cache = _cache
        if cache is not None and time.monotonic() < cache.expires_at:
            return cache.rows, _hit(cache, t_total)

        repository = await _repository()
        loaded, n_entries, read_ms, flatten_ms = await _load(repository, cache)
        if loaded is not cache:
            _cached_attached = None
        _cache = loaded

    return loaded.rows, OccupationsLoadTiming(
        occupation_cache_hit=False,
        occupation_source=loaded.source,
        occupation_source_read_ms=read_ms,
        occupation_flatten_ms=flatten_ms,
        n_occupation_rows=len(loaded.rows),
        n_occupation_entries=n_entries,
        get_all_occupations_total_ms=_ms(t_total),
    )


async def get_occupation_entries() -> list[OccupationFileEntry]:
    """The occupation entries behind the cached corpus (loading it if needed)."""
    await get_all_occupations_with_timing()
    return list(_cache.entries) if _cache is not None else []


def attach_occupation_embeddings(occupations: Sequence[Occupation]) -> list[Occupation]:
    """Return the occupations with ``embedding`` (a shared np.ndarray) attached by code.

    Vector is shared across all county-rows of the same occupation code (skills are identical),
    so memory stays at one array per code. Rows with no matching embedding are returned
    unchanged (the v4 engine skips items without a stage-1 vector).

    :param occupations: The occupation rows
    :return: New occupation rows; the inputs (the cached corpus) are not modified
    """
    global _cached_attached
    cache = _cache
    is_cached_corpus = cache is not None and occupations is cache.rows
    if is_cached_corpus and _cached_attached is not None:
        return list(_cached_attached)
    emb = cache.embeddings if cache is not None else {}
    if not emb:
        return list(occupations)
    out: list[Occupation] = []
    for occ in occupations:
        found = emb.get(occ.originUuid)
        if found is None:
            out.append(occ)
        else:
            # embedding_whitened is True once the vector is in the whitened space (at ingest or at
            # load) -> engine consumes it directly (mirrors DB-whitened jobs); False -> raw.
            vec, whitened = found
            out.append(
                occ.model_copy(
                    update={"embedding": vec, "embedding_whitened": whitened}
                )
            )
    if is_cached_corpus:
        _cached_attached = out
    return list(out)
