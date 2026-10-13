"""Occupation corpus for matching: read once from the taxonomy JSON, flattened, and cached in-process.

Occupations come from committed resource files (``OCCUPATION_JSON_PATH`` and the concat-embeddings
NPZ), not from MongoDB.
"""

import json
import logging
import time
from collections.abc import Sequence

import numpy as np
from pydantic import Field

from app.config import OCCUPATION_CONCAT_EMBEDDINGS_PATH, OCCUPATION_JSON_PATH
from app.domain.base import DomainModel
from app.domain.occupation import Occupation
from app.occupations.flatten import flatten_occupations
from app.occupations.occupation_file import OCCUPATION_FILE

logger = logging.getLogger(__name__)


class OccupationsLoadTiming(DomainModel):
    """Durations and counts of one ``get_all_occupations_with_timing`` call (logged and traced)."""

    occupation_cache_hit: bool = Field(
        description="True when the in-process cache served the corpus."
    )
    occupation_file_read_ms: float = Field(
        description="Time reading and parsing the JSON file; 0 on a cache hit."
    )
    occupation_json_parse_and_flatten_ms: float = Field(
        description="Time validating and flattening the entries; 0 on a cache hit."
    )
    n_occupation_rows: int = Field(description="Occupation × county rows returned.")
    n_raw_occupation_entries: int | None = Field(
        default=None, description="Entries in the file; None on a cache hit."
    )
    get_all_occupations_total_ms: float = Field(description="Total time of the call.")


_cached_occupations: list[Occupation] | None = None
# The cached corpus with embeddings attached; reused across requests so the ~1.7k rows are copied
# with their embedding once, not per request.
_cached_attached: list[Occupation] | None = None
_cached_occ_embeddings: dict[str, np.ndarray] | None = (
    None  # occupation code -> float32 vector
)
_occ_prewhitened = False  # True once the cached occ embeddings are whitened (consumed directly, no per-request whitening)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _load_occupation_embeddings() -> dict[str, np.ndarray]:
    """Lazy/cached load of the committed occupation concat-embeddings NPZ (code -> vector).

    Returns {} (with a warning) if the artifact is missing/unreadable, so occupations are
    simply skipped by the /match_v4 retrieval rather than crashing the request.
    """
    global _cached_occ_embeddings
    if _cached_occ_embeddings is not None:
        return _cached_occ_embeddings
    out: dict[str, np.ndarray] = {}
    try:
        with np.load(OCCUPATION_CONCAT_EMBEDDINGS_PATH, allow_pickle=True) as data:
            codes = [str(c) for c in data["codes"].tolist()]
            vectors = np.asarray(data["vectors"], dtype=np.float32)
        for code, vec in zip(codes, vectors, strict=True):
            out[code] = np.ascontiguousarray(vec, dtype=np.float32)
        logger.info(
            "Loaded %d occupation concat embeddings from %s",
            len(out),
            OCCUPATION_CONCAT_EMBEDDINGS_PATH,
        )
    except FileNotFoundError:
        logger.warning(
            "Occupation embeddings NPZ not found at %s; /match_v4 occupations will be skipped. "
            "Build it via `python -m tools.embeddings.embed_occupations`.",
            OCCUPATION_CONCAT_EMBEDDINGS_PATH,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            "Failed to load occupation embeddings (%s): %s; occupations skipped.",
            OCCUPATION_CONCAT_EMBEDDINGS_PATH,
            e,
        )
    # Whiten the (static) occupation embeddings ONCE here, into the same whitened concat space the
    # matcher uses, so the engine consumes them directly instead of re-whitening ~1700 vectors on every
    # request (the NPZ is raw L2-normalized). Falls back to raw if the concat artifact is unavailable.
    global _occ_prewhitened
    _occ_prewhitened = False
    if out:
        try:
            from app.server_dependencies.model_dependencies import get_concat_whitener

            whitener = get_concat_whitener()
            if whitener.rescale_target() > 0:
                codes_list = list(out.keys())
                wmat = whitener.whiten_rows(
                    np.stack([out[c] for c in codes_list], axis=0)
                ).astype(np.float32)
                for c, wv in zip(codes_list, wmat, strict=True):
                    out[c] = np.ascontiguousarray(wv, dtype=np.float32)
                _occ_prewhitened = True
                logger.info(
                    "Whitened %d occupation embeddings once at load (consumed directly thereafter).",
                    len(out),
                )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "Could not pre-whiten occupation embeddings (%s); whitening in-process per request.",
                e,
            )
            _occ_prewhitened = False
    _cached_occ_embeddings = out
    return out


def attach_occupation_embeddings(occupations: Sequence[Occupation]) -> list[Occupation]:
    """Return the occupations with ``embedding`` (a shared np.ndarray) attached by code.

    Vector is shared across all county-rows of the same occupation code (skills are identical),
    so memory stays at one array per code. Rows with no matching embedding are returned
    unchanged (the v4 engine skips items without a stage-1 vector).

    :param occupations: The occupation rows
    :return: New occupation rows; the inputs (the cached corpus) are not modified
    """
    global _cached_attached
    is_cached_corpus = occupations is _cached_occupations
    if is_cached_corpus and _cached_attached is not None:
        return list(_cached_attached)
    emb = _load_occupation_embeddings()
    if not emb:
        return list(occupations)
    out: list[Occupation] = []
    for occ in occupations:
        vec = emb.get(occ.originUuid)
        if vec is None:
            out.append(occ)
        else:
            # embedding_whitened is True once the occ cache has been whitened at load -> engine
            # consumes it directly (mirrors DB-whitened jobs); False -> raw, whitened in-process.
            out.append(
                occ.model_copy(
                    update={"embedding": vec, "embedding_whitened": _occ_prewhitened}
                )
            )
    if is_cached_corpus:
        _cached_attached = out
    return list(out)


async def get_all_occupations_with_timing() -> tuple[
    list[Occupation], OccupationsLoadTiming
]:
    """Load the occupation corpus, validated and flattened, from the in-process cache or the file.

    :return: ``(occupations, timing)``; on a cache hit the read/flatten times are 0
    :raises RuntimeError: If the file cannot be read or does not have the occupation-file shape
    """
    global _cached_occupations
    t_total = time.perf_counter()

    if _cached_occupations is not None:
        total_ms = _ms(t_total)
        return _cached_occupations, OccupationsLoadTiming(
            occupation_cache_hit=True,
            occupation_file_read_ms=0.0,
            occupation_json_parse_and_flatten_ms=0.0,
            n_occupation_rows=len(_cached_occupations),
            get_all_occupations_total_ms=total_ms,
        )

    try:
        t0 = time.perf_counter()
        with open(OCCUPATION_JSON_PATH, "r", encoding="utf-8") as f:
            raw_occupations = json.load(f)
        file_read_and_json_ms = _ms(t0)

        t1 = time.perf_counter()
        flattened = flatten_occupations(
            OCCUPATION_FILE.validate_python(raw_occupations)
        )
        flatten_ms = _ms(t1)
        _cached_occupations = flattened
        total_ms = _ms(t_total)
        logger.info(
            "Loaded %d occupation-county items from %d raw occupations",
            len(flattened),
            len(raw_occupations),
        )
        return _cached_occupations, OccupationsLoadTiming(
            occupation_cache_hit=False,
            occupation_file_read_ms=file_read_and_json_ms,
            occupation_json_parse_and_flatten_ms=flatten_ms,
            n_occupation_rows=len(flattened),
            n_raw_occupation_entries=len(raw_occupations),
            get_all_occupations_total_ms=total_ms,
        )
    except Exception as e:
        logger.exception(e)
        raise RuntimeError(f"Failed to load occupations: {e}") from e
