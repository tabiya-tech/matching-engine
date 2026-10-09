"""Occupation corpus for matching: read once from the taxonomy JSON, flattened, and cached in-process.

Occupations come from committed resource files (``OCCUPATION_JSON_PATH`` and the concat-embeddings
NPZ), not from MongoDB.
"""

import json
import logging
import time
from collections.abc import Sequence
from typing import Any

from app.config import OCCUPATION_CONCAT_EMBEDDINGS_PATH, OCCUPATION_JSON_PATH
from app.occupations.flatten import flatten_occupations

logger = logging.getLogger(__name__)

_cached_occupations = None
_cached_occ_embeddings = None  # {occupation_code: np.ndarray(float32, EMBEDDING_DIM)}
_occ_prewhitened = False  # True once the cached occ embeddings are whitened (consumed directly, no per-request whitening)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _load_occupation_embeddings() -> dict[str, Any]:
    """Lazy/cached load of the committed occupation concat-embeddings NPZ (code -> vector).

    Returns {} (with a warning) if the artifact is missing/unreadable, so occupations are
    simply skipped by the /match_v4 retrieval rather than crashing the request.
    """
    global _cached_occ_embeddings
    if _cached_occ_embeddings is not None:
        return _cached_occ_embeddings
    out: dict[str, Any] = {}
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
            import numpy as np

            from app.server_dependencies.model_dependencies import get_concat_whitener

            whitener = get_concat_whitener()
            if whitener.rescale_target() > 0:
                codes_list = list(out.keys())
                wmat = whitener.whiten_rows(
                    np.stack([out[c] for c in codes_list], axis=0)
                ).astype(np.float32)
                for c, wv in zip(codes_list, wmat):
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


def attach_occupation_embeddings(occupations: Sequence[dict]) -> list[dict]:
    """Return occupation dicts with ``job_embedding`` (shared np.ndarray) attached by code.

    Vector is shared across all county-rows of the same occupation code (skills are identical),
    so memory stays at one array per code. Rows with no matching embedding are returned
    unchanged (the v4 engine skips items without a stage-1 vector).
    """
    emb = _load_occupation_embeddings()
    if not emb:
        return list(occupations)
    out: list[dict] = []
    for occ in occupations:
        vec = emb.get(str(occ.get("originUuid") or ""))
        if vec is None:
            out.append(occ)
        else:
            o = dict(occ)
            o["job_embedding"] = vec
            # True once the occ cache has been whitened at load -> engine consumes it directly (mirrors
            # DB-whitened jobs); False -> raw, whitened in-process per request.
            o["job_embedding_whitened"] = _occ_prewhitened
            out.append(o)
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
        flattened = flatten_occupations(raw_occupations)
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
