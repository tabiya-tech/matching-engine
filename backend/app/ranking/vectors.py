"""Stage-1 concat embedding vectors on job / occupation dicts, and row normalisation."""

from __future__ import annotations

from typing import Any

import numpy as np


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms > 0, norms, 1.0).astype(np.float32)
    return (mat / norms).astype(np.float32)


def stage1_vector(job: dict[str, Any], *, dim: int) -> np.ndarray | None:
    """Prefer NPZ-sync BSON; fall back to ``job_embedding`` float list on the job doc."""

    sub = job.get("concat_skill_embedding_gemini")
    if isinstance(sub, dict):
        vb = sub.get("vector_bin")
        if vb is not None:
            raw = getattr(vb, "bytes", None) or bytes(vb)
            arr = np.frombuffer(raw, dtype=np.float32)
            if arr.size == dim:
                return arr

    je = job.get("job_embedding")
    # Accept a float list (Mongo job docs) or a numpy array (occupation embeddings attached
    # in-process by OccupationsRepository.attach_embeddings).
    if isinstance(je, np.ndarray):
        if je.ndim == 1 and je.size == dim:
            return je.astype(np.float32, copy=False)
    elif isinstance(je, list) and je:
        arr = np.asarray(je, dtype=np.float32)
        if arr.ndim == 1 and arr.size == dim:
            return arr
    return None


def is_prewhitened(job: dict[str, Any]) -> bool:
    """True iff this job's stage-1 embedding is ALREADY whitened on the DB side (set by
    build_job_dict_from_ranked from llm_reranker_meta.embedding.whitening.enabled). Occupations and
    offline jobs lack the flag -> raw (whitened in-process)."""
    return bool(job.get("job_embedding_whitened"))


def strip_vectors(job: dict[str, Any]) -> dict[str, Any]:
    out = dict(job)
    out.pop("concat_skill_embedding_gemini", None)
    out.pop("job_embedding", None)
    return out


def index_by_uuid(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for it in items:
        uid = str(it.get("uuid") or it.get("_id") or "")
        if uid:
            out[uid] = it
    return out
