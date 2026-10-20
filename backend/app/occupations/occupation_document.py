"""Shape of an occupation document in MongoDB (``MONGO_OCCUPATIONS_COLLECTION``).

One document per occupation, keyed by its code (``_id``). It is the occupation-file entry as written
(``occupation`` / ``skills`` / ``counties_data`` / ``onet_work_activities``, so ``OccupationFileEntry``
still validates it and ``app.occupations.flatten`` expands it to occupation × county rows at read
time), plus the concat skill embedding made at ingest and the provenance needed to tell when it is
stale. Written by ``scripts/ingest_occupations.py`` (mapping in ``app.occupations.ingest``), read by
``app.occupations.mongo_repository``.
"""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import Field

from app.domain.base import SourceRecord
from app.domain.coercion import OptionalFlag, OptionalLevel, OptionalText, TextList
from app.occupations.occupation_file import OccupationFileEntry
from app.ranking.retrieval import build_concat_embedding_text


class StoredWhitening(SourceRecord):
    """``embedding.whitening``: which concat-whitening artifact made ``whitened_vector_bin``."""

    enabled: OptionalFlag = Field(
        default=None, description="True when ``whitened_vector_bin`` is set."
    )
    artifact_sha256: OptionalText = Field(
        default=None,
        description="SHA-256 of ``V4_FULL_CONCAT_WHITENING_PATH`` at ingest (as on job documents).",
    )


class StoredEmbedding(SourceRecord):
    """``embedding``: the occupation's concat skill embedding and how it was made."""

    model: OptionalText = Field(
        default=None, description="Embedding model, e.g. ``gemini-embedding-001``."
    )
    dim: OptionalLevel = Field(default=None, description="Vector length.")
    normalized: OptionalFlag = Field(
        default=None, description="True when ``vector_bin`` is L2-normalised."
    )
    text_sha256: OptionalText = Field(
        default=None,
        description="SHA-256 of the embedded text (see ``occupation_embedding_text``).",
    )
    vector_sha256: OptionalText = Field(
        default=None, description="SHA-256 of ``vector_bin``."
    )
    vector_bin: bytes | None = Field(
        default=None, description="Raw vector, float32 little-endian, as BSON binary."
    )
    whitened_vector_bin: bytes | None = Field(
        default=None,
        description="Whitened vector, float32 little-endian, as BSON binary.",
    )
    whitening: StoredWhitening | None = Field(
        default=None, description="Whitening provenance."
    )


class OccupationDocument(OccupationFileEntry):
    """An occupation document as read from MongoDB: the file entry plus ingest fields."""

    code: OptionalText = Field(
        default=None, description="Occupation code; also the ``_id``."
    )
    position: OptionalLevel = Field(
        default=None,
        description="Index of the entry in the ingested file; the runtime read keeps this order.",
    )
    group_codes: TextList = Field(
        default_factory=list,
        description="Codes of the occupation groups it sits under, broadest first.",
    )
    content_sha256: OptionalText = Field(
        default=None,
        description="SHA-256 of the entry and group codes; unchanged means nothing to update.",
    )
    embedding: StoredEmbedding | None = Field(
        default=None, description="Concat skill embedding."
    )


OCCUPATION_DOCUMENT_FIND_PROJECTION: dict[str, int] = {
    # LLM rationale for each county attribute estimate: kept in the document, never read by matching.
    "counties_data.job_attributes.attributes.reasoning": 0,
}
"""Exclusion projection for the runtime read."""


def occupation_skill_labels(entry: OccupationFileEntry) -> list[str]:
    """Essential then optional skill labels of an entry, blanks dropped."""
    labels: list[str] = []
    for block in (entry.skills.essential, entry.skills.optional):
        if block is not None:
            labels.extend(label for label in block.labels if label)
    return labels


def occupation_embedding_text(entry: OccupationFileEntry) -> str:
    """The text an occupation's concat embedding is made from (same as ``embed_occupations``)."""
    return build_concat_embedding_text(occupation_skill_labels(entry)) or " "


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_sha256(raw_entry: Mapping[str, Any], group_codes: list[str]) -> str:
    """Order-independent hash of an entry as written plus its group codes."""
    payload = json.dumps(
        {"entry": raw_entry, "group_codes": group_codes},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def vector_to_bin(vector: np.ndarray) -> bytes:
    return np.ascontiguousarray(vector, dtype="<f4").tobytes()


def vector_from_bin(data: bytes | None, *, dim: int) -> np.ndarray | None:
    """The float32 vector in ``data``, or None when absent or not ``dim`` long."""
    if data is None or len(data) != dim * 4:
        return None
    return np.frombuffer(bytes(data), dtype="<f4").astype(np.float32)
