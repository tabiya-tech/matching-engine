"""Occupation file -> occupation documents, and what an ingest would change.

Pure (no I/O): ``scripts/ingest_occupations.py`` reads the file and the collection, calls these, and
writes. Steps:

1. ``parse_occupation_file`` validates the whole file (``OccupationFileEntry`` per entry, plus the
   checks the flattening relies on) and raises ``OccupationFileInvalid`` listing every bad entry.
2. ``resolve_embeddings`` picks each occupation's raw vector: the stored one when its text is
   unchanged, else a seed NPZ vector, else a fresh embedding (or "pending" in a dry run).
3. ``build_occupation_document`` maps an entry + vectors to the stored document (shape in
   ``app.occupations.occupation_document``).
4. ``plan_ingest`` compares the documents with the stored ones: added, updated, unchanged, removed.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from app.domain.base import DomainModel
from app.occupations.occupation_document import (
    OccupationDocument,
    content_sha256,
    occupation_embedding_text,
    sha256_bytes,
    sha256_text,
    vector_from_bin,
    vector_to_bin,
)
from app.occupations.occupation_file import OccupationFileEntry

_ENTRY: TypeAdapter[OccupationFileEntry] = TypeAdapter(OccupationFileEntry)
_MAX_PROBLEMS_PER_ENTRY = 5


class OccupationFileInvalid(ValueError):
    """The occupation file cannot be ingested; ``problems`` names every bad entry."""

    def __init__(self, problems: list[str]):
        super().__init__(f"{len(problems)} problem(s) in the occupation file")
        self.problems = problems


class ParsedOccupation(DomainModel):
    """One valid entry of the occupation file."""

    position: int = Field(description="Index in the file.")
    code: str = Field(description="Occupation code (stripped).")
    entry: OccupationFileEntry = Field(description="The validated entry.")
    raw: dict[str, Any] = Field(description="The entry as written, stored verbatim.")


def _where(index: int, raw: object) -> str:
    code = None
    if isinstance(raw, dict):
        occupation = raw.get("occupation")
        if isinstance(occupation, dict):
            code = occupation.get("code")
    return f"entry {index}" + (f" (code {code!r})" if code else "")


def parse_occupation_file(raw: object) -> list[ParsedOccupation]:
    """Validate a whole occupation file (the parsed JSON).

    :raises OccupationFileInvalid: If it is not a list, or any entry is malformed, has no code or
        label, repeats a code, or lists more skill labels than uuids
    """
    if not isinstance(raw, list):
        raise OccupationFileInvalid(
            [f"expected a JSON list of occupations, got {type(raw).__name__}"]
        )
    problems: list[str] = []
    parsed: list[ParsedOccupation] = []
    first_seen: dict[str, int] = {}
    for index, item in enumerate(raw):
        where = _where(index, item)
        if not isinstance(item, dict):
            problems.append(f"{where}: expected an object, got {type(item).__name__}")
            continue
        try:
            entry = _ENTRY.validate_python(item)
        except ValidationError as e:
            for err in e.errors()[:_MAX_PROBLEMS_PER_ENTRY]:
                loc = ".".join(str(p) for p in err["loc"])
                problems.append(f"{where}: {loc}: {err['msg']}")
            continue
        entry_problems: list[str] = []
        code = (entry.occupation.code or "").strip()
        if not code:
            entry_problems.append("occupation.code is missing")
        elif code in first_seen:
            entry_problems.append(
                f"occupation.code {code!r} repeats entry {first_seen[code]}"
            )
        if not (entry.occupation.preferred_label or "").strip():
            entry_problems.append("occupation.preferred_label is missing")
        for name in ("essential", "optional"):
            block = getattr(entry.skills, name)
            if block is not None and len(block.labels) > len(block.uuids):
                entry_problems.append(
                    f"skills.{name} has {len(block.labels)} labels for {len(block.uuids)} uuids"
                )
        if entry_problems:
            problems.extend(f"{where}: {p}" for p in entry_problems)
            continue
        first_seen[code] = index
        parsed.append(
            ParsedOccupation(position=index, code=code, entry=entry, raw=item)
        )
    if problems:
        raise OccupationFileInvalid(problems)
    return parsed


EmbeddingOrigin = Literal["stored", "seed", "embedded", "pending"]


class ResolvedEmbedding(DomainModel):
    """An occupation's raw concat vector and where it came from."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text_sha256: str = Field(description="SHA-256 of the embedded text.")
    vector: np.ndarray | None = Field(
        repr=False, description="L2-normalised float32 vector; None while pending."
    )
    origin: EmbeddingOrigin = Field(
        description="Stored document, seed NPZ, fresh embedding, or not embedded yet (dry run)."
    )


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def resolve_embeddings(
    occupations: Sequence[ParsedOccupation],
    *,
    stored: Mapping[str, OccupationDocument],
    seed: Mapping[str, np.ndarray],
    embed: Callable[[list[str]], np.ndarray] | None,
    model: str,
    dim: int,
) -> dict[str, ResolvedEmbedding]:
    """The raw vector for every occupation, embedding only what is new or changed.

    Order of preference: the stored vector when it was made by ``model`` from the same text (so a
    re-run never re-embeds, and Gemini's non-determinism never shows up as a change); a ``seed``
    vector (e.g. the committed NPZ); a fresh ``embed`` call over all remaining texts; else pending.

    :param embed: Texts -> ``[n, dim]`` vectors; None in a dry run
    """
    out: dict[str, ResolvedEmbedding] = {}
    to_embed: list[tuple[str, str, str]] = []  # (code, text, text_sha256)
    for occ in occupations:
        text = occupation_embedding_text(occ.entry)
        text_sha = sha256_text(text)
        doc = stored.get(occ.code)
        emb = doc.embedding if doc else None
        stored_vec = (
            vector_from_bin(emb.vector_bin, dim=dim)
            if emb and emb.model == model and emb.text_sha256 == text_sha
            else None
        )
        seed_vec = seed.get(occ.code)
        if stored_vec is not None:
            out[occ.code] = ResolvedEmbedding(
                text_sha256=text_sha, vector=stored_vec, origin="stored"
            )
        elif seed_vec is not None and seed_vec.shape == (dim,):
            out[occ.code] = ResolvedEmbedding(
                text_sha256=text_sha,
                vector=np.ascontiguousarray(seed_vec, dtype=np.float32),
                origin="seed",
            )
        else:
            to_embed.append((occ.code, text, text_sha))

    if to_embed and embed is not None:
        vectors = embed([text for _, text, _ in to_embed])
        if vectors.shape != (len(to_embed), dim):
            raise RuntimeError(
                f"embedding returned shape {vectors.shape}, expected {(len(to_embed), dim)}"
            )
        for (code, _, text_sha), vec in zip(
            to_embed, _l2_normalise(vectors), strict=True
        ):
            out[code] = ResolvedEmbedding(
                text_sha256=text_sha, vector=vec, origin="embedded"
            )
    else:
        for code, _, text_sha in to_embed:
            out[code] = ResolvedEmbedding(
                text_sha256=text_sha, vector=None, origin="pending"
            )
    return out


def build_occupation_document(
    occupation: ParsedOccupation,
    embedding: ResolvedEmbedding,
    *,
    whitened: np.ndarray | None,
    whitening_artifact_sha256: str | None,
    model: str,
    dim: int,
    group_codes: list[str],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """The stored document of one occupation (timestamps are added by the writer)."""
    vector = embedding.vector
    vector_bin = vector_to_bin(vector) if vector is not None else None
    whitened_bin = vector_to_bin(whitened) if whitened is not None else None
    return {
        **occupation.raw,
        "_id": occupation.code,
        "code": occupation.code,
        "position": occupation.position,
        "group_codes": group_codes,
        "content_sha256": content_sha256(occupation.raw, group_codes),
        "embedding": {
            "model": model,
            "dim": dim,
            "normalized": True,
            "text_sha256": embedding.text_sha256,
            "vector_sha256": (
                sha256_bytes(vector_bin) if vector_bin is not None else None
            ),
            "vector_bin": vector_bin,
            "whitened_vector_bin": whitened_bin,
            "whitening": {
                "enabled": whitened_bin is not None,
                "artifact_sha256": (
                    whitening_artifact_sha256 if whitened_bin is not None else None
                ),
            },
        },
        "source": dict(source),
    }


class IngestPlan(DomainModel):
    """What writing the desired documents over the stored ones would change (codes, sorted)."""

    added: list[str] = Field(description="Not stored yet.")
    updated: list[str] = Field(
        description="Stored with different content or embedding."
    )
    unchanged: list[str] = Field(description="Stored as is; not written.")
    removed: list[str] = Field(
        description="Stored but not in the file; deleted only with prune."
    )
    prune: bool = Field(description="Whether ``removed`` will be deleted.")


def document_signature(doc: Mapping[str, Any]) -> tuple[object, ...]:
    """What makes two stored documents the same for ingest purposes (not timestamps or source)."""
    emb = doc.get("embedding") or {}
    whitening = emb.get("whitening") or {}
    return (
        doc.get("content_sha256"),
        doc.get("position"),
        emb.get("model"),
        emb.get("text_sha256"),
        emb.get("vector_sha256"),
        whitening.get("artifact_sha256"),
    )


def plan_ingest(
    desired: Mapping[str, Mapping[str, Any]],
    stored: Mapping[str, Mapping[str, Any]],
    *,
    prune: bool,
) -> IngestPlan:
    """Diff desired documents against stored ones, both keyed by ``_id``."""
    added: list[str] = []
    updated: list[str] = []
    unchanged: list[str] = []
    for key, doc in desired.items():
        current = stored.get(key)
        if current is None:
            added.append(key)
        elif document_signature(current) != document_signature(doc):
            updated.append(key)
        else:
            unchanged.append(key)
    removed = [key for key in stored if key not in desired]
    return IngestPlan(
        added=sorted(added),
        updated=sorted(updated),
        unchanged=sorted(unchanged),
        removed=sorted(removed),
        prune=prune,
    )
