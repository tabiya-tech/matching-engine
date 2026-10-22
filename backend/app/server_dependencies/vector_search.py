"""Atlas Vector Search for stage-1 job retrieval: the jobs vector index, the ``$vectorSearch`` query,
and the index checks run by ``scripts/vector_search_indexes.py`` and the startup warmup.

The index is over pre-whitened, unit-length concat embeddings (whitening cannot be moved to the
query side: it re-normalises every document vector), so the search cosine is the whitened cosine
the in-process path computes. Every filter the retriever uses must be declared here as a ``filter``
field, or Atlas rejects the query.

* Jobs (``Collections.JOBS``): ``job_embedding``, filtered by ``is_active``, the whitening
  artifact's SHA-256, ``location_keys`` and ``requires_post_secondary``.

Index definitions change rarely and an Atlas rebuild is slow, so they are created by the script, not
at startup; the warmup only reports whether they exist and are queryable.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pydantic import Field
from pymongo.operations import SearchIndexModel

from app.clients.gemini_embedding_client import EMBEDDING_DIM
from app.config import JOBS_VECTOR_INDEX_NAME
from app.domain.base import DomainModel
from app.jobs.job_document import (
    JOB_EMBEDDING_FIELD,
    JOB_REQUIRES_POST_SECONDARY_FIELD,
    JOB_WHITENING_SHA256_FIELD,
    LOCATION_KEYS_FIELD,
)

from .database_collections import Collections

logger = logging.getLogger(__name__)

# Atlas rejects a numCandidates above this.
MAX_NUM_CANDIDATES = 10_000

# The stored vectors are unit length, so cosine and dotProduct rank the same; cosine keeps the score
# right for a vector a little off unit length after float32 storage.
_SIMILARITY = "cosine"


class VectorIndexSpec(DomainModel):
    """One vector index: the collection it is on and its Atlas definition."""

    collection: str = Field(description="Collection name.")
    name: str = Field(description="Index name, used by ``$vectorSearch.index``.")
    vector_path: str = Field(description="Field holding the indexed vector.")
    filter_paths: tuple[str, ...] = Field(
        description="Fields the query may pre-filter on."
    )

    def definition(self) -> dict[str, Any]:
        """The ``vectorSearch`` index definition sent to Atlas."""
        return {
            "fields": [
                {
                    "type": "vector",
                    "path": self.vector_path,
                    "numDimensions": EMBEDDING_DIM,
                    "similarity": _SIMILARITY,
                },
                *({"type": "filter", "path": p} for p in self.filter_paths),
            ]
        }

    def model(self) -> SearchIndexModel:
        return SearchIndexModel(
            definition=self.definition(), name=self.name, type="vectorSearch"
        )


JOBS_VECTOR_INDEX = VectorIndexSpec(
    collection=Collections.JOBS,
    name=JOBS_VECTOR_INDEX_NAME,
    vector_path=JOB_EMBEDDING_FIELD,
    filter_paths=(
        "is_active",
        JOB_WHITENING_SHA256_FIELD,
        LOCATION_KEYS_FIELD,
        JOB_REQUIRES_POST_SECONDARY_FIELD,
    ),
)

VECTOR_INDEXES: tuple[VectorIndexSpec, ...] = (JOBS_VECTOR_INDEX,)


class VectorSearchQuery(DomainModel):
    """One ``$vectorSearch``: the whitened user vector, the pre-filter, and how many to return."""

    vector: list[float] = Field(
        description="Whitened, unit-length query vector (EMBEDDING_DIM long)."
    )
    filter: dict[str, Any] = Field(
        description="MQL pre-filter on the index's filter fields (applied inside the search)."
    )
    limit: int = Field(ge=1, description="Documents returned.")
    num_candidates: int = Field(
        ge=1,
        description="Nearest neighbours considered (ignored when ``exact``).",
    )
    exact: bool = Field(
        default=False, description="Exact (ENN) instead of approximate search."
    )


def num_candidates_for(limit: int, factor: int) -> int:
    """``factor`` x ``limit``, at least ``limit`` and at most Atlas's cap."""
    return max(limit, min(MAX_NUM_CANDIDATES, limit * max(1, factor)))


def vector_search_pipeline(
    index: VectorIndexSpec, query: VectorSearchQuery, projection: Mapping[str, int]
) -> list[dict[str, Any]]:
    """``$vectorSearch`` then a ``$project`` of ``projection`` plus the search score as ``score``.

    The projection is inclusive and never lists the vector, so it does not come back.
    """
    stage: dict[str, Any] = {
        "index": index.name,
        "path": index.vector_path,
        "queryVector": query.vector,
        "limit": query.limit,
    }
    if query.exact:
        stage["exact"] = True
    else:
        stage["numCandidates"] = max(query.num_candidates, query.limit)
    if query.filter:
        stage["filter"] = query.filter
    return [
        {"$vectorSearch": stage},
        {"$project": {**projection, "score": {"$meta": "vectorSearchScore"}}},
    ]


def score_to_cosine(score: float) -> float:
    """Atlas's cosine ``vectorSearchScore`` is ``(1 + cosine) / 2``; this undoes it."""
    return 2.0 * float(score) - 1.0


class Stage1Coverage(DomainModel):
    """How a collection's documents stand for the vector search against one whitening artifact.

    Only ``n_searchable`` documents can be retrieved; the rest are excluded until the producer (the
    llm-reranker) re-whitens them.
    """

    collection: str = Field(description="Collection name.")
    artifact_sha256: str = Field(
        description="SHA-256 of the service's whitening artifact."
    )
    n_total: int = Field(description="Documents considered (active jobs).")
    n_searchable: int = Field(
        description="Whitened with the service's artifact: in the search."
    )
    n_other_artifact: int = Field(
        description="Whitened with another artifact: excluded (versions are never mixed)."
    )
    n_not_whitened: int = Field(
        description="Embedding present but not whitened: excluded until backfilled."
    )
    n_no_embedding: int = Field(
        description="No embedding at all: excluded until embedded."
    )

    @property
    def n_excluded(self) -> int:
        return self.n_total - self.n_searchable


async def count_stage1_coverage(
    collection: AsyncIOMotorCollection,
    *,
    base_filter: Mapping[str, Any],
    vector_path: str,
    whitening_sha256_path: str,
    artifact_sha256: str,
) -> Stage1Coverage:
    """Count ``Stage1Coverage`` for the documents of ``collection`` matching ``base_filter``."""

    async def count(extra: Mapping[str, Any]) -> int:
        return await collection.count_documents({**base_filter, **extra})

    has_vector = {vector_path: {"$ne": None}}
    n_total = await count({})
    n_no_embedding = await count({vector_path: None})
    n_searchable = await count({**has_vector, whitening_sha256_path: artifact_sha256})
    n_other = await count(
        {**has_vector, whitening_sha256_path: {"$nin": [None, artifact_sha256]}}
    )
    return Stage1Coverage(
        collection=collection.name,
        artifact_sha256=artifact_sha256,
        n_total=n_total,
        n_searchable=n_searchable,
        n_other_artifact=n_other,
        n_not_whitened=max(0, n_total - n_no_embedding - n_searchable - n_other),
        n_no_embedding=n_no_embedding,
    )


class VectorIndexStatus(DomainModel):
    """Whether one vector index exists, is queryable, and has the definition in code."""

    collection: str = Field(description="Collection name.")
    name: str = Field(description="Index name.")
    exists: bool = Field(description="An index with this name exists.")
    queryable: bool = Field(description="Atlas reports it queryable.")
    status: str | None = Field(
        default=None, description="Atlas status (``READY``, ``BUILDING``, …)."
    )
    definition_matches: bool = Field(
        description="The stored definition equals ``VectorIndexSpec.definition()``."
    )


def _same_definition(stored: Mapping[str, Any], wanted: Mapping[str, Any]) -> bool:
    def key(field: Mapping[str, Any]) -> tuple[str, str]:
        return (str(field.get("type")), str(field.get("path")))

    def norm(definition: Mapping[str, Any]) -> list[dict[str, Any]]:
        return sorted((dict(f) for f in definition.get("fields") or []), key=key)

    return norm(stored) == norm(wanted)


async def _find_index(
    collection: AsyncIOMotorCollection, name: str
) -> Mapping[str, Any] | None:
    cursor = collection.list_search_indexes(name)
    async for index in cursor:
        return index
    return None


async def vector_index_status(
    db: AsyncIOMotorDatabase, spec: VectorIndexSpec
) -> VectorIndexStatus:
    """The live state of ``spec`` on ``db``.

    :raises pymongo.errors.OperationFailure: If the deployment does not support search indexes
    """
    found = await _find_index(db.get_collection(spec.collection), spec.name)
    if found is None:
        return VectorIndexStatus(
            collection=spec.collection,
            name=spec.name,
            exists=False,
            queryable=False,
            definition_matches=False,
        )
    stored = found.get("latestDefinition") or found.get("definition") or {}
    return VectorIndexStatus(
        collection=spec.collection,
        name=spec.name,
        exists=True,
        queryable=bool(found.get("queryable")),
        status=found.get("status"),
        definition_matches=_same_definition(stored, spec.definition()),
    )


async def ensure_vector_index(db: AsyncIOMotorDatabase, spec: VectorIndexSpec) -> str:
    """Create ``spec`` if missing, update it if its definition differs; Atlas builds it async.

    :return: ``created``, ``updated`` or ``unchanged``
    :raises pymongo.errors.OperationFailure: If the deployment does not support search indexes
    """
    collection = db.get_collection(spec.collection)
    found = await _find_index(collection, spec.name)
    if found is None:
        await collection.create_search_index(spec.model())
        return "created"
    stored = found.get("latestDefinition") or found.get("definition") or {}
    if _same_definition(stored, spec.definition()):
        return "unchanged"
    await collection.update_search_index(spec.name, spec.definition())
    return "updated"
