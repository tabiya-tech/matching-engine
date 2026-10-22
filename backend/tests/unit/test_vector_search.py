"""Atlas Vector Search plumbing (``app.server_dependencies.vector_search``) and the repositories'
``search_stage1`` / ``count_stage1_coverage``, against in-process fake collections (never Atlas)."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.clients.gemini_embedding_client import EMBEDDING_DIM
from app.jobs.job_document import RANKED_JOB_SEARCH_PROJECTION
from app.jobs.mongo_repository import MongoJobsRepository
from app.ranking.vector_retrieval import LocationTier, job_prefilter
from app.server_dependencies.vector_search import (
    JOBS_VECTOR_INDEX,
    MAX_NUM_CANDIDATES,
    VectorIndexSpec,
    VectorSearchQuery,
    count_stage1_coverage,
    ensure_vector_index,
    num_candidates_for,
    score_to_cosine,
    vector_index_status,
    vector_search_pipeline,
)
from tests.domain_factories import make_user
from tests.unit.test_vector_retrieval import matches

SHA = "sha-deployed"


def _filter_paths(flt: Mapping[str, Any]) -> set[str]:
    if "$and" in flt:
        return set().union(*(_filter_paths(c) for c in flt["$and"]))
    return set(flt)


def _query(**overrides: Any) -> VectorSearchQuery:
    fields: dict[str, Any] = {
        "vector": [0.1, 0.2],
        "filter": {"is_active": {"$eq": True}},
        "limit": 5,
        "num_candidates": 75,
    }
    fields.update(overrides)
    return VectorSearchQuery(**fields)


# --------------------------------------------------------------------------------------------------
# index definitions and query
# --------------------------------------------------------------------------------------------------


class TestIndexDefinitions:
    def test_jobs_index_is_a_cosine_vector_index_on_job_embedding(self):
        vector, *filters = JOBS_VECTOR_INDEX.definition()["fields"]

        assert vector == {
            "type": "vector",
            "path": "job_embedding",
            "numDimensions": EMBEDDING_DIM,
            "similarity": "cosine",
        }
        assert {f["path"] for f in filters} == {
            "is_active",
            "llm_reranker_meta.embedding.whitening.artifact_sha256",
            "location_keys",
            "requires_post_secondary",
        }
        assert all(f["type"] == "filter" for f in filters)

    def test_every_job_prefilter_field_is_declared_on_the_index(self):
        """Atlas rejects a ``$vectorSearch`` filter on a field the index does not declare."""
        flt = job_prefilter(
            make_user(any_post_secondary_educ=0),
            LocationTier(keys=("nairobi",)),
            artifact_sha256=SHA,
        )

        assert _filter_paths(flt) <= set(JOBS_VECTOR_INDEX.filter_paths)

    def test_search_index_model_is_a_vector_search_index(self):
        model = JOBS_VECTOR_INDEX.model()

        assert model.document["type"] == "vectorSearch"
        assert model.document["name"] == JOBS_VECTOR_INDEX.name


class TestPipeline:
    def test_approximate_search_then_projection_with_the_score(self):
        pipeline = vector_search_pipeline(
            JOBS_VECTOR_INDEX, _query(), RANKED_JOB_SEARCH_PROJECTION
        )

        assert pipeline[0] == {
            "$vectorSearch": {
                "index": JOBS_VECTOR_INDEX.name,
                "path": "job_embedding",
                "queryVector": [0.1, 0.2],
                "limit": 5,
                "numCandidates": 75,
                "filter": {"is_active": {"$eq": True}},
            }
        }
        assert pipeline[1]["$project"]["score"] == {"$meta": "vectorSearchScore"}

    def test_the_vectors_never_come_back(self):
        project = vector_search_pipeline(
            JOBS_VECTOR_INDEX, _query(), RANKED_JOB_SEARCH_PROJECTION
        )[1]["$project"]

        assert "job_embedding" not in project
        assert "concat_skill_embedding_gemini" not in project
        assert project["classifier_metadata.title"] == 1

    def test_exact_search_has_no_num_candidates(self):
        stage = vector_search_pipeline(
            JOBS_VECTOR_INDEX, _query(exact=True), {"_id": 1}
        )[0]["$vectorSearch"]

        assert stage["exact"] is True
        assert "numCandidates" not in stage

    @pytest.mark.parametrize(
        "limit,factor,expected",
        [(100, 15, 1500), (10, 1, 10), (5_000, 15, MAX_NUM_CANDIDATES)],
    )
    def test_num_candidates(self, limit, factor, expected):
        assert num_candidates_for(limit, factor) == expected

    @pytest.mark.parametrize("score,cosine", [(1.0, 1.0), (0.5, 0.0), (0.6, 0.2)])
    def test_score_to_cosine(self, score, cosine):
        assert score_to_cosine(score) == pytest.approx(cosine)


# --------------------------------------------------------------------------------------------------
# index management
# --------------------------------------------------------------------------------------------------


class _IndexCollection:
    def __init__(self, existing: list[dict[str, Any]]):
        self.existing = existing
        self.create_search_index = AsyncMock()
        self.update_search_index = AsyncMock()

    def list_search_indexes(self, name: str):
        async def gen():
            for index in self.existing:
                if index["name"] == name:
                    yield index

        return gen()


def _db(collection: object) -> MagicMock:
    db = MagicMock()
    db.get_collection.return_value = collection
    return db


class TestEnsureVectorIndex:
    def test_creates_a_missing_index(self):
        collection = _IndexCollection([])

        assert (
            asyncio.run(ensure_vector_index(_db(collection), JOBS_VECTOR_INDEX))
            == "created"
        )
        (model,), _ = collection.create_search_index.call_args
        assert model.document["definition"] == JOBS_VECTOR_INDEX.definition()

    def test_leaves_a_matching_index_alone_whatever_the_field_order(self):
        definition = JOBS_VECTOR_INDEX.definition()
        stored = {"fields": list(reversed(definition["fields"]))}
        collection = _IndexCollection(
            [{"name": JOBS_VECTOR_INDEX.name, "latestDefinition": stored}]
        )

        assert (
            asyncio.run(ensure_vector_index(_db(collection), JOBS_VECTOR_INDEX))
            == "unchanged"
        )
        collection.create_search_index.assert_not_called()
        collection.update_search_index.assert_not_called()

    def test_updates_an_index_whose_definition_differs(self):
        stale = {"fields": JOBS_VECTOR_INDEX.definition()["fields"][:2]}
        collection = _IndexCollection(
            [{"name": JOBS_VECTOR_INDEX.name, "latestDefinition": stale}]
        )

        assert (
            asyncio.run(ensure_vector_index(_db(collection), JOBS_VECTOR_INDEX))
            == "updated"
        )
        collection.update_search_index.assert_awaited_once_with(
            JOBS_VECTOR_INDEX.name, JOBS_VECTOR_INDEX.definition()
        )

    def test_status_reports_existence_queryability_and_definition(self):
        spec = VectorIndexSpec(
            collection="c", name="i", vector_path="v", filter_paths=("f",)
        )
        collection = _IndexCollection(
            [
                {
                    "name": "i",
                    "status": "BUILDING",
                    "queryable": False,
                    "latestDefinition": spec.definition(),
                }
            ]
        )

        status = asyncio.run(vector_index_status(_db(collection), spec))
        missing = asyncio.run(vector_index_status(_db(_IndexCollection([])), spec))

        assert (status.exists, status.queryable, status.status) == (
            True,
            False,
            "BUILDING",
        )
        assert status.definition_matches
        assert (missing.exists, missing.queryable) == (False, False)


# --------------------------------------------------------------------------------------------------
# coverage
# --------------------------------------------------------------------------------------------------


class _CountingCollection:
    name = "ranked_jobs"

    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs

    async def count_documents(self, flt: Mapping[str, Any]) -> int:
        return sum(1 for d in self.docs if _matches_count(d, flt))


def _matches_count(doc: Mapping[str, Any], flt: Mapping[str, Any]) -> bool:
    # count_documents filters: a top-level AND of equality / $ne / $nin, None meaning missing.
    return matches(doc, {"$and": [{k: v} for k, v in flt.items()]}) if flt else True


def _job_doc(
    *, active: bool = True, vector: bool = True, sha: str | None = SHA
) -> dict:
    doc: dict[str, Any] = {"is_active": active}
    if vector:
        doc["job_embedding"] = [0.1]
    if sha is not None:
        doc["llm_reranker_meta"] = {
            "embedding": {"whitening": {"artifact_sha256": sha}}
        }
    return doc


def test_coverage_splits_active_jobs_by_search_eligibility():
    collection = _CountingCollection(
        [
            _job_doc(),
            _job_doc(),
            _job_doc(sha="sha-old"),
            _job_doc(sha=None),
            _job_doc(vector=False, sha=None),
            _job_doc(active=False),
        ]
    )

    coverage = asyncio.run(
        MongoJobsRepository(_db(collection)).count_stage1_coverage(SHA)
    )

    assert coverage.model_dump() == {
        "collection": "ranked_jobs",
        "artifact_sha256": SHA,
        "n_total": 5,
        "n_searchable": 2,
        "n_other_artifact": 1,
        "n_not_whitened": 1,
        "n_no_embedding": 1,
    }
    assert coverage.n_excluded == 3


def test_coverage_counts_any_collection_with_the_given_paths():
    collection = _CountingCollection(
        [
            {"embedding": {"vector_bin": b"x", "whitening": {"artifact_sha256": SHA}}},
            {"embedding": {"vector_bin": b"x"}},
            {"embedding": {}},
        ]
    )

    coverage = asyncio.run(
        count_stage1_coverage(
            collection,  # type: ignore[arg-type]
            base_filter={},
            vector_path="embedding.vector_bin",
            whitening_sha256_path="embedding.whitening.artifact_sha256",
            artifact_sha256=SHA,
        )
    )

    assert (
        coverage.n_searchable,
        coverage.n_not_whitened,
        coverage.n_no_embedding,
    ) == (1, 1, 1)


# --------------------------------------------------------------------------------------------------
# repositories
# --------------------------------------------------------------------------------------------------


class _AggregateCollection:
    def __init__(self, docs: list[dict[str, Any]]):
        self.docs = docs
        self.pipelines: list[list[dict[str, Any]]] = []

    def aggregate(self, pipeline: list[dict[str, Any]]):
        self.pipelines.append(pipeline)

        async def gen():
            for d in self.docs:
                yield d

        return gen()


def test_jobs_search_maps_hits_and_skips_invalid_and_inactive_documents():
    collection = _AggregateCollection(
        [
            {
                "job_id": "a",
                "is_active": True,
                "classifier_metadata": {"title": "A", "city": "Nairobi"},
                "score": 0.8,
            },
            {
                "job_id": "b",
                "is_active": False,
                "classifier_metadata": {},
                "score": 0.7,
            },
            {
                "job_id": "c",
                "is_active": True,
                "classifier_metadata": "nope",
                "score": 0.6,
            },
        ]
    )
    query = _query()

    result = asyncio.run(MongoJobsRepository(_db(collection)).search_stage1(query))

    assert [(h.job.uuid, round(h.cosine, 6)) for h in result.hits] == [("a", 0.6)]
    assert (result.n_returned, result.n_skipped_inactive, result.n_skipped_invalid) == (
        3,
        1,
        1,
    )
    assert result.hits[0].job.embedding is None
    assert collection.pipelines == [
        vector_search_pipeline(JOBS_VECTOR_INDEX, query, RANKED_JOB_SEARCH_PROJECTION)
    ]


# --------------------------------------------------------------------------------------------------
# startup report
# --------------------------------------------------------------------------------------------------


def test_startup_logs_excluded_documents_loudly(caplog):
    from app.server_dependencies.vector_search import Stage1Coverage
    from app.warmup import log_stage1_coverage

    with caplog.at_level("INFO", logger="app.warmup"):
        log_stage1_coverage(
            Stage1Coverage(
                collection="ranked_jobs",
                artifact_sha256=SHA,
                n_total=10,
                n_searchable=6,
                n_other_artifact=1,
                n_not_whitened=2,
                n_no_embedding=1,
            )
        )

    levels = {r.levelname: r.getMessage() for r in caplog.records}
    assert "6/10 searchable" in levels["INFO"]
    assert "another artifact" in levels["ERROR"] and "never mixed" in levels["ERROR"]
    assert (
        "3 documents" in levels["WARNING"] and "until backfilled" in levels["WARNING"]
    )
