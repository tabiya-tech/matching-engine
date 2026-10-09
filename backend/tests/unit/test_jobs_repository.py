"""Tests for MongoJobsRepository query construction and shaping, against an in-process fake collection."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from app.jobs.job_document import RANKED_JOB_FIND_PROJECTION, RANKED_JOBS_ACTIVE_FILTER
from app.jobs.mongo_repository import MongoJobsRepository
from app.jobs.pagination import InvalidCursor, decode_jobs_cursor, encode_jobs_cursor
from app.server_dependencies.database_collections import Collections


class _FakeCursor:
    """Records sort/limit and yields the given documents (honouring limit) like a Motor cursor."""

    def __init__(self, docs: list[dict]):
        self._docs = docs
        self.sort_spec: list | None = None
        self.limit_n: int | None = None

    def sort(self, spec):
        self.sort_spec = spec
        return self

    def limit(self, n):
        self.limit_n = n
        return self

    def __aiter__(self):
        docs = self._docs if self.limit_n is None else self._docs[: self.limit_n]
        return _aiter(docs)


async def _aiter(docs):
    for d in docs:
        yield d


class _FakeCollection:
    def __init__(
        self, docs: list[dict], distinct_values: dict[str, list] | None = None
    ):
        self._docs = docs
        self.find_calls: list[tuple] = []
        self.cursors: list[_FakeCursor] = []
        self.count_documents = AsyncMock(return_value=len(docs))
        self._distinct_values = distinct_values or {}

    def find(self, *args):
        self.find_calls.append(args)
        cursor = _FakeCursor(self._docs)
        self.cursors.append(cursor)
        return cursor

    async def distinct(self, field: str, _filter: dict[str, Any]):
        return self._distinct_values.get(field, [])


def _repository(collection: _FakeCollection) -> MongoJobsRepository:
    db = MagicMock()
    db.get_collection.return_value = collection
    repository = MongoJobsRepository(db)
    db.get_collection.assert_called_once_with(Collections.JOBS)
    return repository


def _doc(job_id: str, *, active: bool = True, oid: ObjectId | None = None) -> dict:
    return {
        "_id": oid or ObjectId(),
        "job_id": job_id,
        "is_active": active,
        "classifier_metadata": {"title": f"Title {job_id}", "city": "Nairobi"},
    }


class TestFindJobsWithTiming:
    def test_no_filter_loads_every_active_job_unsorted_and_uncapped(self):
        # GIVEN two active jobs and one inactive one
        collection = _FakeCollection([_doc("a"), _doc("b"), _doc("c", active=False)])

        # WHEN loading without a filter (a limit is ignored without a filter)
        jobs, timing = asyncio.run(
            _repository(collection).find_jobs_with_timing(None, limit=1)
        )

        # THEN the active filter and projection are used, with no sort or cap
        assert collection.find_calls == [
            (RANKED_JOBS_ACTIVE_FILTER, RANKED_JOB_FIND_PROJECTION)
        ]
        assert collection.cursors[0].sort_spec is None
        assert collection.cursors[0].limit_n is None
        # AND inactive documents are skipped in the build step
        assert [j["uuid"] for j in jobs] == ["a", "b"]
        assert timing["n_ranked_raw"] == 3
        assert timing["n_jobs"] == 2
        assert timing["n_skipped_inactive"] == 1
        assert timing["jobs_find_use_projection"] is True

    def test_filter_is_sorted_newest_first_and_capped(self):
        collection = _FakeCollection([_doc("a"), _doc("b"), _doc("c")])
        query_filter = {
            "$and": [RANKED_JOBS_ACTIVE_FILTER, {"classifier_metadata.city": "Nairobi"}]
        }

        jobs, _ = asyncio.run(
            _repository(collection).find_jobs_with_timing(query_filter, limit=2)
        )

        assert collection.find_calls[0][0] == query_filter
        assert collection.cursors[0].sort_spec == [("_id", -1)]
        assert collection.cursors[0].limit_n == 2
        assert len(jobs) == 2

    def test_zero_limit_means_no_cap(self):
        collection = _FakeCollection([_doc("a")])

        asyncio.run(
            _repository(collection).find_jobs_with_timing(
                {"is_active": True, "x": 1}, limit=0
            )
        )

        assert collection.cursors[0].limit_n is None


class TestGetJobsPage:
    def test_fetches_one_extra_document_to_detect_another_page(self):
        # GIVEN three jobs and a page size of two
        oids = [ObjectId() for _ in range(3)]
        collection = _FakeCollection([_doc(str(i), oid=o) for i, o in enumerate(oids)])

        # WHEN fetching the first page
        jobs, next_cursor, total, timing = asyncio.run(
            _repository(collection).get_jobs_page_with_timing(limit=2)
        )

        # THEN limit+1 is requested newest-first, two jobs are returned, and the cursor wraps the last one
        assert collection.cursors[0].limit_n == 3
        assert collection.cursors[0].sort_spec == [("_id", -1)]
        assert [j.uuid for j in jobs] == ["0", "1"]
        assert decode_jobs_cursor(next_cursor) == oids[1]
        assert timing["has_more"] is True
        assert total is None

    def test_last_page_has_no_cursor(self):
        collection = _FakeCollection([_doc("a")])

        _, next_cursor, _, timing = asyncio.run(
            _repository(collection).get_jobs_page_with_timing(limit=5)
        )

        assert next_cursor is None
        assert timing["has_more"] is False

    def test_cursor_is_composed_with_the_filter(self):
        oid = ObjectId()
        collection = _FakeCollection([])

        asyncio.run(
            _repository(collection).get_jobs_page_with_timing(
                cursor=encode_jobs_cursor(oid), employment_type="full_time"
            )
        )

        query_filter = collection.find_calls[0][0]
        assert query_filter["$and"][1] == {"_id": {"$lt": oid}}
        assert {"classifier_metadata.employment_type": "full_time"} in query_filter[
            "$and"
        ][0]["$and"]

    def test_total_counts_the_filter_without_the_cursor(self):
        collection = _FakeCollection([_doc("a")])

        _, _, total, _ = asyncio.run(
            _repository(collection).get_jobs_page_with_timing(
                cursor=encode_jobs_cursor(ObjectId()), include_total=True
            )
        )

        assert total == 1
        collection.count_documents.assert_awaited_once_with(
            dict(RANKED_JOBS_ACTIVE_FILTER)
        )

    def test_malformed_cursor_raises(self):
        with pytest.raises(InvalidCursor):
            asyncio.run(
                _repository(_FakeCollection([])).get_jobs_page_with_timing(
                    cursor="garbage"
                )
            )


class TestGetJobsStats:
    def test_counts_sectors_and_platforms_case_insensitively(self):
        collection = _FakeCollection(
            [_doc("a"), _doc("b")],
            distinct_values={
                "classifier_metadata.category": ["Health", "health ", "Education", ""],
                "classifier_metadata.source_platform": ["BrighterMonday"],
                "classifier_metadata.source": ["brightermonday", "Fuzu"],
            },
        )

        actual = asyncio.run(_repository(collection).get_jobs_stats())

        assert (actual.total, actual.sectors, actual.platforms) == (2, 2, 2)

    def test_sectors_fall_back_to_the_isco_group(self):
        collection = _FakeCollection(
            [],
            distinct_values={
                "classifier_metadata.isco_occupation_group": ["Nurses", "Drivers"]
            },
        )

        actual = asyncio.run(_repository(collection).get_jobs_stats())

        assert actual.sectors == 2
