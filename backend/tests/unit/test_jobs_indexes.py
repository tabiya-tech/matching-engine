"""Tests for the jobs collection index definitions and the idempotent ensure step."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app import database
from app.database import JOBS_INDEX_MODELS, ensure_jobs_indexes


class TestJobsIndexModels:
    def test_every_index_has_a_unique_name(self):
        names = [m.document["name"] for m in JOBS_INDEX_MODELS]
        assert len(names) == len(set(names))

    def test_every_index_leads_with_is_active(self):
        # Every jobs query filters is_active, so it must be the leading key of each compound index.
        for model in JOBS_INDEX_MODELS:
            first_key = list(model.document["key"].items())[0][0]
            assert first_key == "is_active"

    def test_browse_index_sorts_by_id_descending(self):
        # The keyset browse (sort _id desc) needs an {is_active: 1, _id: -1} index to avoid a scan.
        keys = {m.document["name"]: list(m.document["key"].items()) for m in JOBS_INDEX_MODELS}
        assert keys["is_active_-_id"] == [("is_active", 1), ("_id", -1)]

    def test_employment_type_index_ends_with_id_so_sort_is_index_served(self):
        # A type-filtered browse still sorts by _id desc; the index must end with _id to serve both
        # the equality and the sort (otherwise the planner falls back to the plain {is_active,_id} index).
        keys = {m.document["name"]: list(m.document["key"].items()) for m in JOBS_INDEX_MODELS}
        assert keys["is_active_employment_type_-_id"] == [
            ("is_active", 1),
            ("classifier_metadata.employment_type", 1),
            ("_id", -1),
        ]


class TestEnsureJobsIndexes:
    def test_calls_create_indexes_with_the_models_and_returns_names(self):
        # GIVEN a collection that reports the created index names
        mock_collection = MagicMock()
        mock_collection.create_indexes = AsyncMock(return_value=["is_active_-_id", "is_active_category"])
        mock_db = MagicMock()
        mock_db.__getitem__.return_value = mock_collection

        with patch.object(database, "db", mock_db):
            # WHEN ensuring indexes
            actual = asyncio.run(ensure_jobs_indexes())

        # THEN create_indexes is called once with the module's index models, and names returned
        mock_collection.create_indexes.assert_awaited_once_with(JOBS_INDEX_MODELS)
        assert actual == ["is_active_-_id", "is_active_category"]
