"""Tests for the jobs collection index definitions and the idempotent ensure step."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.server_dependencies.database_collections import Collections
from app.server_dependencies.db_dependencies import (
    JOBS_INDEX_MODELS,
    MatchingDBProvider,
)


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
        keys = {
            m.document["name"]: list(m.document["key"].items())
            for m in JOBS_INDEX_MODELS
        }
        assert keys["is_active_-_id"] == [("is_active", 1), ("_id", -1)]

    def test_employment_type_index_ends_with_id_so_sort_is_index_served(self):
        # A type-filtered browse still sorts by _id desc; the index must end with _id to serve both
        # the equality and the sort (otherwise the planner falls back to the plain {is_active,_id} index).
        keys = {
            m.document["name"]: list(m.document["key"].items())
            for m in JOBS_INDEX_MODELS
        }
        assert keys["is_active_employment_type_-_id"] == [
            ("is_active", 1),
            ("classifier_metadata.employment_type", 1),
            ("_id", -1),
        ]


class TestInitializeJobsMongoDb:
    def test_calls_create_indexes_with_the_models_and_returns_names(self):
        # GIVEN a jobs database whose collection reports the created index names
        mock_collection = MagicMock()
        mock_collection.create_indexes = AsyncMock(
            return_value=["is_active_-_id", "is_active_category"]
        )
        mock_db = MagicMock()
        mock_db.get_collection.return_value = mock_collection

        # WHEN initializing the database
        actual = asyncio.run(
            MatchingDBProvider.initialize_jobs_mongo_db(
                mock_db, logging.getLogger(__name__)
            )
        )

        # THEN create_indexes is called once on the jobs collection with the index models, and names returned
        mock_db.get_collection.assert_called_once_with(Collections.JOBS)
        mock_collection.create_indexes.assert_awaited_once_with(JOBS_INDEX_MODELS)
        assert actual == ["is_active_-_id", "is_active_category"]

    def test_failure_is_raised_to_the_caller(self):
        # GIVEN a collection whose index creation fails
        mock_collection = MagicMock()
        mock_collection.create_indexes = AsyncMock(side_effect=RuntimeError("boom"))
        mock_db = MagicMock()
        mock_db.get_collection.return_value = mock_collection

        # WHEN initializing the database THEN the error propagates (the warmup decides it is non-fatal)
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(
                MatchingDBProvider.initialize_jobs_mongo_db(
                    mock_db, logging.getLogger(__name__)
                )
            )
