"""Tests for the MongoDB client lifecycle: provider, client options, dependency wiring, and app lifespan."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.jobs.get_jobs_repository import get_jobs_repository, standalone_jobs_repository
from app.jobs.mongo_repository import MongoJobsRepository
from app.server_dependencies.database_collections import Collections
from app.server_dependencies.db_dependencies import (
    MatchingDBProvider,
    MongoDbSettings,
    build_client_kwargs,
)

_LOGGER = logging.getLogger(__name__)
_GET_JOBS_DB = "app.server_dependencies.db_dependencies._get_jobs_db"


@pytest.fixture(autouse=True)
def _fresh_provider():
    MatchingDBProvider.clear_cache()
    yield
    MatchingDBProvider.clear_cache()


def _settings(**overrides) -> MongoDbSettings:
    return MongoDbSettings(
        **{
            "mongo_url": "mongodb://localhost:27017",
            "database_name": "test",
            **overrides,
        }
    )


class TestMatchingDBProvider:
    def test_creates_the_client_once_and_reuses_it(self):
        # GIVEN a client factory
        with patch(_GET_JOBS_DB, return_value=MagicMock()) as factory:
            # WHEN the database is requested several times, concurrently and sequentially
            async def _run():
                first, second = await asyncio.gather(
                    MatchingDBProvider.get_jobs_db(), MatchingDBProvider.get_jobs_db()
                )
                third = await MatchingDBProvider.get_jobs_db()
                return first, second, third

            first, second, third = asyncio.run(_run())

        # THEN one client is created and every caller gets the same database
        factory.assert_called_once()
        assert first is second is third

    def test_close_closes_the_client_and_a_later_call_opens_a_new_one(self):
        # GIVEN an open database
        db_1, db_2 = MagicMock(), MagicMock()
        with patch(_GET_JOBS_DB, side_effect=[db_1, db_2]):
            assert asyncio.run(MatchingDBProvider.get_jobs_db()) is db_1

            # WHEN the provider is closed
            MatchingDBProvider.close()

            # THEN the client is closed, and the next request opens a fresh one
            db_1.client.close.assert_called_once()
            assert asyncio.run(MatchingDBProvider.get_jobs_db()) is db_2

    def test_close_without_an_open_client_is_a_no_op(self):
        MatchingDBProvider.close()

    def test_settings_are_read_on_first_use_not_at_import(self, monkeypatch):
        # GIVEN MONGO_URL is unset (the module was already imported without failing)
        monkeypatch.delenv("MONGO_URL", raising=False)

        # WHEN the database is requested THEN the missing setting is reported
        with pytest.raises(ValueError, match="MONGO_URL"):
            asyncio.run(MatchingDBProvider.get_jobs_db())

    def test_settings_from_env(self, monkeypatch):
        monkeypatch.setenv("MONGO_URL", "mongodb://db:27017")
        monkeypatch.setenv("MONGO_DB_NAME", "Kenya")
        monkeypatch.setenv("MONGO_MAX_POOL_SIZE", "7")
        monkeypatch.setenv("MONGO_TLS_INSECURE", "1")

        actual = MongoDbSettings.from_env()

        assert actual.mongo_url == "mongodb://db:27017"
        assert actual.database_name == "Kenya"
        assert actual.max_pool_size == 7
        assert actual.tls_allow_invalid_certificates is True


class TestBuildClientKwargs:
    def test_pool_and_timeout_options(self):
        actual = build_client_kwargs(
            _settings(
                server_selection_timeout_ms=1000, max_pool_size=0, min_pool_size=3
            ),
            _LOGGER,
        )
        assert actual == {
            "serverSelectionTimeoutMS": 1000,
            "maxPoolSize": 1,
            "minPoolSize": 3,
        }

    def test_plain_uri_gets_no_tls_options(self):
        actual = build_client_kwargs(_settings(), _LOGGER)
        assert "tlsCAFile" not in actual
        assert "tlsAllowInvalidCertificates" not in actual

    def test_tls_uri_uses_the_certifi_bundle(self):
        import certifi

        actual = build_client_kwargs(
            _settings(mongo_url="mongodb+srv://cluster.example.net"), _LOGGER
        )
        assert actual["tlsCAFile"] == certifi.where()

    def test_explicit_ca_file_wins_over_certifi(self):
        actual = build_client_kwargs(
            _settings(
                mongo_url="mongodb+srv://cluster.example.net", tls_ca_file="/etc/ca.pem"
            ),
            _LOGGER,
        )
        assert actual["tlsCAFile"] == "/etc/ca.pem"

    def test_certifi_literal_resolves_to_the_bundle_path(self):
        import certifi

        actual = build_client_kwargs(_settings(tls_ca_file="certifi"), _LOGGER)
        assert actual["tlsCAFile"] == certifi.where()

    def test_allow_invalid_certificates_skips_the_ca_bundle(self):
        actual = build_client_kwargs(
            _settings(
                mongo_url="mongodb+srv://cluster.example.net",
                tls_allow_invalid_certificates=True,
            ),
            _LOGGER,
        )
        assert actual["tlsAllowInvalidCertificates"] is True
        assert "tlsCAFile" not in actual


class TestRepositoryDependencies:
    def test_get_jobs_repository_binds_the_injected_database(self):
        # GIVEN the database the provider injects
        mock_db = MagicMock()

        # WHEN the dependency builds the repository
        actual = asyncio.run(get_jobs_repository(mock_db))

        # THEN it is a MongoJobsRepository reading the jobs collection of that database
        assert isinstance(actual, MongoJobsRepository)
        mock_db.get_collection.assert_called_once_with(Collections.JOBS)

    def test_standalone_repository_closes_the_client_on_exit(self):
        mock_db = MagicMock()
        with patch(_GET_JOBS_DB, return_value=mock_db):

            async def _run():
                async with standalone_jobs_repository() as repository:
                    assert isinstance(repository, MongoJobsRepository)
                    mock_db.client.close.assert_not_called()

            asyncio.run(_run())

        mock_db.client.close.assert_called_once()
        assert MatchingDBProvider._jobs_mongo_db is None


class TestLifespan:
    def test_client_is_created_at_startup_and_closed_at_shutdown(self):
        from app.main import app, lifespan

        mock_db = MagicMock()
        with (
            patch(_GET_JOBS_DB, return_value=mock_db) as factory,
            patch("app.main.warmup_on_startup", new_callable=AsyncMock) as warmup,
            patch("app.main.get_cross_encoder_client"),
            patch("app.main.shutdown_tracing"),
        ):

            async def _run():
                async with lifespan(app):
                    # Startup: one client, handed to the warmup, still open while serving.
                    factory.assert_called_once()
                    warmup.assert_awaited_once_with(mock_db)
                    mock_db.client.close.assert_not_called()

            asyncio.run(_run())

        # Shutdown: the client is closed and the provider forgets it.
        mock_db.client.close.assert_called_once()
        assert MatchingDBProvider._jobs_mongo_db is None

    def test_background_warmup_is_cancelled_before_the_client_closes(self, monkeypatch):
        from app.main import app, lifespan

        monkeypatch.setenv("WARMUP_NON_BLOCKING", "1")
        mock_db = MagicMock()

        async def _slow_warmup(_db):
            await asyncio.sleep(3600)

        with (
            patch(_GET_JOBS_DB, return_value=mock_db),
            patch("app.main.warmup_on_startup", side_effect=_slow_warmup),
            patch("app.main.get_cross_encoder_client"),
            patch("app.main.shutdown_tracing"),
        ):

            async def _run():
                async with lifespan(app):
                    await asyncio.sleep(0)  # let the background warmup start
                return [
                    t for t in asyncio.all_tasks() if t is not asyncio.current_task()
                ]

            remaining = asyncio.run(_run())

        assert remaining == []
        mock_db.client.close.assert_called_once()
