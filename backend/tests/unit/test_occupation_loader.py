"""Tests for the occupation loader (cache, fallbacks, embeddings) and both occupation repositories."""

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.occupations import loader
from app.occupations.ingest import (
    build_occupation_document,
    parse_occupation_file,
    resolve_embeddings,
)
from app.occupations.mongo_repository import MongoOccupationsRepository
from app.occupations.occupation_document import OCCUPATION_DOCUMENT_FIND_PROJECTION
from app.occupations.occupation_file import OccupationFileEntry
from app.occupations.repository import (
    FileOccupationsRepository,
    IOccupationsRepository,
    OccupationRecord,
)
from app.server_dependencies import model_dependencies

DIM = 3072


def _entry(code: str = "2221", counties=("Nairobi", "Mombasa")) -> dict:
    return {
        "occupation": {"code": code, "preferred_label": f"Occ {code}"},
        "skills": {"essential": {"uuids": ["s1"], "labels": ["triage"]}},
        "counties_data": [{"county": c} for c in counties],
    }


def _vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _record(code: str = "2221", **fields) -> OccupationRecord:
    return OccupationRecord(
        entry=OccupationFileEntry.model_validate(_entry(code)), **fields
    )


class _FakeRepository(IOccupationsRepository):
    def __init__(self, records=None, error: Exception | None = None):
        self.records = records or []
        self.error = error
        self.calls = 0

    @property
    def source(self) -> str:
        return "fake"

    async def load_occupations(self) -> list[OccupationRecord]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.records)


class _FakeWhitener:
    def __init__(self, active: bool = True):
        self.active = active
        self.calls = 0

    def rescale_target(self) -> float:
        return 0.5 if self.active else 0.0

    def whiten_rows(self, m: np.ndarray) -> np.ndarray:
        self.calls += 1
        return m[:, ::-1].copy()


@pytest.fixture(autouse=True)
def _isolated_loader():
    yield
    loader.set_occupations_repository(None)


def _load(repository, *, whitener=None, deployed_sha="sha-w"):
    loader.set_occupations_repository(repository)
    with (
        patch.object(
            model_dependencies,
            "get_concat_whitener",
            return_value=whitener or _FakeWhitener(),
        ),
        patch.object(loader, "_current_whitening_sha256", return_value=deployed_sha),
    ):
        return asyncio.run(loader.get_all_occupations_with_timing())


def _expire_cache() -> None:
    assert loader._cache is not None
    loader._cache = loader._cache.model_copy(update={"expires_at": 0.0})


# --- cache ---------------------------------------------------------------------------------------


def test_load_flattens_and_reports_the_source():
    rows, timing = _load(_FakeRepository([_record("1"), _record("2")]))

    assert [r.uuid for r in rows] == [
        "1_Nairobi",
        "1_Mombasa",
        "2_Nairobi",
        "2_Mombasa",
    ]
    assert timing.occupation_cache_hit is False
    assert timing.occupation_source == "fake"
    assert (timing.n_occupation_entries, timing.n_occupation_rows) == (2, 4)


def test_second_call_is_a_cache_hit():
    repository = _FakeRepository([_record()])
    first, _ = _load(repository)

    second, timing = asyncio.run(loader.get_all_occupations_with_timing())

    assert second is first
    assert timing.occupation_cache_hit is True
    assert repository.calls == 1


def test_expired_cache_is_reloaded():
    repository = _FakeRepository([_record("1")])
    _load(repository)
    repository.records = [_record("1"), _record("2")]
    _expire_cache()

    rows, timing = asyncio.run(loader.get_all_occupations_with_timing())

    assert repository.calls == 2
    assert timing.occupation_cache_hit is False
    assert len(rows) == 4


def test_ttl_zero_never_expires():
    with patch.object(loader, "OCCUPATIONS_CACHE_TTL_S", 0.0):
        _load(_FakeRepository([_record()]))
    assert loader._cache is not None and loader._cache.expires_at == float("inf")


# --- empty / unreachable source ------------------------------------------------------------------


def test_empty_source_is_logged_and_yields_no_occupations(caplog):
    with caplog.at_level(logging.ERROR, logger=loader.__name__):
        rows, timing = _load(_FakeRepository([]))

    assert rows == []
    assert timing.n_occupation_rows == 0
    assert "No occupations in fake" in caplog.text


def test_unreachable_source_is_logged_and_does_not_raise(caplog):
    repository = _FakeRepository(error=ConnectionError("no route"))

    with caplog.at_level(logging.ERROR, logger=loader.__name__):
        rows, _ = _load(repository)

    assert rows == []
    assert (
        "Occupations unavailable from fake (ConnectionError: no route)" in caplog.text
    )
    # The failure is cached briefly: the next request does not hit the source again.
    asyncio.run(loader.get_all_occupations_with_timing())
    assert repository.calls == 1


def test_failed_reload_keeps_serving_the_loaded_occupations():
    repository = _FakeRepository([_record()])
    first, _ = _load(repository)
    repository.error = TimeoutError("server selection")
    _expire_cache()

    rows, _ = asyncio.run(loader.get_all_occupations_with_timing())

    assert [r.uuid for r in rows] == [r.uuid for r in first]
    assert repository.calls == 2


# --- embeddings ----------------------------------------------------------------------------------


def _attached(rows):
    return {r.originUuid: r for r in loader.attach_occupation_embeddings(rows)}


def test_vector_whitened_with_the_deployed_artifact_is_used_as_is():
    whitener = _FakeWhitener()
    rows, _ = _load(
        _FakeRepository(
            [
                _record(
                    embedding=_vec(1),
                    whitened_embedding=_vec(9),
                    whitening_artifact_sha256="sha-w",
                )
            ]
        ),
        whitener=whitener,
    )

    occ = _attached(rows)["2221"]
    np.testing.assert_array_equal(occ.embedding, _vec(9))
    assert occ.embedding_whitened is True
    assert whitener.calls == 0


def test_stale_or_missing_whitened_vectors_are_whitened_once_at_load(caplog):
    whitener = _FakeWhitener()
    with caplog.at_level(logging.WARNING, logger=loader.__name__):
        rows, _ = _load(
            _FakeRepository(
                [
                    _record(
                        "1",
                        embedding=_vec(1),
                        whitened_embedding=_vec(9),
                        whitening_artifact_sha256="sha-old",
                    ),
                    _record("2", embedding=_vec(2)),
                ]
            ),
            whitener=whitener,
        )

    attached = _attached(rows)
    np.testing.assert_array_equal(attached["1"].embedding, _vec(1)[::-1])
    np.testing.assert_array_equal(attached["2"].embedding, _vec(2)[::-1])
    assert attached["1"].embedding_whitened and attached["2"].embedding_whitened
    assert whitener.calls == 1  # one batch for both
    assert "1 occupations were whitened at ingest with another artifact" in caplog.text


def test_without_a_whitening_artifact_raw_vectors_are_kept():
    rows, _ = _load(
        _FakeRepository(
            [
                _record(
                    embedding=_vec(1),
                    whitened_embedding=_vec(9),
                    whitening_artifact_sha256="sha-w",
                )
            ]
        ),
        whitener=_FakeWhitener(active=False),
    )

    occ = _attached(rows)["2221"]
    np.testing.assert_array_equal(occ.embedding, _vec(1))
    assert occ.embedding_whitened is False


def test_rows_share_one_vector_and_attach_is_cached():
    rows, _ = _load(_FakeRepository([_record(embedding=_vec(1))]))

    first = loader.attach_occupation_embeddings(rows)
    second = loader.attach_occupation_embeddings(rows)

    assert first[0].embedding is first[1].embedding
    assert first == second and first is not second


def test_occupations_without_a_vector_are_returned_unchanged():
    rows, _ = _load(_FakeRepository([_record("1", embedding=_vec(1)), _record("2")]))

    attached = _attached(rows)

    assert attached["2"].embedding is None


def test_occupation_entries_back_the_wa_lookup():
    _load(_FakeRepository([_record("1"), _record("2")]))

    entries = asyncio.run(loader.get_occupation_entries())

    assert [e.occupation.code for e in entries] == ["1", "2"]


# --- FileOccupationsRepository -------------------------------------------------------------------


def _write_file_source(tmp_path: Path, *, with_npz: bool = True):
    json_path = tmp_path / "occupations.json"
    json_path.write_text(json.dumps([_entry("1"), _entry("2")]), encoding="utf-8")
    npz_path = tmp_path / "embeddings.npz"
    if with_npz:
        np.savez(
            npz_path,
            codes=np.array(["2"], dtype=object),
            vectors=np.stack([_vec(2)]),
        )
    return str(json_path), str(npz_path)


def test_file_repository_reads_the_file_and_npz(tmp_path):
    repository = FileOccupationsRepository(*_write_file_source(tmp_path))

    records = asyncio.run(repository.load_occupations())

    assert [r.entry.occupation.code for r in records] == ["1", "2"]
    assert records[0].embedding is None
    np.testing.assert_array_equal(records[1].embedding, _vec(2))
    assert repository.source.startswith("file:")


def test_file_repository_without_npz_has_no_embeddings(tmp_path):
    repository = FileOccupationsRepository(
        *_write_file_source(tmp_path, with_npz=False)
    )

    records = asyncio.run(repository.load_occupations())

    assert all(r.embedding is None for r in records)


def test_file_repository_raises_when_the_file_is_missing(tmp_path):
    repository = FileOccupationsRepository(str(tmp_path / "missing.json"), None)

    with pytest.raises(FileNotFoundError):
        asyncio.run(repository.load_occupations())


# --- MongoOccupationsRepository ------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, docs):
        self._docs = docs
        self.sort_spec = None

    def sort(self, spec):
        self.sort_spec = spec
        return self

    def __aiter__(self):
        async def gen():
            for d in self._docs:
                yield d

        return gen()


def _stored(code: str, position: int) -> dict:
    [occ] = parse_occupation_file([_entry(code)])
    occ = occ.model_copy(update={"position": position})
    [resolved] = resolve_embeddings(
        [occ], stored={}, seed={code: _vec(position)}, embed=None, model="m", dim=DIM
    ).values()
    return build_occupation_document(
        occ,
        resolved,
        whitened=_vec(position + 100),
        whitening_artifact_sha256="sha-w",
        model="m",
        dim=DIM,
        group_codes=[],
        source={},
    )


def _mongo_repository(docs):
    cursor = _FakeCursor(docs)
    collection = MagicMock()
    collection.find.return_value = cursor
    db = MagicMock()
    db.get_collection.return_value = collection
    return MongoOccupationsRepository(db), collection, cursor


def test_mongo_repository_reads_in_catalogue_order_with_projection():
    repository, collection, cursor = _mongo_repository(
        [_stored("1", 0), _stored("2", 1)]
    )

    records = asyncio.run(repository.load_occupations())

    collection.find.assert_called_once_with({}, OCCUPATION_DOCUMENT_FIND_PROJECTION)
    assert cursor.sort_spec == [("position", 1), ("_id", 1)]
    assert [r.entry.occupation.code for r in records] == ["1", "2"]
    np.testing.assert_array_equal(records[1].embedding, _vec(1))
    np.testing.assert_array_equal(records[1].whitened_embedding, _vec(101))
    assert records[1].whitening_artifact_sha256 == "sha-w"


def test_mongo_repository_skips_invalid_documents(caplog):
    bad = {"_id": "x", "counties_data": "not a list"}
    repository, _, _ = _mongo_repository([bad, _stored("1", 0)])

    with caplog.at_level(logging.WARNING):
        records = asyncio.run(repository.load_occupations())

    assert [r.entry.occupation.code for r in records] == ["1"]
    assert "Skipping invalid occupation document _id=x" in caplog.text
