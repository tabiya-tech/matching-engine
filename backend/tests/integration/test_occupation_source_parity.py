"""Parity: the committed occupation file and the same file ingested into Mongo feed the engine alike.

"Before" is the file source (JSON + NPZ, whitened once at load). "After" is the ingest of the same
file and NPZ (whitened at ingest) read back through ``MongoOccupationsRepository``
(``scripts.occupation_parity_report.documents_repository``). The engine is deterministic in its
inputs, so identical rows in identical order with bit-identical stage-1 vectors means ``/match_v4``
returns the same occupations in the same order. Uses the real committed catalogue and whitening
artifact; no network or Mongo. ``scripts/occupation_parity_report.py`` runs the engine itself.
"""

import asyncio

import numpy as np
import pytest

from app.config import OCCUPATION_CONCAT_EMBEDDINGS_PATH, OCCUPATION_JSON_PATH
from app.domain.occupation import Occupation
from app.occupations import loader
from app.occupations.repository import FileOccupationsRepository
from app.server_dependencies.model_dependencies import get_concat_whitener
from scripts.occupation_parity_report import documents_repository


def _corpus(repository) -> list[Occupation]:
    loader.set_occupations_repository(repository)
    rows, _ = asyncio.run(loader.get_all_occupations_with_timing())
    return loader.attach_occupation_embeddings(rows)


@pytest.fixture(scope="module")
def corpora():
    if get_concat_whitener().rescale_target() <= 0:
        pytest.skip("concat whitening artifact unavailable")
    before = _corpus(
        FileOccupationsRepository(
            OCCUPATION_JSON_PATH, OCCUPATION_CONCAT_EMBEDDINGS_PATH
        )
    )
    after = _corpus(
        documents_repository(OCCUPATION_JSON_PATH, OCCUPATION_CONCAT_EMBEDDINGS_PATH)
    )
    loader.set_occupations_repository(None)
    return before, after


def test_same_rows_in_the_same_order(corpora):
    before, after = corpora

    assert len(before) == len(after) > 1000
    for b, a in zip(before, after, strict=True):
        assert b.model_dump(exclude={"embedding"}) == a.model_dump(
            exclude={"embedding"}
        )


def test_bit_identical_stage1_vectors(corpora):
    before, after = corpora

    for b, a in zip(before, after, strict=True):
        assert (b.embedding is None) == (a.embedding is None)
        assert b.embedding_whitened == a.embedding_whitened
        if b.embedding is not None and a.embedding is not None:
            assert np.array_equal(b.embedding, a.embedding), b.uuid
    assert all(a.embedding_whitened for a in after if a.embedding is not None)
