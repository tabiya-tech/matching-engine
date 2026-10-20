"""Tests for the occupation file -> document mapping and the ingest plan, without Mongo.

``run_ingest`` (``scripts/ingest_occupations.py``) runs against an in-memory collection, so the
idempotency and change-reporting acceptance criteria are checked end to end.
"""

import copy
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from app.clients.gemini_embedding_client import EMBEDDING_DIM, MODEL_NAME
from app.occupations.flatten import flatten_occupations
from app.occupations.ingest import (
    OccupationFileInvalid,
    build_occupation_document,
    parse_occupation_file,
    plan_ingest,
    resolve_embeddings,
)
from app.occupations.mongo_repository import record_from_document
from app.occupations.occupation_document import (
    OccupationDocument,
    occupation_embedding_text,
    sha256_text,
    vector_to_bin,
)
from scripts.ingest_occupations import Whitening, run_ingest

DIM = EMBEDDING_DIM
NOW = datetime(2026, 10, 19, tzinfo=timezone.utc)


def _entry(code: str = "2221", label: str = "Nurse", skills=("triage",)) -> dict:
    return {
        "occupation": {"code": code, "preferred_label": label, "description": "d"},
        "skills": {
            "essential": {
                "uuids": [f"u-{s}" for s in skills],
                "labels": list(skills),
            },
            "optional": {"uuids": [], "labels": []},
        },
        "counties_data": [
            {
                "county": "Nairobi",
                "job_attributes": {
                    "attributes": [
                        {
                            "attribute_name": "earnings_per_month",
                            "selected_level_id": "earn_70k",
                            "reasoning": "long LLM rationale",
                        }
                    ]
                },
                "labor_demand": {"expected_demand": "High Expected Demand"},
            }
        ],
        "onet_work_activities": [
            {"WA_code": "4.A.1", "WA_label": "x", "WA_Importance": "3", "WA_Level": "4"}
        ],
    }


def _vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class _Embedder:
    """Deterministic fake embedding API that records what it was asked to embed."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.stack([_vec(len(t)) * 3 for t in texts])  # un-normalised on purpose


class _FakeCollection:
    def __init__(self):
        self.docs: dict[str, dict[str, Any]] = {}
        self.writes = 0

    def find(self, filter: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
        return [copy.deepcopy(d) for d in self.docs.values()]

    def replace_one(
        self, filter: Mapping[str, Any], replacement: Mapping[str, Any], upsert: bool
    ) -> object:
        self.writes += 1
        self.docs[filter["_id"]] = copy.deepcopy(dict(replacement))
        return None

    def delete_many(self, filter: Mapping[str, Any]) -> object:
        for key in filter["_id"]["$in"]:
            self.docs.pop(key, None)
        return None


# --- parse_occupation_file ---------------------------------------------------------------------


def test_parse_keeps_the_raw_entry_and_position():
    raw = [_entry("1"), _entry("2")]

    parsed = parse_occupation_file(raw)

    assert [(p.position, p.code) for p in parsed] == [(0, "1"), (1, "2")]
    assert parsed[0].raw == raw[0]


def test_parse_refuses_a_non_list():
    with pytest.raises(OccupationFileInvalid) as e:
        parse_occupation_file({"occupation": {}})

    assert "expected a JSON list" in e.value.problems[0]


def test_parse_lists_every_bad_entry_with_its_index_and_code():
    bad_shape = _entry("3")
    bad_shape["counties_data"] = "Nairobi"
    no_label = _entry("4", label="")
    no_code = _entry("")
    labels_over_uuids = _entry("5")
    labels_over_uuids["skills"]["essential"]["labels"].append("extra")

    with pytest.raises(OccupationFileInvalid) as e:
        parse_occupation_file(
            [
                _entry("1"),
                _entry("1"),
                bad_shape,
                no_label,
                no_code,
                labels_over_uuids,
                7,
            ]
        )

    problems = e.value.problems
    assert any(
        p.startswith("entry 1 (code '1')") and "repeats entry 0" in p for p in problems
    )
    assert any(
        p.startswith("entry 2 (code '3')") and "counties_data" in p for p in problems
    )
    assert any(
        p.startswith("entry 3 (code '4')") and "preferred_label" in p for p in problems
    )
    assert any(
        p.startswith("entry 4") and "occupation.code is missing" in p for p in problems
    )
    assert any(
        p.startswith("entry 5") and "2 labels for 1 uuids" in p for p in problems
    )
    assert any(p.startswith("entry 6") and "expected an object" in p for p in problems)
    assert len(problems) == 6


# --- resolve_embeddings ------------------------------------------------------------------------


def _stored_doc(
    occupation, vector: np.ndarray, *, text_sha: str | None = None, model=MODEL_NAME
):
    return OccupationDocument.model_validate(
        {
            **occupation.raw,
            "code": occupation.code,
            "embedding": {
                "model": model,
                "text_sha256": text_sha
                or sha256_text(occupation_embedding_text(occupation.entry)),
                "vector_bin": vector_to_bin(vector),
            },
        }
    )


def test_stored_vector_is_reused_while_the_text_is_unchanged():
    [occ] = parse_occupation_file([_entry()])
    embed = _Embedder()

    resolved = resolve_embeddings(
        [occ],
        stored={occ.code: _stored_doc(occ, _vec(1))},
        seed={occ.code: _vec(2)},
        embed=embed,
        model=MODEL_NAME,
        dim=DIM,
    )

    assert resolved[occ.code].origin == "stored"
    np.testing.assert_array_equal(resolved[occ.code].vector, _vec(1))
    assert embed.calls == []


def test_changed_text_or_model_falls_back_to_seed_then_embedding():
    a, b, c = parse_occupation_file([_entry("a"), _entry("b"), _entry("c")])
    embed = _Embedder()

    resolved = resolve_embeddings(
        [a, b, c],
        stored={
            "a": _stored_doc(a, _vec(1), text_sha="old text"),
            "b": _stored_doc(b, _vec(1), model="other-model"),
        },
        seed={"a": _vec(2)},
        embed=embed,
        model=MODEL_NAME,
        dim=DIM,
    )

    assert resolved["a"].origin == "seed"
    np.testing.assert_array_equal(resolved["a"].vector, _vec(2))
    assert resolved["b"].origin == resolved["c"].origin == "embedded"
    assert len(embed.calls) == 1 and len(embed.calls[0]) == 2  # one batch
    vector = resolved["c"].vector
    assert vector is not None
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_without_an_embedder_new_occupations_are_pending():
    [occ] = parse_occupation_file([_entry()])

    resolved = resolve_embeddings(
        [occ], stored={}, seed={}, embed=None, model=MODEL_NAME, dim=DIM
    )

    assert resolved[occ.code].origin == "pending"
    assert resolved[occ.code].vector is None


# --- document mapping --------------------------------------------------------------------------


def _document(occ, vector, whitened=None, group_codes=()):
    [resolved] = resolve_embeddings(
        [occ], stored={}, seed={occ.code: vector}, embed=None, model=MODEL_NAME, dim=DIM
    ).values()
    return build_occupation_document(
        occ,
        resolved,
        whitened=whitened,
        whitening_artifact_sha256="sha-w" if whitened is not None else None,
        model=MODEL_NAME,
        dim=DIM,
        group_codes=list(group_codes),
        source={"file": "occupations.json"},
    )


def test_document_keeps_the_entry_verbatim_and_records_provenance():
    [occ] = parse_occupation_file([_entry()])

    doc = _document(occ, _vec(1), whitened=_vec(9), group_codes=["2", "22", "2221"])

    assert doc["_id"] == doc["code"] == "2221"
    assert (
        doc["counties_data"] == occ.raw["counties_data"]
    )  # rationale kept in the document
    assert doc["group_codes"] == ["2", "22", "2221"]
    emb = doc["embedding"]
    assert (emb["model"], emb["dim"], emb["normalized"]) == (MODEL_NAME, DIM, True)
    assert emb["whitening"] == {"enabled": True, "artifact_sha256": "sha-w"}
    assert len(emb["vector_bin"]) == DIM * 4
    assert emb["vector_sha256"]


def test_document_round_trips_to_the_same_rows_and_vectors():
    [occ] = parse_occupation_file([_entry()])
    doc = _document(occ, _vec(1), whitened=_vec(9))

    record = record_from_document(OccupationDocument.model_validate(doc))

    assert flatten_occupations([record.entry]) == flatten_occupations([occ.entry])
    np.testing.assert_array_equal(record.embedding, _vec(1))
    np.testing.assert_array_equal(record.whitened_embedding, _vec(9))
    assert record.whitening_artifact_sha256 == "sha-w"


def test_record_drops_vectors_of_the_wrong_length():
    [occ] = parse_occupation_file([_entry()])
    doc = _document(occ, _vec(1))
    doc["embedding"]["vector_bin"] = b"\x00" * 8

    record = record_from_document(OccupationDocument.model_validate(doc))

    assert record.embedding is None
    assert record.whitening_artifact_sha256 is None


# --- plan_ingest -------------------------------------------------------------------------------


def test_plan_classifies_added_updated_unchanged_and_removed():
    a, b, c = parse_occupation_file([_entry("a"), _entry("b"), _entry("c")])
    stored = {"a": _document(a, _vec(1)), "b": _document(b, _vec(1)), "z": {"_id": "z"}}
    changed_b = parse_occupation_file([_entry("b", label="Midwife")])[0]
    desired = {
        "a": _document(a, _vec(1)),
        "b": _document(changed_b, _vec(1)),
        "c": _document(c, _vec(1)),
    }

    plan = plan_ingest(desired, stored, prune=False)

    assert (plan.added, plan.updated, plan.unchanged, plan.removed) == (
        ["c"],
        ["b"],
        ["a"],
        ["z"],
    )
    assert plan.prune is False


def test_a_new_whitening_artifact_is_an_update():
    [occ] = parse_occupation_file([_entry()])
    stored = {occ.code: _document(occ, _vec(1), whitened=_vec(9))}
    desired = {occ.code: _document(occ, _vec(1), whitened=_vec(9))}
    desired[occ.code]["embedding"]["whitening"]["artifact_sha256"] = "sha-new"

    assert plan_ingest(desired, stored, prune=False).updated == [occ.code]


# --- run_ingest (the script, against a fake collection) ----------------------------------------


def _whitening() -> Whitening:
    return Whitening(whiten_rows=lambda m: m[:, ::-1].copy(), artifact_sha256="sha-w")


def _run(
    collection,
    entries,
    *,
    embed=None,
    seed=None,
    prune=False,
    dry_run=False,
    groups=None,
):
    group_collection = _FakeCollection() if groups is not None else None
    result = run_ingest(
        parse_occupation_file(entries),
        occupations_collection=collection,
        occupations_collection_name="occupations",
        groups_collection=group_collection,
        groups_collection_name="occupation_groups",
        group_documents=groups,
        ancestry={"2221": ["2", "2221"]},
        seed=seed or {},
        embed=embed,
        whitening=_whitening(),
        source={"file": "f.json"},
        dry_run=dry_run,
        prune=prune,
        now=NOW,
    )
    return result, group_collection


def test_second_run_with_the_same_file_changes_nothing():
    collection = _FakeCollection()
    embed = _Embedder()
    entries = [_entry("2221"), _entry("3000", skills=("cook",))]

    first, _ = _run(collection, entries, embed=embed)
    snapshot = copy.deepcopy(collection.docs)
    second, _ = _run(collection, entries, embed=embed)

    assert first.occupations.plan.added == ["2221", "3000"]
    assert second.occupations.plan.added == second.occupations.plan.updated == []
    assert second.occupations.plan.unchanged == ["2221", "3000"]
    assert collection.writes == 2
    assert collection.docs == snapshot
    assert len(embed.calls) == 1  # the second run re-embeds nothing
    assert second.embedding_origins == {"stored": 2}


def test_stored_documents_hold_both_vectors_and_group_codes():
    collection = _FakeCollection()

    _run(collection, [_entry()], seed={"2221": _vec(1)})

    doc = collection.docs["2221"]
    raw = np.frombuffer(doc["embedding"]["vector_bin"], dtype="<f4")
    whitened = np.frombuffer(doc["embedding"]["whitened_vector_bin"], dtype="<f4")
    np.testing.assert_array_equal(raw, _vec(1))
    np.testing.assert_array_equal(whitened, _vec(1)[::-1])
    assert doc["embedding"]["whitening"]["artifact_sha256"] == "sha-w"
    assert doc["group_codes"] == ["2", "2221"]
    assert doc["ingested_at"] == doc["updated_at"] == NOW


def test_an_updated_file_reports_and_re_embeds_only_what_changed():
    collection = _FakeCollection()
    embed = _Embedder()
    _run(collection, [_entry("2221"), _entry("3000", skills=("cook",))], embed=embed)

    result, _ = _run(
        collection,
        [_entry("2221", label="Registered nurse"), _entry("3000", skills=("bake",))],
        embed=embed,
    )

    assert result.occupations.plan.updated == ["2221", "3000"]
    assert embed.calls[-1] == ["bake"]  # the label change keeps its vector
    assert (
        collection.docs["2221"]["occupation"]["preferred_label"] == "Registered nurse"
    )
    assert collection.docs["2221"]["ingested_at"] == NOW


def test_occupations_missing_from_the_file_are_reported_and_pruned_only_on_request():
    collection = _FakeCollection()
    _run(
        collection,
        [_entry("2221"), _entry("3000")],
        seed={"2221": _vec(1), "3000": _vec(2)},
    )

    kept, _ = _run(collection, [_entry("2221")])
    assert kept.occupations.plan.removed == ["3000"]
    assert "3000" in collection.docs

    pruned, _ = _run(collection, [_entry("2221")], prune=True)
    assert pruned.occupations.plan.removed == ["3000"]
    assert set(collection.docs) == {"2221"}


def test_dry_run_writes_nothing_and_reports_pending_embeddings():
    collection = _FakeCollection()

    result, _ = _run(collection, [_entry()], dry_run=True)

    assert collection.docs == {}
    assert result.occupations.plan.added == ["2221"]
    assert result.occupations.written is False
    assert result.embedding_origins == {"pending": 1}


def test_a_real_run_refuses_to_write_occupations_without_embeddings():
    collection = _FakeCollection()

    with pytest.raises(RuntimeError, match="no embedding"):
        _run(collection, [_entry()])
    assert collection.docs == {}


def test_groups_are_written_alongside():
    collection = _FakeCollection()
    groups = {"2": {"_id": "2", "code": "2", "content_sha256": "x"}}

    result, group_collection = _run(
        collection, [_entry()], seed={"2221": _vec(1)}, groups=groups
    )

    assert result.groups is not None and result.groups.plan.added == ["2"]
    assert group_collection is not None and set(group_collection.docs) == {"2"}
