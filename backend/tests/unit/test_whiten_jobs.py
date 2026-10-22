"""Tests for ``scripts/whiten_jobs.py`` against an in-memory jobs collection, without Mongo."""

import copy
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.clients.gemini_embedding_client import EMBEDDING_DIM
from scripts.whiten_jobs import (
    APPLIED_BY,
    JobUpdate,
    Outcome,
    Whitening,
    run_whitening,
)

DIM = EMBEDDING_DIM
NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)
SHA = "a" * 64
LOADED_AT = "2026-10-01T00:00:00+00:00"


def _vec(seed: int) -> list[float]:
    v = np.random.default_rng(seed).standard_normal(DIM)
    return [float(x) for x in v / np.linalg.norm(v)]


def _whiten(rows: np.ndarray) -> np.ndarray:
    """A stand-in transform that is easy to recognise: negate."""
    return -np.asarray(rows, dtype=np.float64)


WHITENING = Whitening(whiten_rows=_whiten, artifact_sha256=SHA, target=0.5)


def _get(doc: Mapping[str, Any], path: str) -> object:
    for part in path.split("."):
        if not isinstance(doc, Mapping):
            return None
        doc = doc.get(part)  # type: ignore[assignment]
    return doc


class _FakeJobs:
    """Implements the ``find`` filter the script uses and applies ``JobUpdate``s like ``$set``."""

    def __init__(self, docs: list[dict]):
        self.docs = {d["_id"]: copy.deepcopy(d) for d in docs}

    def find(
        self, filter: Mapping[str, Any], projection: Mapping[str, Any]
    ) -> Iterable[Mapping[str, Any]]:
        out = []
        for doc in self.docs.values():
            if all(_get(doc, path) != cond["$ne"] for path, cond in filter.items()):
                out.append(copy.deepcopy(doc))
        return out

    def write(self, updates: list[JobUpdate]) -> int:
        modified = 0
        for u in updates:
            doc = self.docs[u.filter["_id"]]
            if (
                _get(doc, "llm_reranker_meta.embedding.loaded_at")
                != u.filter["llm_reranker_meta.embedding.loaded_at"]
            ):
                continue
            for path, value in u.set.items():
                *parents, leaf = path.split(".")
                target = doc
                for p in parents:
                    target = target.setdefault(p, {})
                target[leaf] = copy.deepcopy(value)
            modified += 1
        return modified


def _run(jobs: _FakeJobs, *, dry_run: bool = False, batch_size: int = 200):
    return run_whitening(
        jobs,
        whitening=WHITENING,
        write=jobs.write,
        dry_run=dry_run,
        now=NOW,
        batch_size=batch_size,
    )


def _raw_job(_id: str, seed: int, **extra) -> dict:
    """What the reranker writes with whitening off."""
    return {
        "_id": _id,
        "job_embedding": _vec(seed),
        "llm_reranker_meta": {
            "schema_version": 3,
            "embedding": {"model": "m", "dim": DIM, "loaded_at": LOADED_AT},
        },
        **extra,
    }


def test_raw_job_is_whitened_from_job_embedding_and_keeps_its_meta():
    raw = _vec(1)
    jobs = _FakeJobs([_raw_job("j1", 1)])

    result = _run(jobs)

    doc = jobs.docs["j1"]
    assert result.outcomes[Outcome.WHITENED.value] == ["j1"]
    assert result.n_written == 1
    np.testing.assert_allclose(doc["job_embedding"], -np.asarray(raw))
    assert doc["raw_embedding"] == raw
    assert doc["llm_reranker_meta"]["schema_version"] == 3
    assert doc["llm_reranker_meta"]["embedding"]["model"] == "m"
    assert doc["llm_reranker_meta"]["embedding"]["whitening"] == {
        "enabled": True,
        "artifact_sha256": SHA,
        "target": 0.5,
        "applied_at": NOW.isoformat(),
        "applied_by": APPLIED_BY,
    }


def test_stale_raw_embedding_is_ignored_when_job_is_not_marked_whitened():
    # Re-ranked with whitening off after an earlier whitened run: raw_embedding is left over.
    jobs = _FakeJobs([_raw_job("j1", 1, raw_embedding=_vec(99))])

    _run(jobs)

    assert jobs.docs["j1"]["raw_embedding"] == _vec(1)


def test_job_whitened_without_artifact_or_with_another_is_rewhitened_from_raw():
    raw = _vec(2)
    no_sha = _raw_job("no_sha", 0, raw_embedding=raw)
    no_sha["llm_reranker_meta"]["embedding"]["whitening"] = {
        "enabled": True,
        "artifact_sha256": None,
        "target": 0.0,
    }
    other = _raw_job("other", 0, raw_embedding=raw)
    other["llm_reranker_meta"]["embedding"]["whitening"] = {
        "enabled": True,
        "artifact_sha256": "b" * 64,
    }
    jobs = _FakeJobs([no_sha, other])

    result = _run(jobs)

    assert sorted(result.outcomes[Outcome.REWHITENED.value]) == ["no_sha", "other"]
    for key in ("no_sha", "other"):
        np.testing.assert_allclose(jobs.docs[key]["job_embedding"], -np.asarray(raw))
        assert jobs.docs[key]["raw_embedding"] == raw


def test_already_whitened_and_unfixable_jobs_are_not_written():
    done = _raw_job("done", 3)
    done["llm_reranker_meta"]["embedding"]["whitening"] = {
        "enabled": True,
        "artifact_sha256": SHA,
    }
    no_raw = _raw_job("no_raw", 4)
    no_raw["llm_reranker_meta"]["embedding"]["whitening"] = {"enabled": True}
    short = _raw_job("short", 5)
    short["job_embedding"] = [0.1, 0.2]
    no_embedding = {"_id": "none", "job_embedding": None}
    jobs = _FakeJobs([done, no_raw, short, no_embedding])
    before = copy.deepcopy(jobs.docs)

    result = _run(jobs)

    assert result.outcomes[Outcome.NO_RAW_EMBEDDING.value] == ["no_raw"]
    assert result.outcomes[Outcome.BAD_DIM.value] == ["short"]
    assert result.n_written == 0
    assert jobs.docs == before


def test_job_without_embedding_meta_gets_one():
    jobs = _FakeJobs(
        [{"_id": "j1", "job_embedding": _vec(6), "llm_reranker_meta": None}]
    )

    _run(jobs)

    whitening = jobs.docs["j1"]["llm_reranker_meta"]["embedding"]["whitening"]
    assert whitening["artifact_sha256"] == SHA


def test_second_run_finds_nothing_to_do():
    jobs = _FakeJobs([_raw_job(f"j{i}", i) for i in range(5)])

    first = _run(jobs, batch_size=2)
    second = _run(jobs)

    assert first.n_written == 5
    assert all(not ids for ids in second.outcomes.values())
    assert second.n_written == 0


def test_dry_run_writes_nothing():
    jobs = _FakeJobs([_raw_job("j1", 1)])
    before = copy.deepcopy(jobs.docs)

    result = _run(jobs, dry_run=True)

    assert result.outcomes[Outcome.WHITENED.value] == ["j1"]
    assert not result.written
    assert jobs.docs == before


def test_job_reembedded_since_it_was_read_is_left_alone():
    jobs = _FakeJobs([_raw_job("j1", 1)])

    def reembed_then_write(updates: list[JobUpdate]) -> int:
        jobs.docs["j1"]["llm_reranker_meta"]["embedding"]["loaded_at"] = "later"
        return jobs.write(updates)

    result = run_whitening(
        jobs, whitening=WHITENING, write=reembed_then_write, dry_run=False, now=NOW
    )

    assert result.n_written == 0
    assert "raw_embedding" not in jobs.docs["j1"]
