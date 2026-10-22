#!/usr/bin/env python3
"""Whiten the job embeddings the llm-reranker left unwhitened (or whitened with another artifact).

Stage-1 vector search only retrieves jobs whose ``job_embedding`` was whitened with the service's
artifact (``V4_FULL_CONCAT_WHITENING_PATH``, matched by SHA-256 at
``llm_reranker_meta.embedding.whitening.artifact_sha256``). The llm-reranker leaves a job outside
that set when:

* ``RERANKER_CONCAT_WHITENING_ENABLED`` was off (the IaC default) when the job was ranked:
  ``job_embedding`` is the raw vector and there is no whitening block;
* the artifact was missing and ``RERANKER_CONCAT_WHITENING_REQUIRED=0``: the whitening block says
  ``enabled`` but has no SHA-256, and ``job_embedding`` is only L2-normalised;
* a different artifact whitened it: the SHA-256 differs.

For each such job this script takes the raw vector (``job_embedding`` when the job is not marked
whitened; ``raw_embedding`` when it is, since ``job_embedding`` is then already transformed),
whitens it with the service's artifact, and sets ``job_embedding`` (whitened), ``raw_embedding``
(raw) and the whitening block, the same shape the llm-reranker writes. Jobs with no embedding at
all, or marked whitened with no ``raw_embedding`` to start from, need the llm-reranker to re-embed
them and are only reported.

* Idempotent: a second run finds nothing to do.
* A job re-ranked while the script runs is left alone (each write is conditioned on the embedding's
  ``loaded_at`` being the one read).

Usage (from ``backend/``; connection from ``--env-file`` or ``MONGO_URL`` / ``MONGO_DB_NAME``)::

    python -m scripts.whiten_jobs --env-file .env.shp --dry-run
    python -m scripts.whiten_jobs --env-file .env.shp
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import numpy as np  # noqa: E402

from app.clients.gemini_embedding_client import EMBEDDING_DIM  # noqa: E402
from app.domain.base import DomainModel  # noqa: E402
from app.jobs.job_document import (  # noqa: E402
    JOB_EMBEDDING_FIELD,
    JOB_WHITENING_SHA256_FIELD,
)

RAW_EMBEDDING_FIELD = "raw_embedding"
APPLIED_BY = "matching-service/scripts/whiten_jobs.py"
_SHOW = 10
logger = logging.getLogger("whiten_jobs")


class JobsCollection(Protocol):
    """The part of a ``pymongo`` collection the script reads (a fake in tests)."""

    def find(
        self, filter: Mapping[str, Any], projection: Mapping[str, Any]
    ) -> Iterable[Mapping[str, Any]]: ...


class Whitening(DomainModel):
    """The service's concat-whitening transform."""

    whiten_rows: Callable[[np.ndarray], np.ndarray]
    artifact_sha256: str
    target: float


class Outcome(str, Enum):
    WHITENED = "whitened"
    """Raw ``job_embedding`` (no whitening block): whitened from ``job_embedding``."""
    REWHITENED = "rewhitened"
    """Marked whitened without this artifact's SHA-256: re-whitened from ``raw_embedding``."""
    NO_RAW_EMBEDDING = "no_raw_embedding"
    """Marked whitened without this artifact's SHA-256 and no ``raw_embedding``: needs re-embedding."""
    BAD_DIM = "bad_dim"
    """The source vector is not ``EMBEDDING_DIM`` long: needs re-embedding."""


class JobUpdate(DomainModel):
    """One planned write: the guarded filter and the ``$set``."""

    job_id: str
    filter: dict[str, Any]
    set: dict[str, Any]


class WhiteningResult(DomainModel):
    """What one run found (and, unless a dry run, wrote)."""

    outcomes: dict[str, list[str]]
    n_written: int
    written: bool


def needs_whitening_filter(artifact_sha256: str) -> dict[str, Any]:
    """Jobs with an embedding that is not whitened with ``artifact_sha256``."""
    return {
        JOB_EMBEDDING_FIELD: {"$ne": None},
        JOB_WHITENING_SHA256_FIELD: {"$ne": artifact_sha256},
    }


_PROJECTION = {
    JOB_EMBEDDING_FIELD: 1,
    RAW_EMBEDDING_FIELD: 1,
    "llm_reranker_meta.embedding": 1,
}


def _embedding_meta(doc: Mapping[str, Any]) -> Mapping[str, Any] | None:
    meta = doc.get("llm_reranker_meta")
    embedding = meta.get("embedding") if isinstance(meta, Mapping) else None
    return embedding if isinstance(embedding, Mapping) else None


def _marked_whitened(doc: Mapping[str, Any]) -> bool:
    embedding = _embedding_meta(doc)
    whitening = embedding.get("whitening") if embedding else None
    return isinstance(whitening, Mapping) and whitening.get("enabled") is True


def classify_job(doc: Mapping[str, Any]) -> tuple[Outcome, list[float] | None]:
    """How to whiten one job, and the raw vector to whiten (None when it cannot be)."""
    if _marked_whitened(doc):
        outcome, source = Outcome.REWHITENED, doc.get(RAW_EMBEDDING_FIELD)
        if not isinstance(source, list):
            return Outcome.NO_RAW_EMBEDDING, None
    else:
        # Not whitened: job_embedding is the raw vector. A raw_embedding left over from an earlier
        # whitened run may be stale (the reranker's $set does not unset it), so it is not used.
        outcome, source = Outcome.WHITENED, doc.get(JOB_EMBEDDING_FIELD)
    if not isinstance(source, list) or len(source) != EMBEDDING_DIM:
        return Outcome.BAD_DIM, None
    return outcome, source


def _whitening_set(
    doc: Mapping[str, Any], whitening_block: dict[str, Any]
) -> dict[str, Any]:
    """The ``$set`` for the whitening block, without clobbering the rest of ``llm_reranker_meta``."""
    meta = doc.get("llm_reranker_meta")
    if not isinstance(meta, Mapping):
        return {"llm_reranker_meta": {"embedding": {"whitening": whitening_block}}}
    if not isinstance(meta.get("embedding"), Mapping):
        return {"llm_reranker_meta.embedding": {"whitening": whitening_block}}
    return {"llm_reranker_meta.embedding.whitening": whitening_block}


def plan_updates(
    docs: Iterable[Mapping[str, Any]], whitening: Whitening, *, now: datetime
) -> tuple[list[JobUpdate], dict[str, list[str]]]:
    """Plan the writes for ``docs``; also return every job id by ``Outcome``."""
    outcomes: dict[str, list[str]] = {o.value: [] for o in Outcome}
    pending: list[tuple[Mapping[str, Any], list[float]]] = []
    for doc in docs:
        outcome, source = classify_job(doc)
        outcomes[outcome.value].append(str(doc["_id"]))
        if source is not None:
            pending.append((doc, source))
    if not pending:
        return [], outcomes

    whitened = whitening.whiten_rows(
        np.asarray([source for _, source in pending], dtype=np.float64)
    )
    block = {
        "enabled": True,
        "artifact_sha256": whitening.artifact_sha256,
        "target": whitening.target,
        "applied_at": now.isoformat(),
        "applied_by": APPLIED_BY,
    }
    updates = []
    for (doc, source), row in zip(pending, whitened, strict=True):
        embedding = _embedding_meta(doc)
        updates.append(
            JobUpdate(
                job_id=str(doc["_id"]),
                # Skip the write if the reranker re-embedded the job since it was read.
                filter={
                    "_id": doc["_id"],
                    "llm_reranker_meta.embedding.loaded_at": (
                        embedding.get("loaded_at") if embedding else None
                    ),
                },
                set={
                    JOB_EMBEDDING_FIELD: [float(x) for x in row],
                    RAW_EMBEDDING_FIELD: [float(x) for x in source],
                    **_whitening_set(doc, block),
                },
            )
        )
    return updates, outcomes


def run_whitening(
    collection: JobsCollection,
    *,
    whitening: Whitening,
    write: Callable[[list[JobUpdate]], int],
    dry_run: bool,
    now: datetime,
    batch_size: int = 200,
) -> WhiteningResult:
    """Whiten every job not whitened with ``whitening``'s artifact, ``batch_size`` jobs at a time.

    ``write`` applies one batch and returns how many documents it modified.
    """
    outcomes: dict[str, list[str]] = {o.value: [] for o in Outcome}
    n_written = 0
    batch: list[Mapping[str, Any]] = []

    def flush() -> None:
        nonlocal n_written
        updates, found = plan_updates(batch, whitening, now=now)
        for key, ids in found.items():
            outcomes[key].extend(ids)
        if updates and not dry_run:
            n_written += write(updates)
        batch.clear()

    for doc in collection.find(
        needs_whitening_filter(whitening.artifact_sha256), _PROJECTION
    ):
        batch.append(doc)
        if len(batch) >= batch_size:
            flush()
    if batch:
        flush()
    return WhiteningResult(outcomes=outcomes, n_written=n_written, written=not dry_run)


def _load_whitening() -> Whitening | None:
    from app.artifacts.repository import get_artifacts_repository
    from app.config import V4_FULL_CONCAT_WHITENING_PATH
    from app.server_dependencies.model_dependencies import get_concat_whitener

    whitener = get_concat_whitener()
    target = whitener.rescale_target()
    if target <= 0:
        return None
    return Whitening(
        whiten_rows=whitener.whiten_rows,
        artifact_sha256=get_artifacts_repository().file_sha256(
            V4_FULL_CONCAT_WHITENING_PATH
        ),
        target=target,
    )


def _print_result(result: WhiteningResult, n_no_embedding: int) -> None:
    def ids(values: list[str]) -> str:
        more = f" (+{len(values) - _SHOW} more)" if len(values) > _SHOW else ""
        return f": {', '.join(values[:_SHOW])}{more}" if values else ""

    verb = "" if result.written else "would be "
    for outcome, label in (
        (Outcome.WHITENED, f"{verb}whitened (raw job_embedding)"),
        (Outcome.REWHITENED, f"{verb}re-whitened from raw_embedding"),
        (Outcome.NO_RAW_EMBEDDING, "skipped, marked whitened but no raw_embedding"),
        (Outcome.BAD_DIM, f"skipped, vector is not {EMBEDDING_DIM}-dim"),
    ):
        values = result.outcomes[outcome.value]
        print(f"    {label}: {len(values)}{ids(values)}")
    print(f"    skipped, no job_embedding at all: {n_no_embedding}")
    if result.written:
        print(f"    documents modified: {result.n_written}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Whiten the job embeddings the llm-reranker left unwhitened."
    )
    ap.add_argument(
        "--env-file",
        type=Path,
        help="Env file with MONGO_URL / MONGO_DB_NAME; overrides the environment.",
    )
    ap.add_argument("--mongo-url", help="Overrides MONGO_URL.")
    ap.add_argument("--db", help="Overrides MONGO_DB_NAME.")
    ap.add_argument("--collection", help="Overrides MONGO_JOBS_COLLECTION.")
    ap.add_argument(
        "--batch-size", type=int, default=200, help="Jobs per write (default 200)."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change; write nothing.",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    # Env file first, so app.config sees it.
    if args.env_file is not None:
        from dotenv import load_dotenv

        if not load_dotenv(args.env_file, override=True):
            print(f"[whiten_jobs] cannot read {args.env_file}", file=sys.stderr)
            return 2
    for flag, name in (
        (args.mongo_url, "MONGO_URL"),
        (args.db, "MONGO_DB_NAME"),
        (args.collection, "MONGO_JOBS_COLLECTION"),
    ):
        if flag:
            os.environ[name] = flag

    from pymongo import MongoClient, UpdateOne

    from app.server_dependencies.database_collections import Collections
    from app.server_dependencies.db_dependencies import (
        MongoDbSettings,
        build_client_kwargs,
    )

    whitening = _load_whitening()
    if whitening is None:
        print(
            "[whiten_jobs] the whitening artifact (V4_FULL_CONCAT_WHITENING_PATH) is unavailable",
            file=sys.stderr,
        )
        return 2

    settings = MongoDbSettings.from_env()
    jobs_name = os.getenv("MONGO_JOBS_COLLECTION") or Collections.JOBS
    print(
        f"[whiten_jobs] target: db={settings.database_name} collection={jobs_name} "
        f"artifact={whitening.artifact_sha256[:12]}…"
        + (" (dry run)" if args.dry_run else "")
    )
    client: MongoClient = MongoClient(
        settings.mongo_url, **build_client_kwargs(settings, logger)
    )
    try:
        collection = client[settings.database_name][jobs_name]

        def write(updates: list[JobUpdate]) -> int:
            result = collection.bulk_write(
                [UpdateOne(u.filter, {"$set": u.set}) for u in updates], ordered=False
            )
            return result.modified_count

        result = run_whitening(
            collection,
            whitening=whitening,
            write=write,
            dry_run=args.dry_run,
            now=datetime.now(timezone.utc),
            batch_size=args.batch_size,
        )
        n_no_embedding = collection.count_documents({JOB_EMBEDDING_FIELD: None})
    finally:
        client.close()

    _print_result(result, n_no_embedding)
    print("[whiten_jobs] " + ("done" if result.written else "dry run: nothing written"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
