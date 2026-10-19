"""Backfill ``location_keys`` on the stored ranked-job documents (CORE-825).

The /match job filter is ``location_keys $in <the users' keys>`` (``app.services.location_query``), so
every document needs the array. The llm-reranker writes it on every new or re-ranked job; this tool
writes it on the documents ranked before that, from the same ``classifier_metadata`` city / county /
province via ``app.location.keys.location_keys_for``. Run it once per database BEFORE deploying the
backend that filters on the field: un-backfilled jobs are not loaded by /match.

Idempotent: documents whose stored keys already match are left untouched. Inactive documents are
backfilled too, so a re-activated job is matchable at once.

Usage (from backend/; uses the app's MONGO_* settings for the connection)::

    python -m tools.location.backfill_location_keys [--db KenyaJobs_V2] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from typing import Any

from pymongo import UpdateOne

from app.config import MONGO_JOBS_COLLECTION
from app.jobs.job_document import LOCATION_KEYS_FIELD
from app.location.keys import location_keys_for
from app.server_dependencies.db_dependencies import MatchingDBProvider


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def keys_of(doc: dict[str, Any]) -> list[str]:
    """The location keys of one stored ranked-job document."""
    meta = doc.get("classifier_metadata") or {}
    return list(
        location_keys_for(
            _text(meta.get("city")),
            _text(meta.get("county")),
            _text(meta.get("province")),
        )
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill location_keys on ranked jobs.")
    ap.add_argument(
        "--db", help="Database name (default: the configured MONGO_DB_NAME)."
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Count changes; write nothing."
    )
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()

    jobs_db = await MatchingDBProvider.get_jobs_db()
    db = jobs_db.client[args.db] if args.db else jobs_db
    collection = db[MONGO_JOBS_COLLECTION]
    n_seen = n_unchanged = n_written = 0
    key_counts: Counter[str] = Counter()
    ops: list[UpdateOne] = []

    async def flush() -> None:
        nonlocal n_written, ops
        if ops and not args.dry_run:
            result = await collection.bulk_write(ops, ordered=False)
            n_written += result.modified_count
        ops = []

    try:
        cursor = collection.find(
            {},
            {
                "classifier_metadata.city": 1,
                "classifier_metadata.county": 1,
                "classifier_metadata.province": 1,
                LOCATION_KEYS_FIELD: 1,
            },
        )
        async for doc in cursor:
            n_seen += 1
            keys = keys_of(doc)
            key_counts.update(keys)
            if doc.get(LOCATION_KEYS_FIELD) == keys:
                n_unchanged += 1
                continue
            ops.append(
                UpdateOne({"_id": doc["_id"]}, {"$set": {LOCATION_KEYS_FIELD: keys}})
            )
            if len(ops) >= args.batch_size:
                await flush()
        n_to_write = n_seen - n_unchanged
        await flush()
    finally:
        MatchingDBProvider.close()

    print(f"database={db.name} collection={MONGO_JOBS_COLLECTION}")
    print(f"documents={n_seen} unchanged={n_unchanged} to_write={n_to_write}")
    print(f"written={n_written}" + (" (dry run)" if args.dry_run else ""))
    print(
        "most common keys:",
        ", ".join(f"{k}={n}" for k, n in key_counts.most_common(15)),
    )


if __name__ == "__main__":
    asyncio.run(main())
