#!/usr/bin/env python3
"""Create or update the stage-1 jobs vector index of a deployment, and report what it can search.

The Atlas Vector Search index (``app.server_dependencies.vector_search``) is defined in code so
every stack (dev, shp, zmb, sa, eth) gets the same one; run this once per stack before setting
``STAGE1_RETRIEVAL_BACKEND=db``, and again whenever the definition changes. Atlas builds an index
asynchronously: wait for ``READY`` / queryable in the report before switching the stack.

The report also counts the active jobs the search can see (whitened with this service's
``V4_FULL_CONCAT_WHITENING_PATH``) and those it excludes: whitened with another artifact, not
whitened (llm-reranker ``RERANKER_CONCAT_WHITENING_ENABLED`` off, no backfill yet), or without an
embedding. Exit code 1 when an index is missing or not queryable.

Usage (from ``backend/``; connection from ``--env-file`` or ``MONGO_URL`` / ``MONGO_DB_NAME``)::

    python -m scripts.vector_search_indexes --env-file .env.dev            # report only
    python -m scripts.vector_search_indexes --env-file .env.dev --apply    # create / update
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


async def _run(apply: bool) -> int:
    from app.jobs.mongo_repository import MongoJobsRepository
    from app.server_dependencies.db_dependencies import MatchingDBProvider
    from app.server_dependencies.model_dependencies import (
        get_concat_whitening_sha256,
    )
    from app.server_dependencies.vector_search import (
        VECTOR_INDEXES,
        ensure_vector_index,
        vector_index_status,
    )

    db = await MatchingDBProvider.get_jobs_db()
    ok = True
    try:
        print(f"[vector_search_indexes] database {db.name}")
        for spec in VECTOR_INDEXES:
            if apply:
                outcome = await ensure_vector_index(db, spec)
                print(f"  {spec.collection}.{spec.name}: {outcome}")
            status = await vector_index_status(db, spec)
            ok = ok and status.exists and status.queryable
            print(
                f"  {spec.collection}.{spec.name}: exists={status.exists} "
                f"queryable={status.queryable} status={status.status} "
                f"definition_matches={status.definition_matches}"
            )

        artifact_sha256 = get_concat_whitening_sha256()
        if artifact_sha256 is None:
            print(
                "  whitening artifact unreadable: coverage not counted", file=sys.stderr
            )
            return 1
        print(f"  whitening artifact sha256 {artifact_sha256}")
        c = await MongoJobsRepository(db).count_stage1_coverage(artifact_sha256)
        print(
            f"  {c.collection}: searchable {c.n_searchable}/{c.n_total}; excluded: "
            f"other artifact {c.n_other_artifact}, not whitened {c.n_not_whitened}, "
            f"no embedding {c.n_no_embedding}"
        )
    finally:
        MatchingDBProvider.close()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Create/update and report the stage-1 jobs vector index."
    )
    ap.add_argument("--apply", action="store_true", help="Create or update the index.")
    ap.add_argument(
        "--env-file", type=Path, help="Env file (MONGO_URL, MONGO_DB_NAME, …)."
    )
    args = ap.parse_args(argv)
    if args.env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=True)
    return asyncio.run(_run(args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
