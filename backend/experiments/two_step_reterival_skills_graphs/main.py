#!/usr/bin/env python3
"""CLI: run the final ranker for one user."""

from __future__ import annotations

import logging
import sys

from graph_engine.context import DEFAULT_TAXONOMY_DIR, load_context
from graph_engine.user_profile import load_user_by_id
from paths import JOBS_PATH, USERS_PATH
from registry import run_final

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
PRINT_LIMIT = 10


def main() -> int:
    user_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not user_id:
        print("Usage: python main.py <user_id>")
        return 1

    ctx = load_context(DEFAULT_TAXONOMY_DIR, JOBS_PATH)
    user = load_user_by_id(USERS_PATH, user_id)
    if user is None:
        print(f"User '{user_id}' not found in {USERS_PATH}")
        return 1

    print(f"User: {user.user_id} ({user.city}) · {len(user.skills)} skills")
    print(f"Jobs: {len(ctx.jobs)}\n")

    recs = run_final(user, ctx, top_n=30)
    if not recs:
        print("No recommendations.")
        return 0

    print("Final recommendations (exact → graph):")
    for rec in recs[:PRINT_LIMIT]:
        tag = rec.get("src", "?")
        print(f"  #{rec['r']:2d} [{tag:5s}]  {rec['t']} @ {rec['e']}")
    if len(recs) > PRINT_LIMIT:
        print(f"  ... and {len(recs) - PRINT_LIMIT} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
