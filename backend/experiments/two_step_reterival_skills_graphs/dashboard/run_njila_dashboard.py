#!/usr/bin/env python3
"""Run skills-graph matching for all Njila users and build comparison dashboard."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from build_dashboard import write_dashboard

DASHBOARD_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = DASHBOARD_DIR.parent
OUTPUT_DIR = DASHBOARD_DIR / "output"
DEFAULT_USERS = EXPERIMENT_DIR / "data" / "njila_users.jsonl"
DEFAULT_JOBS = EXPERIMENT_DIR / "data" / "ranked_jobs_v2.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    sys.path.insert(0, str(EXPERIMENT_DIR))
    from graph_engine.context import DEFAULT_TAXONOMY_DIR, load_context
    from graph_engine.user_profile import load_users_jsonl, parse_user
    from registry import ALL, dashboard_meta, run_all, user_dashboard_record

    parser = argparse.ArgumentParser(description="Run Njila skills-graph dashboard")
    parser.add_argument("--users", type=Path, default=DEFAULT_USERS)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_DIR)
    parser.add_argument(
        "--json-out", type=Path, default=OUTPUT_DIR / "final_dashboard.json"
    )
    parser.add_argument(
        "--html-out", type=Path, default=OUTPUT_DIR / "final_dashboard.html"
    )
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    t0 = time.perf_counter()
    logger.info("Loading taxonomy + %s jobs …", args.jobs.name)
    ctx = load_context(args.taxonomy, args.jobs)
    logger.info("Graph: %d nodes · jobs=%d", ctx.tax.number_of_nodes(), len(ctx.jobs))

    users_raw = load_users_jsonl(args.users)
    logger.info("Loaded %d users from %s", len(users_raw), args.users)

    records: list[dict] = []
    for i, raw in enumerate(users_raw, 1):
        user = parse_user(raw)
        logger.info(
            "[%d/%d] Matching user %s (%d skills) …",
            i,
            len(users_raw),
            user.user_id,
            len(user.skills),
        )
        rankings = run_all(user, ctx, top_n=args.top_n)
        records.append(user_dashboard_record(raw, user, rankings))

    payload = {
        "meta": {"jobs": len(ctx.jobs), **dashboard_meta(len(records))},
        "data": {mod.NAME: records for mod in ALL},
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote JSON → %s", args.json_out)

    write_dashboard(args.json_out, args.html_out)
    logger.info("Wrote HTML → %s", args.html_out)
    logger.info("Done in %.1fs", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
