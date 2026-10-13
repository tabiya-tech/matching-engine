"""
Standalone script to run the matching algorithm locally without MongoDB.
Loads jobs from data/demand.jsonl and matches them against supply.jsonl users.

Usage (from backend/ directory):
    python run_local.py
    python run_local.py --user user_001
    python run_local.py --all

From root of repo, to write output to file:
    python backend/run_local.py --output data/output_full.json
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load backend/.env before any app imports (paths, MONGO_*).
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── 0. Fix OpenMP conflict on Windows with multiple conda packages ────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── 1. Import app modules in the right order (import order matters for DLL loading) ──
# No Mongo client is created on import (the app opens it in its lifespan), and this script passes
# jobs/occupations from local files straight to the matcher, so no database is needed.
sys.path.insert(0, str(Path(__file__).parent))
from app.config import OCCUPATION_JSON_PATH  # noqa: E402
from app.domain.job import Job  # noqa: E402
from app.domain.occupation import Occupation  # noqa: E402
from app.domain.user import UserProfile  # noqa: E402
from app.occupations.flatten import flatten_occupations  # noqa: E402
from app.occupations.occupation_file import OCCUPATION_FILE  # noqa: E402
from app.schemas import MatchResponse  # noqa: E402
from app.services.offline_io import load_jobs_file, load_users  # noqa: E402

# ── 2. Load local JSONL data (override with SUPPLY_JSONL_PATH / DEMAND_JSONL_PATH) ──
REPO_ROOT = Path(__file__).parent.parent
SUPPLY_PATH = Path(
    os.getenv("SUPPLY_JSONL_PATH", str(REPO_ROOT / "data" / "supply.jsonl"))
)
DEMAND_PATH = Path(
    os.getenv("DEMAND_JSONL_PATH", str(REPO_ROOT / "data" / "demand.jsonl"))
)

JOBS: list[Job] = load_jobs_file(DEMAND_PATH)

# Each occupation × county becomes one row, exactly as the service flattens the occupation file.
OCCUPATIONS: list[Occupation] = flatten_occupations(
    OCCUPATION_FILE.validate_json(Path(OCCUPATION_JSON_PATH).read_bytes())
)


import app.services.matching_service as ms_module  # noqa: E402

# ── 3. Run matching ───────────────────────────────────────────────────────────


async def run(users: list[UserProfile]) -> list[MatchResponse]:
    """Match each user against the local jobs and occupations."""
    return [await ms_module.match_single_user(u, JOBS, OCCUPATIONS) for u in users]


def main():
    parser = argparse.ArgumentParser(description="Run matching algorithm locally")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--user", metavar="USER_ID", help="Match a specific user ID")
    group.add_argument(
        "--all", action="store_true", help="Match all users in supply.jsonl"
    )
    parser.add_argument("--output", metavar="FILE", help="Write JSON output to file")
    args = parser.parse_args()

    supply = load_users(SUPPLY_PATH)

    if args.user:
        users = [u for u in supply if u.user_id == args.user]
        if not users:
            print(f"User '{args.user}' not found in supply.jsonl", file=sys.stderr)
            sys.exit(1)
    elif args.all:
        users = supply
    else:
        # Default: first user only
        users = supply[:1]
        print(
            f"No flag given — matching first user: {users[0].user_id}\n"
            f"Use --user <id> or --all to match other users.\n",
            file=sys.stderr,
        )

    results = asyncio.run(run(users))

    output_json = json.dumps([r.model_dump(mode="json") for r in results], indent=2)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
