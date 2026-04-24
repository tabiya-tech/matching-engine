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

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from unittest.mock import MagicMock

from dotenv import load_dotenv

# Load backend/.env before any app imports (paths, MONGO_*).
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── 0. Fix OpenMP conflict on Windows with multiple conda packages ────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── 1. Fake env vars so database.py doesn't raise at import ──────────────────
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB_NAME", "test")

# ── 2. Mock motor so AsyncIOMotorClient doesn't try to connect ────────────────
motor_mock = MagicMock()
sys.modules["motor"] = motor_mock
sys.modules["motor.motor_asyncio"] = motor_mock

# ── 3. Import app modules in the right order (import order matters for DLL loading) ──
sys.path.insert(0, str(Path(__file__).parent))
import app.database as db_module
import app.config  # must be imported before skill_score/matching_service
import app.services.preference_score
import app.services.demand_score
import app.services.skill_score
from app.config import OCCUPATION_JSON_PATH

# ── 4. Load local JSONL data (override with SUPPLY_JSONL_PATH / DEMAND_JSONL_PATH) ──
REPO_ROOT = Path(__file__).parent.parent
SUPPLY_PATH = Path(os.getenv("SUPPLY_JSONL_PATH", str(REPO_ROOT / "data" / "supply.jsonl")))
DEMAND_PATH = Path(os.getenv("DEMAND_JSONL_PATH", str(REPO_ROOT / "data" / "demand.jsonl")))


def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


JOBS = load_jsonl(DEMAND_PATH)

OCCUPATION_DB_PATH = Path(OCCUPATION_JSON_PATH)
with open(OCCUPATION_DB_PATH, "r", encoding="utf-8") as _f:
    _raw_occupations = json.load(_f)

# Flatten: each occupation × county becomes one record in the format the matching
# service expects (mirrors how data would be stored flat in MongoDB).
OCCUPATIONS = []
for entry in _raw_occupations:
    occ = entry.get("occupation", {})
    skills = entry.get("skills", {})
    essential_uuids = skills.get("essential", {}).get("uuids", [])
    optional_uuids = skills.get("optional", {}).get("uuids", [])
    onet_was = entry.get("onet_work_activities", [])

    for county_entry in entry.get("counties_data", []):
        county = county_entry.get("county", "")
        raw_attrs = (county_entry.get("job_attributes") or {}).get("attributes", [])
        flat_attrs = {a["attribute_name"]: a["selected_level_id"] for a in raw_attrs if "attribute_name" in a and "selected_level_id" in a}

        OCCUPATIONS.append({
            "uuid": f"{occ.get('code', '')}_{county}",
            "originUuid": occ.get("code", ""),
            "occupation_label": occ.get("preferred_label", ""),
            "occupation_description": occ.get("description", ""),
            "province": county,
            "city": county,
            "location": county,
            "essential_skills_origin_uuids": essential_uuids,
            "optional_skills_origin_uuids": optional_uuids,
            "skill_groups_origin_uuids": [],
            "attributes": flat_attrs,
            "onet_work_activities": onet_was,
        })


# ── 6. Patch database helpers to use local files ──────────────────────────────
async def _get_all_jobs():
    return JOBS


async def _get_all_occupations():
    return OCCUPATIONS


db_module.get_all_jobs = _get_all_jobs
db_module.get_all_occupations = _get_all_occupations

# Re-import matching service AFTER patching so it picks up patched db helpers
import importlib
import app.services.matching_service as ms_module
importlib.reload(ms_module)

# ── 7. Run matching ───────────────────────────────────────────────────────────

async def run(users: list) -> list:
    results = []
    for user in users:
        result = await ms_module.match_single_user(user)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run matching algorithm locally")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--user", metavar="USER_ID", help="Match a specific user ID")
    group.add_argument("--all", action="store_true", help="Match all users in supply.jsonl")
    parser.add_argument("--output", metavar="FILE", help="Write JSON output to file")
    args = parser.parse_args()

    supply = load_jsonl(SUPPLY_PATH)

    if args.user:
        users = [u for u in supply if u.get("user_id") == args.user]
        if not users:
            print(f"User '{args.user}' not found in supply.jsonl", file=sys.stderr)
            sys.exit(1)
    elif args.all:
        users = supply
    else:
        # Default: first user only
        users = supply[:1]
        print(f"No flag given — matching first user: {users[0].get('user_id')}\n"
              f"Use --user <id> or --all to match other users.\n", file=sys.stderr)

    results = asyncio.run(run(users))

    output_json = json.dumps(results, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
