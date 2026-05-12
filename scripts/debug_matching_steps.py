#!/usr/bin/env python3
"""Run matching-service steps in isolation and write JSON under output_results/.

Step 1 — same parallel fetch as ``POST /match`` (see ``backend/app/routes.py``):
  ``asyncio.gather(get_all_jobs_with_timing(users=...), get_all_occupations_with_timing())``

Requires working ``backend/.env`` (Mongo, collection names, occupation JSON path).

Usage (from repo root):
  python scripts/debug_matching_steps.py --step 1
  python scripts/debug_matching_steps.py --step 2    # reads output_results/step_01_fetch_jobs_and_occupations/*
  python scripts/debug_matching_steps.py --step 3    # opp scoring from step 1 (+ location gate inside _match_items)
  python scripts/debug_matching_steps.py --step 4    # skill gaps (needs step 1 artefacts)
  python scripts/debug_matching_steps.py --step 5    # full match_user_with_data (golden /match payload)
  python scripts/debug_matching_steps.py --step all  # 1 … 5 in order

Optional:
  --supply PATH     default: data/njila_match_input.jsonl
  --n-users N       default: 1
  --full-occupations  write every occupation row (can be large); default writes sample only
  --jobs-jsonl PATH load jobs from JSONL instead of Mongo (must match post-``build_job_dict_from_ranked`` shape)
  --step2-full-jobs write full filtered job dicts per user (large); default summaries + UUIDs only
  --step3-opportunity-k N   max ranked opportunities written per user (default: env MATCH_TOP_K_OPPORTUNITIES)
  --step3-score-all-opportunities rank all filtered jobs per user (ignores MATCH_TOP_K; can be huge)
  --step3-include-occupations also run occupation _match_items (needs step_01/occupations.json or occupations_sample.json)
  --step4-top-k N    skill-gap rows before response filter (default: MATCH_TOP_K_SKILL_GAPS)

Step 5 recomputes the full endpoint payload via ``match_user_with_data`` (sanity check vs steps 3+4).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "backend"
OUT_ROOT_DEFAULT = REPO_ROOT / "output_results"
STEP_01_DIRNAME = "step_01_fetch_jobs_and_occupations"
STEP_02_DIRNAME = "step_02_location_filter"
STEP_03_DIRNAME = "step_03_score_opportunity_occupation"
STEP_04_DIRNAME = "step_04_skill_gap_analysis"
STEP_05_DIRNAME = "step_05_full_match_match_user_with_data"


def _setup_path_and_env() -> None:
    sys.path.insert(0, str(BACKEND_ROOT))
    from dotenv import load_dotenv

    load_dotenv(BACKEND_ROOT / ".env")


def _load_jsonl_users(path: Path, limit: int) -> list[dict]:
    users: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            users.append(json.loads(line))
            if len(users) >= limit:
                break
    return users


def _load_jsonl_all(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


async def step_01_fetch_jobs_and_occupations(
    users: list[dict],
    out_dir: Path,
    *,
    full_occupations: bool,
    occ_sample_cap: int = 100,
    jobs_jsonl: Path | None = None,
) -> None:
    from app.database import get_all_jobs_with_timing, get_all_occupations_with_timing

    t0 = time.perf_counter()
    if jobs_jsonl is not None:
        jobs = _load_jsonl_all(jobs_jsonl.resolve())
        jobs_timing = {
            "source": "jobs_jsonl",
            "path": str(jobs_jsonl.resolve()),
            "n_jobs": len(jobs),
            "mongo_skipped": True,
        }
        occ, occ_timing = await get_all_occupations_with_timing()
    else:
        (jobs, jobs_timing), (occ, occ_timing) = await asyncio.gather(
            get_all_jobs_with_timing(users=users),
            get_all_occupations_with_timing(),
        )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    step_dir = out_dir / STEP_01_DIRNAME
    meta = {
        "description": "Mirrors routes.match() parallel fetch before match_user_with_data",
        "jobs_source": "jsonl_file" if jobs_jsonl else "mongodb_get_all_jobs_with_timing",
        "n_users_supplied": len(users),
        "user_ids": [str(u.get("user_id")) for u in users],
        "fetch_parallel_wall_ms": round(wall_ms, 2),
        "n_jobs": len(jobs),
        "n_occupation_rows": len(occ),
        "jobs_timing": jobs_timing,
        "occupations_timing": occ_timing,
        "full_occupations_written": bool(full_occupations),
        "occupations_sample_rows": 0 if full_occupations else min(occ_sample_cap, len(occ)),
    }

    _write_json(step_dir / "meta.json", meta)
    _write_json(step_dir / "users.json", users)
    _write_json(step_dir / "jobs.json", jobs)

    if full_occupations:
        _write_json(step_dir / "occupations.json", occ)
    else:
        sample = occ[:occ_sample_cap]
        _write_json(step_dir / "occupations_sample.json", sample)
        meta_note = {
            "note": f"Full occupation list not written (n={len(occ)}). "
            "Re-run with --full-occupations to write occupations.json",
            "sample_size": len(sample),
        }
        _write_json(step_dir / "occupations_sample_meta.json", meta_note)

    print(f"Step 1 wrote to {step_dir}")
    print(f"  users={len(users)}  jobs={len(jobs)}  occupations={len(occ)}  wall_ms={wall_ms:.1f}")


def step_02_location_filter_jobs(
    out_dir: Path,
    *,
    step1_subdir: Path | None = None,
    write_full_jobs: bool = False,
) -> None:
    """Same gate as matching_service._match_items before scoring: `_job_matches_user_location`.

    Input: ``users.json`` + ``jobs.json`` from Step 1 (dependency).
    Occupations unchanged — not referenced here.

    Mirrors:
      MATCH_APPLY_LOCATION_FILTER + `_job_matches_user_location(job, user)`
    """
    from app.config import MATCH_APPLY_LOCATION_FILTER
    from app.services.matching_service import _job_matches_user_location

    base = step1_subdir if step1_subdir is not None else (out_dir / STEP_01_DIRNAME)
    users_path = base / "users.json"
    jobs_path = base / "jobs.json"
    for p in (users_path, jobs_path):
        if not p.is_file():
            sys.stderr.write(
                f"Step 2 depends on Step 1. Missing {p}. Run:\n"
                f"  python scripts/debug_matching_steps.py --step 1\n"
            )
            sys.exit(1)

    users: list[dict] = json.loads(users_path.read_text(encoding="utf-8"))
    jobs: list[dict] = json.loads(jobs_path.read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    summaries: list[dict] = []
    full_payload: dict[str, list[dict]] = {}

    for user in users:
        uid = str(user.get("user_id") or "?")
        n_in = len(jobs)
        t_u = time.perf_counter()
        if MATCH_APPLY_LOCATION_FILTER:
            kept = [j for j in jobs if _job_matches_user_location(j, user)]
        else:
            kept = list(jobs)
        filter_ms = (time.perf_counter() - t_u) * 1000.0

        summaries.append(
            {
                "user_id": uid,
                "n_jobs_before": n_in,
                "n_jobs_after": len(kept),
                "filter_wall_ms": round(filter_ms, 4),
                "kept_job_uuids": [str(j.get("uuid") or "") for j in kept],
            }
        )
        if write_full_jobs:
            full_payload[uid] = kept

    total_ms = (time.perf_counter() - t0) * 1000.0

    dest = out_dir / STEP_02_DIRNAME
    meta = {
        "description": "In-memory location filter identical to matching_service._match_items (jobs only)",
        "depends_on": {
            "step": 1,
            "users_json": str(users_path.resolve()),
            "jobs_json": str(jobs_path.resolve()),
        },
        "MATCH_APPLY_LOCATION_FILTER": MATCH_APPLY_LOCATION_FILTER,
        "n_users": len(users),
        "n_jobs_unfiltered": len(jobs),
        "total_wall_ms": round(total_ms, 4),
        "full_jobs_written": bool(write_full_jobs),
    }

    _write_json(dest / "meta.json", meta)
    _write_json(dest / "per_user_summaries.json", summaries)
    if write_full_jobs:
        _write_json(dest / "filtered_jobs_by_user_id.json", full_payload)

    print(f"Step 2 wrote to {dest}")
    print(f"  MATCH_APPLY_LOCATION_FILTER={MATCH_APPLY_LOCATION_FILTER}  users={len(users)}  jobs_in={len(jobs)}")
    for s in summaries:
        print(f"    user {s['user_id']}: {s['n_jobs_before']} → {s['n_jobs_after']} jobs")


def step_03_score_matching(
    out_dir: Path,
    *,
    step1_subdir: Path | None = None,
    include_occupations: bool = False,
    opportunity_top_k_override: int | None = None,
    score_all_opportunities: bool = False,
) -> None:
    """`_match_items` for opportunities (+ optional occupations) — same scoring as ``match_user_with_data``.

    Reads ``users.json`` + ``jobs.json`` from Step 1. Location filtering inside ``_match_items``
    duplicates Step 2 (intentionally identical code path).

    Output: formatted rows like the API (`_format_opportunity` / `_format_occupation`).
    """
    from app.config import (
        MATCH_TOP_K_OCCUPATIONS,
        MATCH_TOP_K_OPPORTUNITIES,
        SCORING_MODE,
    )
    from app.services.matching_service import _match_items

    base = step1_subdir if step1_subdir is not None else (out_dir / STEP_01_DIRNAME)
    users_path = base / "users.json"
    jobs_path = base / "jobs.json"
    for p in (users_path, jobs_path):
        if not p.is_file():
            sys.stderr.write(f"Step 3 depends on Step 1. Missing {p}\n")
            sys.exit(1)

    users: list[dict] = json.loads(users_path.read_text(encoding="utf-8"))
    jobs: list[dict] = json.loads(jobs_path.read_text(encoding="utf-8"))

    occ_path = base / "occupations.json"
    sample_path = base / "occupations_sample.json"
    occupations: Optional[list]
    occupations_source: Optional[str]

    occupations = None
    occupations_source = None
    if include_occupations:
        if occ_path.is_file():
            occupations = json.loads(occ_path.read_text(encoding="utf-8"))
            occupations_source = str(occ_path.resolve())
        elif sample_path.is_file():
            occupations = json.loads(sample_path.read_text(encoding="utf-8"))
            occupations_source = f"{sample_path.resolve()} (SAMPLE ONLY — rerun step 1 with --full-occupations for full parity)"
            sys.stderr.write(
                "Warning: occupation scoring uses occupations_sample.json, not full taxonomy.\n"
            )
        else:
            sys.stderr.write(
                "Step 3 --step3-include-occupations needs step_01/occupations.json or occupations_sample.json\n"
            )
            sys.exit(1)

    if score_all_opportunities:
        opp_k = max(len(jobs), 1)
    elif opportunity_top_k_override is not None:
        opp_k = max(1, opportunity_top_k_override)
    else:
        opp_k = MATCH_TOP_K_OPPORTUNITIES

    dest = out_dir / STEP_03_DIRNAME
    per_user: list[dict] = []
    t_all = time.perf_counter()

    for user in users:
        uid = str(user.get("user_id") or "?")
        t0 = time.perf_counter()
        opp_rows, opp_timing = _match_items(user, jobs, item_type="opportunity", top_k=opp_k)
        wall_opp_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        occ_rows: Optional[list] = None
        occ_timing: Optional[dict] = None
        wall_occ_ms: Optional[float] = None

        if include_occupations and occupations is not None:
            t1 = time.perf_counter()
            occ_rows, occ_timing = _match_items(
                user,
                occupations,
                item_type="occupation",
                top_k=MATCH_TOP_K_OCCUPATIONS,
            )
            wall_occ_ms = round((time.perf_counter() - t1) * 1000.0, 2)

        per_user.append(
            {
                "user_id": uid,
                "opportunity_recommendations": opp_rows,
                "opportunity_timing": {**opp_timing, "wall_ms": wall_opp_ms},
                "occupation_recommendations": occ_rows,
                "occupation_timing": occ_timing,
                "occupation_wall_ms": wall_occ_ms,
            }
        )

    total_ms = round((time.perf_counter() - t_all) * 1000.0, 2)

    meta = {
        "description": "match_user_with_data scoring via _match_items (opportunities; optional occupations)",
        "depends_on": {
            "step": 1,
            "users_json": str(users_path.resolve()),
            "jobs_json": str(jobs_path.resolve()),
        },
        "SCORING_MODE": SCORING_MODE,
        "opportunity_top_k_used": opp_k,
        "score_all_opportunities_flag": score_all_opportunities,
        "include_occupations": include_occupations,
        "occupations_source": occupations_source,
        "MATCH_TOP_K_OCCUPATIONS": MATCH_TOP_K_OCCUPATIONS,
        "n_users_scored": len(users),
        "n_jobs_loaded": len(jobs),
        "total_wall_ms": total_ms,
    }

    _write_json(dest / "meta.json", meta)
    _write_json(dest / "per_user_scores.json", per_user)
    print(f"Step 3 wrote to {dest}")
    print(f"  SCORING_MODE={SCORING_MODE}  opportunity_top_k={opp_k}  users={len(users)}")


def step_04_skill_gap_analysis(
    out_dir: Path,
    *,
    step1_subdir: Path | None = None,
    top_k_override: int | None = None,
) -> None:
    """Same as ``match_user_with_data`` skill-gap phase: ``analyze_skill_gaps`` + response filter.

    Uses **full** ``jobs.json`` from Step 1 (not location-filtered), matching production —
    gaps are mined over the same corpus as `/match``.
    """
    from app.config import MATCH_RESPONSE_SKILL_MIN_SCORE, MATCH_TOP_K_SKILL_GAPS
    from app.services.matching_service import _filter_skill_gap_recommendations, scorer_skill
    from app.services.skill_gap_analysis import analyze_skill_gaps

    base = step1_subdir if step1_subdir is not None else (out_dir / STEP_01_DIRNAME)
    users_path = base / "users.json"
    jobs_path = base / "jobs.json"
    for p in (users_path, jobs_path):
        if not p.is_file():
            sys.stderr.write(
                f"Step 4 depends on Step 1. Missing {p}. Run:\n"
                f"  python scripts/debug_matching_steps.py --step 1\n"
            )
            sys.exit(1)

    users: list[dict] = json.loads(users_path.read_text(encoding="utf-8"))
    jobs: list[dict] = json.loads(jobs_path.read_text(encoding="utf-8"))

    gap_k = top_k_override if top_k_override is not None else MATCH_TOP_K_SKILL_GAPS
    gap_k = max(1, gap_k)

    dest = out_dir / STEP_04_DIRNAME
    per_user: list[dict] = []
    t_all = time.perf_counter()

    for user in users:
        uid = str(user.get("user_id") or "?")
        t0 = time.perf_counter()
        skill_gaps = analyze_skill_gaps(
            user,
            jobs,
            scorer_skill.engine,
            scorer_skill.skill_labels,
            top_k=gap_k,
            resolve_id=scorer_skill._resolve_label,
            timing_out=None,
        )
        filtered = _filter_skill_gap_recommendations(skill_gaps)
        wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        per_user.append(
            {
                "user_id": uid,
                "skill_gap_recommendations": filtered,
                "n_raw_gaps_before_filter": len(skill_gaps),
                "n_after_response_threshold": len(filtered),
                "wall_ms": wall_ms,
            }
        )

    total_ms = round((time.perf_counter() - t_all) * 1000.0, 2)

    meta = {
        "description": (
            "skill_gap_analysis.analyze_skill_gaps then _filter_skill_gap_recommendations "
            "(same as match_user_with_data)"
        ),
        "depends_on": {"step": 1, "users_json": str(users_path), "jobs_json": str(jobs_path)},
        "note": (
            "Candidate jobs = full Step 1 list (Mongo/jsonl corpus), "
            "not Step 2 location-filtered set — mirrors production API."
        ),
        "MATCH_TOP_K_SKILL_GAPS_requested": gap_k,
        "MATCH_RESPONSE_SKILL_MIN_SCORE": MATCH_RESPONSE_SKILL_MIN_SCORE,
        "n_users": len(users),
        "n_jobs": len(jobs),
        "total_wall_ms": total_ms,
    }

    _write_json(dest / "meta.json", meta)
    _write_json(dest / "per_user_skill_gaps.json", per_user)
    print(f"Step 4 wrote to {dest}")
    print(f"  top_k_request={gap_k}  MATCH_RESPONSE_SKILL_MIN_SCORE={MATCH_RESPONSE_SKILL_MIN_SCORE}  users={len(users)}")
    for row in per_user:
        print(f"    user {row['user_id']}: {row['n_after_response_threshold']} gap rows (filtered)")


def step_05_full_match_user_with_data(
    out_dir: Path,
    *,
    step1_subdir: Path | None = None,
) -> None:
    """Canonical end-to-end path: ``match_user_with_data(user, jobs, occupations)``.

    Output matches what ``POST /match`` returns per user (opportunity + occupation + gap lists).
    Use to verify steps 3/4 piecewise output against one code path.

    Requires Step 1 ``users.json``, ``jobs.json``. Occupations: ``occupations.json`` or
    fallback ``occupations_sample.json`` (stderr warning; not production parity).
    """
    from app.services.matching_service import match_user_with_data

    base = step1_subdir if step1_subdir is not None else (out_dir / STEP_01_DIRNAME)
    users_path = base / "users.json"
    jobs_path = base / "jobs.json"
    for p in (users_path, jobs_path):
        if not p.is_file():
            sys.stderr.write(
                f"Step 5 depends on Step 1. Missing {p}. Run:\n"
                f"  python scripts/debug_matching_steps.py --step 1\n"
            )
            sys.exit(1)

    users: list[dict] = json.loads(users_path.read_text(encoding="utf-8"))
    jobs: list[dict] = json.loads(jobs_path.read_text(encoding="utf-8"))

    occ_json = base / "occupations.json"
    occ_sample = base / "occupations_sample.json"
    occupations_source: str
    if occ_json.is_file():
        occupations = json.loads(occ_json.read_text(encoding="utf-8"))
        occupations_source = str(occ_json.resolve())
    elif occ_sample.is_file():
        occupations = json.loads(occ_sample.read_text(encoding="utf-8"))
        occupations_source = f"{occ_sample.resolve()} (SAMPLE ONLY — run step 1 with --full-occupations for parity)"
        sys.stderr.write(
            "Step 5: using occupations_sample.json — occupation_recommendations are incomplete vs production.\n"
        )
    else:
        occupations = []
        occupations_source = "none"
        sys.stderr.write(
            "Step 5: no occupations.json or occupations_sample.json — occupation_recommendations will be empty.\n"
        )

    dest = out_dir / STEP_05_DIRNAME
    responses: list[dict] = []
    walls: list[float] = []
    t_all = time.perf_counter()

    for user in users:
        uid = str(user.get("user_id") or "?")
        t0 = time.perf_counter()
        try:
            body = match_user_with_data(user, jobs, occupations)
        except ValueError as e:
            walls.append(0.0)
            responses.append(
                {
                    "user_id": uid,
                    "error": str(e),
                    "opportunity_recommendations": [],
                    "occupation_recommendations": [],
                    "skill_gap_recommendations": [],
                }
            )
            sys.stderr.write(f"Step 5: user {uid}: {e}\n")
            continue
        walls.append(round((time.perf_counter() - t0) * 1000.0, 2))
        responses.append(body)

    total_ms = round((time.perf_counter() - t_all) * 1000.0, 2)

    meta = {
        "description": "match_user_with_data → same three lists as production /match body element",
        "depends_on": {
            "step": 1,
            "users_json": str(users_path.resolve()),
            "jobs_json": str(jobs_path.resolve()),
            "occupations_source": occupations_source,
        },
        "n_users_requested": len(users),
        "n_users_responses": len(responses),
        "user_ids_order": [str(u.get("user_id")) for u in users],
        "match_user_with_data_wall_ms_each": walls,
        "total_wall_ms": total_ms,
        "n_jobs": len(jobs),
        "n_occupation_rows_used": len(occupations),
    }

    _write_json(dest / "meta.json", meta)
    _write_json(dest / "match_response.json", responses)
    print(f"Step 5 wrote to {dest}")
    print(f"  responses={len(responses)} users  total_ms={total_ms:.1f}")


async def _amain() -> None:
    parser = argparse.ArgumentParser(description="Debug matching_service steps → output_results/")
    parser.add_argument(
        "--step",
        choices=["1", "2", "3", "4", "5", "all"],
        default="1",
        help="1=fetch | 2=location | 3=scoring | 4=gaps | 5=full match_user_with_data | all=1…5",
    )
    parser.add_argument(
        "--supply",
        type=Path,
        default=REPO_ROOT / "data" / "njila_match_input.jsonl",
        help="JSONL of user profiles (MatchRequest-shaped dicts)",
    )
    parser.add_argument("--n-users", type=int, default=1, help="How many users to take from the top of the file")
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_ROOT_DEFAULT,
        help="Output root directory",
    )
    parser.add_argument(
        "--full-occupations",
        action="store_true",
        help="Write full occupations.json (can be very large)",
    )
    parser.add_argument("--occ-sample-cap", type=int, default=100, help="Rows in occupations_sample.json when not full")
    parser.add_argument(
        "--jobs-jsonl",
        type=Path,
        default=None,
        help="Offline: load jobs from this JSONL (skip Mongo). Rows must match API job dict shape.",
    )
    parser.add_argument(
        "--step2-full-jobs",
        action="store_true",
        help="Step 2: also write filtered_jobs_by_user_id.json (full job dicts; can be large)",
    )
    parser.add_argument(
        "--step3-include-occupations",
        action="store_true",
        help="Step 3: also score occupations (needs step_01 occupations.json or occupations_sample.json)",
    )
    parser.add_argument(
        "--step3-opportunity-k",
        type=int,
        default=None,
        metavar="N",
        help="Step 3: opportunity top-k per user (default: MATCH_TOP_K_OPPORTUNITIES from env)",
    )
    parser.add_argument(
        "--step3-score-all-opportunities",
        action="store_true",
        help="Step 3: rank all jobs that pass location filter (can be large JSON)",
    )
    parser.add_argument(
        "--step4-top-k",
        type=int,
        default=None,
        metavar="N",
        help="Step 4: analyze_skill_gaps top_k (default MATCH_TOP_K_SKILL_GAPS)",
    )
    args = parser.parse_args()

    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.step in ("1", "all"):
        supply = args.supply.resolve()
        if not supply.is_file():
            print(f"Supply file not found: {supply}", file=sys.stderr)
            sys.exit(1)

        users = _load_jsonl_users(supply, args.n_users)
        if not users:
            print(f"No users loaded from {supply}", file=sys.stderr)
            sys.exit(1)

        missing_id = [i for i, u in enumerate(users) if not u.get("user_id")]
        if missing_id:
            print(f"WARNING: rows without user_id at indices {missing_id}", file=sys.stderr)

        jobs_jsonl = args.jobs_jsonl.resolve() if args.jobs_jsonl else None
        if jobs_jsonl is not None and not jobs_jsonl.is_file():
            print(f"--jobs-jsonl not found: {jobs_jsonl}", file=sys.stderr)
            sys.exit(1)

        await step_01_fetch_jobs_and_occupations(
            users,
            out_root,
            full_occupations=args.full_occupations,
            occ_sample_cap=max(1, args.occ_sample_cap),
            jobs_jsonl=jobs_jsonl,
        )

    if args.step in ("2", "all"):
        step_02_location_filter_jobs(
            out_root,
            step1_subdir=out_root / STEP_01_DIRNAME,
            write_full_jobs=args.step2_full_jobs,
        )

    if args.step in ("3", "all"):
        step_03_score_matching(
            out_root,
            step1_subdir=out_root / STEP_01_DIRNAME,
            include_occupations=args.step3_include_occupations,
            opportunity_top_k_override=args.step3_opportunity_k,
            score_all_opportunities=args.step3_score_all_opportunities,
        )

    if args.step in ("4", "all"):
        step_04_skill_gap_analysis(
            out_root,
            step1_subdir=out_root / STEP_01_DIRNAME,
            top_k_override=args.step4_top_k,
        )

    if args.step in ("5", "all"):
        step_05_full_match_user_with_data(
            out_root,
            step1_subdir=out_root / STEP_01_DIRNAME,
        )


def main() -> None:
    _setup_path_and_env()
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
