"""
Run the live ``POST /match_v4`` algorithm locally against offline datasets — no MongoDB.

This drives the *real* route handler (``app.routes.match_v4``), so the run goes through the
identical code path the deployed service uses:
    Pydantic validation -> Gemini user embedding -> job-vector cosine retrieval
    -> cross-encoder rerank -> u_hat/p_hat preference final score -> response model.

The only swap is the Mongo job read: ``app.database.get_all_jobs_with_timing`` is monkeypatched
to serve jobs from a local JSON file instead of MongoDB. Occupations are not used by /match_v4.

Data sources (override via env or CLI):
    JOBS   : data/kenya_jobs_for_pipeline.json   (JSON array, already in build_job_dict shape,
             each job carries a 3072-dim ``job_embedding`` used as the stage-1 vector)
    USERS  : data/kenya_match_input.jsonl        (one MatchRequest-shaped dict per line)

Completeness gate (decided requirement): only users with a *full* set of information are run.
By default a user is "complete" when it has non-empty ``skills_vector.top_skills`` AND a populated
``preference_vector`` AND non-empty ``preference_vector.bws_scores`` (equivalently: none of the
blocking prep_status_tags {skills_missing, preferences_missing, bws_missing}).

Caveats (also recorded in manifest.json):
  * PREFERENCE_SCORER_MODE: we run whatever the active branch's .env sets (faithful to live);
    app.config validates/aliases it (valid values differ per branch — e.g. 'unified'/'legacy' on
    post-secondary_and_v1_response, 'legacy'/'hybrid_v1' on main). Override via --preference-scorer-mode.
  * HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are off by default, so the cross-encoder model
    (cross-encoder/ms-marco-MiniLM-L-6-v2, ~80MB) downloads from HuggingFace on first run, then
    caches. First run needs internet; later runs are offline.
  * GEMINI_API_KEY is consumed to embed users (one small network batch per run).

Datasets / toggling input:
    --dataset kenya   -> data/kenya_match_input.jsonl  (default; 8 complete users, all with BWS)
    --dataset njila   -> data/njila_match_input.jsonl  (71 users, no BWS collected)
    --users <path>    -> any JSONL of MatchRequest-shaped users (overrides --dataset)
  The jobs corpus is shared across datasets (--jobs to change). The BWS completeness requirement
  auto-adapts: required only if the chosen dataset actually contains BWS (kenya yes, njila no);
  force it with --require-bws / --no-require-bws.

Usage (from backend/ directory):
    python run_match_v4_local.py                          # kenya (default): 8 complete users, full corpus
    python run_match_v4_local.py --dataset njila          # njila: 71 users (BWS auto-relaxed)
    python run_match_v4_local.py --users data/my.jsonl    # arbitrary users file
    python run_match_v4_local.py --no-require-bws          # kenya without bws gate (10 users)
    python run_match_v4_local.py --user <user_id>         # one user
    python run_match_v4_local.py --limit 3                # first N complete users
    python run_match_v4_local.py --all-users              # bypass completeness gate (debug)
    python run_match_v4_local.py --preference-scorer-mode hybrid_v1
    python run_match_v4_local.py --final-top-k 30 --retrieve-top-k 50 --final-score-combiner product
"""

import os
import sys
import csv
import json
import asyncio
import argparse
import datetime as _dt
from pathlib import Path
from unittest.mock import MagicMock

from dotenv import load_dotenv

# ── 0. Load backend/.env before any app import (paths, MONGO_*, GEMINI_API_KEY, scorer mode) ──
BACKEND_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_ROOT.parent
load_dotenv(BACKEND_ROOT / ".env")

# ── 1. Fix OpenMP conflict on Windows with multiple conda packages ────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── 2. Fake env vars so database.py doesn't raise at import, and mock motor so no DB connect ──
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB_NAME", "test")
_motor_mock = MagicMock()
sys.modules["motor"] = _motor_mock
sys.modules["motor.motor_asyncio"] = _motor_mock

# Named dataset presets for --dataset (users file per dataset; jobs corpus is shared unless --jobs given).
# njila users carry no BWS, so the BWS requirement auto-relaxes for it (see _resolve_require_bws).
_DATA = REPO_ROOT / "data"
DATASETS = {
    "kenya": _DATA / "kenya_match_input.jsonl",
    "njila": _DATA / "njila_match_input.jsonl",
}
DEFAULT_DATASET = "kenya"

# Default data paths (override with JOBS_JSON_PATH / USERS_JSONL_PATH env or CLI).
DEFAULT_JOBS_PATH = Path(os.getenv("JOBS_JSON_PATH", str(_DATA / "kenya_jobs_for_pipeline.json")))
DEFAULT_USERS_PATH = Path(os.getenv("USERS_JSONL_PATH", str(DATASETS[DEFAULT_DATASET])))

BLOCKING_TAGS = {"skills_missing", "preferences_missing", "bws_missing"}


def _load_jobs(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    if not isinstance(jobs, list):
        raise ValueError(f"Expected a JSON array of jobs in {path}, got {type(jobs).__name__}")
    return jobs


def _load_users_jsonl(path: Path) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _has_skills(u: dict) -> bool:
    return len(((u.get("skills_vector") or {}).get("top_skills")) or []) > 0


def _has_preferences(u: dict) -> bool:
    pv = u.get("preference_vector") or {}
    return pv.get("earnings_per_month") is not None


def _has_bws(u: dict) -> bool:
    return bool((u.get("preference_vector") or {}).get("bws_scores"))


def _completeness(u: dict, require_bws: bool) -> tuple[bool, str]:
    """Return (is_complete, reason). Reason is empty when complete."""
    missing = []
    if not _has_skills(u):
        missing.append("skills")
    if not _has_preferences(u):
        missing.append("preferences")
    if require_bws and not _has_bws(u):
        missing.append("bws")
    if missing:
        return False, "missing: " + ", ".join(missing)
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run /match_v4 locally against offline files (no Mongo)")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--user", metavar="USER_ID", help="Run a single user_id (must pass completeness unless --all-users)")
    grp.add_argument("--all-users", action="store_true", help="Bypass the completeness gate (debug/contrast)")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Cap to first N selected users")
    bws_grp = parser.add_mutually_exclusive_group()
    bws_grp.add_argument("--require-bws", dest="require_bws", action="store_true", default=None,
                         help="Force the bws requirement on in the completeness gate")
    bws_grp.add_argument("--no-require-bws", dest="require_bws", action="store_false", default=None,
                         help="Force the bws requirement off (e.g. kenya: 8 -> 10 users)")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=None,
                        help=f"Named users-input preset (default {DEFAULT_DATASET}). Overridden by --users. "
                             f"BWS requirement auto-adapts per dataset unless --require-bws/--no-require-bws is given.")
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS_PATH, help="Jobs JSON array path (shared across datasets)")
    parser.add_argument("--users", type=Path, default=None, help="Users JSONL path (overrides --dataset)")
    parser.add_argument("--retrieve-top-k", type=int, default=None, help="Stage-1 cosine shortlist (default: env COSINE_CROSS_ENCODER_RETRIEVE_TOP_K)")
    parser.add_argument("--final-top-k", type=int, default=30, help="Final ranked rows per user (default 30)")
    parser.add_argument("--final-score-combiner", choices=["product", "geometric_mean"], default=None,
                        help="u_hat/p_hat combiner (default: env FINAL_SCORE_COMBINER)")
    parser.add_argument("--preference-scorer-mode", default=None,
                        help="Override PREFERENCE_SCORER_MODE (e.g. hybrid_v1) before scorers are imported")
    parser.add_argument("--no-system-trust", action="store_true",
                        help="Do not route TLS through the OS trust store (truststore). Use only if certifi "
                             "already trusts the Gemini endpoint and no TLS-inspecting proxy/AV is present.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output_results" / "match_v4_local",
                        help="Output root dir (a timestamped subdir is created)")
    args = parser.parse_args()

    # Resolve which users file to run: explicit --users wins, else --dataset preset, else default.
    if args.users is not None:
        users_path = Path(args.users)
        dataset_label = f"custom ({users_path.name})"
    else:
        dataset_label = args.dataset or DEFAULT_DATASET
        users_path = Path(os.getenv("USERS_JSONL_PATH") or DATASETS[dataset_label]) if args.dataset is None else DATASETS[dataset_label]
    args.users = users_path

    # Route TLS through the OS trust store so the live Gemini call works behind TLS-inspecting
    # antivirus/proxies (e.g. Norton Web/Mail Shield), whose root lives in the Windows cert store
    # but not in certifi's bundle. This trusts the OS store (secure) — it does NOT disable verification.
    if not args.no_system_trust:
        try:
            import truststore

            truststore.inject_into_ssl()
            print("TLS: using OS trust store via truststore.inject_into_ssl()", file=sys.stderr)
        except ImportError:
            print(
                "WARNING: truststore not installed; relying on certifi. If you see "
                "CERTIFICATE_VERIFY_FAILED, run `pip install truststore` or clear --no-system-trust.",
                file=sys.stderr,
            )

    # PREFERENCE_SCORER_MODE: the harness is faithful to whatever the active branch's .env sets
    # (e.g. 'unified' on post-secondary_and_v1_response, 'legacy'/'hybrid_v1' elsewhere). We do NOT
    # second-guess it — app.config is the source of truth and validates/aliases the value at import.
    # Only --preference-scorer-mode overrides it.
    if args.preference_scorer_mode:
        os.environ["PREFERENCE_SCORER_MODE"] = args.preference_scorer_mode.strip().lower()

    # Safety net (harmless when unneeded): if the DCE attribute schema env var is unset and the
    # canonical schema file is present in the package, point HYBRID_PREF_SCHEMA_PATH at it. On
    # branches where the schema is tracked + the default path is correct this is a no-op; it only
    # matters on branches whose default path is stale/missing. The scorer degrades gracefully
    # (BWS-only) if neither resolves.
    if not (os.getenv("HYBRID_PREF_SCHEMA_PATH") or "").strip():
        canonical_schema = BACKEND_ROOT / "app" / "services" / "preference_score_v1" / "job_attributes_schema.json"
        if canonical_schema.is_file():
            os.environ["HYBRID_PREF_SCHEMA_PATH"] = str(canonical_schema)

    # ── 3. Import app modules (order mirrors run_local.py: config & scorers first for DLL load order) ──
    # NOTE: we deliberately do NOT import app.routes — it pulls in fastapi (not installed in this
    # env, and not part of the algorithm). Instead we replicate the /match_v4 route body verbatim
    # around the real service function + the same Pydantic response model, so the matching code path
    # is identical to production.
    sys.path.insert(0, str(BACKEND_ROOT))
    try:
        import app.config  # noqa: F401  (must precede skill/preference scorers for DLL load order)
    except ValueError as e:
        # app.config raises for an invalid PREFERENCE_SCORER_MODE (value differs per branch).
        print(
            f"app.config rejected the configuration: {e}\n"
            f"PREFERENCE_SCORER_MODE={os.getenv('PREFERENCE_SCORER_MODE')!r}. "
            f"Fix backend/.env or pass --preference-scorer-mode with a value this branch accepts.",
            file=sys.stderr,
        )
        sys.exit(1)
    import app.database as db_module
    import app.services.preference_score  # noqa: F401
    import app.services.demand_score  # noqa: F401
    import app.services.skill_score  # noqa: F401
    from app.config import COSINE_CROSS_ENCODER_RETRIEVE_TOP_K, FINAL_SCORE_COMBINER, PREFERENCE_SCORER_MODE
    from app.services.cross_encoder.gemini_embeddings import EMBEDDING_DIM
    from app.services.match_concat_gemini_ce_preference_service import (
        run_match_concat_gemini_ce_with_preferences,
    )
    from app.schemas import MatchRequest, MatchConcatGeminiCeResponse

    # ── 4. Patch the Mongo job loader to serve the local corpus (full active set; ignore users=) ──
    jobs = _load_jobs(args.jobs)
    jobs_timing = {
        "mongo_ranked_find_ms": 0.0,
        "python_build_jobs_ms": 0.0,
        "n_ranked_raw": len(jobs),
        "n_jobs": len(jobs),
        "n_skipped_inactive": 0,
        "get_all_jobs_total_ms": 0.0,
        "jobs_retrieval_filter_applied": False,  # offline: no per-user location prefilter
        "jobs_find_use_projection": False,
        "source": "local_json",
        "path": str(args.jobs.resolve()),
    }

    async def _get_all_jobs_with_timing(users=None):  # noqa: ANN001 — signature matches production
        return list(jobs), dict(jobs_timing)

    async def _get_all_jobs(users=None):  # noqa: ANN001
        return list(jobs)

    db_module.get_all_jobs_with_timing = _get_all_jobs_with_timing
    db_module.get_all_jobs = _get_all_jobs

    # ── 5. Load + filter users ────────────────────────────────────────────────
    all_users = _load_users_jsonl(args.users)

    # BWS requirement: explicit flag wins; otherwise auto — require BWS only if the dataset
    # actually contains any (kenya has some → require; njila has none → relax). Keeps both
    # datasets working with no extra flags.
    if args.require_bws is None:
        require_bws = any(_has_bws(u) for u in all_users)
        bws_decision = f"auto ({'on' if require_bws else 'off'}; dataset {'has' if require_bws else 'has no'} BWS)"
    else:
        require_bws = args.require_bws
        bws_decision = f"explicit ({'on' if require_bws else 'off'})"

    report = []  # inclusion/exclusion report
    selected = []
    for u in all_users:
        uid = str(u.get("user_id") or "")
        if args.all_users:
            kept, reason = True, "(--all-users: gate bypassed)"
        else:
            kept, reason = _completeness(u, require_bws)
        report.append({"user_id": uid, "kept": kept, "reason": reason or "complete"})
        if kept:
            selected.append(u)

    if args.user:
        selected = [u for u in selected if str(u.get("user_id") or "") == args.user]
        if not selected:
            # Allow explicit single-user run even if it failed the gate, but warn.
            match = [u for u in all_users if str(u.get("user_id") or "") == args.user]
            if not match:
                print(f"User '{args.user}' not found in {args.users}", file=sys.stderr)
                sys.exit(1)
            _, why = _completeness(match[0], require_bws)
            print(f"User '{args.user}' is not complete ({why}). Use --all-users to run anyway.", file=sys.stderr)
            sys.exit(1)

    if args.limit is not None:
        selected = selected[: max(0, args.limit)]

    if not selected:
        print("No users selected after filtering. Nothing to run.", file=sys.stderr)
        sys.exit(1)

    retrieve_top_k = args.retrieve_top_k if args.retrieve_top_k is not None else COSINE_CROSS_ENCODER_RETRIEVE_TOP_K
    combiner = args.final_score_combiner if args.final_score_combiner is not None else FINAL_SCORE_COMBINER

    n_complete = sum(1 for r in report if r["kept"])
    print(
        f"dataset={dataset_label}  users_file={args.users.name}  require_bws={bws_decision}",
        file=sys.stderr,
    )
    print(
        f"n_users_total={len(all_users)}  n_users_complete={n_complete}  "
        f"n_users_selected={len(selected)}  n_jobs={len(jobs)}",
        file=sys.stderr,
    )
    print(
        f"PREFERENCE_SCORER_MODE={PREFERENCE_SCORER_MODE}  retrieve_top_k={retrieve_top_k}  "
        f"final_top_k={args.final_top_k}  final_score_combiner={combiner}",
        file=sys.stderr,
    )

    # ── 6. Run the /match_v4 code path (replicates app.routes.match_v4 body verbatim) ─────────
    # Production: validate -> model_dump -> get_all_jobs_with_timing -> service -> response model.
    payload = [MatchRequest(**u) for u in selected]
    users_dicts = [m.model_dump() for m in payload]
    t0 = _dt.datetime.now()
    jobs_list, mongo_timing = asyncio.run(_get_all_jobs_with_timing(users=users_dicts))
    raw = run_match_concat_gemini_ce_with_preferences(
        users_dicts,
        jobs_list,
        retrieve_top_k=retrieve_top_k,
        final_top_k=args.final_top_k,
        mongo_timing=mongo_timing,
        final_score_combiner=combiner,
    )
    responses = [MatchConcatGeminiCeResponse(**row) for row in raw]
    wall_s = (_dt.datetime.now() - t0).total_seconds()

    # response_model objects -> plain dicts
    resp_dicts = [r.model_dump() if hasattr(r, "model_dump") else dict(r) for r in responses]

    # ── 7. Write outputs ──────────────────────────────────────────────────────
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "match_v4_response.json").write_text(
        json.dumps(resp_dicts, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )

    # Flat CSV for quick correctness/completeness scanning.
    csv_path = out_dir / "recommendations.csv"
    csv_cols = [
        "user_id", "rank", "rank_cosine", "job_uuid", "opportunity_title", "employer", "location",
        "concat_cosine_similarity", "cross_encoder_score", "u_hat", "p_hat", "final_score",
    ]
    n_rows = 0
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for resp in resp_dicts:
            uid = resp.get("user_id")
            for rec in resp.get("concat_gemini_ce_recommendations") or []:
                w.writerow({
                    "user_id": uid,
                    "rank": rec.get("rank"),
                    "rank_cosine": rec.get("rank_cosine"),
                    "job_uuid": rec.get("job_uuid"),
                    "opportunity_title": rec.get("opportunity_title"),
                    "employer": rec.get("employer"),
                    "location": rec.get("location"),
                    "concat_cosine_similarity": rec.get("concat_cosine_similarity"),
                    "cross_encoder_score": rec.get("cross_encoder_score"),
                    "u_hat": rec.get("u_hat"),
                    "p_hat": rec.get("p_hat"),
                    "final_score": rec.get("final_score"),
                })
                n_rows += 1

    per_user_counts = {
        resp.get("user_id"): len(resp.get("concat_gemini_ce_recommendations") or [])
        for resp in resp_dicts
    }

    manifest = {
        "endpoint": "/match_v4",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(wall_s, 2),
        "data": {
            "dataset": dataset_label,
            "jobs_path": str(args.jobs.resolve()),
            "users_path": str(args.users.resolve()),
            "n_jobs": len(jobs),
            "job_loader": "local_json (full active corpus; users= location prefilter intentionally ignored)",
        },
        "config": {
            "preference_scorer_mode": PREFERENCE_SCORER_MODE,
            "retrieve_top_k": retrieve_top_k,
            "final_top_k": args.final_top_k,
            "final_score_combiner": combiner,
            "gemini_user_embed_dim": EMBEDDING_DIM,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
        "completeness": {
            "require_bws": require_bws,
            "require_bws_decision": bws_decision,
            "all_users_flag": args.all_users,
            "blocking_tags_definition": sorted(BLOCKING_TAGS),
            "n_users_total": len(all_users),
            "n_users_complete": n_complete,
            "n_users_selected": len(selected),
            "selected_user_ids": [str(u.get("user_id") or "") for u in selected],
            "inclusion_report": report,
        },
        "output": {
            "response_json": str((out_dir / "match_v4_response.json").resolve()),
            "recommendations_csv": str(csv_path.resolve()),
            "n_recommendation_rows": n_rows,
            "recommendations_per_user": per_user_counts,
        },
        "caveats": [
            "Job vectors come from each job's 3072-dim 'job_embedding' (stage-1 fallback); same space as gemini-embedding-001.",
            "Offline job loader returns the full active corpus and ignores per-user location prefilter (users have city='Unknown').",
            "Cross-encoder downloads from HuggingFace on first run unless HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set.",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nWrote outputs to {out_dir}", file=sys.stderr)
    print(f"  match_v4_response.json  ({len(resp_dicts)} users)", file=sys.stderr)
    print(f"  recommendations.csv     ({n_rows} rows)", file=sys.stderr)
    print(f"  manifest.json", file=sys.stderr)
    for uid, c in per_user_counts.items():
        print(f"    {uid}: {c} recommendations", file=sys.stderr)


if __name__ == "__main__":
    main()
