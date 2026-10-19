"""
Run the live ``POST /match`` engine (v4) — or one of the retired engines (legacy, v3, v5), which are
no longer served — locally against offline datasets, no MongoDB. Pick the engine with
``--version {match,v3,v4,v5}`` (default v4).

Each version calls the SAME engine function its route calls (or called, for retired versions), so
the local output is identical to what that endpoint's consumers receive(d):
  * ``--version v3`` -> ``run_match_v3_full`` (``app.matching.retired_routes.match_v3``): Gemini concat-cosine -> cross-
    encoder rerank. ``final_score`` is the concat cosine (whitened space when the artifact is present);
    NO preference layer (u_hat/p_hat empty). Same ``MatchResponse`` shape. Route defaults: 50/30.
  * ``--version v4`` -> ``MatchingService.rank`` (``app.matching.routes``, served at ``POST /match``): Pydantic validation -> Gemini
    user embedding -> concat-cosine retrieval -> cross-encoder rerank -> u_hat/p_hat preference final
    score -> ``MatchResponse`` with opportunities (jobs), occupations/careers (demand-gamma weighting
    + per-user location filter + top-k) and Node2Vec skill-gap recommendations.
  * ``--version v5`` -> the SAME ``MatchingService.rank`` engine, then per-opportunity ZQF (Zambia)
    eligibility annotation (zqf_eligible/zqf_gap + labels) from the user's ``zqf_level`` and each job's
    ``zqf_min`` (``app.matching.retired_routes.match_v5``) -> ``MatchResponseV5``. Null unless the data carries ZQF fields.
  * ``--version match`` -> ``match_user_with_data`` (``app.matching.retired_routes.match_legacy``): the legacy skill/Node2Vec
    engine — no Gemini user embedding, no cross-encoder, no retrieve/final top-k. Same ``MatchResponse``
    shape (opportunities + occupations + skill-gaps), so all the CSV/manifest outputs below apply; the
    v4-only u_hat/p_hat/demand columns are simply empty.

The only swap is the Mongo job read: jobs are served from a local JSON file instead of MongoDB
(occupations always load from local resource files). With ``--live-jobs`` even the job read goes to
MongoDB.

Live jobs (``--live-jobs``): read jobs straight from the MongoDB
configured in ``backend/.env`` (``MONGO_URL`` / ``MONGO_DB_NAME`` / ``MONGO_JOBS_COLLECTION``) —
useful when the live corpus carries ``requires_post_secondary``. Needs ``motor`` installed and
network access. Defaults to the full active corpus; ``--jobs-location-filter`` enables the
per-user Mongo location prefilter. Credentials live ONLY in ``backend/.env`` (never on the CLI).

Data sources (override via env or CLI):
    JOBS   : data/kenya_jobs_for_pipeline.json   (JSON array, already in build_job_dict shape,
             each job carries a 3072-dim ``job_embedding`` used as the stage-1 vector)
    USERS  : data/kenya_match_input.jsonl        (one MatchRequest-shaped dict per line)

Completeness gate (decided requirement): only users with a *full* set of information are run.
By default a user is "complete" when it has non-empty ``skills_vector.top_skills`` AND a populated
``preference_vector`` (``earnings_per_month`` present) AND non-empty ``preference_vector.bws_scores``.
The gate is computed directly from the payload — the live product input no longer carries
prep_status / prep_status_tags fields, so completeness is derived from the data itself. The
{skills_missing, preferences_missing, bws_missing} labels are just the human-readable reasons we
emit (BLOCKING_TAGS below), not fields read from the request.

Caveats (also recorded in manifest.json):
  * PREFERENCE_SCORER_MODE: we run whatever the active branch's .env sets (faithful to live);
    app.config validates/aliases it (valid values differ per branch — e.g. 'unified'/'legacy' on
    post-secondary_and_v1_response, 'legacy'/'hybrid_v1' on main). Override via --preference-scorer-mode.
  * HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are off by default, so the cross-encoder model
    (cross-encoder/ms-marco-MiniLM-L-6-v2, ~80MB) downloads from HuggingFace on first run, then
    caches. First run needs internet; later runs are offline.
  * GEMINI_API_KEY is consumed to embed users (one small network batch per run).

Datasets / toggling input:
    --dataset kenya   -> data/kenya_match_input.jsonl  (default; 85 users, 38 complete with BWS)
    --dataset njila   -> data/njila_match_input.jsonl  (71 users, no BWS collected)
    --users <path>    -> any JSONL of MatchRequest-shaped users (overrides --dataset)
  The jobs corpus is shared across datasets (--jobs to change). The BWS completeness requirement
  auto-adapts: required only if the chosen dataset actually contains BWS (kenya yes, njila no);
  force it with --require-bws / --no-require-bws.

Usage (from backend/ directory):
    python run_match_v4_local.py                          # v4 (default), kenya: 38 complete users, full corpus
    python run_match_v4_local.py --version match          # legacy /match engine (no Gemini/cross-encoder)
    python run_match_v4_local.py --version v5             # v4 engine + ZQF annotation (/experiments/v5/match)
    python run_match_v4_local.py --dataset njila          # njila: 71 users (BWS auto-relaxed)
    python run_match_v4_local.py --users data/my.jsonl    # arbitrary users file
    python run_match_v4_local.py --no-require-bws          # kenya without bws gate (51 users)
    python run_match_v4_local.py --user <user_id>         # one user
    python run_match_v4_local.py --limit 3                # first N complete users
    python run_match_v4_local.py --all-users              # bypass completeness gate (debug)
    python run_match_v4_local.py --preference-scorer-mode hybrid_v1
    python run_match_v4_local.py --final-top-k 30 --retrieve-top-k 50 --final-score-combiner product
"""

import argparse
import asyncio
import csv
import datetime as _dt
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

# ── 0. Load backend/.env before any app import (paths, MONGO_*, GEMINI_API_KEY, scorer mode) ──
BACKEND_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_ROOT.parent
load_dotenv(BACKEND_ROOT / ".env")

# ── 1. Fix OpenMP conflict on Windows with multiple conda packages ────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── 2. DB access mode (pre-scanned from sys.argv so it is known before argparse runs). ──
#   * default       : no DB is ever touched; jobs come from the local --jobs JSON file.
#   * --live-jobs   : a standalone jobs repository connects to the Mongo configured in backend/.env
#                     (MONGO_URL / MONGO_DB_NAME / MONGO_JOBS_COLLECTION) and jobs are read live
#                     (incl. requires_post_secondary).
USE_LIVE_JOBS = "--live-jobs" in sys.argv

# Named dataset presets for --dataset (users file per dataset; jobs corpus is shared unless --jobs given).
# njila users carry no BWS, so the BWS requirement auto-relaxes for it (see _resolve_require_bws).
_DATA = REPO_ROOT / "data"
DATASETS = {
    "kenya": _DATA / "kenya_match_input.jsonl",
    "njila": _DATA / "njila_match_input.jsonl",
}
DEFAULT_DATASET = "kenya"

# Default data paths (override with JOBS_JSON_PATH / USERS_JSONL_PATH env or CLI).
DEFAULT_JOBS_PATH = Path(
    os.getenv("JOBS_JSON_PATH", str(_DATA / "kenya_jobs_for_pipeline.json"))
)
DEFAULT_USERS_PATH = Path(os.getenv("USERS_JSONL_PATH", str(DATASETS[DEFAULT_DATASET])))

# Human-readable reasons surfaced when a user fails the completeness gate. These are derived from
# the payload by _completeness() below — they are NOT prep_status_tags read off the request (the
# live product input no longer carries prep_status / prep_status_tags). The post-secondary
# education fields the new payload does carry (any_post_secondary_educ / number_post_secondary_educ
# / total_duration_postsec) feed the algorithm's education gate, not this completeness gate.
BLOCKING_TAGS = {"skills_missing", "preferences_missing", "bws_missing"}


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
    parser = argparse.ArgumentParser(
        description="Run the POST /match engine (v4) or a retired engine locally against offline files (no Mongo)"
    )
    parser.add_argument(
        "--version",
        choices=["match", "v3", "v4", "v5"],
        default="v4",
        help=(
            "Which endpoint engine to run (default v4):\n"
            "  match -> retired legacy /match  : legacy skill/Node2Vec engine (match_user_with_data); "
            "no Gemini/cross-encoder, no retrieve/final top-k.\n"
            "  v3    -> retired v3 (not served) : Gemini concat-cosine -> cross-encoder rerank "
            "(run_match_v3_full); final_score is the concat cosine (whitened when the artifact is "
            "present), NO preference layer (u_hat/p_hat empty). Defaults retrieve=50/final=30.\n"
            "  v4    -> POST /match            : Gemini concat-cosine -> cross-encoder -> u_hat x p_hat "
            "(MatchingService.rank). Defaults retrieve=100/final=50.\n"
            "  v5    -> retired v5 (not served) : same engine as v4 plus per-opportunity ZQF "
            "(Zambia) eligibility annotation from the user's zqf_level and each job's zqf_min."
        ),
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--user",
        metavar="USER_ID",
        help="Run a single user_id (must pass completeness unless --all-users)",
    )
    grp.add_argument(
        "--all-users",
        action="store_true",
        help="Bypass the completeness gate (debug/contrast)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap to first N selected users",
    )
    bws_grp = parser.add_mutually_exclusive_group()
    bws_grp.add_argument(
        "--require-bws",
        dest="require_bws",
        action="store_true",
        default=None,
        help="Force the bws requirement on in the completeness gate",
    )
    bws_grp.add_argument(
        "--no-require-bws",
        dest="require_bws",
        action="store_false",
        default=None,
        help="Force the bws requirement off (e.g. kenya: 8 -> 10 users)",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default=None,
        help=f"Named users-input preset (default {DEFAULT_DATASET}). Overridden by --users. "
        f"BWS requirement auto-adapts per dataset unless --require-bws/--no-require-bws is given.",
    )
    parser.add_argument(
        "--jobs",
        type=Path,
        default=DEFAULT_JOBS_PATH,
        help="Jobs JSON array path (shared across datasets)",
    )
    parser.add_argument(
        "--users",
        type=Path,
        default=None,
        help="Users JSONL path (overrides --dataset)",
    )
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=None,
        help="Stage-1 cosine shortlist (default: MATCH_V4_RETRIEVE_TOP_K, the live /match_v4 default)",
    )
    parser.add_argument(
        "--final-top-k",
        type=int,
        default=None,
        help="Final ranked rows per user / pool sent to whitening (default: MATCH_V4_FINAL_TOP_K, the live /match_v4 default)",
    )
    parser.add_argument(
        "--final-score-combiner",
        choices=["product", "geometric_mean"],
        default=None,
        help="u_hat/p_hat combiner (default: env FINAL_SCORE_COMBINER)",
    )
    parser.add_argument(
        "--preference-scorer-mode",
        default=None,
        help="Override PREFERENCE_SCORER_MODE (e.g. hybrid_v1) before scorers are imported",
    )
    parser.add_argument(
        "--no-system-trust",
        action="store_true",
        help="Do not route TLS through the OS trust store (truststore). Use only if certifi "
        "already trusts the Gemini endpoint and no TLS-inspecting proxy/AV is present.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "output_results" / "match_v4_local",
        help="Output root dir (a timestamped subdir is created)",
    )
    parser.add_argument(
        "--live-jobs",
        action="store_true",
        help="Read jobs from the live MongoDB in backend/.env (MONGO_URL / "
        "MONGO_DB_NAME / MONGO_JOBS_COLLECTION) instead of the --jobs JSON file. "
        "Needs 'motor' installed and network access to the DB.",
    )
    parser.add_argument(
        "--jobs-location-filter",
        action="store_true",
        help="With --live-jobs, apply the per-user Mongo location prefilter "
        "(JOBS_RETRIEVAL_FILTER=true). Off by default so the full active corpus "
        "is returned (mirrors the offline harness; test users have city='Unknown').",
    )
    args = parser.parse_args()

    # Resolve which users file to run: explicit --users wins, else --dataset preset, else default.
    if args.users is not None:
        users_path = Path(args.users)
        dataset_label = f"custom ({users_path.name})"
    else:
        dataset_label = args.dataset or DEFAULT_DATASET
        users_path = (
            Path(os.getenv("USERS_JSONL_PATH") or DATASETS[dataset_label])
            if args.dataset is None
            else DATASETS[dataset_label]
        )
    args.users = users_path

    # Route TLS through the OS trust store so the live Gemini call works behind TLS-inspecting
    # antivirus/proxies (e.g. Norton Web/Mail Shield), whose root lives in the Windows cert store
    # but not in certifi's bundle. This trusts the OS store (secure) — it does NOT disable verification.
    if not args.no_system_trust:
        try:
            import truststore  # pyright: ignore[reportMissingImports] — optional; the except below covers its absence

            truststore.inject_into_ssl()
            print(
                "TLS: using OS trust store via truststore.inject_into_ssl()",
                file=sys.stderr,
            )
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
        os.environ["PREFERENCE_SCORER_MODE"] = (
            args.preference_scorer_mode.strip().lower()
        )

    # Safety net (harmless when unneeded): if the DCE attribute schema env var is unset and the
    # canonical schema file is present in the package, point HYBRID_PREF_SCHEMA_PATH at it. On
    # branches where the schema is tracked + the default path is correct this is a no-op; it only
    # matters on branches whose default path is stale/missing. The scorer degrades gracefully
    # (BWS-only) if neither resolves.
    if not (os.getenv("HYBRID_PREF_SCHEMA_PATH") or "").strip():
        canonical_schema = (
            BACKEND_ROOT
            / "app"
            / "services"
            / "preference_score_v1"
            / "job_attributes_schema.json"
        )
        if canonical_schema.is_file():
            os.environ["HYBRID_PREF_SCHEMA_PATH"] = str(canonical_schema)

    # ── Live-jobs preflight: fail fast with actionable messages, and pick the Mongo prefilter mode
    # BEFORE app.config is imported (config reads JOBS_RETRIEVAL_FILTER at import time). ──
    if USE_LIVE_JOBS:
        import importlib.util

        if importlib.util.find_spec("motor") is None:
            print(
                "ERROR: --live-jobs needs the 'motor' MongoDB driver. Install it:\n"
                "    pip install motor",
                file=sys.stderr,
            )
            sys.exit(1)
        mongo_url = (os.getenv("MONGO_URL") or "").strip()
        if not mongo_url or mongo_url == "mongodb://localhost:27017":
            print(
                "ERROR: --live-jobs needs a real MONGO_URL. Put your credentials in backend/.env:\n"
                f"  (current MONGO_URL={mongo_url!r} — placeholder/unset)\n"
                "    MONGO_URL=mongodb+srv://<user>:<password>@<cluster-host>/?retryWrites=true&w=majority\n"
                "    MONGO_DB_NAME=<database name>\n"
                "    MONGO_JOBS_COLLECTION=<jobs collection>   # default: RankedJobsEnriched\n"
                "  For Atlas TLS, also set:  MONGO_TLS_CA_FILE=certifi",
                file=sys.stderr,
            )
            sys.exit(1)
        # Full active corpus by default; opt into the per-user location prefilter with the flag
        # (or by setting JOBS_RETRIEVAL_FILTER explicitly in the environment / .env).
        if args.jobs_location_filter:
            os.environ["JOBS_RETRIEVAL_FILTER"] = "true"
        elif "JOBS_RETRIEVAL_FILTER" not in os.environ:
            os.environ["JOBS_RETRIEVAL_FILTER"] = "false"
        # Log the host only (split off any user:pass@ credentials) so secrets never hit the console.
        _host = mongo_url.split("@")[-1].split("/")[0][:60]
        print(
            f"LIVE JOBS: MongoDB host={_host}  db={os.getenv('MONGO_DB_NAME')!r}  "
            f"collection={os.getenv('MONGO_JOBS_COLLECTION') or 'RankedJobsEnriched'!r}  "
            f"JOBS_RETRIEVAL_FILTER={os.getenv('JOBS_RETRIEVAL_FILTER')}",
            file=sys.stderr,
        )

    # ── 3. Import app modules (order mirrors run_local.py: config & scorers first for DLL load order) ──
    # NOTE: we deliberately do NOT import the routes — they pull in fastapi (not part of the
    # algorithm). Instead we call MatchingService.rank (the exact method POST /match calls), so the
    # code path is identical to prod.
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
    from app.artifacts.repository import get_artifacts_repository
    from app.clients.gemini_embedding_client import EMBEDDING_DIM
    from app.config import (
        COSINE_CROSS_ENCODER_RETRIEVE_TOP_K,
        FINAL_SCORE_COMBINER,
        MATCH_TOP_K_OCCUPATIONS,
        MATCH_TOP_K_OPPORTUNITIES,
        MATCH_TOP_K_SKILL_GAPS,
        MATCH_V4_FINAL_TOP_K,
        MATCH_V4_OCC_DEMAND_GAMMA,
        MATCH_V4_RETRIEVE_TOP_K,
        MATCH_V4_TOP_K_OCCUPATIONS,
        PREFERENCE_SCORER_MODE,
    )
    from app.domain.job import Job
    from app.domain.occupation import Occupation
    from app.domain.user import UserProfile
    from app.jobs.get_jobs_repository import standalone_jobs_repository
    from app.jobs.repository import IJobsRepository, JobsLoadTiming
    from app.matching.get_matching_service import create_matching_service
    from app.occupations.loader import (
        attach_occupation_embeddings,
        get_all_occupations_with_timing,
    )
    from app.ranking.retrieval import (
        job_requires_post_secondary,
        user_lacks_post_secondary,
        zqf_annotation,
    )
    from app.ranking.types import MatchItem
    from app.schemas import (
        JobsStats,
        MatchRequest,
        MatchRequestV5,
        MatchResponse,
        MatchResponseV5,
        OpportunityRecommendationV5,
    )
    from app.services.job_retrieval import retrieve_jobs_with_timing
    from app.services.location_query import location_hub_chains, resolve_location_keys
    from app.services.match_v3_full_service import run_match_v3_full
    from app.services.matching_service import match_user_with_data
    from app.services.offline_io import load_jobs_file

    class _LoadedJobsRepository(IJobsRepository):
        """The already-loaded job corpus behind the jobs-repository interface.

        ``MatchingService.rank`` never reads the repository (jobs are passed in); this keeps the
        service construction free of a Mongo connection for offline runs.
        """

        def __init__(self, jobs: list[Job], timing: JobsLoadTiming):
            self._jobs = jobs
            self._timing = timing

        async def find_jobs_with_timing(self, query_filter=None, *, limit=0):  # noqa: ANN001
            return list(self._jobs), self._timing

        async def get_jobs_page_with_timing(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise NotImplementedError("paging is not used by this script")

        async def get_jobs_stats(self) -> JobsStats:
            raise NotImplementedError("stats are not used by this script")

    # v3/v4/v5 all run the Gemini concat → CE engine (need user embedding + cross-encoder +
    # occupation embeddings + retrieve/final top-k). HAS_PREF is the narrower v4/v5 family that
    # adds the u_hat/p_hat preference layer + final_score_combiner (v3 has no preference signal).
    IS_GEMINI = args.version in ("v3", "v4", "v5")
    HAS_PREF = args.version in ("v4", "v5")
    # v3 keeps the route's own shortlist defaults (retrieve=50/final=30); v4/v5 use the wider 100/50.
    V3_DEFAULT_FINAL_TOP_K = 30

    # ── 4. Job source. Offline (default): serve the local JSON corpus (full active set; ignore users).
    # Live (--live-jobs): read through a standalone jobs repository, with the same location prefilter
    # policy as the routes (app.services.job_retrieval). ──
    if not USE_LIVE_JOBS:
        # Ranked-job documents or flat job objects, validated into Job models.
        local_jobs = load_jobs_file(args.jobs)
        local_timing = JobsLoadTiming(
            mongo_ranked_find_ms=0.0,
            python_build_jobs_ms=0.0,
            n_ranked_raw=len(local_jobs),
            n_jobs=len(local_jobs),
            n_skipped_inactive=0,
            n_skipped_invalid=0,
            get_all_jobs_total_ms=0.0,
            jobs_find_use_projection=False,
            jobs_retrieval_filter_applied=False,  # offline: no per-user location prefilter
        )

        async def _fetch_jobs(
            users: list[UserProfile],
        ) -> tuple[list[Job], JobsLoadTiming]:
            return list(local_jobs), local_timing

    else:

        async def _fetch_jobs(
            users: list[UserProfile],
        ) -> tuple[list[Job], JobsLoadTiming]:
            async with standalone_jobs_repository() as jobs_repository:
                return await retrieve_jobs_with_timing(
                    jobs_repository,
                    resolve_location_keys(
                        users, location_hub_chains(get_artifacts_repository())
                    ),
                )

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
            print(
                f"User '{args.user}' is not complete ({why}). Use --all-users to run anyway.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.limit is not None:
        selected = selected[: max(0, args.limit)]

    if not selected:
        print("No users selected after filtering. Nothing to run.", file=sys.stderr)
        sys.exit(1)

    # Shortlist sizing mirrors each route's own defaults: v3 uses COSINE_CROSS_ENCODER_RETRIEVE_TOP_K
    # (50) / 30; v4/v5 use the wider MATCH_V4_RETRIEVE_TOP_K (100) / MATCH_V4_FINAL_TOP_K (50). CLI
    # --retrieve-top-k / --final-top-k override either. (Unused by --version match.)
    if args.version == "v3":
        default_retrieve, default_final = (
            COSINE_CROSS_ENCODER_RETRIEVE_TOP_K,
            V3_DEFAULT_FINAL_TOP_K,
        )
    else:
        default_retrieve, default_final = MATCH_V4_RETRIEVE_TOP_K, MATCH_V4_FINAL_TOP_K
    retrieve_top_k = (
        args.retrieve_top_k if args.retrieve_top_k is not None else default_retrieve
    )
    if args.final_top_k is None:
        args.final_top_k = default_final
    combiner = (
        args.final_score_combiner
        if args.final_score_combiner is not None
        else FINAL_SCORE_COMBINER
    )

    # Validate users once (production parity), then fetch jobs from the active source. Both modes
    # go through _fetch_jobs — offline it's the local JSON closure defined above; with --live-jobs
    # it's the real Mongo-backed repository. Done before the banner so the
    # n_jobs count and all downstream lookups reflect the corpus actually used. v5 validates with
    # MatchRequestV5 so the user's optional ``zqf_level`` reaches the UserProfile.
    request_model = MatchRequestV5 if args.version == "v5" else MatchRequest
    users = [UserProfile.from_request(request_model(**u)) for u in selected]
    t0 = _dt.datetime.now()
    try:
        jobs_list, mongo_timing = asyncio.run(_fetch_jobs(users))
    except Exception as e:  # noqa: BLE001 — surface live-DB failures with an actionable hint
        if USE_LIVE_JOBS:
            print(
                f"ERROR: live Mongo job load failed: {e.__class__.__name__}: {e}\n"
                "  Check MONGO_URL / MONGO_DB_NAME / MONGO_JOBS_COLLECTION in backend/.env, network "
                "access / Atlas IP allowlist, and TLS (Atlas needs a CA bundle: set "
                "MONGO_TLS_CA_FILE=certifi).",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    jobs: list[Job] = (
        jobs_list  # corpus actually used (banner, job_by_uuid lookups, education-gate counts, manifest)
    )

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
    endpoint_label = {
        "match": "/match (retired legacy engine)",
        "v3": "/experiments/v3/match (retired)",
        "v4": "/match",
        "v5": "/experiments/v5/match (retired)",
    }[args.version]
    if HAS_PREF:
        print(
            f"version={args.version} ({endpoint_label})  "
            f"PREFERENCE_SCORER_MODE={PREFERENCE_SCORER_MODE}  retrieve_top_k={retrieve_top_k}  "
            f"final_top_k={args.final_top_k}  final_score_combiner={combiner}",
            file=sys.stderr,
        )
    elif args.version == "v3":
        print(
            f"version=v3 ({endpoint_label})  Gemini concat-cosine -> cross-encoder (no preference layer; "
            f"final_score=concat cosine [whitened])  retrieve_top_k={retrieve_top_k}  final_top_k={args.final_top_k}",
            file=sys.stderr,
        )
    else:
        print(
            f"version={args.version} ({endpoint_label})  legacy skill/Node2Vec engine  "
            f"top_k_opportunities={MATCH_TOP_K_OPPORTUNITIES}  "
            f"top_k_occupations={MATCH_TOP_K_OCCUPATIONS}  "
            f"skill_gap_top_k={MATCH_TOP_K_SKILL_GAPS}",
            file=sys.stderr,
        )

    # ── 6. Run the chosen endpoint's code path — mirrors the matching app.routes EXACTLY ──────
    # Occupations load from local resource files via app.occupations (works offline and with --live-jobs).
    occ_corpus: list[Occupation]
    occ_corpus, _occ_timing = asyncio.run(get_all_occupations_with_timing())

    if IS_GEMINI:
        # v3/v4/v5 route bodies all: validate -> UserProfile -> (jobs, occ) ->
        # attach_occupation_embeddings -> the engine. We call the same engine the live route calls,
        # so opportunities, occupations (county-scoped + top-k) and skill-gaps are produced
        # identically to live consumers.
        occ_corpus = attach_occupation_embeddings(occ_corpus)

    matching_service = create_matching_service(
        _LoadedJobsRepository(jobs_list, mongo_timing)
    )
    responses: Sequence[MatchResponse | MatchResponseV5]
    if args.version == "v3":
        # /match_v3: Gemini concat-cosine -> cross-encoder rerank. final_score is the concat
        # cosine; NO preference layer (u_hat/p_hat empty). No combiner / mongo_timing args.
        responses = run_match_v3_full(
            users,
            jobs_list,
            occ_corpus,
            retrieve_top_k=retrieve_top_k,
            final_top_k=args.final_top_k,
            skill_gap_top_k=MATCH_TOP_K_SKILL_GAPS,
            matching_service=matching_service,
        )
    elif HAS_PREF:
        # POST /match (v4) + /experiments/v5/match: Gemini concat-cosine -> CE -> u_hat x p_hat
        # (demand-gamma + per-user location filter + top-k). v5 then annotates each opportunity with
        # ZQF eligibility (same loop as the retired v5 route).
        v4 = matching_service.rank(
            users,
            jobs_list,
            occ_corpus,
            retrieve_top_k=retrieve_top_k,
            final_top_k=args.final_top_k,
            final_score_combiner=combiner,
            skill_gap_top_k=MATCH_TOP_K_SKILL_GAPS,
            mongo_timing=mongo_timing,
        )
        if args.version == "v5":
            job_by_uuid_v5 = {j.uuid: j for j in jobs_list}
            v5_responses: list[MatchResponseV5] = []
            for response, user in zip(v4, users, strict=True):
                opportunities: list[OpportunityRecommendationV5] = []
                for opp in response.opportunity_recommendations:
                    job = job_by_uuid_v5.get(opp.uuid)
                    eligible, gap = zqf_annotation(
                        user.zqf_level, job.zqf_min if job else None
                    )
                    opportunities.append(
                        OpportunityRecommendationV5(
                            **dict(opp),
                            zqf_eligible=eligible,
                            zqf_gap=gap,
                            zqf_min_label=job.zqf_min_label if job else None,
                            zqf_max_label=job.zqf_max_label if job else None,
                        )
                    )
                v5_responses.append(
                    MatchResponseV5(
                        user_id=response.user_id,
                        occupation_recommendations=response.occupation_recommendations,
                        opportunity_recommendations=opportunities,
                        skill_gap_recommendations=response.skill_gap_recommendations,
                    )
                )
            responses = v5_responses
        else:
            responses = v4
    else:
        # Retired legacy /match: match_user_with_data per user (CPU-bound, thread-pooled live;
        # sequential here). Legacy skill/Node2Vec engine — no Gemini user embedding, no
        # cross-encoder, no retrieve/final top-k. Uses occupations as loaded (no
        # attach_occupation_embeddings; that is a v3/v4/v5-only step).
        responses = [match_user_with_data(u, jobs_list, occ_corpus) for u in users]

    # Output edge: the CSV / JSON writers below read the serialised responses.
    resp_dicts = [r.model_dump(mode="json") for r in responses]
    wall_s = (_dt.datetime.now() - t0).total_seconds()

    # ── 7. Write outputs ──────────────────────────────────────────────────────
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    response_filename = {
        "match": "match_response.json",
        "v3": "match_v3_response.json",
        "v4": "match_v4_response.json",
        "v5": "match_v5_response.json",
    }[args.version]
    (out_dir / response_filename).write_text(
        json.dumps(resp_dicts, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    # Lookups for joining recs back to the source items + the validated users.
    job_by_uuid: dict[str, Job] = {j.uuid: j for j in jobs}
    occ_by_uuid: dict[str, Occupation] = {o.uuid: o for o in occ_corpus}
    user_by_id: dict[str, UserProfile] = {u.user_id or "": u for u in users}

    def _fmt_user_skills(u: UserProfile) -> str:
        parts = []
        for s in u.skills_vector.top_skills:
            label = s.preferredLabel or s.originUUID or ""
            prof = s.proficiency
            parts.append(f"{label} ({prof})" if prof is not None else str(label))
        return " | ".join(parts)

    def _fmt_item_skills(item: MatchItem | None, essential: bool) -> str:
        if item is None:
            return ""
        skills = item.essential_skills if essential else item.optional_skills
        return " | ".join(s.label or s.id for s in skills)

    def _join(xs) -> str:
        return " | ".join(str(x) for x in (xs or []))

    PREF_KEYS = [
        "earnings_per_month",
        "task_content",
        "physical_demand",
        "work_flexibility",
        "social_interaction",
        "career_growth",
        "social_meaning",
    ]

    def _sb(rec: dict, key: str):
        """Pull a field from the rich MatchResponse row's score_breakdown (u_hat/p_hat/demand live there)."""
        return (rec.get("score_breakdown") or {}).get(key)

    def _user_vec_cols(u: UserProfile | None) -> dict[str, object]:
        if u is None:
            return {}
        pv = u.preference_vector
        cols: dict[str, object] = {
            "user_top_skills": _fmt_user_skills(u),
            "user_skill_groups_origin_uuids": _join(u.skill_groups_origin_uuids),
            "user_bws_scores": json.dumps(pv.bws_scores or {}, ensure_ascii=False),
            "user_top_10_bws": _join(pv.top_10_bws),
        }
        for k in PREF_KEYS:
            cols[f"user_pref_{k}"] = getattr(pv, k)
        return cols

    # ── 7a. Opportunities (jobs): flat CSV + detailed CSV with vectors ──
    # v5 carries the per-opportunity ZQF annotation; expose those columns only for v5.
    is_v5 = args.version == "v5"
    zqf_cols = ["zqf_eligible", "zqf_gap", "zqf_min_label", "zqf_max_label"]
    opp_base_cols = [
        "user_id",
        "rank",
        "job_uuid",
        "opportunity_title",
        "employer",
        "location",
        "is_eligible",
        "u_hat",
        "p_hat",
        "final_score",
        "demand_label",
    ] + (zqf_cols if is_v5 else [])

    def _opp_base_row(uid, rec):
        row = {
            "user_id": uid,
            "rank": rec.get("rank"),
            "job_uuid": rec.get("uuid"),
            "opportunity_title": rec.get("opportunity_title"),
            "employer": rec.get("employer"),
            "location": rec.get("location"),
            "is_eligible": rec.get("is_eligible"),
            "u_hat": _sb(rec, "u_hat"),
            "p_hat": _sb(rec, "p_hat"),
            "final_score": rec.get("final_score"),
            "demand_label": _sb(rec, "demand_label"),
        }
        if is_v5:
            for k in zqf_cols:
                row[k] = rec.get(k)
        return row

    csv_path = out_dir / "recommendations.csv"
    n_rows = 0
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=opp_base_cols)
        w.writeheader()
        for resp in resp_dicts:
            uid = resp.get("user_id")
            for rec in resp.get("opportunity_recommendations") or []:
                w.writerow(_opp_base_row(uid, rec))
                n_rows += 1

    vec_csv_path = out_dir / "recommendations_with_vectors.csv"
    vec_cols = (
        opp_base_cols
        + ["user_top_skills", "user_skill_groups_origin_uuids"]
        + [f"user_pref_{k}" for k in PREF_KEYS]
        + ["user_bws_scores", "user_top_10_bws"]
        + [
            "job_essential_skills",
            "job_optional_skills",
            "job_skill_groups_origin_uuids",
            "job_attributes",
        ]
    )
    n_vec_rows = 0
    with open(vec_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=vec_cols)
        w.writeheader()
        for resp in resp_dicts:
            uid = resp.get("user_id")
            uc = _user_vec_cols(user_by_id.get(str(uid or "")))
            for rec in resp.get("opportunity_recommendations") or []:
                j = job_by_uuid.get(str(rec.get("uuid") or ""))
                w.writerow(
                    {
                        **_opp_base_row(uid, rec),
                        "job_essential_skills": _fmt_item_skills(j, True),
                        "job_optional_skills": _fmt_item_skills(j, False),
                        "job_skill_groups_origin_uuids": _join(
                            j.skill_groups_origin_uuids if j else []
                        ),
                        "job_attributes": j.attributes.model_dump_json(
                            exclude_none=True
                        )
                        if j
                        else "{}",
                        **uc,
                    }
                )
                n_vec_rows += 1

    # ── 7b. Careers (occupations): flat CSV + detailed CSV with vectors ──
    career_base_cols = [
        "user_id",
        "rank",
        "occupation_uuid",
        "occupation_code",
        "occupation_label",
        "province",
        "is_eligible",
        "u_hat",
        "p_hat",
        "final_score",
        "demand_label",
        "salary_range",
    ]

    def _career_base_row(uid, rec):
        return {
            "user_id": uid,
            "rank": rec.get("rank"),
            "occupation_uuid": rec.get("uuid"),
            "occupation_code": rec.get("originUuid"),
            "occupation_label": rec.get("occupation_label"),
            "province": rec.get("province"),
            "is_eligible": rec.get("is_eligible"),
            "u_hat": _sb(rec, "u_hat"),
            "p_hat": _sb(rec, "p_hat"),
            "final_score": rec.get("final_score"),
            "demand_label": _sb(rec, "demand_label"),
            "salary_range": rec.get("salary_range"),
        }

    career_csv_path = out_dir / "career_recommendations.csv"
    n_career_rows = 0
    with open(career_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=career_base_cols)
        w.writeheader()
        for resp in resp_dicts:
            uid = resp.get("user_id")
            for rec in resp.get("occupation_recommendations") or []:
                w.writerow(_career_base_row(uid, rec))
                n_career_rows += 1

    career_vec_csv_path = out_dir / "career_recommendations_with_vectors.csv"
    career_vec_cols = (
        career_base_cols
        + ["user_top_skills", "user_skill_groups_origin_uuids"]
        + [f"user_pref_{k}" for k in PREF_KEYS]
        + ["user_bws_scores", "user_top_10_bws"]
        + [
            "occupation_essential_skills",
            "occupation_optional_skills",
            "occupation_skill_groups_origin_uuids",
            "occupation_attributes",
            "occupation_requires_post_secondary",
        ]
    )
    n_career_vec_rows = 0
    with open(career_vec_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=career_vec_cols)
        w.writeheader()
        for resp in resp_dicts:
            uid = resp.get("user_id")
            uc = _user_vec_cols(user_by_id.get(str(uid or "")))
            for rec in resp.get("occupation_recommendations") or []:
                occ = occ_by_uuid.get(str(rec.get("uuid") or ""))
                w.writerow(
                    {
                        **_career_base_row(uid, rec),
                        "occupation_essential_skills": _fmt_item_skills(occ, True),
                        "occupation_optional_skills": _fmt_item_skills(occ, False),
                        "occupation_skill_groups_origin_uuids": "",
                        "occupation_attributes": occ.attributes.model_dump_json(
                            exclude_none=True
                        )
                        if occ
                        else "{}",
                        "occupation_requires_post_secondary": occ.requires_post_secondary
                        if occ
                        else None,
                        **uc,
                    }
                )
                n_career_vec_rows += 1

    # ── 7c. Skill-gap recommendations (Node2Vec; same as /match_v4) ──
    sg_csv_path = out_dir / "skill_gap_recommendations.csv"
    sg_cols = [
        "user_id",
        "skill_id",
        "skill_label",
        "proximity_score",
        "job_unlock_count",
        "combined_score",
        "reasoning",
    ]
    n_sg_rows = 0
    with open(sg_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sg_cols)
        w.writeheader()
        for resp in resp_dicts:
            uid = resp.get("user_id")
            for sg in resp.get("skill_gap_recommendations") or []:
                row = {"user_id": uid}
                row.update({k: sg.get(k) for k in sg_cols[1:]})
                w.writerow(row)
                n_sg_rows += 1

    per_user_counts = {
        resp.get("user_id"): len(resp.get("opportunity_recommendations") or [])
        for resp in resp_dicts
    }
    career_per_user_counts = {
        resp.get("user_id"): len(resp.get("occupation_recommendations") or [])
        for resp in resp_dicts
    }
    skill_gap_per_user_counts = {
        resp.get("user_id"): len(resp.get("skill_gap_recommendations") or [])
        for resp in resp_dicts
    }

    # ── 7d. Education-gate observability ──────────────────────────────────────
    # The post-secondary gate is applied per-user in run_match_concat_gemini_ce (retrieval, before
    # the top-k cutoff) for BOTH jobs and occupations. Proof: for a user who lacks post-secondary,
    # the *_requiring_post_secondary counts below should be 0 even though the corpora contain such
    # items (n_*_requiring_post_secondary > 0). If a corpus count is 0, the gate is inert on this data.
    n_jobs_requiring_ps = sum(1 for j in jobs if job_requires_post_secondary(j))
    n_occ_requiring_ps = sum(1 for o in occ_corpus if job_requires_post_secondary(o))
    education_gate = {
        "n_jobs_requiring_post_secondary": n_jobs_requiring_ps,
        "n_occupation_rows_requiring_post_secondary": n_occ_requiring_ps,
        "note": (
            "Gate applied per-user in run_match_concat_gemini_ce (retrieval, before top-k), for both "
            "jobs and occupations. For a lacking user the requiring counts should be 0."
        ),
        "per_user": [],
    }
    for resp in resp_dicts:
        uid = str(resp.get("user_id") or "")
        u = user_by_id.get(uid)
        opps = resp.get("opportunity_recommendations") or []
        occs = resp.get("occupation_recommendations") or []
        opp_items = [job_by_uuid.get(str(rec.get("uuid") or "")) for rec in opps]
        occ_items = [occ_by_uuid.get(str(rec.get("uuid") or "")) for rec in occs]
        n_opp_req = sum(
            1 for j in opp_items if j is not None and job_requires_post_secondary(j)
        )
        n_occ_req = sum(
            1 for o in occ_items if o is not None and job_requires_post_secondary(o)
        )
        education_gate["per_user"].append(
            {
                "user_id": uid,
                "any_post_secondary_educ": u.any_post_secondary_educ if u else None,
                "lacks_post_secondary": user_lacks_post_secondary(u) if u else False,
                "n_opportunities": len(opps),
                "n_opportunities_requiring_post_secondary": n_opp_req,
                "n_occupations": len(occs),
                "n_occupations_requiring_post_secondary": n_occ_req,
            }
        )
    print(
        f"education_gate: {n_jobs_requiring_ps} jobs / {n_occ_requiring_ps} occupation-rows require "
        f"post-secondary; {sum(1 for p in education_gate['per_user'] if p['lacks_post_secondary'])} of "
        f"{len(education_gate['per_user'])} users lack it",
        file=sys.stderr,
    )

    manifest = {
        "version": args.version,
        "endpoint": endpoint_label,
        "engine": (
            "run_match_v3_full (identical to the retired POST /experiments/v3/match route; "
            "Gemini concat-cosine -> cross-encoder rerank, concat-cosine [whitened] final_score, no preference layer)"
            if args.version == "v3"
            else f"MatchingService.rank (identical to the POST {endpoint_label} route)"
            + (" + per-opportunity ZQF eligibility annotation" if is_v5 else "")
            if IS_GEMINI
            else "match_user_with_data (identical to the retired legacy POST /match route; legacy skill/Node2Vec engine)"
        ),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": round(wall_s, 2),
        "data": {
            "dataset": dataset_label,
            "jobs_source": "live_mongo" if USE_LIVE_JOBS else "local_json",
            "jobs_path": (
                f"mongo://{os.getenv('MONGO_DB_NAME')}/{os.getenv('MONGO_JOBS_COLLECTION') or 'RankedJobsEnriched'}"
                if USE_LIVE_JOBS
                else str(args.jobs.resolve())
            ),
            "users_path": str(args.users.resolve()),
            "n_jobs": len(jobs),
            "n_occupation_rows_corpus": len(occ_corpus),
            "job_loader": (
                f"live_mongo (app.services.job_retrieval.retrieve_jobs_with_timing; JOBS_RETRIEVAL_FILTER={os.getenv('JOBS_RETRIEVAL_FILTER')})"
                if USE_LIVE_JOBS
                else "local_json (full active corpus; users= location prefilter intentionally ignored)"
            ),
            "mongo_timing": mongo_timing.model_dump(),
        },
        "config": (
            {
                # v3: Gemini concat-cosine -> CE rerank, concat-cosine [whitened] final_score. No preference layer,
                # so no scorer mode / combiner / demand-gamma.
                "retrieve_top_k": retrieve_top_k,
                "final_top_k": args.final_top_k,
                "occupation_top_k": MATCH_V4_TOP_K_OCCUPATIONS,
                "skill_gap_top_k": MATCH_TOP_K_SKILL_GAPS,
                "gemini_user_embed_dim": EMBEDDING_DIM,
                "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            }
            if args.version == "v3"
            else {
                "preference_scorer_mode": PREFERENCE_SCORER_MODE,
                "retrieve_top_k": retrieve_top_k,
                "final_top_k": args.final_top_k,
                "occupation_top_k": MATCH_V4_TOP_K_OCCUPATIONS,
                "occupation_demand_gamma": MATCH_V4_OCC_DEMAND_GAMMA,
                "skill_gap_top_k": MATCH_TOP_K_SKILL_GAPS,
                "final_score_combiner": combiner,
                "gemini_user_embed_dim": EMBEDDING_DIM,
                "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            }
            if HAS_PREF
            else {
                # Legacy /match engine: no Gemini/cross-encoder, no retrieve/final top-k or combiner.
                "preference_scorer_mode": PREFERENCE_SCORER_MODE,
                "top_k_opportunities": MATCH_TOP_K_OPPORTUNITIES,
                "top_k_occupations": MATCH_TOP_K_OCCUPATIONS,
                "skill_gap_top_k": MATCH_TOP_K_SKILL_GAPS,
            }
        ),
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
            "response_json": str((out_dir / response_filename).resolve()),
            "recommendations_csv": str(csv_path.resolve()),
            "recommendations_with_vectors_csv": str(vec_csv_path.resolve()),
            "n_recommendation_rows": n_rows,
            "recommendations_per_user": per_user_counts,
            "career_recommendations_csv": str(career_csv_path.resolve()),
            "career_recommendations_with_vectors_csv": str(
                career_vec_csv_path.resolve()
            ),
            "n_career_recommendation_rows": n_career_rows,
            "career_recommendations_per_user": career_per_user_counts,
            "skill_gap_recommendations_csv": str(sg_csv_path.resolve()),
            "n_skill_gap_rows": n_sg_rows,
            "skill_gaps_per_user": skill_gap_per_user_counts,
        },
        "education_gate": education_gate,
        "caveats": [
            (
                f"Drives run_match_v3_full directly — same Gemini concat-cosine -> cross-encoder engine, county-scoped occupations, top-k and skill-gaps as the deployed POST {endpoint_label}. final_score is the concat cosine (whitened space when the artifact is present); the v4-only u_hat/p_hat/demand columns are empty for this engine. Only the job source (local JSON unless --live-jobs) and the absence of the FastAPI/Mongo wrapper differ."
                if args.version == "v3"
                else f"Drives MatchingService.rank directly — same engine, scoring, occupation demand-gamma + per-user location filter, top-k, and skill-gaps as the deployed POST {endpoint_label}. Only the job source (local JSON unless --live-jobs) and the absence of the FastAPI/Mongo wrapper differ."
                if IS_GEMINI
                else "Drives match_user_with_data directly — same legacy skill/Node2Vec engine, scoring, top-k and skill-gaps as the retired legacy POST /match. Only the job source (local JSON unless --live-jobs) and the absence of the FastAPI/Mongo wrapper differ."
            ),
            (
                "Job vectors come from each job's 3072-dim 'job_embedding' (stage-1 fallback); same space as gemini-embedding-001."
                if IS_GEMINI
                else "Legacy /match scores from skills (essential/optional/skill-groups + Node2Vec), not the Gemini 'job_embedding'; u_hat/p_hat/demand columns are empty for this engine."
            ),
            (
                "Live Mongo loader: jobs read from MONGO_JOBS_COLLECTION; JOBS_RETRIEVAL_FILTER "
                "controls the per-user location prefilter (off here unless --jobs-location-filter)."
                if USE_LIVE_JOBS
                else "Offline job loader returns the full active corpus and ignores per-user location prefilter (users have city='Unknown')."
            ),
        ]
        + (
            [
                "Cross-encoder downloads from HuggingFace on first run unless HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set."
            ]
            if IS_GEMINI
            else []
        )
        + (
            [
                "v5 ZQF annotation (zqf_eligible/zqf_gap/labels) is null unless users carry 'zqf_level' AND jobs carry 'zqf_min' (a Zambia/ZQF corpus). On the Kenya data these are null by design."
            ]
            if is_v5
            else []
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nWrote outputs to {out_dir}", file=sys.stderr)
    print(f"  {response_filename}  ({len(resp_dicts)} users)", file=sys.stderr)
    print(
        f"  recommendations.csv               ({n_rows} opportunity rows)",
        file=sys.stderr,
    )
    print(f"  recommendations_with_vectors.csv  ({n_vec_rows} rows)", file=sys.stderr)
    print(
        f"  career_recommendations.csv               ({n_career_rows} occupation rows)",
        file=sys.stderr,
    )
    print(
        f"  career_recommendations_with_vectors.csv  ({n_career_vec_rows} rows)",
        file=sys.stderr,
    )
    print(f"  skill_gap_recommendations.csv     ({n_sg_rows} rows)", file=sys.stderr)
    print("  manifest.json", file=sys.stderr)
    for uid in per_user_counts:
        print(
            f"    {uid}: {per_user_counts[uid]} opportunities, "
            f"{career_per_user_counts.get(uid, 0)} careers, "
            f"{skill_gap_per_user_counts.get(uid, 0)} skill-gaps",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
