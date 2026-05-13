"""Batch cosine skill matching: users × jobs → JSON (same envelope as BM25 runner).

Output shape mirrors ``bm25_scoring/bm25library``:

- ``config``, ``index_stats``, ``n_users``, ``n_jobs``, ``results[]``
- each result: ``user_id``, ``city``, ``province``, ``n_resolved_user_skills``,
  ``resolved_user_skill_labels``, ``recommendations[]``
- each recommendation: ``rank``, ``job_uuid``, ``job_title``, ``employer``,
  ``location``, ``mean_best_cosine``, ``min_best_cosine``, counts, ``per_job_skill``

Usage::

    cd backend
    python -m app.services.cosine_similarity.run_cosine_matching \\
        --users app/services/index_based_matching/njila_match_input.resolved.jsonl \\
        --from-mongo \\
        --output ./path/to/cosine_results.json \\
        --top-k 10

    python -m app.services.cosine_similarity.build_cosine_dashboard \\
        --input ./path/to/cosine_results.json \\
        --output ./path/to/cosine_dashboard.html
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from app.config import EMBEDDING_MODEL_PATH, SKILLS_CSV_PATH, SKILL_TO_ROW_PATH

from .skill_score import CosineSkillMatcher


def _backend_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "backend":
            return parent
    raise RuntimeError(f"Could not locate 'backend/' from {here}")


def _ensure_backend_on_syspath() -> None:
    root = str(_backend_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_backend_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # pylint: disable=import-outside-toplevel
    except ImportError:
        return
    p = _backend_root() / ".env"
    if p.is_file():
        load_dotenv(p, override=False)


def _is_jsonl(path: Path) -> bool:
    return path.suffix.lower() == ".jsonl"


def _load_users(path: Path) -> List[dict]:
    if _is_jsonl(path):
        users: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{ln}: invalid JSON ({e})") from e
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"{path}:{ln}: expected an object, got {type(obj).__name__}"
                    )
                users.append(obj)
        return users
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list, got {type(data).__name__}")
    return data


def _load_jobs_file(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list, got {type(data).__name__}")
    return data


async def _async_fetch_jobs_mongo(
    users_for_filter: Optional[List[dict]],
) -> Tuple[List[dict], Dict[str, Any]]:
    _load_backend_dotenv()
    _ensure_backend_on_syspath()
    from app.database import get_all_jobs_with_timing  # pylint: disable=import-outside-toplevel

    return await get_all_jobs_with_timing(users_for_filter)


def fetch_jobs_from_mongo(
    users_for_filter: Optional[List[dict]],
) -> Tuple[List[dict], Dict[str, Any]]:
    return asyncio.run(_async_fetch_jobs_mongo(users_for_filter))


def load_jobs(
    jobs_source: Literal["file", "mongo"],
    jobs_path: Optional[Path],
    users: List[dict],
    mongo_filter_by_users: bool,
) -> Tuple[List[dict], Optional[Dict[str, Any]]]:
    """Return ``jobs`` and optional ``mongo_timing``. Shared by cosine runners."""

    mongo_timing: Optional[Dict[str, Any]] = None

    if jobs_source == "file":
        if jobs_path is None:
            raise ValueError("jobs_path is required when jobs_source is 'file'")
        jobs = _load_jobs_file(jobs_path)
    else:
        users_arg = users if mongo_filter_by_users else None
        jobs, mongo_timing = fetch_jobs_from_mongo(users_arg)
        if not mongo_filter_by_users:
            print(
                "[cosine] mongo: loaded all active jobs (no per-user location filter)",
                file=sys.stderr,
            )
    return jobs, mongo_timing


def _trim_recommendations(
    recs: List[Dict[str, Any]], *, max_per_job_skills: int
) -> None:
    for r in recs:
        pj = r.get("per_job_skill") or []
        if len(pj) > max_per_job_skills:
            r["per_job_skill"] = pj[:max_per_job_skills]
            r["per_job_skill_truncated"] = True


def run(
    users_path: Path,
    jobs_source: Literal["file", "mongo"],
    jobs_path: Optional[Path],
    output_path: Optional[Path],
    *,
    top_k: int = 10,
    max_per_job_skills: int = 200,
    mongo_filter_by_users: bool = True,
) -> Dict[str, Any]:
    users = _load_users(users_path)

    jobs, mongo_timing = load_jobs(
        jobs_source, jobs_path, users, mongo_filter_by_users
    )

    matcher = CosineSkillMatcher()
    print(
        f"[cosine] loaded {len(users)} users, {len(jobs)} jobs "
        f"(embedding dim={matcher.W.shape[1]})",
        file=sys.stderr,
    )

    results: List[Dict[str, Any]] = []
    for user in users:
        labels = matcher.resolved_user_skill_labels_ordered(user)
        recs = matcher.rank_jobs(user, jobs, top_k=top_k)
        _trim_recommendations(recs, max_per_job_skills=max_per_job_skills)
        results.append({
            "user_id": user.get("user_id"),
            "city": user.get("city"),
            "province": user.get("province"),
            "n_resolved_user_skills": len(labels),
            "resolved_user_skill_labels": labels,
            "recommendations": recs,
        })

    config: Dict[str, Any] = {
        "users_path": str(users_path),
        "jobs_source": jobs_source,
        "top_k": top_k,
        "scorer": "cosine_embedding_skill_match",
        "max_per_job_skills_in_output": max_per_job_skills,
        "embedding_model_path": str(EMBEDDING_MODEL_PATH),
        "skill_to_row_path": str(SKILL_TO_ROW_PATH),
        "skills_csv_path": str(SKILLS_CSV_PATH),
    }
    if jobs_source == "file":
        config["jobs_path"] = str(jobs_path) if jobs_path else None
    else:
        config["mongo_filter_by_users"] = mongo_filter_by_users

    payload: Dict[str, Any] = {
        "config": config,
        "index_stats": {
            "n_jobs": len(jobs),
            "embedding_dim": int(matcher.W.shape[1]),
            "n_embedding_rows": int(matcher.W.shape[0]),
        },
        "n_users": len(users),
        "n_jobs": len(jobs),
        "results": results,
    }
    if mongo_timing is not None:
        payload["mongo_timing"] = mongo_timing

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[cosine] wrote {output_path}", file=sys.stderr)
    else:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    examples = """\
Examples:
  # Jobs from Mongo (same pipeline as bm25library — uses MONGO_* in backend/.env)
  %(prog)s --users app/services/index_based_matching/njila_match_input.resolved.jsonl \\
      --from-mongo --output app/services/cosine_similarity/cosine_match_mongo_results.json

  # Jobs from a local JSON list
  %(prog)s --users app/services/index_based_matching/users.json \\
      --jobs app/services/index_based_matching/jobs.json \\
      --output app/services/cosine_similarity/cosine_match_results.json
"""

    p = argparse.ArgumentParser(
        description=(
            "Cosine skill embedding match over users × jobs → JSON "
            "(envelope matches bm25_scoring bm25library)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    p.add_argument("--users", required=True, type=Path,
                   help="JSON or JSONL list of user records.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jobs", type=Path,
                     help="Local jobs list JSON. Mutually exclusive with --from-mongo.")
    src.add_argument("--from-mongo", action="store_true",
                     help="Load active jobs from MongoDB using backend/.env.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--max-per-job-skills", type=int, default=200,
        help="Cap per_job_skill rows per recommendation in the JSON file.",
    )
    p.add_argument(
        "--mongo-all-active", action="store_true",
        help=(
            "With --from-mongo: ignore JOBS_RETRIEVAL_FILTER and load every "
            "is_active=true job (see JOBS_RETRIEVAL_LIMIT)."
        ),
    )
    args = p.parse_args(argv)

    jobs_source: Literal["file", "mongo"] = "mongo" if args.from_mongo else "file"
    jobs_path = None if jobs_source == "mongo" else args.jobs

    if jobs_source == "file" and jobs_path is not None:
        jpath = jobs_path.expanduser()
        if not jpath.is_file():
            print(
                "[cosine] jobs file not found:\n"
                f"       {jobs_path}\n"
                "\n       To load jobs from MongoDB (same as bm25library), omit --jobs and pass:\n"
                "       --from-mongo\n\n"
                "       Set MONGO_URL, MONGO_DB_NAME, MONGO_JOBS_COLLECTION in backend/.env.",
                file=sys.stderr,
            )
            return 2

    run(
        users_path=args.users,
        jobs_source=jobs_source,
        jobs_path=jobs_path,
        output_path=args.output,
        top_k=args.top_k,
        max_per_job_skills=args.max_per_job_skills,
        mongo_filter_by_users=not args.mongo_all_active,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
