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

    python -m tools.cosine_similarity.build_cosine_dashboard \\
        --input ./path/to/cosine_results.json \\
        --output ./path/to/cosine_dashboard.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from app.config import EMBEDDING_MODEL_PATH, SKILLS_CSV_PATH, SKILL_TO_ROW_PATH
from app.jobs.repository import JobsLoadTiming
from app.ranking.types import SkillMatchDetail
from app.services.offline_io import JobsSource, load_jobs, load_users


class CosineRecommendation(BaseModel):
    """One job ranked by pooled per-skill cosine (``CosineSkillMatcher.score_pair``)."""

    rank: int = Field(description="1-based rank.")
    job_uuid: str = Field(description="Job id.")
    job_title: str = Field(description="Job title.")
    employer: str | None = Field(default=None, description="Employer.")
    location: str = Field(description="Location.")
    mean_best_cosine: float = Field(description="Mean best cosine over job skills.")
    min_best_cosine: float = Field(description="Minimum best cosine over job skills.")
    n_user_skills_embedded: int = Field(description="User skills with a vector.")
    n_job_skills_embedded: int = Field(description="Job skills with a vector.")
    per_job_skill: list[SkillMatchDetail] = Field(description="Per-skill matches.")
    per_job_skill_truncated: bool = Field(
        default=False, description="True when ``per_job_skill`` was capped."
    )


class CosineUserResult(BaseModel):
    """Cosine recommendations for one user."""

    user_id: str | None = Field(description="The user's id.")
    city: str = Field(description="The user's city.")
    province: str = Field(description="The user's province.")
    n_resolved_user_skills: int = Field(description="User skills with a vector.")
    resolved_user_skill_labels: list[str] = Field(description="Their labels.")
    recommendations: list[CosineRecommendation] = Field(description="Top-k jobs.")


class CosineRunConfig(BaseModel):
    """Configuration of one cosine run."""

    users_path: str = Field(description="Users file.")
    jobs_source: JobsSource = Field(description="``file`` or ``mongo``.")
    top_k: int = Field(description="Recommendations per user.")
    scorer: str = Field(description="Scorer name.")
    max_per_job_skills_in_output: int = Field(description="Per-skill row cap.")
    embedding_model_path: str = Field(description="Embedding artifact.")
    skill_to_row_path: str = Field(description="Skill -> row map.")
    skills_csv_path: str = Field(description="Taxonomy skills CSV.")
    jobs_path: str | None = Field(default=None, description="Jobs file (file source).")
    mongo_filter_by_users: bool | None = Field(
        default=None, description="Location prefilter (mongo source)."
    )


class CosineIndexStats(BaseModel):
    """Size of the scored corpus and of the embedding matrix."""

    n_jobs: int = Field(description="Jobs scored.")
    embedding_dim: int = Field(description="Embedding dimension.")
    n_embedding_rows: int = Field(description="Embedding rows.")


class CosineRunResult(BaseModel):
    """The JSON payload of one cosine run."""

    config: CosineRunConfig = Field(description="Run configuration.")
    index_stats: CosineIndexStats = Field(description="Corpus / embedding size.")
    n_users: int = Field(description="Users scored.")
    n_jobs: int = Field(description="Jobs scored.")
    results: list[CosineUserResult] = Field(description="One entry per user.")
    mongo_timing: JobsLoadTiming | None = Field(
        default=None, description="Job-load timing (mongo source)."
    )


def run(
    users_path: Path,
    jobs_source: JobsSource,
    jobs_path: Optional[Path],
    output_path: Optional[Path],
    *,
    top_k: int = 10,
    max_per_job_skills: int = 200,
    mongo_filter_by_users: bool = True,
) -> CosineRunResult:
    """Rank jobs for every user by pooled skill cosine and write the JSON payload."""
    users = load_users(users_path)
    jobs, mongo_timing = load_jobs(
        jobs_source, jobs_path, users, mongo_filter_by_users, log_prefix="[cosine]"
    )

    from app.server_dependencies.model_dependencies import build_skill_matcher

    matcher = build_skill_matcher()
    print(
        f"[cosine] loaded {len(users)} users, {len(jobs)} jobs "
        f"(embedding dim={matcher.W.shape[1]})",
        file=sys.stderr,
    )

    results: List[CosineUserResult] = []
    for user in users:
        labels = matcher.resolved_user_skill_labels_ordered(user)
        recs: List[CosineRecommendation] = []
        for rank, (job, score) in enumerate(
            matcher.rank_jobs(user, jobs, top_k=top_k), 1
        ):
            per = score.per_job_skill
            recs.append(
                CosineRecommendation(
                    rank=rank,
                    job_uuid=job.uuid,
                    job_title=job.title,
                    employer=getattr(job, "employer", None),
                    location=job.location,
                    mean_best_cosine=score.mean_best_cosine,
                    min_best_cosine=score.min_best_cosine,
                    n_user_skills_embedded=score.n_user_skills_embedded,
                    n_job_skills_embedded=score.n_job_skills_embedded,
                    per_job_skill=per[:max_per_job_skills],
                    per_job_skill_truncated=len(per) > max_per_job_skills,
                )
            )
        results.append(
            CosineUserResult(
                user_id=user.user_id,
                city=user.city,
                province=user.province,
                n_resolved_user_skills=len(labels),
                resolved_user_skill_labels=labels,
                recommendations=recs,
            )
        )

    payload = CosineRunResult(
        config=CosineRunConfig(
            users_path=str(users_path),
            jobs_source=jobs_source,
            top_k=top_k,
            scorer="cosine_embedding_skill_match",
            max_per_job_skills_in_output=max_per_job_skills,
            embedding_model_path=str(EMBEDDING_MODEL_PATH),
            skill_to_row_path=str(SKILL_TO_ROW_PATH),
            skills_csv_path=str(SKILLS_CSV_PATH),
            jobs_path=(str(jobs_path) if jobs_path else None)
            if jobs_source == "file"
            else None,
            mongo_filter_by_users=mongo_filter_by_users
            if jobs_source == "mongo"
            else None,
        ),
        index_stats=CosineIndexStats(
            n_jobs=len(jobs),
            embedding_dim=int(matcher.W.shape[1]),
            n_embedding_rows=int(matcher.W.shape[0]),
        ),
        n_users=len(users),
        n_jobs=len(jobs),
        results=results,
        mongo_timing=mongo_timing,
    )

    text = payload.model_dump_json(indent=2, exclude_none=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[cosine] wrote {output_path}", file=sys.stderr)
    else:
        sys.stdout.write(text + "\n")

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
    p.add_argument(
        "--users", required=True, type=Path, help="JSON or JSONL list of user records."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--jobs",
        type=Path,
        help="Local jobs list JSON. Mutually exclusive with --from-mongo.",
    )
    src.add_argument(
        "--from-mongo",
        action="store_true",
        help="Load active jobs from MongoDB using backend/.env.",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--max-per-job-skills",
        type=int,
        default=200,
        help="Cap per_job_skill rows per recommendation in the JSON file.",
    )
    p.add_argument(
        "--mongo-all-active",
        action="store_true",
        help=(
            "With --from-mongo: ignore JOBS_RETRIEVAL_FILTER and load every "
            "is_active=true job (see JOBS_RETRIEVAL_LIMIT)."
        ),
    )
    args = p.parse_args(argv)

    jobs_source: JobsSource = "mongo" if args.from_mongo else "file"
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
