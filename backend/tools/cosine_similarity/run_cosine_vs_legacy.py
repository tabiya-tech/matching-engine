"""Compare pooled cosine skill match vs legacy ``compute_U_complete`` (full U stack).

Columns:

1. **Pooled cosine** — :meth:`CosineSkillMatcher.score_pair` /
   ``run_cosine_matching`` semantics: essential ∪ optional, row-wise max cosine per
   job skill, mean (no rescaling hook in cosine module; embeddings as loaded).

2. **Legacy utility** — :class:`app.ranking.skills.SkillScorer` /
   ``compute_U_complete``: location, groups, weighted ess/opt/grp blend, rescaling when
   the scorer hydrates targets from artefact metadata, gap penalty, etc.

Produces JSON; pair with ``build_cosine_legacy_compare_dashboard`` for HTML.

Usage::

    cd backend
    python -m tools.cosine_similarity.run_cosine_vs_legacy \\
        --users …/njila_match_input.resolved.jsonl \\
        --from-mongo \\
        --output …/cosine_vs_legacy_results.json \\
        --top-compare 15

    python -m app.services.cosine_similarity.build_cosine_legacy_compare_dashboard \\
        --input …/cosine_vs_legacy_results.json \\
        --output …/cosine_vs_legacy_dashboard.html
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from app.config import EMBEDDING_MODEL_PATH, SKILLS_CSV_PATH, SKILL_TO_ROW_PATH
from app.domain.user import UserProfile
from app.jobs.repository import JobsLoadTiming
from app.ranking.skills import CosineSkillMatcher, SkillScorer, SkillUtilityComponents
from app.ranking.types import MatchItem
from app.server_dependencies.model_dependencies import (
    build_skill_matcher,
    build_skill_scorer,
)
from app.services.cosine_similarity.run_cosine_matching import CosineIndexStats
from app.services.hybrid_scoring.run_bm25_cosine_hybrid import ResolutionStats
from app.services.offline_io import JobsSource, load_jobs, load_users


class ComparisonRow(BaseModel):
    """One job scored by pooled cosine and by the legacy skill utility U."""

    job_uuid: str = Field(description="Job id.")
    job_title: str = Field(description="Job title.")
    employer: str | None = Field(default=None, description="Employer.")
    location: str = Field(description="Location.")
    mean_best_cosine_union_pool: float = Field(description="Pooled mean best cosine.")
    min_best_cosine_union_pool: float = Field(description="Pooled min best cosine.")
    n_job_skills_embedded_union_pool: int = Field(
        description="Job skills with a vector."
    )
    n_user_skills_embedded: int = Field(description="User skills with a vector.")
    legacy_U_final: float = Field(description="Legacy U.")
    legacy_is_eligible: bool = Field(description="Legacy eligibility.")
    legacy_penalty: float = Field(description="Legacy gap penalty.")
    legacy_components: SkillUtilityComponents = Field(
        description="Legacy U components."
    )
    rank_pooled_cosine: int | None = Field(default=None, description="Rank by cosine.")
    rank_legacy_U: int | None = Field(default=None, description="Rank by legacy U.")
    cross_rank_pooled_cosine: int | None = Field(
        default=None, description="Cosine rank (legacy column)."
    )
    delta_rank_cosine_minus_legacy: int | None = Field(
        default=None, description="Cosine rank − legacy rank."
    )


class ComparisonUserResult(BaseModel):
    """Both rankings for one user."""

    user_id: str | None = Field(description="The user's id.")
    city: str = Field(description="The user's city.")
    province: str = Field(description="The user's province.")
    n_resolved_user_skills: int = Field(description="User skills with a vector.")
    resolved_user_skill_labels: list[str] = Field(description="Their labels.")
    overlap_jaccard_top_cosine_vs_legacy: float = Field(
        description="Jaccard overlap of the two top-k lists."
    )
    top_pooled_cosine: list[ComparisonRow] = Field(description="Top-k by cosine.")
    top_legacy_U: list[ComparisonRow] = Field(description="Top-k by legacy U.")


class ComparisonConfig(BaseModel):
    """Configuration of one comparison run."""

    users_path: str = Field(description="Users file.")
    jobs_source: JobsSource = Field(description="``file`` or ``mongo``.")
    top_compare: int = Field(description="Rows per column.")
    mode: str = Field(description="Comparison mode.")
    embedding_model_path: str = Field(description="Embedding artifact.")
    skill_to_row_path: str = Field(description="Skill -> row map.")
    skills_csv_path: str = Field(description="Taxonomy skills CSV.")
    comparison_note: str = Field(description="What each column ranks by.")
    jobs_path: str | None = Field(default=None, description="Jobs file (file source).")
    mongo_filter_by_users: bool | None = Field(
        default=None, description="Location prefilter (mongo source)."
    )


class ComparisonResult(BaseModel):
    """The JSON payload of one comparison run."""

    config: ComparisonConfig = Field(description="Run configuration.")
    index_stats: CosineIndexStats = Field(description="Corpus / embedding size.")
    n_users: int = Field(description="Users compared.")
    n_jobs: int = Field(description="Jobs scored.")
    results: list[ComparisonUserResult] = Field(description="One entry per user.")
    mongo_timing: JobsLoadTiming | None = Field(
        default=None, description="Job-load timing (mongo source)."
    )
    skill_scorer_resolution_stats: ResolutionStats = Field(
        description="SkillScorer label misses."
    )
    cosine_resolution_stats: ResolutionStats = Field(
        description="CosineSkillMatcher label misses."
    )


def _jacc_top_k(uuids_a: List[str], uuids_b: List[str]) -> float:
    aa, bb = set(uuids_a), set(uuids_b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / float(len(aa | bb))


def _one_row_skeleton(
    matcher: CosineSkillMatcher,
    scorer: SkillScorer,
    user: UserProfile,
    job: MatchItem,
) -> ComparisonRow:
    uni = matcher.score_pair(user, job)
    legacy = scorer.calculate_score(user, job)
    return ComparisonRow(
        job_uuid=job.uuid,
        job_title=job.title,
        employer=getattr(job, "employer", None),
        location=job.location,
        mean_best_cosine_union_pool=uni.mean_best_cosine,
        min_best_cosine_union_pool=uni.min_best_cosine,
        n_job_skills_embedded_union_pool=uni.n_job_skills_embedded,
        n_user_skills_embedded=uni.n_user_skills_embedded,
        legacy_U_final=legacy.U_final,
        legacy_is_eligible=legacy.is_eligible,
        legacy_penalty=legacy.penalty,
        legacy_components=legacy.components,
    )


def _ranked(rows: Sequence[ComparisonRow], *, by_cosine: bool) -> List[ComparisonRow]:
    """Rows sorted by pooled cosine (then min cosine), or by legacy U (then its ``ess``)."""
    if by_cosine:
        return sorted(
            rows,
            key=lambda r: (
                -r.mean_best_cosine_union_pool,
                -r.min_best_cosine_union_pool,
            ),
        )
    return sorted(rows, key=lambda r: (-r.legacy_U_final, -r.legacy_components.ess))


def run_comparison(
    users_path: Path,
    jobs_source: JobsSource,
    jobs_path: Optional[Path],
    output_path: Optional[Path],
    *,
    top_compare: int = 15,
    mongo_filter_by_users: bool = True,
) -> ComparisonResult:
    """Rank every job per user by pooled cosine and by legacy U, and write both top-k lists."""
    users = load_users(users_path)
    jobs, mongo_timing = load_jobs(
        jobs_source,
        jobs_path,
        users,
        mongo_filter_by_users,
        log_prefix="[cosine_vs_legacy]",
    )

    matcher = build_skill_matcher()
    scorer = build_skill_scorer()

    print(
        f"[cosine_vs_legacy] {len(users)} users × {len(jobs)} jobs "
        f"(dim={matcher.W.shape[1]})",
        file=sys.stderr,
    )

    results_out: List[ComparisonUserResult] = []

    for user in users:
        labels = matcher.resolved_user_skill_labels_ordered(user)
        skinny = [_one_row_skeleton(matcher, scorer, user, job) for job in jobs]

        cos_ranked = _ranked(skinny, by_cosine=True)
        leg_ranked = _ranked(skinny, by_cosine=False)
        cos_rank = {r.job_uuid: i for i, r in enumerate(cos_ranked, 1)}
        leg_rank = {r.job_uuid: i for i, r in enumerate(leg_ranked, 1)}

        top_cosine = [
            r.model_copy(
                update={
                    "rank_pooled_cosine": i,
                    "rank_legacy_U": leg_rank.get(r.job_uuid, 999999),
                    "delta_rank_cosine_minus_legacy": i
                    - leg_rank.get(r.job_uuid, 999999),
                }
            )
            for i, r in enumerate(cos_ranked, 1)
        ][: max(0, top_compare)]
        top_legacy = [
            r.model_copy(
                update={
                    "rank_legacy_U": i,
                    "cross_rank_pooled_cosine": cos_rank.get(r.job_uuid, 999999),
                    "delta_rank_cosine_minus_legacy": cos_rank.get(r.job_uuid, 999999)
                    - i,
                }
            )
            for i, r in enumerate(leg_ranked, 1)
        ][: max(0, top_compare)]

        results_out.append(
            ComparisonUserResult(
                user_id=user.user_id,
                city=user.city,
                province=user.province,
                n_resolved_user_skills=len(labels),
                resolved_user_skill_labels=labels,
                overlap_jaccard_top_cosine_vs_legacy=round(
                    _jacc_top_k(
                        [r.job_uuid for r in top_cosine],
                        [r.job_uuid for r in top_legacy],
                    ),
                    4,
                ),
                top_pooled_cosine=top_cosine,
                top_legacy_U=top_legacy,
            )
        )

    payload = ComparisonResult(
        config=ComparisonConfig(
            users_path=str(users_path),
            jobs_source=jobs_source,
            top_compare=top_compare,
            mode="cosine_pooled_union_vs_legacy_skills_match_U",
            embedding_model_path=str(EMBEDDING_MODEL_PATH),
            skill_to_row_path=str(SKILL_TO_ROW_PATH),
            skills_csv_path=str(SKILLS_CSV_PATH),
            comparison_note=(
                "Left column ranks by pooled cosine (CosineSkillMatcher.score_pair: "
                "essential ∪ optional, mean of row-wise max cosines vs user skills). "
                "Right column ranks by SkillScorer.calculate_score → compute_U_complete "
                "(location + skill groups + weighted ess/opt/grp + gap penalty; embedding "
                "rescaling applies when SKILL_RESCALE_TARGET is set from artefact/env). "
                "Both scorers read the user's skills_vector.top_skills labels."
            ),
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
        results=results_out,
        mongo_timing=mongo_timing,
        skill_scorer_resolution_stats=ResolutionStats.model_validate(
            scorer.get_resolution_stats()
        ),
        cosine_resolution_stats=ResolutionStats.model_validate(
            matcher.get_resolution_stats()
        ),
    )

    text = payload.model_dump_json(indent=2, exclude_none=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[cosine_vs_legacy] wrote {output_path}", file=sys.stderr)
    else:
        sys.stdout.write(text + "\n")

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    examples = """\
Examples:
  %(prog)s --users …/njila_match_input.resolved.jsonl \\
      --from-mongo --output …/cosine_vs_legacy_results.json --top-compare 15
"""

    p = argparse.ArgumentParser(
        description=(
            "Compare pooled cosine embeddings vs legacy skills_match U_complete rankings."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )
    p.add_argument("--users", required=True, type=Path)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jobs", type=Path)
    src.add_argument("--from-mongo", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--top-compare",
        type=int,
        default=15,
        help="Jobs per column in JSON (still scores all jobs globally).",
    )
    p.add_argument("--mongo-all-active", action="store_true")

    args = p.parse_args(argv)

    js: JobsSource = "mongo" if args.from_mongo else "file"
    jp = None if js == "mongo" else args.jobs

    if js == "file" and jp is not None and not jp.expanduser().is_file():
        print(
            "[cosine_vs_legacy] jobs file not found — use --from-mongo for Mongo.",
            file=sys.stderr,
        )
        return 2

    run_comparison(
        users_path=args.users,
        jobs_source=js,
        jobs_path=jp,
        output_path=args.output,
        top_compare=args.top_compare,
        mongo_filter_by_users=not args.mongo_all_active,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
