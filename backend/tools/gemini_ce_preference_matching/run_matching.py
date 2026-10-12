"""match_v3 concat cosine + CE → u_hat × p_hat (p_hat = raw cosine, not CE 0–1).

Stage 1–2: same as ``POST /match_v3`` (:meth:`~app.matching.service.MatchingService.shortlist_and_rerank`):
Gemini user concat × Mongo job vectors → **concat_cosine_similarity**, then CE rerank.

Stage 3: the configured preference scorer (``get_preference_scorer``) → ``u_hat``;
``p_hat`` = ``concat_cosine_similarity``; ``final = u_hat × p_hat``.

Usage (from ``backend/``)::

    python -m tools.gemini_ce_preference_matching.run_matching \\
        --users data/njila/njila_match_input.jsonl \\
        --from-mongo \\
        --retrieve-top-k 50 \\
        --final-top-k 10 \\
        --output output/results_gemini_ce_preference.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from app.artifacts.repository import get_artifacts_repository
from app.clients.gemini_embedding_client import (
    EMBEDDING_DIM,
    MODEL_NAME as GEMINI_EMBEDDING_MODEL_NAME,
)
from app.config import (
    COSINE_CROSS_ENCODER_RETRIEVE_TOP_K,
    CROSS_ENCODER_BATCH_SIZE,
    CROSS_ENCODER_MODEL_NAME,
    FINAL_SCORE_COMBINER,
    PREFERENCE_SCORER_MODE,
)
from app.domain.user import UserProfile
from app.jobs.get_jobs_repository import standalone_jobs_repository
from app.jobs.repository import JobsLoadTiming
from app.matching.get_matching_service import create_matching_service
from app.ranking.preference import attribute_label, detect_bws_score_type
from app.ranking.retrieval import (
    job_skill_labels_for_concat,
    user_skill_labels_for_concat,
)
from app.ranking.scoring import (
    WorkActivityDashboard,
    enrich_recommendations_with_preferences,
    preference_details_for_dashboard,
    work_activity_match_for_dashboard,
)
from app.ranking.types import (
    AttributePreference,
    Candidate,
    FinalScoreBreakdown,
    ScoredCandidate,
    UserShortlist,
)
from app.server_dependencies.model_dependencies import get_preference_scorer
from app.services.offline_io import JobsSource, load_jobs, load_users


class CeRow(BaseModel):
    """One job after the concat-cosine shortlist and the cross-encoder rerank."""

    rank: int = Field(
        description="Rank (CE order, or final-score order for scored rows)."
    )
    job_uuid: str = Field(description="Job id.")
    job_title: str = Field(description="Job title.")
    employer: str | None = Field(default=None, description="Employer.")
    location: str = Field(description="Location.")
    concat_cosine_similarity: float = Field(description="Stage-1 concat cosine.")
    cross_encoder_logit: float | None = Field(default=None, description="CE logit.")
    cross_encoder_score: float | None = Field(
        default=None, description="CE score in [0, 1]."
    )
    rank_cosine: int | None = Field(default=None, description="Stage-1 rank.")
    job_concat_skills: list[str] = Field(description="The job's concat skill labels.")

    @classmethod
    def from_candidate(cls, c: Candidate) -> CeRow:
        """The CE row of a reranked candidate."""
        return cls(**_ce_fields(c))


class PreferenceRow(CeRow):
    """A CE row re-ranked by ``u_hat × p_hat``."""

    rank_cross_encoder: int = Field(description="Rank after the cross-encoder.")
    u_hat: float = Field(description="Preference utility.")
    p_hat: float = Field(description="Success propensity (concat cosine).")
    p_hat_source: str = Field(description="Signal p_hat came from.")
    final_score: float = Field(description="Final score.")
    score_breakdown: FinalScoreBreakdown = Field(
        description="Factors behind the score."
    )
    pref_match: list[AttributePreference] = Field(
        description="Schema-scored attribute rows, largest |contribution| first."
    )
    wa_match: WorkActivityDashboard = Field(description="BWS work-activity block.")
    S_attrs: float | None = Field(
        default=None, description="Harmonised attribute utility."
    )
    S_wa: float | None = Field(default=None, description="Harmonised task utility.")
    preference_include_work_activities: bool = Field(
        description="Whether work activities were scored."
    )

    @classmethod
    def from_scored(
        cls, row: ScoredCandidate, *, include_work_activities: bool
    ) -> PreferenceRow:
        """The dashboard row of a preference-scored candidate."""
        c = row.candidate
        return cls(
            **{**_ce_fields(c), "rank": row.rank},
            rank_cross_encoder=c.rank,
            u_hat=row.preference.u_hat,
            p_hat=row.p_hat,
            p_hat_source=row.p_hat_source,
            final_score=row.final_score,
            score_breakdown=row.breakdown,
            pref_match=preference_details_for_dashboard(row.preference),
            wa_match=work_activity_match_for_dashboard(row.preference),
            S_attrs=row.preference.S_attrs,
            S_wa=row.preference.S_wa,
            preference_include_work_activities=include_work_activities,
        )


def _ce_fields(c: Candidate) -> dict[str, object]:
    item = c.item
    return {
        "rank": c.rank,
        "job_uuid": item.uuid,
        "job_title": item.title,
        "employer": getattr(item, "employer", None),
        "location": item.location,
        "concat_cosine_similarity": c.concat_cosine_similarity,
        "cross_encoder_logit": c.cross_encoder_logit,
        "cross_encoder_score": c.cross_encoder_score,
        "rank_cosine": c.rank_cosine,
        "job_concat_skills": job_skill_labels_for_concat(item),
    }


class PreferenceFactor(BaseModel):
    """One of the user's attribute preference weights."""

    attribute: str = Field(description="Attribute name.")
    label: str = Field(description="Attribute label.")
    importance: float = Field(description="Preference value clamped to [0, 1].")


class BwsRow(BaseModel):
    """One of the user's BWS work-activity scores."""

    wa_code: str = Field(description="Work-activity code.")
    bws: float = Field(description="BWS score.")


class UserBwsSummary(BaseModel):
    """The user's top BWS scores, for the dashboard header."""

    has_bws: bool = Field(description="BWS keys are work-activity codes.")
    score_type: str | None = Field(description="BWS key type; None without BWS.")
    rows: list[BwsRow] = Field(description="Up to 12 activities.")


class UserPreferenceRun(BaseModel):
    """All columns for one user."""

    user_id: str | None = Field(description="The user's id.")
    city: str = Field(description="The user's city.")
    province: str = Field(description="The user's province.")
    n_user_concat_skills: int = Field(description="User concat skill labels.")
    user_concat_skills: list[str] = Field(description="Their labels.")
    has_user_skills: bool = Field(description="The user has skill labels.")
    user_preference_factors: list[PreferenceFactor] = Field(
        description="Preference weights."
    )
    user_bws: UserBwsSummary = Field(description="BWS summary.")
    n_jobs_scored: int = Field(description="Jobs with a stage-1 vector.")
    cross_encoder_recommendations: list[CeRow] = Field(
        description="Column 1: CE order."
    )
    recommendations_attrs_only: list[PreferenceRow] = Field(
        description="Column 2: attributes only."
    )
    recommendations: list[PreferenceRow] = Field(
        description="Column 3: attributes + work activities."
    )
    recommendations_alt: list[PreferenceRow] | None = Field(
        default=None, description="Column 3 with the other combiner."
    )


class PreferenceRunConfig(BaseModel):
    """Configuration of one preference run."""

    users_path: str = Field(description="Users file.")
    jobs_source: JobsSource = Field(description="``file`` or ``mongo``.")
    retrieve_top_k: int = Field(description="Stage-1 / CE pool size.")
    final_top_k: int = Field(description="Rows per column.")
    stage1_scorer: str = Field(description="Stage-1 method.")
    stage2_scorer: str = Field(description="Stage-2 method.")
    stage3_scorer: str = Field(description="Stage-3 method.")
    stage3_variants: list[str] = Field(description="Stage-3 columns.")
    scoring_mode: str = Field(description="Scoring mode.")
    final_score_combiner: str = Field(description="Combiner.")
    final_formula: str = Field(description="Final-score formula.")
    p_hat_source: str = Field(description="Signal p_hat comes from.")
    preference_scorer_mode: str = Field(description="Scorer mode.")
    preference_module: str = Field(description="Scorer class path.")
    match_v3_service: str = Field(description="Stage-1/2 service.")
    gemini_user_embed_model: str = Field(description="User embedding model.")
    cross_encoder_model: str = Field(description="Cross-encoder checkpoint.")
    cross_encoder_batch_size: int = Field(description="CE batch size.")
    embedding_dim: int = Field(description="Embedding dimension.")
    n_jobs_with_stage1_embedding: int | None = Field(description="Jobs with a vector.")
    max_per_job_skills_in_output: int = Field(description="Per-skill row cap (unused).")
    include_per_job_skill_detail: bool = Field(description="Always False.")
    skip_users_without_skills: bool = Field(description="Always False.")
    jobs_limit: int | None = Field(description="Job cap.")
    n_jobs_loaded_before_limit: int = Field(description="Jobs before the cap.")
    compare_final_combiners: bool = Field(description="Alternate column computed.")
    jobs_path: str | None = Field(default=None, description="Jobs file (file source).")
    mongo_filter_by_users: bool | None = Field(
        default=None, description="Location prefilter (mongo source)."
    )


class PreferenceIndexStats(BaseModel):
    """Corpus size."""

    n_jobs: int = Field(description="Jobs.")
    n_jobs_with_concat_embedding: int | None = Field(description="Jobs with a vector.")
    embedding_dim: int = Field(description="Embedding dimension.")


class PreferenceRunResult(BaseModel):
    """The JSON payload of one preference run (input of the dashboards)."""

    config: PreferenceRunConfig = Field(description="Run configuration.")
    index_stats: PreferenceIndexStats = Field(description="Corpus size.")
    n_users: int = Field(description="Users.")
    n_jobs: int = Field(description="Jobs.")
    results: list[UserPreferenceRun] = Field(description="One entry per user.")
    mongo_timing: JobsLoadTiming | None = Field(
        default=None, description="Job-load timing (mongo source)."
    )


def user_preference_factors(user: UserProfile) -> List[PreferenceFactor]:
    """User importance weights for dashboard (sorted high → low)."""
    schema = get_artifacts_repository().load_attribute_schema()
    rows: List[PreferenceFactor] = []
    for spec in schema.attributes:
        if not spec.name:
            continue
        w = getattr(user.preference_vector, spec.name, None)
        w = w if isinstance(w, float) else 0.0
        rows.append(
            PreferenceFactor(
                attribute=spec.name,
                label=attribute_label(spec.name, schema),
                importance=round(max(0.0, min(1.0, w)), 4),
            )
        )
    rows.sort(key=lambda r: (-r.importance, r.label))
    return rows


def user_bws_summary(user: UserProfile) -> UserBwsSummary:
    """Top BWS scores on work-activity codes for dashboard header."""
    bws_scores = user.preference_vector.bws_scores or {}
    if not bws_scores:
        return UserBwsSummary(has_bws=False, score_type=None, rows=[])

    score_type = detect_bws_score_type(bws_scores)
    top_codes = list(user.preference_vector.top_10_bws or [])

    rows: List[BwsRow] = []
    seen: set[str] = set()

    def _add(code: str) -> None:
        c = code.strip()
        if not c or c in seen:
            return
        seen.add(c)
        rows.append(BwsRow(wa_code=c, bws=round(bws_scores.get(c, 0.0), 2)))

    for code in top_codes[:10]:
        _add(code)
    for code, _val in sorted(bws_scores.items(), key=lambda kv: (-abs(kv[1]), kv[0])):
        if len(rows) >= 12:
            break
        _add(code)

    return UserBwsSummary(
        has_bws=score_type == "work_activity_id", score_type=score_type, rows=rows
    )


def run_pipeline(
    users_path: Path,
    jobs_source: JobsSource,
    jobs_path: Optional[Path],
    *,
    retrieve_top_k: int,
    final_top_k: int,
    output_path: Optional[Path],
    max_per_job_skills: int,
    mongo_filter_by_users: bool,
    cross_encoder_model: Optional[str],
    cross_encoder_batch_size: Optional[int],
    jobs_limit: Optional[int] = None,
    compare_final_combiners: bool = False,
) -> PreferenceRunResult:
    """Shortlist + rerank every user's jobs, then re-rank by ``u_hat × p_hat`` three ways."""
    retrieve_top_k = max(1, int(retrieve_top_k))
    final_top_k = max(1, int(final_top_k))

    users = load_users(users_path)
    jobs, mongo_timing = load_jobs(
        jobs_source,
        jobs_path,
        users,
        mongo_filter_by_users,
        log_prefix="[gemini_ce_pref]",
    )
    n_jobs_loaded = len(jobs)
    if jobs_limit is not None and int(jobs_limit) > 0:
        jobs = jobs[: int(jobs_limit)]
        print(
            f"[gemini_ce_pref] jobs capped to {len(jobs)} (from {n_jobs_loaded})",
            file=sys.stderr,
        )

    pref_scorer = get_preference_scorer()
    pref_cls = f"{pref_scorer.__class__.__module__}.{pref_scorer.__class__.__name__}"

    print(
        f"[gemini_ce_pref] users={len(users)} jobs={len(jobs)} "
        f"stage1=match_v3_concat_cosine retrieve_top_k={retrieve_top_k} "
        f"final_top_k={final_top_k} preferences={PREFERENCE_SCORER_MODE} ({pref_cls}) "
        f"scoring={FINAL_SCORE_COMBINER} p_hat=concat_cosine",
        file=sys.stderr,
    )

    async def _shortlist_and_rerank() -> List[UserShortlist]:
        async with standalone_jobs_repository() as jobs_repository:
            return create_matching_service(jobs_repository).shortlist_and_rerank(
                users,
                jobs,
                retrieve_top_k=retrieve_top_k,
                final_top_k=retrieve_top_k,
                mongo_timing=mongo_timing,
            )

    slates = asyncio.run(_shortlist_and_rerank())

    def _scored(
        user: UserProfile,
        pool: Sequence[Candidate],
        *,
        include_work_activities: bool,
        combiner: str | None = None,
    ) -> List[PreferenceRow]:
        rows = enrich_recommendations_with_preferences(
            user,
            pool,
            preference_scorer=pref_scorer,
            include_work_activities=include_work_activities,
            final_score_combiner=combiner,
        )[:final_top_k]
        return [
            PreferenceRow.from_scored(
                r, include_work_activities=include_work_activities
            )
            for r in rows
        ]

    results: List[UserPreferenceRun] = []
    for user, slate in zip(users, slates):
        user_concat_skills = user_skill_labels_for_concat(user)
        ce_pool = slate.candidates

        alt: Optional[List[PreferenceRow]] = None
        if compare_final_combiners:
            # Alternate ranking for dashboards (no re-embedding).
            other = "geometric_mean" if FINAL_SCORE_COMBINER == "product" else "product"
            alt = _scored(user, ce_pool, include_work_activities=True, combiner=other)

        results.append(
            UserPreferenceRun(
                user_id=user.user_id,
                city=user.city,
                province=user.province,
                n_user_concat_skills=len(user_concat_skills),
                user_concat_skills=user_concat_skills,
                has_user_skills=bool(user_concat_skills),
                user_preference_factors=user_preference_factors(user),
                user_bws=user_bws_summary(user),
                n_jobs_scored=slate.n_jobs_scored,
                # Column 1: CE order, p_hat = concat cosine only (no u_hat yet).
                cross_encoder_recommendations=[
                    CeRow.from_candidate(c) for c in ce_pool[:final_top_k]
                ],
                # Column 2: hybrid Part A only (no BWS / work activities).
                recommendations_attrs_only=_scored(
                    user, ce_pool, include_work_activities=False
                ),
                # Column 3: hybrid Part A + Part B (BWS × O*NET work activities).
                recommendations=_scored(
                    user,
                    ce_pool,
                    include_work_activities=True,
                    combiner=FINAL_SCORE_COMBINER,
                ),
                recommendations_alt=alt,
            )
        )

    summary0 = slates[0].config_summary if slates else None
    payload = PreferenceRunResult(
        config=PreferenceRunConfig(
            users_path=str(users_path),
            jobs_source=jobs_source,
            retrieve_top_k=retrieve_top_k,
            final_top_k=final_top_k,
            stage1_scorer="match_v3_concat_gemini_cosine_mongo_job_vectors",
            stage2_scorer="cross_encoder_rerank",
            stage3_scorer="combine_u_hat_and_p_hat",
            stage3_variants=["attrs_only", "attrs_plus_work_activities"],
            scoring_mode="multiplicative",
            final_score_combiner=FINAL_SCORE_COMBINER,
            final_formula=(
                "u_hat * p_hat"
                if FINAL_SCORE_COMBINER == "product"
                else "sqrt(u_hat * p_hat)"
            ),
            p_hat_source="concat_cosine_similarity",
            preference_scorer_mode=PREFERENCE_SCORER_MODE,
            preference_module=pref_cls,
            match_v3_service="app.matching.service",
            gemini_user_embed_model=(
                summary0.gemini_user_embed_model if summary0 else None
            )
            or GEMINI_EMBEDDING_MODEL_NAME,
            cross_encoder_model=(summary0.cross_encoder_model if summary0 else None)
            or cross_encoder_model
            or CROSS_ENCODER_MODEL_NAME,
            cross_encoder_batch_size=int(
                cross_encoder_batch_size or CROSS_ENCODER_BATCH_SIZE
            ),
            embedding_dim=(summary0.embedding_dim if summary0 else None)
            or EMBEDDING_DIM,
            n_jobs_with_stage1_embedding=summary0.n_jobs_with_stage1_embedding
            if summary0
            else None,
            max_per_job_skills_in_output=max_per_job_skills,
            # MatchingService.shortlist_and_rerank does not emit per-job skill detail nor skip
            # skill-less users; labels reflect actual service behaviour.
            include_per_job_skill_detail=False,
            skip_users_without_skills=False,
            jobs_limit=jobs_limit,
            n_jobs_loaded_before_limit=n_jobs_loaded,
            compare_final_combiners=compare_final_combiners,
            jobs_path=(str(jobs_path) if jobs_path else None)
            if jobs_source == "file"
            else None,
            mongo_filter_by_users=mongo_filter_by_users
            if jobs_source == "mongo"
            else None,
        ),
        index_stats=PreferenceIndexStats(
            n_jobs=len(jobs),
            n_jobs_with_concat_embedding=summary0.n_jobs_with_stage1_embedding
            if summary0
            else None,
            embedding_dim=EMBEDDING_DIM,
        ),
        n_users=len(users),
        n_jobs=len(jobs),
        results=results,
        mongo_timing=mongo_timing,
    )

    text = payload.model_dump_json(indent=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[gemini_ce_pref] wrote → {output_path}", file=sys.stderr)
    else:
        sys.stdout.write(text + "\n")

    return payload


def main(argv: Optional[List[str]] = None) -> int:
    epilog = """\
Examples:
  %(prog)s --users data/njila/njila_match_input.jsonl --from-mongo \\
      --retrieve-top-k 50 --final-top-k 10 \\
      --output output/results_gemini_ce_preference.json

Requires GEMINI_API_KEY and jobs with job_embedding or concat_skill_embedding_gemini in Mongo.
"""
    p = argparse.ArgumentParser(
        description=(
            "match_v3 (concat Gemini cosine + CE) → u_hat × p_hat "
            "(p_hat = concat_cosine_similarity [whitened space], not CE min–max score)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "--users", required=True, type=Path, help="JSON or JSONL user records."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jobs", type=Path, help="Local jobs JSON list.")
    src.add_argument(
        "--from-mongo", action="store_true", help="Load jobs from Mongo (.env)."
    )

    p.add_argument(
        "--retrieve-top-k",
        type=int,
        default=COSINE_CROSS_ENCODER_RETRIEVE_TOP_K,
        help="match_v3 stage-1 shortlist / CE pool size per user.",
    )
    p.add_argument(
        "--final-top-k",
        type=int,
        default=10,
        help="Rows per user after u_hat × p_hat re-rank.",
    )
    p.add_argument(
        "--output", type=Path, default=None, help="Write JSON; omit → stdout."
    )
    p.add_argument(
        "--max-per-job-skills",
        type=int,
        default=200,
        help="Cap per_job_skill rows per recommendation.",
    )
    p.add_argument(
        "--mongo-all-active",
        action="store_true",
        help="With --from-mongo: load all active jobs (no per-user location filter).",
    )
    p.add_argument("--cross-encoder-model", type=str, default=None)
    p.add_argument("--cross-encoder-batch-size", type=int, default=None)
    p.add_argument(
        "--jobs-limit",
        type=int,
        default=None,
        help="Cap active jobs loaded from Mongo/file (e.g. 100 for testing).",
    )
    p.add_argument(
        "--compare-final-combiners",
        action="store_true",
        help="Also compute an alternate ranking using the other combiner (product vs geometric_mean).",
    )

    args = p.parse_args(argv)
    jobs_source: JobsSource = "mongo" if args.from_mongo else "file"
    jobs_path = None if jobs_source == "mongo" else args.jobs

    if jobs_source == "file" and jobs_path is not None:
        jp = jobs_path.expanduser()
        if not jp.is_file():
            print(f"[gemini_ce_pref] jobs file not found: {jobs_path}", file=sys.stderr)
            return 2

    run_pipeline(
        users_path=args.users,
        jobs_source=jobs_source,
        jobs_path=jobs_path,
        retrieve_top_k=args.retrieve_top_k,
        final_top_k=args.final_top_k,
        output_path=args.output,
        max_per_job_skills=args.max_per_job_skills,
        mongo_filter_by_users=not args.mongo_all_active,
        cross_encoder_model=args.cross_encoder_model,
        cross_encoder_batch_size=args.cross_encoder_batch_size,
        jobs_limit=args.jobs_limit,
        compare_final_combiners=args.compare_final_combiners,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
