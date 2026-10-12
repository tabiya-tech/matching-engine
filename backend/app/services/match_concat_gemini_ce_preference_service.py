"""match_v3 retrieval (Gemini concat cosine + CE) → hybrid preference final score.

Used by ``POST /match_v4`` — same response envelope as ``POST /match_v3`` with
``u_hat``, ``p_hat``, and ``final_score`` on each recommendation (ranked by final).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.config import FINAL_SCORE_COMBINER, PREFERENCE_SCORER_MODE
from app.domain.job import Job
from app.domain.user import UserProfile
from app.jobs.repository import JobsLoadTiming
from app.matching.service import IMatchingService
from app.ranking.scoring import enrich_recommendations_with_preferences
from app.ranking.types import Candidate, ScoredCandidate
from app.schemas import (
    MatchConcatGeminiCeJobRecommendation,
    MatchConcatGeminiCeResponse,
    PreferenceStageSummary,
)
from app.server_dependencies.model_dependencies import get_preference_scorer

__all__ = ["run_match_concat_gemini_ce_with_preferences"]


def _job_fields(c: Candidate) -> dict[str, str | None]:
    item = c.item
    return {
        "job_uuid": item.uuid,
        "opportunity_title": item.title,
        "employer": item.employer if isinstance(item, Job) else None,
        "location": item.location,
        "URL": item.url if isinstance(item, Job) else None,
    }


def _scored_to_http(row: ScoredCandidate) -> MatchConcatGeminiCeJobRecommendation:
    c = row.candidate
    return MatchConcatGeminiCeJobRecommendation(
        **_job_fields(c),
        rank=row.rank,
        rank_cosine=c.rank_cosine,
        rank_cross_encoder=c.rank,
        concat_cosine_similarity=c.concat_cosine_similarity,
        cross_encoder_logit=c.cross_encoder_logit,
        cross_encoder_score=c.cross_encoder_score,
        u_hat=row.preference.u_hat,
        p_hat=row.p_hat,
        final_score=row.final_score,
        score_breakdown=row.breakdown,
    )


def _candidate_to_http(c: Candidate) -> MatchConcatGeminiCeJobRecommendation:
    """A reranked candidate without preference scoring (p_hat = the concat cosine)."""
    return MatchConcatGeminiCeJobRecommendation(
        **_job_fields(c),
        rank=c.rank,
        rank_cosine=c.rank_cosine,
        rank_cross_encoder=None,
        concat_cosine_similarity=c.concat_cosine_similarity,
        cross_encoder_logit=c.cross_encoder_logit,
        cross_encoder_score=c.cross_encoder_score,
        u_hat=None,
        p_hat=c.concat_cosine_similarity,
        final_score=None,
    )


def _final_formula(combiner: str) -> str:
    return "u_hat * p_hat" if combiner == "product" else "sqrt(u_hat * p_hat)"


def run_match_concat_gemini_ce_with_preferences(
    users: Sequence[UserProfile],
    jobs: Sequence[Job],
    *,
    retrieve_top_k: int,
    final_top_k: int,
    mongo_timing: JobsLoadTiming | None = None,
    final_score_combiner: str | None = None,
    include_work_activities: bool = True,
    user_unit_vectors: np.ndarray | None = None,
    matching_service: IMatchingService,
) -> list[MatchConcatGeminiCeResponse]:
    """One ``MatchConcatGeminiCeResponse`` per user, with preference fields on each job.

    ``user_unit_vectors`` (optional) lets a caller embed users once and reuse the matrix across
    corpora (jobs + occupations); passed straight through to ``MatchingService.shortlist_and_rerank``.

    :raises ValueError: If ``final_score_combiner`` is not ``product`` / ``geometric_mean``
    """

    combiner = (final_score_combiner or FINAL_SCORE_COMBINER).strip().lower()
    if combiner not in ("product", "geometric_mean"):
        raise ValueError("final_score_combiner must be 'product' or 'geometric_mean'")

    slates = matching_service.shortlist_and_rerank(
        users,
        jobs,
        retrieve_top_k=retrieve_top_k,
        final_top_k=final_top_k,
        mongo_timing=mongo_timing,
        user_unit_vectors=user_unit_vectors,
    )
    users_by_id = {u.user_id or "": u for u in users}
    pref_scorer = get_preference_scorer()
    pref_cls = type(pref_scorer).__module__ + "." + type(pref_scorer).__name__
    preference_summary = PreferenceStageSummary(
        stage3="combine_u_hat_and_p_hat",
        stage3_scorer="hybrid_preference_final",
        preference_scorer_mode=PREFERENCE_SCORER_MODE,
        preference_module=pref_cls,
        final_score_combiner=combiner,
        final_formula=_final_formula(combiner),
        p_hat_source="concat_cosine_similarity",
        include_work_activities=include_work_activities,
    )

    out: list[MatchConcatGeminiCeResponse] = []
    for slate in slates:
        user = users_by_id.get(slate.user_id)
        if slate.candidates and user is not None:
            recs = [
                _scored_to_http(r)
                for r in enrich_recommendations_with_preferences(
                    user,
                    slate.candidates,
                    preference_scorer=pref_scorer,
                    include_work_activities=include_work_activities,
                    final_score_combiner=combiner,
                )
            ]
        else:
            recs = [_candidate_to_http(c) for c in slate.candidates]

        out.append(
            MatchConcatGeminiCeResponse(
                user_id=slate.user_id,
                n_jobs_scored=slate.n_jobs_scored,
                n_jobs_active_loaded=slate.n_jobs_active_loaded,
                concat_gemini_ce_recommendations=recs,
                config_summary=slate.config_summary,
                preference_summary=preference_summary,
            )
        )

    return out
