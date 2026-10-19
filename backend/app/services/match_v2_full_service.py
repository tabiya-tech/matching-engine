"""`/experiments/v2/match` full response: occupations + opportunities + skill-gaps via the v2 engine.

Keeps the v2 **matching formula** unchanged (BM25 × embedding-cosine pool min-max fusion,
``hybrid_match_users_with_jobs``) but assembles the same ``MatchResponse`` shape ``/match_v4``
returns: one ``MatchResponse`` per user with opportunity, occupation and skill-gap
recommendations.

Differences from the v2 ``hybrid_recommendations`` envelope (by design):
* ``final_score`` = the hybrid ``fusion_score``; v4-only ``u_hat``/``p_hat``/preference fields are
  left empty (the v2 engine computes no preference / p_hat signal).
* Occupations are scored with the **same** hybrid engine over the occupation corpus, then filtered
  to the user's county (with v4's random-county fallback) and de-duped by occupation code.
* ``matched_skills`` is rebuilt from the shared ``CosineSkillMatcher`` (taxonomy/uuid mapping),
  identical to the v4 path.
* Skill gaps reuse the existing ``/match_v4`` analysis unchanged.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence

from app import observability
from app.config import (
    MATCH_TOP_K_SKILL_GAPS,
    MATCH_V4_TOP_K_OCCUPATIONS,
    SKILL_RESCALE_TARGET,
    V4_FULL_MIN_ESS_SHARE,
    V4_FULL_SIM_THRESHOLD,
)
from app.domain.job import Job
from app.domain.occupation import Occupation
from app.domain.user import UserProfile
from app.location.keys import canonical_location, user_home_keys, user_location_keys
from app.matching import formatting as fmt
from app.ranking.retrieval import index_by_uuid
from app.ranking.scoring import skill_detail
from app.ranking.skill_gaps import skill_gaps_for
from app.schemas import (
    MatchResponse,
    OccupationRecommendation,
    OpportunityRecommendation,
    SkillGapRecommendation,
)
from app.server_dependencies.model_dependencies import get_skill_scorer
from app.services.hybrid_scoring.run_bm25_cosine_hybrid import (
    HybridFusedRow,
    HybridRunResult,
    get_cosine_matcher_singleton,
    hybrid_match_users_with_jobs,
)

__all__ = ["run_match_v2_full"]

logger = logging.getLogger(__name__)


def _skill_gaps_for(
    user: UserProfile, jobs: Sequence[Job], top_k: int
) -> list[SkillGapRecommendation]:
    """The /match skill-gap analysis (engine-agnostic)."""
    return skill_gaps_for(
        user,
        jobs,
        top_k,
        scorer=get_skill_scorer(),
        rescale_target=SKILL_RESCALE_TARGET,
    )


def _fused_rows(envelope: HybridRunResult) -> dict[str, list[HybridFusedRow]]:
    """Map user_id -> that user's ``column_fused_weighted_minmax`` rows."""
    return {b.user_id or "": b.column_fused_weighted_minmax for b in envelope.results}


def run_match_v2_full(
    users: Sequence[UserProfile],
    jobs: Sequence[Job],
    occupations: Sequence[Occupation],
    *,
    fusion_top_k: int,
    alpha_on_cosine: float,
    skill_gap_top_k: int = MATCH_TOP_K_SKILL_GAPS,
) -> list[MatchResponse]:
    """One ``MatchResponse`` per user using the v2 hybrid matching formula.

    ``final_score`` is the hybrid fusion score; ``u_hat``/``p_hat``/preferences are empty (the v2
    engine computes none). ``matched_skills`` comes from the shared ``CosineSkillMatcher``.
    """
    if not users:
        return []

    matcher = get_cosine_matcher_singleton()
    job_index = index_by_uuid(jobs)
    occ_index = index_by_uuid(occupations)

    # Opportunities — v2 hybrid over the active job corpus (engine + education gate unchanged).
    opp_by_uid: dict[str, list[HybridFusedRow]] = {}
    if jobs:
        with observability.stage(
            "shortlist", corpus="jobs", engine="bm25_cosine_hybrid", n_users=len(users)
        ):
            opp_by_uid = _fused_rows(
                hybrid_match_users_with_jobs(
                    users,
                    jobs,
                    col_display_k=fusion_top_k,
                    alpha_on_cosine=alpha_on_cosine,
                )
            )

    # Occupations — SAME hybrid engine over the occupation corpus (its full-text documents use the
    # occupation label and description). Score wide, then apply the v4 county filter / fallback /
    # dedupe-by-code so ~top_k distinct occupation codes survive.
    occ_by_uid: dict[str, list[HybridFusedRow]] = {}
    if occupations:
        occ_breadth = max(fusion_top_k, MATCH_V4_TOP_K_OCCUPATIONS * 8)
        with observability.stage(
            "shortlist",
            corpus="occupations",
            engine="bm25_cosine_hybrid",
            n_users=len(users),
        ):
            occ_by_uid = _fused_rows(
                hybrid_match_users_with_jobs(
                    users,
                    occupations,
                    col_display_k=occ_breadth,
                    alpha_on_cosine=alpha_on_cosine,
                )
            )

    occ_counties = sorted({o.county for o in occupations if o.county})
    occ_keys = {k for o in occupations for k in o.location_keys}

    out: list[MatchResponse] = []
    for user in users:
        uid = user.user_id or ""
        user_keys = user_location_keys(user)

        opportunities: list[OpportunityRecommendation] = []
        with observability.stage("formatting", corpus="jobs"):
            for row in opp_by_uid.get(uid, []):
                job = job_index.get(row.job_uuid)
                if job is None:
                    continue
                if user_keys.tier_for(job.location_keys) <= 0.0:
                    continue  # strict same-location: urban-pull pool broadening is v4-only
                opportunities.append(
                    fmt.build_opportunity_row(
                        job,
                        skill_detail(matcher, user, job, whitened_gate=False),
                        rank=len(opportunities) + 1,
                        final_score=row.fusion_score,
                        scored=None,
                        sim_threshold=V4_FULL_SIM_THRESHOLD,
                        min_ess_share=V4_FULL_MIN_ESS_SHARE,
                    )
                )

        # Occupation county scoping (mirrors /match_v4): keep the user's own county row per code;
        # if the user's province matches no occupation county, fall back to a random available one
        # (location only — the user still drives skill matching).
        occupations_out: list[OccupationRecommendation] = []
        with observability.stage("formatting", corpus="occupations"):
            loc_keys = frozenset(user_home_keys(user))
            if occ_counties and not loc_keys & occ_keys:
                fallback = random.choice(occ_counties)
                loc_keys = frozenset({canonical_location(fallback)})
                logger.warning(
                    "User %r matches no occupation county %s; using random fallback county %r.",
                    uid,
                    occ_counties,
                    fallback,
                )

            seen_codes: set[str] = set()
            for row in occ_by_uid.get(uid, []):
                occ = occ_index.get(row.job_uuid)
                if occ is None:
                    continue
                if loc_keys.isdisjoint(occ.location_keys):
                    continue
                code = occ.originUuid or occ.uuid
                if not code or code in seen_codes:
                    continue
                seen_codes.add(code)
                occupations_out.append(
                    fmt.build_occupation_row(
                        occ,
                        skill_detail(matcher, user, occ, whitened_gate=False),
                        rank=len(occupations_out) + 1,
                        final_score=row.fusion_score,
                        scored=None,
                        sim_threshold=V4_FULL_SIM_THRESHOLD,
                        min_ess_share=V4_FULL_MIN_ESS_SHARE,
                    )
                )
                if len(occupations_out) >= MATCH_V4_TOP_K_OCCUPATIONS:
                    break

        with observability.stage("skill_gaps"):
            skill_gaps = _skill_gaps_for(user, jobs, skill_gap_top_k)

        out.append(
            MatchResponse(
                user_id=uid,
                occupation_recommendations=occupations_out,
                opportunity_recommendations=opportunities,
                skill_gap_recommendations=skill_gaps,
            )
        )

    return out
