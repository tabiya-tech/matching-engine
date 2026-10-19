"""`/experiments/v3/match` full response: occupations + opportunities + skill-gaps via the v3 engine.

Keeps the v3 **matching logic** unchanged (Gemini concat-cosine shortlist → cross-encoder rerank,
``MatchingService.shortlist_and_rerank``) but assembles the same ``MatchResponse`` shape ``/match_v4``
returns. This is ``MatchingService.rank`` *without* the preference (u_hat × p_hat) step.

By design:
* ``final_score`` = raw ``concat_cosine_similarity`` (the chosen v3 score); results are ordered by
  it so rank matches score. v4-only ``u_hat``/``p_hat``/preference fields are left empty (the v3
  engine computes no preference signal).
* Occupations are scored with the **same** v3 engine over the occupation corpus (precomputed concat
  embeddings), then county-scoped + random-fallback + de-duped by code, like ``/match_v4``. No
  occupation demand tilt (that is a v4-only re-rank).
* ``matched_skills`` is rebuilt from the shared ``CosineSkillMatcher`` (taxonomy/uuid mapping).
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
from app.matching.service import IMatchingService
from app.ranking.scoring import skill_detail
from app.ranking.skill_gaps import skill_gaps_for
from app.ranking.types import Candidate, UserShortlist
from app.schemas import (
    MatchResponse,
    OccupationRecommendation,
    OpportunityRecommendation,
    SkillGapRecommendation,
)
from app.server_dependencies.model_dependencies import (
    get_skill_matcher,
    get_skill_scorer,
)

__all__ = ["run_match_v3_full"]

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


def _by_cosine(slate: UserShortlist | None) -> list[Candidate]:
    """The slate ordered by concat cosine (the chosen v3 score), best first.

    ``final_score`` is the concat cosine (whitened — stage-1 ranks in the whitened concat space
    when the artifact is present).
    """
    if slate is None:
        return []
    return sorted(
        slate.candidates, key=lambda c: c.concat_cosine_similarity, reverse=True
    )


def run_match_v3_full(
    users: Sequence[UserProfile],
    jobs: Sequence[Job],
    occupations: Sequence[Occupation],
    *,
    retrieve_top_k: int,
    final_top_k: int,
    skill_gap_top_k: int = MATCH_TOP_K_SKILL_GAPS,
    matching_service: IMatchingService,
) -> list[MatchResponse]:
    """One ``MatchResponse`` per user using the v3 matching logic.

    The deployment's ``TARGET_LANGUAGE`` selects the cross-encoder checkpoint (see
    ``MatchingService.shortlist_and_rerank``); skill resolution itself is language-neutral.
    ``u_hat``/``p_hat``/preferences are empty (the v3 engine computes none).
    """
    if not users:
        return []

    u_norm = matching_service.embed_users(
        users
    )  # embed users ONCE, reuse for both corpora
    matcher = get_skill_matcher()

    # Opportunities — v3 engine over the active job corpus (engine + education gate unchanged).
    job_v3 = (
        matching_service.shortlist_and_rerank(
            users,
            jobs,
            retrieve_top_k=retrieve_top_k,
            final_top_k=final_top_k,
            user_unit_vectors=u_norm,
        )
        if jobs
        else []
    )
    # Occupations — SAME v3 engine over the occupation corpus. Score wide, then apply the v4 county
    # filter / fallback / dedupe-by-code so ~top_k distinct occupation codes survive.
    occ_breadth = max(retrieve_top_k, final_top_k, MATCH_V4_TOP_K_OCCUPATIONS * 8)
    occ_v3 = (
        matching_service.shortlist_and_rerank(
            users,
            occupations,
            retrieve_top_k=occ_breadth,
            final_top_k=occ_breadth,
            user_unit_vectors=u_norm,
            corpus="occupations",
        )
        if occupations
        else []
    )

    job_by_uid = {r.user_id: r for r in job_v3}
    occ_by_uid = {r.user_id: r for r in occ_v3}
    occ_counties = sorted({o.county for o in occupations if o.county})
    occ_keys = {k for o in occupations for k in o.location_keys}

    out: list[MatchResponse] = []
    for user in users:
        uid = user.user_id or ""
        user_keys = user_location_keys(user)

        with observability.stage("formatting", corpus="jobs"):
            opportunities: list[OpportunityRecommendation] = []
            for c in _by_cosine(job_by_uid.get(uid)):
                job = c.item
                if not isinstance(job, Job):
                    continue
                if user_keys.tier_for(job.location_keys) <= 0.0:
                    continue  # strict same-location: urban-pull pool broadening is v4-only
                opportunities.append(
                    fmt.build_opportunity_row(
                        job,
                        skill_detail(matcher, user, job, whitened_gate=False),
                        rank=len(opportunities) + 1,
                        final_score=c.concat_cosine_similarity,
                        scored=None,
                        sim_threshold=V4_FULL_SIM_THRESHOLD,
                        min_ess_share=V4_FULL_MIN_ESS_SHARE,
                    )
                )

        # Occupation county scoping (mirrors /match_v4): keep the user's own county row per code;
        # random fallback county if the user's province matches none.
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

            occupations_out: list[OccupationRecommendation] = []
            seen_codes: set[str] = set()
            for c in _by_cosine(occ_by_uid.get(uid)):
                occ = c.item
                if not isinstance(occ, Occupation):
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
                        final_score=c.concat_cosine_similarity,
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
