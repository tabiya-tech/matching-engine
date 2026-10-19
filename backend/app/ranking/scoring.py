"""Per-candidate scoring after retrieval: skill coverage, Phase-2 overrides, demand, success propensity, preference enrichment (final score) and occupation slate selection."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from pydantic import Field

from app.config import (
    DEMAND_SCORE_MAPPING,
    FINAL_SCORE_COMBINER,
    SUCCESS_PROPENSITY_CONFIG,
    V4_FULL_BADGE_PARTIAL,
    V4_FULL_BADGE_STRONG,
    V4_FULL_COVERAGE_FLOOR,
)
from app.domain.base import DomainModel
from app.domain.scores import PHatComponents
from app.domain.user import UserProfile
from app.ranking.preference import combine_final_score
from app.ranking.retrieval import ConcatWhitener
from app.ranking.skills import CosineSkillMatcher, Feasibility
from app.ranking.types import (
    AttributePreference,
    Candidate,
    DemandScore,
    FinalScoreBreakdown,
    MatchItem,
    PreferenceScore,
    RankOverrideSet,
    ScoredCandidate,
    SkillDetail,
    SkillMatchDetail,
    WorkActivityContribution,
)
from app.schemas import MatchedSkill, MatchedSkills, OptionalSkillMatch

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# demand
# --------------------------------------------------------------------------------------------------


class DemandScorer:
    """Maps an item's ``attributes.expected_demand`` label to a [0, 1] demand score."""

    def __init__(self):
        self.mapping = DEMAND_SCORE_MAPPING

    def calculate_score(self, item: MatchItem) -> DemandScore:
        """The item's demand score (0.5 and ``present=False`` when the label is absent or unknown)."""
        label = item.attributes.expected_demand
        if label and label in self.mapping:
            return DemandScore(score=self.mapping[label], label=label, present=True)
        return DemandScore(score=0.5, label=label, present=False)


# --------------------------------------------------------------------------------------------------
# coverage
#
# Per-item skill coverage: matched-skill split, essential coverage, coverage badge and eligibility.
# --------------------------------------------------------------------------------------------------


def build_matched_skills(
    per_job_skill: Sequence[SkillMatchDetail],
    essential_ids: frozenset[str],
    *,
    sim_threshold: float,
) -> MatchedSkills:
    """Split per-skill match rows into the response's ``MatchedSkills``.

    ``essential_ids`` MUST be the matcher-RESOLVED ids of the item's essential skills (resolved via
    the same label->id function the matcher uses to populate ``job_skill_id``), so the split is in
    the same id space. Every ``per_job_skill`` row is a resolved essential ∪ optional skill: rows in
    ``essential_ids`` -> essential (with per-skill cosine + meets_threshold); the rest are optional
    (kept as OptionalSkillMatch when cosine clears the threshold). Skill groups are not computed by
    the Gemini engine (empty list).
    """
    essential: list[MatchedSkill] = []
    optional: list[OptionalSkillMatch] = []
    for r in per_job_skill:
        jid = r.job_skill_id
        if not jid:
            continue
        sim = r.cosine_similarity
        # exact-id overlap always counts as "has it"
        meets = r.exact or sim >= sim_threshold
        tier = "exact" if r.exact else ("embedding" if sim >= sim_threshold else "none")
        if jid in essential_ids:
            essential.append(
                MatchedSkill(
                    job_skill_id=jid,
                    job_skill_label=r.job_skill_label,
                    best_user_skill_id=r.best_user_skill_id,
                    best_user_skill_label=r.best_user_skill_label,
                    similarity=round(sim, 4),
                    meets_threshold=meets,
                    match_tier=tier,
                )
            )
        elif meets:
            optional.append(
                OptionalSkillMatch(skill_id=jid, skill_label=r.job_skill_label)
            )
    return MatchedSkills(
        essential_skill_matches=essential,
        optional_exact_matches=optional,
        skill_group_matches=[],
    )


def essential_coverage(
    essential_matches: Sequence[MatchedSkill], n_essential_total: int
) -> float:
    """Share of a job's essential skills the user meets, in [0,1]. Denominator is the job's TOTAL
    essential count, so unresolved essentials (absent from ``essential_matches``) count as not-met."""
    n = max(n_essential_total, len(essential_matches))
    if n == 0:
        return 1.0
    met = sum(1 for m in essential_matches if m.meets_threshold)
    return met / n


def unparsed_ranking_coverage(
    parsed_covs: Sequence[float], *, override: float
) -> float:
    """Ranking coverage to assign a posting with NO parsed essential skills (unparsed).

    ``essential_coverage`` returns 1.0 for such postings, which gives them a demotion-free ride in
    the v4 Phase-2 ranking (``final *= coverage ** gamma``) despite zero verifiable skill overlap.
    Instead we treat an unparsed posting as a TYPICAL one:
      * ``override >= 0``  -> use that fixed value (1.0 restores the old free-passage behaviour).
      * else, if there are parsed coverages -> their mean (the live, per-shortlist average).
      * else (no parsed items at all) -> a neutral 0.5 fallback.
    Only the RANKING coverage is affected; the displayed essential_coverage is recomputed elsewhere.
    """
    if override >= 0:
        return float(override)
    if parsed_covs:
        return sum(parsed_covs) / len(parsed_covs)
    return 0.5


def skill_match_level(coverage: float, n_essential_total: int) -> str:
    """Graded badge from essential-coverage: strong / partial / weak ('unknown' if no essentials)."""
    if not n_essential_total:
        return "unknown"
    if coverage >= V4_FULL_BADGE_STRONG:
        return "strong"
    if coverage >= V4_FULL_BADGE_PARTIAL:
        return "partial"
    return "weak"


def is_eligible_from_skills(
    essential_matches: Sequence[MatchedSkill],
    *,
    n_essential_total: int,
    min_ess_share: float,
) -> bool:
    """Eligible iff essential-coverage >= min_ess_share. No essential skills -> eligible (nothing to
    gate on). The post-secondary education gate is applied upstream during retrieval.
    """
    if not n_essential_total and not essential_matches:
        return True
    return essential_coverage(essential_matches, n_essential_total) >= min_ess_share


# --------------------------------------------------------------------------------------------------
# phase2
#
# Per-item skill detail and the v4 Phase-2 ranking inputs (whitened p_hat + coverage demotion).
# --------------------------------------------------------------------------------------------------


def skill_detail(
    matcher: CosineSkillMatcher,
    user: UserProfile,
    item: MatchItem,
    *,
    whitened_gate: bool,
) -> SkillDetail:
    """Per-skill gate rows and the matcher-resolved essential id set of one (user, item) pair.

    Both sides go through the matcher's label resolver, so the essential id set is in the same
    (label-resolved) id space as ``per_job_skill[].job_skill_id`` — the split is robust to id/label
    mismatches.
    """
    try:
        per = (
            matcher.score_pair_v4(user, item)
            if whitened_gate
            else matcher.score_pair(user, item).per_job_skill
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("score_pair failed for %s: %s", item.uuid, e)
        per = []
    return SkillDetail(per_job_skill=per, essential_ids=matcher.essential_ids(item))


class RankOverrides:
    """Phase-2 per-candidate ranking inputs for one user's shortlist."""

    def __init__(
        self,
        *,
        matcher: CosineSkillMatcher,
        whitener: ConcatWhitener,
        whitened_gate: bool,
        sim_threshold: float,
        unparsed_coverage: float,
    ):
        self._matcher = matcher
        self._whitener = whitener
        self._whitened_gate = whitened_gate
        self._sim_threshold = sim_threshold
        self._unparsed_coverage = unparsed_coverage

    def compute(
        self,
        user: UserProfile,
        candidates: Sequence[Candidate],
        concat_by_uuid: Mapping[str, tuple[np.ndarray, bool]],
        u_white_vec: np.ndarray | None,
    ) -> RankOverrideSet:
        """Phase-2 inputs for one user's reranked slate.

        - p_hat override = whitened+rescaled concat cosine(user, item) in [0,1]
        - coverage = essential-coverage in [0,1] (drives the achievability demotion)
        - skill detail per item, reused by the formatters (no re-score)

        :param user: The user
        :param candidates: The user's reranked slate
        :param concat_by_uuid: uuid -> (stage-1 vector, is already whitened), snapshotted before retrieval
        :param u_white_vec: The user's whitened embedding, or None to skip the p_hat override
        """
        p_over: dict[str, float] = {}
        cov_over: dict[str, float] = {}
        details: dict[str, SkillDetail] = {}
        parsed_covs: list[float] = []  # coverages of items WITH parsed essential skills
        unparsed_uuids: list[
            str
        ] = []  # items with no essential skills (back-filled below)
        target = self._whitener.rescale_target()
        for c in candidates:
            uuid = c.uuid
            if not uuid or uuid in details:
                continue
            jrec = concat_by_uuid.get(uuid)
            if jrec is not None and u_white_vec is not None and target > 0:
                jv, jv_is_white = jrec
                # DB-whitened jobs are already in whitened space (same artifact) -> use directly (with
                # an L2-norm guard); raw vectors (occupations, offline, not-yet-whitened jobs) are
                # whitened in-process once.
                if jv_is_white:
                    _n = float(np.linalg.norm(jv))
                    jw = jv / _n if _n else jv
                else:
                    jw = self._whitener.whiten_rows(jv.reshape(1, -1))[0]
                cos = float(np.dot(u_white_vec, jw))
                p_over[uuid] = min(1.0, max(0.0, cos) / target)
            detail = skill_detail(
                self._matcher, user, c.item, whitened_gate=self._whitened_gate
            )
            details[uuid] = detail
            n_ess = len(c.item.essential_skills)
            if n_ess:
                ms = build_matched_skills(
                    detail.per_job_skill,
                    detail.essential_ids,
                    sim_threshold=self._sim_threshold,
                )
                cov = essential_coverage(ms.essential_skill_matches, n_ess)
                cov_over[uuid] = cov
                parsed_covs.append(cov)
            else:
                unparsed_uuids.append(uuid)
        # Back-fill unparsed items with the typical (mean) parsed coverage of this shortlist (or the
        # configured override / neutral fallback) so they no longer escape the achievability demotion.
        if unparsed_uuids:
            fill = unparsed_ranking_coverage(
                parsed_covs, override=self._unparsed_coverage
            )
            for uuid in unparsed_uuids:
                cov_over[uuid] = fill
        return RankOverrideSet(p_hat=p_over, coverage=cov_over, skill_details=details)


# --------------------------------------------------------------------------------------------------
# success_propensity
#
# Success Propensity Scorer (p_hat)
#
# Computes a cold-start hiring-probability proxy for a (seeker, job) pair.
# This is the recruiter-side feasibility signal, distinct from the seeker-side
# utility proxy (u_hat) produced by PreferenceScorer.
#
# The score is multiplicative, not additive:
#
#     p_hat = G_ij * E_ij^alpha * R_ij^beta * M_ij^gamma
#
# where:
#     G_ij  - hard feasibility gate (0 or 1)
#     E_ij  - essential-skill coverage (geometric mean of Node2Vec cosine sims)
#     R_ij  - recruiter-side readiness (optional skills + skill group recall)
#     M_ij  - market opportunity (demand, freshness)
#
# All per-skill similarities are Node2Vec graph-based cosine distances computed
# via SimilarityEngine — identical to the existing skill matching pipeline.
# --------------------------------------------------------------------------------------------------


class SuccessPropensity(DomainModel):
    """Legacy success propensity ``p_hat`` of a pair (``SuccessPropensityScorer``)."""

    p_hat: float = Field(description="Success propensity in [0, 1], rounded.")
    components: PHatComponents = Field(description="The G, E, R, M factors.")
    demand_label: str = Field(
        description="The item's demand label; ``Unknown`` when absent."
    )


class SuccessPropensityScorer:
    """Legacy multiplicative ``p_hat`` from feasibility signals and market demand."""

    def __init__(self):
        self.cfg = SUCCESS_PROPENSITY_CONFIG
        self.demand_mapping = DEMAND_SCORE_MAPPING

    # ------------------------------------------------------------------
    # M_ij: market opportunity
    # ------------------------------------------------------------------
    def _market_opportunity(self, item: MatchItem) -> float:
        """Compute market-opportunity signal from demand label.

        Currently uses the same demand-label mapping as the legacy DemandScorer.
        When richer market data become available (tightness, penetration,
        absorptive capacity, freshness), this method is the place to extend.
        """
        demand_label = item.attributes.expected_demand
        if demand_label and demand_label in self.demand_mapping:
            return self.demand_mapping[demand_label]
        return 0.5  # neutral default

    # ------------------------------------------------------------------
    # R_ij: recruiter-side readiness
    # ------------------------------------------------------------------
    @staticmethod
    def _recruiter_readiness(
        optional_sim: float,
        skill_group_recall: float,
        has_optional_skills: bool,
        has_skill_groups: bool,
    ) -> float:
        """Combine optional-skill alignment and skill-group recall into a
        single recruiter-readiness signal.

        Both inputs are already in [0, 1].  When a dimension has no data
        (the job lists no optional skills / no skill groups), that dimension
        is excluded rather than counted as zero — otherwise data sparsity
        is penalised as poor fit.  If *neither* dimension has data, R_ij
        returns 1.0 (neutral).
        """
        terms = []
        weights = []
        if has_optional_skills:
            terms.append(optional_sim)
            weights.append(0.6)
        if has_skill_groups:
            terms.append(skill_group_recall)
            weights.append(0.4)
        if not terms:
            return 1.0  # no data → neutral, don't penalise
        return sum(w * t for w, t in zip(weights, terms, strict=True)) / sum(weights)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def calculate_score(
        self, item: MatchItem, feasibility: Feasibility
    ) -> SuccessPropensity:
        """Compute p_hat from pre-computed feasibility signals.

        :param item: The job or occupation (its demand label drives M)
        :param feasibility: ``SkillScorer`` feasibility signals of the pair
        :return: p_hat, its components and the demand label
        """
        alpha = self.cfg["alpha_essential"]
        beta = self.cfg["beta_readiness"]
        gamma = self.cfg["gamma_market"]

        # --- components ---
        g_ij = 1.0 if feasibility.gate_passed else 0.0
        e_ij = feasibility.essential_fit
        r_ij = self._recruiter_readiness(
            feasibility.optional_sim,
            feasibility.skill_group_recall,
            has_optional_skills=feasibility.has_optional_skills,
            has_skill_groups=feasibility.has_skill_groups,
        )

        demand_label = item.attributes.expected_demand
        m_ij: float | None = (
            self.demand_mapping[demand_label]
            if demand_label and demand_label in self.demand_mapping
            else None
        )

        # --- dynamic exponent redistribution ---
        # When a component's data is absent, redistribute its exponent
        # proportionally to the remaining components so that missing data
        # doesn't drag the score down.
        if m_ij is None:
            remaining = alpha + beta
            total = alpha + beta + gamma
            if remaining > 0:
                alpha = total * (alpha / remaining)
                beta = total * (beta / remaining)
            gamma = 0.0
            m_ij = 1.0

        # Floor at 0.01 so a zero component doesn't annihilate the product
        e_ij = max(e_ij, 0.01)
        r_ij = max(r_ij, 0.01)
        m_ij = max(m_ij, 0.01)

        p_hat = g_ij * (e_ij**alpha) * (r_ij**beta) * (m_ij**gamma)

        return SuccessPropensity(
            p_hat=round(p_hat, 4),
            components=PHatComponents(
                gate=round(g_ij, 4),
                essential_fit=round(e_ij, 4),
                recruiter_readiness=round(r_ij, 4),
                market_opportunity=round(m_ij, 4),
            ),
            demand_label=demand_label or "Unknown",
        )


# --------------------------------------------------------------------------------------------------
# enrichment
#
# Ranker: combine u_hat (preferences) and p_hat (skill retrieval score).
# --------------------------------------------------------------------------------------------------

# Engine-agnostic demand scorer (reads item attributes.expected_demand); torch-free.
_DEMAND_SCORER = DemandScorer()


class IPreferenceScorer(Protocol):
    """A preference scorer (``UnifiedPreferenceScorer`` or the legacy ``PreferenceScorer``)."""

    def calculate_score(
        self,
        user: UserProfile,
        item: MatchItem,
        *,
        include_work_activities: bool = True,
    ) -> PreferenceScore: ...


class WorkActivityDashboard(DomainModel):
    """Part B (BWS × work activities) of one row, for the offline HTML dashboards."""

    S_wa: float | None = Field(description="Task utility; None when not scored.")
    n_work_activities: int = Field(description="Activities scored.")
    wa_aggregation: str = Field(description="Aggregation used.")
    rows: list[WorkActivityContribution] = Field(
        description="Up to 25 activities, largest |contribution| first."
    )


def work_activity_match_for_dashboard(pref: PreferenceScore) -> WorkActivityDashboard:
    """Extract Part B (BWS × job work activities) for one job row."""
    wa = pref.work_activities
    if wa is None:
        return WorkActivityDashboard(
            S_wa=None, n_work_activities=0, wa_aggregation="mean", rows=[]
        )
    rows = sorted(wa.wa_details, key=lambda r: (-abs(r.wa_contribution), r.wa_code))
    return WorkActivityDashboard(
        S_wa=wa.wa_score_sum,
        n_work_activities=wa.n_work_activities or len(rows),
        wa_aggregation=wa.wa_aggregation or "mean",
        rows=rows[:25],
    )


def preference_details_for_dashboard(
    pref: PreferenceScore,
) -> list[AttributePreference]:
    """Schema-scored attribute rows (unified scorer), largest |contribution| first."""
    rows = [r for r in pref.attributes if r.attr_label is not None]
    rows.sort(key=lambda r: (-abs(r.contribution), r.attr_label or ""))
    return rows


def p_hat_from_candidate(candidate: Candidate) -> tuple[float, str]:
    """p_hat = raw stage-1 concat cosine (not the CE min–max score).

    ``cross_encoder_score`` is only used for CE *ordering*; it is per-user min–max in [0, 1] so the
    top row is often 1.0 and must not be used as p_hat.
    """
    return candidate.concat_cosine_similarity, "concat_cosine_similarity"


def compute_final_score(
    pref: PreferenceScore,
    p_hat: float,
    *,
    p_hat_source: str,
    combiner: str,
) -> tuple[float, FinalScoreBreakdown]:
    """Compute final score from u_hat and p_hat (configurable)."""
    u_hat = pref.u_hat
    final = combine_final_score(u_hat, p_hat, combiner=combiner)  # type: ignore[arg-type]
    return final, FinalScoreBreakdown(
        final_score_combiner=combiner,
        u_hat=round(u_hat, 4),
        p_hat=round(p_hat, 4),
        p_hat_source=p_hat_source,
        final_score=round(final, 4),
        preference_score_legacy=round(pref.score, 4),
    )


def enrich_recommendations_with_preferences(
    user: UserProfile,
    candidates: Sequence[Candidate],
    *,
    preference_scorer: IPreferenceScorer,
    include_work_activities: bool = True,
    final_score_combiner: str | None = None,
    include_demand: bool = False,
    demand_gamma: float = 0.0,
    p_hat_by_uuid: Mapping[str, float] | None = None,
    coverage_by_uuid: Mapping[str, float] | None = None,
    coverage_gamma: float = 0.0,
    location_tier_by_uuid: Mapping[str, float] | None = None,
) -> list[ScoredCandidate]:
    """Stage 3 only: compute u_hat per item, p_hat from stage 1–2 cosine, re-rank by u_hat × p_hat.

    ``candidates`` stay in CE order; this returns a new list sorted by ``final_score``.

    ``include_demand`` (with ``demand_gamma`` > 0) applies an occupation-only labour-market tilt:
    ``final *= M ** demand_gamma`` where M = expected-demand score in [0,1] (neutral 1.0 when the
    item has no/unknown demand). Opportunities call with ``include_demand=False`` (the default),
    so their ranking is unaffected.
    """
    combiner = (final_score_combiner or FINAL_SCORE_COMBINER).strip().lower()
    scored: list[tuple[float, float, int, ScoredCandidate]] = []

    for i, c in enumerate(candidates):
        uid = c.uuid
        item = c.item

        pref = preference_scorer.calculate_score(
            user, item, include_work_activities=include_work_activities
        )
        p_hat, p_hat_source = p_hat_from_candidate(c)
        # Phase-2 (V4_FULL_RANK_DEMOTE): override p_hat's skills-fit with the WHITENED+rescaled concat
        # cosine (de-anisotropised, discriminative). Still in [0,1]; final = u_hat x p_hat unchanged.
        if p_hat_by_uuid is not None and uid in p_hat_by_uuid:
            p_hat = max(0.0, min(1.0, float(p_hat_by_uuid[uid])))
            p_hat_source = "concat_cosine_whitened"
        final, breakdown = compute_final_score(
            pref, p_hat, p_hat_source=p_hat_source, combiner=combiner
        )
        updates: dict[str, float | str | None] = {}

        # Occupation-only demand tilt (opportunities pass include_demand=False): multiply the
        # final score by M**gamma — M = expected-demand score in [0,1], neutral 1.0 when the item
        # has no/unknown demand. Mirrors the legacy p_hat market factor; this re-ranks occupations.
        if include_demand and demand_gamma > 0:
            demand = _DEMAND_SCORER.calculate_score(item)
            if demand.present:
                m = max(0.0, min(1.0, demand.score))
                final = final * (m**demand_gamma)
                updates.update(
                    demand_score=round(m, 4),
                    demand_label=demand.label,
                    demand_gamma=round(float(demand_gamma), 4),
                )

        # Phase-2 achievability demotion: final *= floor + (1-floor) * essential_coverage**gamma (all
        # factors in [0,1], so final stays in range and the combination u_hat x p_hat is preserved —
        # coverage is just an extra [0,1] factor of p_hat). Demotes high-fit-but-missing-must-haves
        # items (woodworker x dev). The floor keeps a coverage of 0 a demotion rather than an
        # annihilation: without it every zero-coverage candidate ties at final_score 0.0, which drops
        # the u_hat x p_hat ordering and reports a non-zero breakdown beside a 0.0 score.
        if coverage_by_uuid is not None and coverage_gamma > 0:
            cov = coverage_by_uuid.get(uid)
            if cov is not None:
                cov = max(0.0, min(1.0, float(cov)))
                floor = max(0.0, min(1.0, V4_FULL_COVERAGE_FLOOR))
                factor = floor + (1.0 - floor) * cov**coverage_gamma
                final = final * factor
                updates.update(
                    essential_coverage=round(cov, 4),
                    coverage_gamma=round(float(coverage_gamma), 4),
                    coverage_floor=round(floor, 4),
                    coverage_factor=round(factor, 4),
                )

        # Urban-pull location tier (flat multiply): final *= tier, where tier in {1.0 local,
        # W_REGIONAL, W_NATIONAL, 0.0 off-chain}. The tier values are themselves the tuning knobs, so
        # no exponent. Local jobs preferred; off-chain (e.g. another batch user's locations) is an
        # exclusion, not a demotion — drop the candidate rather than return it scored 0.0.
        if location_tier_by_uuid is not None:
            tier = location_tier_by_uuid.get(uid)
            if tier is not None:
                tier = max(0.0, min(1.0, float(tier)))
                if tier <= 0.0:
                    continue
                final = final * tier
                updates["location_tier_factor"] = round(tier, 4)

        updates["final_score"] = round(final, 4)
        breakdown = breakdown.model_copy(update=updates)
        scored.append(
            (
                final,
                p_hat,
                i,
                ScoredCandidate(
                    candidate=c,
                    rank=0,
                    preference=pref,
                    p_hat=round(p_hat, 4),
                    p_hat_source=p_hat_source,
                    final_score=breakdown.final_score,
                    breakdown=breakdown,
                ),
            )
        )

    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [
        row.model_copy(update={"rank": rank})
        for rank, (_fs, _ph, _i, row) in enumerate(scored, start=1)
    ]


def enriched_recommendations(
    user: UserProfile,
    candidates: Sequence[Candidate],
    pref_scorer: IPreferenceScorer,
    combiner: str | None,
    *,
    location_keys: frozenset[str] | None = None,
    include_demand: bool = False,
    demand_gamma: float = 0.0,
    p_hat_by_uuid: Mapping[str, float] | None = None,
    coverage_by_uuid: Mapping[str, float] | None = None,
    coverage_gamma: float = 0.0,
    location_tier_by_uuid: Mapping[str, float] | None = None,
) -> list[ScoredCandidate]:
    """CE slate for one user -> preference-enriched, final-score-sorted candidates.

    With ``location_keys``, candidates are first restricted to items having one of those location
    keys (``app.location.keys``). The keys may differ from the ``user``'s own WITHOUT changing whose
    preferences drive ``u_hat`` — used for the random-county fallback. For occupations this keeps
    one row/code (a single county) with that county's attributes. None = no location step.

    Note: the post-secondary education gate is already applied upstream in stage-1 retrieval
    (it skips ineligible items before the top-k cutoff), so no education filtering is needed here.
    """
    slate = list(candidates)
    if location_keys is not None and slate:
        slate = [c for c in slate if not location_keys.isdisjoint(c.item.location_keys)]
    if not slate:
        return []
    return enrich_recommendations_with_preferences(
        user,
        slate,
        preference_scorer=pref_scorer,
        include_work_activities=True,
        final_score_combiner=combiner,
        include_demand=include_demand,
        demand_gamma=demand_gamma,
        p_hat_by_uuid=p_hat_by_uuid,
        coverage_by_uuid=coverage_by_uuid,
        coverage_gamma=coverage_gamma,
        location_tier_by_uuid=location_tier_by_uuid,
    )
