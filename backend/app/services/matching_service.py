"""Legacy matching engine (retired ``/match``): skill utility × success propensity per item.

Supports two scoring modes (``config.SCORING_MODE``):
  - ``multiplicative``: final_score = u_hat * p_hat  (paper-aligned)
  - ``additive``:       final_score = w1*S_skills + w2*S_pref + w3*S_demand  (legacy)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app import observability
from app.config import (
    DEMAND_SCORE_MAPPING,
    GLOBAL_WEIGHTS,
    MATCH_APPLY_LOCATION_FILTER,
    MATCH_RESPONSE_SKILL_MIN_SCORE,
    MATCH_TOP_K_OCCUPATIONS,
    MATCH_TOP_K_OPPORTUNITIES,
    MATCH_TOP_K_SKILL_GAPS,
    SCORING_MODE,
    SKILL_RESCALE_TARGET,
)
from app.domain.job import Job
from app.domain.occupation import Occupation
from app.domain.scores import PHatComponents, ScoreBreakdown, SkillComponents
from app.domain.user import UserProfile
from app.match_timing_log import log_match_step
from app.matching.formatting import matched_preferences, matched_work_activities
from app.ranking.retrieval import (
    Place,
    filter_jobs_by_education,
    job_matches_user_location,
)
from app.ranking.scoring import DemandScorer, SuccessPropensity, SuccessPropensityScorer
from app.ranking.skill_gaps import (
    analyze_skill_gaps,
)
from app.ranking.skill_gaps import (
    filter_skill_gap_recommendations as _filter_skill_gap_recommendations,
)
from app.ranking.skill_gaps import (
    skill_gap_candidate_pool_k as _skill_gap_candidate_pool_k,
)
from app.ranking.skills import SkillUtility
from app.ranking.types import AttributePreference, MatchItem, PreferenceScore
from app.schemas import (
    MatchedSkill,
    MatchedSkills,
    MatchResponse,
    OccupationRecommendation,
    OpportunityRecommendation,
)
from app.server_dependencies.model_dependencies import (
    get_preference_scorer,
    get_skill_scorer,
)

# Initialize scorers once at module level
scorer_skill = get_skill_scorer()
scorer_pref = get_preference_scorer()
scorer_success = SuccessPropensityScorer()

ScoringMode = Literal["multiplicative", "additive"]
ItemType = Literal["opportunity", "occupation"]


class LegacyRecommendation(BaseModel):
    """One scored item of the legacy engine, before formatting."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: Job | Occupation = Field(description="The scored job or occupation.")
    score: float = Field(description="Final score.")
    scoring_mode: str = Field(description="``multiplicative`` or ``additive``.")
    u_hat: float = Field(description="Preference utility.")
    propensity: SuccessPropensity = Field(
        description="Success propensity (zero p_hat / no components in additive mode)."
    )
    skill: SkillUtility = Field(description="Skill utility U.")
    match_details: MatchedSkills = Field(description="Per-skill match rows.")
    preference: PreferenceScore = Field(description="Preference scorer output.")
    demand_score: float = Field(default=0.5, description="Demand score (additive).")
    demand_label: str = Field(description="Demand label.")


class MatchItemsTiming(BaseModel):
    """Per-stage timings of one ``_match_items`` call."""

    item_type: ItemType = Field(description="Opportunities or occupations.")
    n_in: int = Field(description="Items passed in.")
    n_after_filter: int = Field(
        description="Items after the location / education filter."
    )
    filter_ms: float = Field(description="Filter time.")
    n_scored: int = Field(description="Items scored.")
    scoring_mode: str = Field(description="Scoring mode.")
    u_hat_ms: float = Field(description="Preference scoring time.")
    skill_feasibility_ms: float = Field(description="Skill + feasibility time.")
    p_hat_ms: float = Field(description="Success-propensity time.")
    legacy_skill_ms: float = Field(description="Additive skill time.")
    legacy_demand_ms: float = Field(description="Additive demand time.")
    total_scoring_ms: float = Field(description="Total scoring time.")
    format_recommendations_ms: float = Field(description="Formatting time.")


class DashboardOpportunityRow(BaseModel):
    """Compact opportunity row for the static comparison dashboards (embedded JSON in HTML)."""

    r: int = Field(description="Rank.")
    t: str = Field(description="Title.")
    e: str = Field(description="Employer.")
    l: str = Field(description="Location.")  # noqa: E741
    el: bool = Field(description="Eligible.")
    f: float = Field(description="Final score.")
    u: float = Field(description="u_hat.")
    p: float = Field(description="p_hat.")
    pg: float = Field(description="p_hat gate.")
    pe: float = Field(description="p_hat essential fit.")
    pr: float = Field(description="p_hat recruiter readiness.")
    pm: float = Field(description="p_hat market opportunity.")
    sl: float = Field(description="Skill location component.")
    se: float = Field(description="Skill essential component.")
    so: float = Field(description="Skill optional component.")
    sg: float = Field(description="Skill group component.")
    ms: list[tuple[str, str, float, bool]] = Field(
        description="(user skill, job skill, similarity, meets threshold) per essential skill."
    )
    j: str = Field(description="Justification.")
    url: str = Field(description="Posting link.")
    je: list[str] = Field(description="Essential skill labels.")
    jo: list[str] = Field(description="Optional skill labels.")


def _ms(t0: float) -> float:
    """Elapsed milliseconds since t0 = time.perf_counter()."""
    return (time.perf_counter() - t0) * 1000.0


def _filter_essential_skill_matches(match_details: MatchedSkills) -> list[MatchedSkill]:
    """Keep only essential skill matches whose similarity passes response threshold."""
    return [
        m
        for m in match_details.essential_skill_matches
        if m.similarity >= MATCH_RESPONSE_SKILL_MIN_SCORE
    ]


def _build_justification(
    match_details: MatchedSkills,
    pref_rows: Sequence[AttributePreference],
    demand_label: str,
    score: float,
) -> str:
    """Build a human-readable justification for a match."""
    top_prefs = [p for p in pref_rows if p.matched][:3]
    pref_text = "; ".join(
        f"{p.attribute.replace('_', ' ').title()}: {p.job_value_label}"
        for p in top_prefs
    )

    top_skills = match_details.essential_skill_matches[:2]
    skill_text = ". ".join(
        f"{s.best_user_skill_label or 'Unknown'} ↔ {s.job_skill_label or 'Unknown'} (sim {s.similarity})"
        for s in top_skills
        if s.meets_threshold
    )

    parts = []
    if skill_text:
        parts.append(f"Top skill matches: {skill_text}")
    if pref_text:
        parts.append(f"Top preference matches: {pref_text}")

    if demand_label:
        parts.append(f"Demand label: {demand_label}")

    return ". ".join(parts) if parts else f"Match Score: {round(float(score), 2)}"


def _skill_components(skill: SkillUtility) -> SkillComponents:
    c = skill.components
    return SkillComponents(loc=c.loc, ess=c.ess, opt=c.opt, grp=c.grp)


def _score_breakdown(
    rec: LegacyRecommendation, *, include_demand: bool = False
) -> ScoreBreakdown:
    """Interpretable score breakdown of one recommendation (multiplicative or additive)."""
    skill = rec.skill
    if rec.scoring_mode == "multiplicative":
        components = rec.propensity.components
        demand_label = rec.propensity.demand_label if include_demand else None
        return ScoreBreakdown(
            u_hat=round(rec.u_hat, 4),
            p_hat=round(rec.propensity.p_hat, 4),
            p_hat_components=PHatComponents(
                gate=round(components.gate, 4),
                essential_fit=round(components.essential_fit, 4),
                recruiter_readiness=round(components.recruiter_readiness, 4),
                market_opportunity=round(components.market_opportunity, 4),
            ),
            total_skill_utility=round(skill.U_final, 4),
            skill_components=_skill_components(skill),
            skill_penalty_applied=round(skill.penalty, 4),
            preference_score=round(rec.preference.score, 4),
            demand_score=(
                round(float(DEMAND_SCORE_MAPPING.get(demand_label, 0.5)), 4)
                if demand_label
                else 0.5
            )
            if include_demand
            else None,
            demand_label=demand_label,
        )
    # Legacy additive score breakdown (kept for A/B testing).
    return ScoreBreakdown(
        total_skill_utility=round(skill.U_final, 4),
        skill_components=_skill_components(skill),
        skill_penalty_applied=round(skill.penalty, 4),
        preference_score=round(rec.preference.score, 4),
        demand_score=round(rec.demand_score, 4),
        demand_label=rec.demand_label,
    )


def _matched_skills(rec: LegacyRecommendation) -> MatchedSkills:
    return MatchedSkills(
        essential_skill_matches=_filter_essential_skill_matches(rec.match_details),
        optional_exact_matches=rec.match_details.optional_exact_matches,
        skill_group_matches=rec.match_details.skill_group_matches,
    )


def _justification(rec: LegacyRecommendation) -> str:
    return _build_justification(
        rec.match_details, rec.preference.attributes, rec.demand_label, rec.score
    )


def _format_opportunity(
    item: Job, rank: int, rec: LegacyRecommendation
) -> OpportunityRecommendation:
    """Format a single opportunity recommendation."""
    return OpportunityRecommendation(
        uuid=item.uuid,
        URL=item.url or f"www.example.com/{item.uuid}",
        rank=rank,
        opportunity_title=item.opportunity_title,
        opportunity_isco_occupation_group=item.opportunity_isco_occupation_group,
        opportunity_isco_occupation_group_id=item.opportunity_isco_occupation_group_id,
        location=item.location,
        employer=item.employer,
        employment_type=item.employment_type,
        salary_text=item.salary_text,
        closing_date=item.closing_date,
        is_eligible=rec.skill.is_eligible,
        justification=_justification(rec),
        opportunity_description=item.opportunity_description or item.contract_type,
        contract_type=item.contract_type,
        final_score=round(rec.score, 4),
        score_breakdown=_score_breakdown(rec),
        matched_skills=_matched_skills(rec),
        matched_preferences=matched_preferences(rec.preference),
        matched_work_activities=matched_work_activities(rec.preference),
    )


def _format_opportunity_dashboard_row(
    item: Job, rank: int, rec: LegacyRecommendation
) -> DashboardOpportunityRow:
    """Compact opportunity row for static comparison dashboards (embedded JSON in HTML).

    Unlike :func:`_format_opportunity`, includes *all* essential skill match rows in ``ms``,
    with ``meets_threshold`` driving the fourth tuple element (dashboard shows ✓/✗ per job skill).
    """
    breakdown = _score_breakdown(rec)
    essentials = rec.match_details.essential_skill_matches
    ph = breakdown.p_hat_components
    sc = breakdown.skill_components or SkillComponents()
    return DashboardOpportunityRow(
        r=rank,
        t=item.opportunity_title,
        e=item.employer or "",
        l=item.location,
        el=rec.skill.is_eligible,
        f=round(rec.score, 4),
        u=round(rec.u_hat, 4),
        p=round(rec.propensity.p_hat, 4),
        pg=round(ph.gate, 4) if ph else 0.0,
        pe=round(ph.essential_fit, 4) if ph else 0.0,
        pr=round(ph.recruiter_readiness, 4) if ph else 0.0,
        pm=round(ph.market_opportunity, 4) if ph else 0.0,
        sl=round(sc.loc or 0.0, 4),
        se=round(sc.ess or 0.0, 4),
        so=round(sc.opt or 0.0, 4),
        sg=round(sc.grp or 0.0, 4),
        ms=[
            (
                m.best_user_skill_label or "",
                m.job_skill_label or "",
                round(m.similarity, 4),
                m.meets_threshold,
            )
            for m in essentials
        ],
        j=_justification(rec),
        url=item.url or "",
        je=[m.job_skill_label or "" for m in essentials],
        jo=[s.label.strip() for s in item.optional_skills if s.label.strip()],
    )


def _format_occupation(
    item: Occupation, rank: int, rec: LegacyRecommendation
) -> OccupationRecommendation:
    """Format a single occupation recommendation."""
    return OccupationRecommendation(
        uuid=item.uuid,
        originUuid=item.originUuid,
        rank=rank,
        occupation_label=item.occupation_label,
        province=item.province,
        is_eligible=rec.skill.is_eligible,
        justification=_justification(rec),
        occupation_description=item.description,
        final_score=round(rec.score, 4),
        score_breakdown=_score_breakdown(rec, include_demand=True),
        matched_skills=_matched_skills(rec),
        matched_preferences=matched_preferences(rec.preference),
        matched_work_activities=matched_work_activities(rec.preference),
    )


def _score_items(
    user: UserProfile,
    items: Sequence[MatchItem],
    item_type: ItemType,
    top_k: int,
) -> tuple[list[LegacyRecommendation], MatchItemsTiming]:
    """Filter, score and keep the top ``top_k`` items for one user (formatting time left 0).

    Returns (recommendations, timing). Timing includes per-stage ms for u_hat (preference),
    skill+feasibility (embedding pair feeding p_hat), and p_hat (SuccessPropensityScorer) in
    multiplicative mode.
    """
    uid = user.user_id or "?"
    n_in = len(items)
    corpus = "occupations" if item_type == "occupation" else "jobs"

    t0 = time.perf_counter()
    with observability.stage("shortlist", corpus=corpus, n_candidates=n_in) as sl:
        kept: list[MatchItem] = list(items)
        if MATCH_APPLY_LOCATION_FILTER:
            where = Place.of_user(user)
            kept = [
                item
                for item in kept
                if job_matches_user_location(Place.of_item(item), where)
            ]
        # Post-secondary education gate (no-op for occupations, which lack the field).
        kept = filter_jobs_by_education(user, kept)
        observability.update_observation(sl, metadata={"n_after_filter": len(kept)})
    filter_ms = _ms(t0)

    scoring_mode = SCORING_MODE
    if not kept:
        log_match_step(
            "matching_service",
            "match_items (no matches after location filter)",
            user_id=uid,
            item_type=item_type,
            n_in=n_in,
            n_after_filter=0,
            filter_ms=filter_ms,
        )
        return [], MatchItemsTiming(
            item_type=item_type,
            n_in=n_in,
            n_after_filter=0,
            filter_ms=filter_ms,
            n_scored=0,
            scoring_mode=scoring_mode,
            u_hat_ms=0.0,
            skill_feasibility_ms=0.0,
            p_hat_ms=0.0,
            legacy_skill_ms=0.0,
            legacy_demand_ms=0.0,
            total_scoring_ms=0.0,
            format_recommendations_ms=0.0,
        )

    demand_scorer = DemandScorer() if scoring_mode != "multiplicative" else None

    with observability.stage(
        "preference_scoring", corpus=corpus, scoring_mode=scoring_mode
    ):
        t_score = time.perf_counter()
        sum_u = sum_sk = sum_p = sum_leg_s = sum_leg_d = 0.0
        recommendations: list[LegacyRecommendation] = []
        for item in kept:
            t_u = time.perf_counter()
            pref = scorer_pref.calculate_score(user, item)
            sum_u += _ms(t_u)

            if demand_scorer is None:
                # Single embedding / matmul pass for U + feasibility (p_hat)
                t_sk = time.perf_counter()
                skill, feasibility = scorer_skill.score_utility_and_feasibility(
                    user, item
                )
                sum_sk += _ms(t_sk)
                t_p = time.perf_counter()
                propensity = scorer_success.calculate_score(item, feasibility)
                sum_p += _ms(t_p)

                recommendations.append(
                    LegacyRecommendation(
                        item=item,
                        score=pref.u_hat * propensity.p_hat,
                        scoring_mode=scoring_mode,
                        u_hat=pref.u_hat,
                        propensity=propensity,
                        skill=skill,
                        match_details=feasibility.match_details,
                        preference=pref,
                        demand_label=propensity.demand_label,
                    )
                )
            else:
                # Legacy additive: skill utility only (no feasibility / p_hat)
                t_s = time.perf_counter()
                skill = scorer_skill.calculate_score(user, item)
                sum_leg_s += _ms(t_s)
                t_d = time.perf_counter()
                demand = demand_scorer.calculate_score(item)
                sum_leg_d += _ms(t_d)

                w1 = GLOBAL_WEIGHTS["w1_skills"]
                w2 = GLOBAL_WEIGHTS["w2_preference"]
                w3 = GLOBAL_WEIGHTS["w3_market"]

                if not demand.present:
                    # Demand absent — redistribute its weight to skills + prefs
                    remaining = w1 + w2
                    if remaining > 0:
                        w1 = w1 / remaining
                        w2 = w2 / remaining
                    w3 = 0.0
                    demand_score_val = 0.0
                else:
                    demand_score_val = demand.score

                demand_label = demand.label or "Unknown"
                recommendations.append(
                    LegacyRecommendation(
                        item=item,
                        score=w1 * skill.U_final
                        + w2 * pref.score
                        + w3 * demand_score_val,
                        scoring_mode=scoring_mode,
                        u_hat=pref.u_hat,
                        propensity=SuccessPropensity(
                            p_hat=0.0,
                            components=PHatComponents(),
                            demand_label=demand_label,
                        ),
                        skill=skill,
                        match_details=skill.match_details,
                        preference=pref,
                        demand_score=demand_score_val,
                        demand_label=demand_label,
                    )
                )

        # Sort by final score; use demand as tie-breaker in multiplicative mode
        if scoring_mode == "multiplicative":
            recommendations.sort(
                key=lambda x: (x.score, x.propensity.components.market_opportunity),
                reverse=True,
            )
        else:
            recommendations.sort(key=lambda x: x.score, reverse=True)

        # Keep only top_k so we release large per-item detail for the rest
        recommendations = recommendations[:top_k]
    total_scoring_ms = _ms(t_score)

    return recommendations, MatchItemsTiming(
        item_type=item_type,
        n_in=n_in,
        n_after_filter=len(kept),
        filter_ms=filter_ms,
        n_scored=len(kept),
        scoring_mode=scoring_mode,
        u_hat_ms=round(sum_u, 2),
        skill_feasibility_ms=round(sum_sk, 2),
        p_hat_ms=round(sum_p, 2),
        legacy_skill_ms=round(sum_leg_s, 2),
        legacy_demand_ms=round(sum_leg_d, 2),
        total_scoring_ms=round(total_scoring_ms, 2),
        format_recommendations_ms=0.0,
    )


def _match_opportunities(
    user: UserProfile, jobs: Sequence[Job], top_k: int
) -> tuple[list[OpportunityRecommendation], MatchItemsTiming]:
    recs, timing = _score_items(user, jobs, "opportunity", top_k)
    with observability.stage("formatting", corpus="jobs"):
        t_fmt = time.perf_counter()
        out = [
            _format_opportunity(r.item, i, r)
            for i, r in enumerate(recs, 1)
            if isinstance(r.item, Job)
        ]
    return out, timing.model_copy(
        update={"format_recommendations_ms": round(_ms(t_fmt), 2)}
    )


def _match_occupations(
    user: UserProfile, occupations: Sequence[Occupation], top_k: int
) -> tuple[list[OccupationRecommendation], MatchItemsTiming]:
    recs, timing = _score_items(user, occupations, "occupation", top_k)
    with observability.stage("formatting", corpus="occupations"):
        t_fmt = time.perf_counter()
        out = [
            _format_occupation(r.item, i, r)
            for i, r in enumerate(recs, 1)
            if isinstance(r.item, Occupation)
        ]
    return out, timing.model_copy(
        update={"format_recommendations_ms": round(_ms(t_fmt), 2)}
    )


def match_user_with_data(
    user: UserProfile,
    jobs: Sequence[Job],
    occupations: Sequence[Occupation],
) -> MatchResponse:
    """Synchronous matching using pre-fetched job and occupation lists (CPU-bound; run in a thread from async).

    :raises ValueError: If the user has no ``user_id``
    """
    if not user.user_id:
        raise ValueError("user must include user_id")

    uid = user.user_id
    t_total = time.perf_counter()

    if not jobs and not occupations:
        log_match_step(
            "matching_service",
            "match_user_with_data (empty input)",
            user_id=uid,
            n_jobs=0,
            n_occ=0,
            total_ms=_ms(t_total),
        )
        return MatchResponse(user_id=uid)

    t0 = time.perf_counter()
    opportunity_recommendations, _opp_timing = _match_opportunities(
        user, jobs, MATCH_TOP_K_OPPORTUNITIES
    )
    t_opp = _ms(t0)

    t0 = time.perf_counter()
    occupation_recommendations, _occ_timing = _match_occupations(
        user, occupations, MATCH_TOP_K_OCCUPATIONS
    )
    t_occ = _ms(t0)

    t0 = time.perf_counter()
    with observability.stage("skill_gaps"):
        skill_gaps = _filter_skill_gap_recommendations(
            analyze_skill_gaps(
                user,
                jobs,
                scorer_skill.engine,
                scorer_skill.skill_labels,
                top_k=_skill_gap_candidate_pool_k(MATCH_TOP_K_SKILL_GAPS),
                resolve_id=scorer_skill._resolve_label,
                rescale_target=SKILL_RESCALE_TARGET,
            ),
            top_k=MATCH_TOP_K_SKILL_GAPS,
        )
    t_gaps = _ms(t0)

    log_match_step(
        "matching_service",
        "match_user_with_data (phase totals vs pipeline)",
        user_id=uid,
        n_jobs=len(jobs),
        n_occupation_rows=len(occupations),
        opportunities_ms=t_opp,
        occupations_ms=t_occ,
        skill_gaps_ms=t_gaps,
        match_pipeline_total_ms=_ms(t_total),
    )

    return MatchResponse(
        user_id=uid,
        occupation_recommendations=occupation_recommendations,
        opportunity_recommendations=opportunity_recommendations,
        skill_gap_recommendations=skill_gaps,
    )


def match_user_opportunities_for_dashboard(
    user: UserProfile,
    jobs: Sequence[Job],
    top_k: int | None = None,
) -> list[DashboardOpportunityRow]:
    """Run opportunity matching and return compact rows for ``comparison_dashboard``-style HTML."""
    k = top_k if top_k is not None else MATCH_TOP_K_OPPORTUNITIES
    recs, _timing = _score_items(user, jobs, "opportunity", k)
    return [
        _format_opportunity_dashboard_row(r.item, i, r)
        for i, r in enumerate(recs, 1)
        if isinstance(r.item, Job)
    ]


async def match_single_user(
    user: UserProfile,
    jobs: Sequence[Job],
    occupations: Sequence[Occupation],
) -> MatchResponse:
    """:func:`match_user_with_data` off the event loop."""
    return await asyncio.to_thread(match_user_with_data, user, jobs, occupations)
