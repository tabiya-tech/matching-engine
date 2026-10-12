"""Pure formatters: ranked candidates -> ``POST /match`` recommendation models.

No heavy deps (torch/Gemini) so these are unit-testable in isolation. They map a ranked item (its
``ScoredCandidate`` from the v4 ranker: u_hat, p_hat, final_score, preference contributions) plus
the per-skill gate detail (``SkillDetail``) into ``OpportunityRecommendation`` /
``OccupationRecommendation`` (best-effort fidelity; Node2Vec-only fields left null).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.artifacts.repository import get_artifacts_repository
from app.domain.job import Job
from app.domain.occupation import Occupation
from app.domain.scores import ScoreBreakdown, SkillComponents
from app.ranking.preference import level_label
from app.ranking.scoring import (
    DemandScorer,
    build_matched_skills,
    essential_coverage,
    is_eligible_from_skills,
    skill_match_level,
)
from app.ranking.types import (
    AttributePreference,
    MatchItem,
    PreferenceScore,
    ScoredCandidate,
    SkillDetail,
    SkillMatchDetail,
)
from app.schemas import (
    MatchedPreference,
    MatchedSkills,
    MatchedWorkActivity,
    OccupationRecommendation,
    OpportunityRecommendation,
    WorkActivityBWS,
)

# Engine-agnostic demand scorer (reads item attributes.expected_demand); torch-free.
_DEMAND_SCORER = DemandScorer()

# Max skills listed in a justification (the strongest by cosine similarity). Keeps the sentence
# readable when an item has many matched essential skills (real occupations can have 40+).
_MAX_JUSTIFICATION_SKILLS = 10


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def matched_preferences(pref: PreferenceScore | None) -> list[MatchedPreference]:
    """The response rows of the scorer's per-attribute contributions (empty without a score)."""
    if pref is None:
        return []
    return [
        MatchedPreference(
            attribute=r.attribute,
            job_value=r.job_value,
            job_value_label=r.job_value_label,
            user_weight=r.user_weight,
            beta=r.beta,
            encoded_value=r.encoded_value,
            contribution=r.contribution,
            matched=r.matched,
        )
        for r in pref.attributes
    ]


def matched_work_activities(pref: PreferenceScore | None) -> WorkActivityBWS | None:
    """The response view of the task (BWS) utility; None when it was not scored."""
    wa = pref.work_activities if pref is not None else None
    if wa is None:
        return None
    return WorkActivityBWS(
        wa_score_sum=wa.wa_score_sum,
        details=[
            MatchedWorkActivity(
                wa_code=d.wa_code,
                wa_label=d.wa_label,
                user_bws=d.user_bws,
                wa_importance=d.wa_importance,
                wa_level=d.wa_level,
                norm_importance=d.norm_importance,
                norm_level=d.norm_level,
                wa_contribution=d.wa_contribution,
                weight=d.weight,
                beta=d.beta,
            )
            for d in wa.wa_details
        ],
        wa_aggregation=wa.wa_aggregation,
        n_work_activities=wa.n_work_activities,
        V_task=wa.V_task,
        V_task_hat=wa.V_task_hat,
    )


def _skill_components_from_cosine(
    per_job_skill: Sequence[SkillMatchDetail],
    essential_ids: frozenset[str],
    *,
    sim_threshold: float,
) -> tuple[SkillComponents, float | None, float | None]:
    """Approximate the legacy Node2Vec skill breakdown from v4 per-skill cosines.

    Reuses the SAME cosine signals v4 already computes (``per_job_skill[].cosine_similarity``,
    clamped to [0,1]), so every numeric field is in [0,1]:
      - ``ess``: mean best cosine over essential skills (higher = better fit)
      - ``opt``: mean best cosine over optional (non-essential) rows
      - ``loc``/``grp``: ``None`` — v4 computes neither a location-similarity nor group recall
        (explicitly null, not 0, so "not computed" is never read as "no match")
    Direction matches the legacy engine (utility/fit higher = better; penalty higher = worse),
    but the magnitudes are v4-derived approximations, not identical to the Node2Vec values.

    :return: ``(components, total_skill_utility, skill_penalty_applied)``: the essential-fit
        aggregate (skill-fit proxy, not ``p_hat``) and the share of essentials below threshold
    """
    ess_sims: list[float] = []
    opt_sims: list[float] = []
    for r in per_job_skill:
        if not r.job_skill_id:
            continue
        sim = _clamp01(r.cosine_similarity)
        (ess_sims if r.job_skill_id in essential_ids else opt_sims).append(sim)

    ess = round(sum(ess_sims) / len(ess_sims), 4) if ess_sims else None
    opt = round(sum(opt_sims) / len(opt_sims), 4) if opt_sims else None
    penalty = (
        round(
            _clamp01(sum(1 for s in ess_sims if s < sim_threshold) / len(ess_sims)), 4
        )
        if ess_sims
        else None
    )
    return SkillComponents(loc=None, ess=ess, opt=opt, grp=None), ess, penalty


def _score_breakdown(
    scored: ScoredCandidate | None,
    item: MatchItem,
    detail: SkillDetail,
    *,
    sim_threshold: float,
    essential_coverage: float,
    skill_match_level: str,
) -> ScoreBreakdown:
    """The response breakdown of one ranked item, from the ranker's breakdown and the skill detail."""
    components, total_skill_utility, penalty = _skill_components_from_cosine(
        detail.per_job_skill, detail.essential_ids, sim_threshold=sim_threshold
    )
    demand = _DEMAND_SCORER.calculate_score(item)
    sb = scored.breakdown if scored is not None else None
    return ScoreBreakdown(
        u_hat=scored.preference.u_hat if scored else None,
        p_hat=scored.p_hat if scored else None,
        # 'concat_cosine_whitened' when Phase-2 demotion is on
        p_hat_source=scored.p_hat_source if scored else None,
        preference_score=scored.preference.u_hat if scored else None,
        preference_score_legacy=sb.preference_score_legacy if sb else None,
        # v4 cosine approximations of the legacy Node2Vec skill breakdown (all in [0,1];
        # loc/grp null = not computed). See _skill_components_from_cosine.
        total_skill_utility=total_skill_utility,
        skill_components=components,
        skill_penalty_applied=penalty,
        # Engine-agnostic demand (reads item attributes.expected_demand); null when absent.
        demand_score=demand.score if demand.present else None,
        demand_label=demand.label if demand.present else None,
        essential_coverage=round(essential_coverage, 4),
        skill_match_level=skill_match_level,
        # The demotions the ranker actually applied, so final_score is reproducible from the
        # breakdown: final = u_hat x p_hat x coverage_factor x location_tier_factor. Null when the
        # corresponding stage did not run (Phase-2 off / occupations / tiering disabled).
        coverage_gamma=sb.coverage_gamma if sb else None,
        coverage_factor=sb.coverage_factor if sb else None,
        location_tier_factor=sb.location_tier_factor if sb else None,
    )


def _salary_range(item: Occupation) -> str | None:
    """Occupation salary as the earnings-level label (e.g. 'earn_70k' -> '~70k')."""
    earn = item.attributes.earnings_per_month
    if not earn:
        return None
    lbl = level_label(
        "earnings_per_month", earn, get_artifacts_repository().load_attribute_schema()
    )
    return lbl if lbl and lbl != "—" else None


def _split_included_tasks(raw: str) -> list[str]:
    """Split freeform ``included_tasks`` text into a task list.

    Source uses lettered enumerations like ``"Tasks include -\\r\\n(a) ...; (b) ..."``: split on
    the ``(a)/(b)/...`` markers, else on newlines/semicolons; drop a leading "Tasks include".
    """
    text = (raw or "").replace("\r\n", "\n").strip()
    parts = re.split(r"\s*\([a-z]\)\s*", text)
    items = parts[1:] if len(parts) > 1 else re.split(r"[\n;]+", text)
    out: list[str] = []
    for it in items:
        s = re.sub(
            r"^tasks include\s*[-:]?\s*", "", it.strip(), flags=re.IGNORECASE
        ).strip(" .;-")
        if s:
            out.append(s)
    return out


def _typical_tasks(item: Occupation, *, max_tasks: int = 8) -> list[str]:
    """Occupation tasks: prefer occupation-specific ``included_tasks``, else top O*NET WA labels."""
    if item.included_tasks.strip():
        tasks = _split_included_tasks(item.included_tasks)
        if tasks:
            return tasks[:max_tasks]
    wa_sorted = sorted(
        (w for w in item.onet_work_activities if w.label),
        key=lambda w: w.importance,
        reverse=True,
    )
    out: list[str] = []
    seen: set[str] = set()
    for w in wa_sorted:
        lab = w.label.strip()
        if lab and lab not in seen:
            seen.add(lab)
            out.append(lab)
        if len(out) >= max_tasks:
            break
    return out


def _natural_list(items: Sequence[str]) -> str:
    """['a','b','c'] -> 'a, b and c'; drops blanks. Empty list -> ''."""
    items = [str(i).strip() for i in items if str(i).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _humanize_demand(label: str | None) -> str | None:
    """'Very High Expected Demand' -> 'very high demand'; None when absent/unknown."""
    if not label:
        return None
    t = str(label).strip()
    if t.lower().endswith(" expected demand"):
        t = t[: -len(" expected demand")].strip()
    return f"{t.lower()} demand" if t else None


def _justification(
    matched_skills: MatchedSkills,
    preferences: Sequence[AttributePreference],
    item: MatchItem,
    *,
    kind: str = "role",
) -> str:
    """Deterministic, second-person justification built from the match facts (no LLM).

    Draws on the user's own matched skills, the preferences they actually expressed (with the
    item's value), and the item's expected demand. Each part is omitted when absent, with a
    graceful fallback when nothing matched.
    """
    sentences: list[str] = []

    # 1. Skills — list the top matched essential skills by cosine similarity (deduped), up to
    # _MAX_JUSTIFICATION_SKILLS; prefer the user's own skill label, fall back to the item's.
    matched_ess = sorted(
        (m for m in matched_skills.essential_skill_matches if m.meets_threshold),
        key=lambda m: m.similarity,
        reverse=True,
    )
    names: list[str] = []
    seen: set[str] = set()
    for m in matched_ess:
        nm = m.best_user_skill_label or m.job_skill_label or m.job_skill_id
        if nm and nm not in seen:
            seen.add(nm)
            names.append(nm)
        if len(names) >= _MAX_JUSTIFICATION_SKILLS:
            break
    if names:
        sentences.append(f"Strong match on your {_natural_list(names)} skills.")

    # 2. Preferences the user expressed (all of them), with the item's value where available.
    parts: list[str] = []
    for p in preferences:
        if not p.matched:
            continue
        lab = (p.attr_label or p.attribute).strip().lower()
        val = p.job_value_label
        parts.append(f"{lab} ({val})" if val and val not in ("", "—") else lab)
    phrase = _natural_list(parts)
    if phrase:
        sentences.append(f"It fits your preferences for {phrase}.")

    # 3. Labour-market demand (occupations carry this; jobs usually don't).
    demand = _humanize_demand(item.attributes.expected_demand)
    if demand:
        sentences.append(f"This {kind} is in {demand}.")

    if not sentences:
        return f"Recommended {kind} based on your overall profile."
    return " ".join(sentences)


def _skill_view(
    item: MatchItem,
    detail: SkillDetail,
    *,
    sim_threshold: float,
    min_ess_share: float,
) -> tuple[MatchedSkills, float, bool]:
    """(matched skills, essential coverage, eligibility) of one item."""
    matched_skills = build_matched_skills(
        detail.per_job_skill, detail.essential_ids, sim_threshold=sim_threshold
    )
    n_ess_total = len(item.essential_skills)
    coverage = essential_coverage(matched_skills.essential_skill_matches, n_ess_total)
    eligible = is_eligible_from_skills(
        matched_skills.essential_skill_matches,
        n_essential_total=n_ess_total,
        min_ess_share=min_ess_share,
    )
    return matched_skills, coverage, eligible


def build_opportunity_row(
    item: Job,
    detail: SkillDetail,
    *,
    rank: int,
    final_score: float,
    scored: ScoredCandidate | None,
    sim_threshold: float,
    min_ess_share: float,
) -> OpportunityRecommendation:
    """One opportunity of the response.

    :param item: The job
    :param detail: Per-skill gate detail of (user, job)
    :param rank: 1-based rank in the user's opportunities
    :param final_score: The ranking score
    :param scored: The v4 ranker's scoring of the job; None for engines without preference
        scoring (u_hat/p_hat/preferences are then null/empty)
    :param sim_threshold: Per-skill similarity threshold for "meets"
    :param min_ess_share: Essential coverage needed to be eligible
    """
    matched_skills, coverage, eligible = _skill_view(
        item, detail, sim_threshold=sim_threshold, min_ess_share=min_ess_share
    )
    prefs = matched_preferences(scored.preference if scored else None)
    return OpportunityRecommendation(
        uuid=item.uuid,
        originUuid=item.originUuid,
        URL=item.url or f"www.example.com/{item.uuid}",
        rank=rank,
        opportunity_title=item.opportunity_title,
        opportunity_isco_occupation_group=item.opportunity_isco_occupation_group,
        opportunity_isco_occupation_group_id=item.opportunity_isco_occupation_group_id,
        related_occupation_id=item.related_occupation_id,
        location=item.location,
        employer=item.employer,
        employment_type=item.employment_type,
        salary_text=item.salary_text,
        closing_date=item.closing_date,
        posted_date=item.posted_date,
        is_eligible=eligible,
        justification=_justification(
            matched_skills, scored.preference.attributes if scored else [], item
        ),
        opportunity_description=item.opportunity_description or item.contract_type,
        contract_type=item.contract_type,
        final_score=round(final_score, 4),
        score_breakdown=_score_breakdown(
            scored,
            item,
            detail,
            sim_threshold=sim_threshold,
            essential_coverage=coverage,
            skill_match_level=skill_match_level(coverage, len(item.essential_skills)),
        ),
        matched_skills=matched_skills,
        matched_preferences=prefs,
        matched_work_activities=matched_work_activities(
            scored.preference if scored else None
        ),
    )


def build_occupation_row(
    item: Occupation,
    detail: SkillDetail,
    *,
    rank: int,
    final_score: float,
    scored: ScoredCandidate | None,
    sim_threshold: float,
    min_ess_share: float,
) -> OccupationRecommendation:
    """One occupation of the response (parameters as ``build_opportunity_row``)."""
    matched_skills, coverage, eligible = _skill_view(
        item, detail, sim_threshold=sim_threshold, min_ess_share=min_ess_share
    )
    prefs = matched_preferences(scored.preference if scored else None)
    return OccupationRecommendation(
        uuid=item.uuid,
        originUuid=item.originUuid,
        rank=rank,
        occupation_label=item.occupation_label,
        province=item.province,
        is_eligible=eligible,
        justification=_justification(
            matched_skills, scored.preference.attributes if scored else [], item
        ),
        occupation_description=item.description,
        salary_range=_salary_range(item),
        typical_tasks=_typical_tasks(item),
        career_path_next_steps=[],  # no source in occupation data — see plan
        final_score=round(final_score, 4),
        score_breakdown=_score_breakdown(
            scored,
            item,
            detail,
            sim_threshold=sim_threshold,
            essential_coverage=coverage,
            skill_match_level=skill_match_level(coverage, len(item.essential_skills)),
        ),
        matched_skills=matched_skills,
        matched_preferences=prefs,
        matched_work_activities=matched_work_activities(
            scored.preference if scored else None
        ),
    )
