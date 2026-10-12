"""
Skill Gap Analysis

Identifies skills that would be most useful for a user to learn based on:
1. Proximity to their existing skills (in embedding space)
2. Number of jobs that would be unlocked or improved by learning the skill
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

import numpy as np

from app.config import MATCH_RESPONSE_SKILL_MIN_SCORE, MATCH_TOP_K_SKILL_GAPS
from app.domain.user import UserProfile
from app.ranking.skills import SkillScorer
from app.ranking.types import MatchItem
from app.schemas import SkillGapRecommendation


class SkillRows(Protocol):
    """Row lookup into a skill-embedding matrix (``SimilarityEngine``)."""

    def _rows(self, skill_ids: Iterable[str]) -> np.ndarray: ...


def analyze_skill_gaps(
    user: UserProfile,
    items: Sequence[MatchItem],
    engine: SkillRows,
    skill_labels: dict[str, str],
    top_k: int = 5,
    resolve_id: Callable[[str], str | None] | None = None,
    rescale_target: float = 0.0,
) -> list[SkillGapRecommendation]:
    """
    Analyzes skill gaps for a user by finding skills that:
    1. Are close to their existing skills in embedding space
    2. Would unlock or improve matches with jobs in the database

    Args:
        user: The user (skills from ``skills_vector.top_skills``)
        items: The jobs (or occupations) whose skills are candidates
        engine: Embedding row lookup (``SkillScorer.engine``)
        skill_labels: Dict mapping skill IDs to preferred labels
        top_k: Number of recommendations to return
        resolve_id: Label -> embedding id resolver (``SkillScorer._resolve_label``); identity
                     when omitted.
        rescale_target: Per-cosine rescale target for proximity scores; <= 0 disables
                     rescaling.

    Returns:
        Up to ``top_k`` recommendations, best combined score first
    """

    def _identity(label: str) -> str | None:
        return label

    # `resolve_id` is the SkillScorer's label-primary resolver. The historical
    # parameter name predates the UUID→label switch; it now takes a preferredLabel.
    _resolve = resolve_id if resolve_id is not None else _identity

    # 1. Extract user's existing skills (resolved to internal IDs via label).
    user_skill_ids: set[str] = set()
    user_skill_labels: dict[str, str] = {}
    for s in user.skills_vector.top_skills:
        label = s.preferredLabel
        if not label:
            continue
        resolved = _resolve(label)
        if resolved is None:
            continue
        user_skill_ids.add(resolved)
        user_skill_labels[resolved] = label

    if not user_skill_ids:
        return []

    # 2. Collect all candidate skills from all jobs (their taxonomy ids)
    candidate_skills: set[str] = set()
    _ess_sets: list[set[str]] = []
    _opt_sets: list[set[str]] = []
    for item in items:
        ess_resolved = {s.id for s in item.essential_skills if s.id}
        opt_resolved = {s.id for s in item.optional_skills if s.id}
        _ess_sets.append(ess_resolved)
        _opt_sets.append(opt_resolved)
        candidate_skills.update(ess_resolved | opt_resolved)

    candidate_skills -= user_skill_ids

    if not candidate_skills:
        return []

    # 3. Compute proximity score for each candidate. Apply per-cosine rescaling so
    # proximity_score lives in the same calibrated frame as the matching service's
    # essential_fit / final_score (whitened cosines compress otherwise). When rescaling
    # is disabled (target<=0), rescaled == raw and behaviour is identical.
    user_skill_ids_list = list(user_skill_ids)
    user_mat = engine._rows(user_skill_ids_list)

    _target = rescale_target
    _rescale_enabled = _target > 0.0

    def _rescale_value(v: float) -> float:
        return min(1.0, v / _target) if _rescale_enabled else v

    proximity_scores: dict[str, float] = {}
    closest_user_skills: dict[str, tuple[str | None, float]] = {}
    for cand_id in candidate_skills:
        cand_mat = engine._rows([cand_id])
        if cand_mat.size == 0 or user_mat.size == 0:
            proximity_scores[cand_id] = 0.0
            closest_user_skills[cand_id] = (None, 0.0)
        else:
            S = cand_mat @ user_mat.T
            argmax_idx = int(np.argmax(S))
            max_sim_raw = float(np.max(S))
            max_sim = _rescale_value(max(0.0, max_sim_raw))
            closest_user_skill_id = user_skill_ids_list[argmax_idx]
            proximity_scores[cand_id] = max_sim
            closest_user_skills[cand_id] = (closest_user_skill_id, max_sim)

    # 4. Compute job unlock potential for each candidate (using pre-resolved sets)
    job_unlock_counts: dict[str, int] = {}
    for cand_id in candidate_skills:
        unlock_count = 0
        for ess_set, opt_set in zip(_ess_sets, _opt_sets):
            if cand_id in ess_set:
                unlock_count += 2
            elif cand_id in opt_set:
                unlock_count += 1
        job_unlock_counts[cand_id] = unlock_count

    # 5. Combine scores (normalized)
    #    proximity: 0-1 range already
    #    job_unlock: normalize by max
    max_unlock = max(job_unlock_counts.values()) if job_unlock_counts else 1

    combined_scores: dict[str, float] = {}
    for cand_id in candidate_skills:
        prox = proximity_scores.get(cand_id, 0.0)
        unlock = job_unlock_counts.get(cand_id, 0) / max(max_unlock, 1)

        # Weighted average: 40% proximity, 60% job unlock potential
        combined_scores[cand_id] = 0.4 * prox + 0.6 * unlock

    # 6. Sort by combined score and return top_k
    sorted_candidates = sorted(
        combined_scores.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    recommendations: list[SkillGapRecommendation] = []
    for cand_id, combined_score in sorted_candidates:
        prox = proximity_scores.get(cand_id, 0.0)
        unlock_count = job_unlock_counts.get(cand_id, 0)
        skill_label = skill_labels.get(cand_id, cand_id)
        closest_user_skill_id, closest_sim = closest_user_skills.get(
            cand_id, (None, 0.0)
        )
        closest_user_skill_label = (
            user_skill_labels.get(closest_user_skill_id)
            if closest_user_skill_id
            else None
        )

        # Create reasoning text
        if closest_user_skill_label and closest_sim > 0:
            reasoning = (
                f"Similar to your '{closest_user_skill_label}' skill. "
                f"Would help unlock or improve {unlock_count} job{'s' if unlock_count != 1 else ''}."
            )
        elif unlock_count > 0:
            job_text = "job" if unlock_count == 1 else "jobs"
            reasoning = f"Would help unlock or improve {unlock_count} {job_text}."
        else:
            reasoning = f"Close match to your existing skills (proximity: {prox:.2f})."

        recommendations.append(
            SkillGapRecommendation(
                skill_id=cand_id,
                skill_label=skill_label,
                proximity_score=round(float(prox), 4),
                job_unlock_count=int(unlock_count),
                combined_score=round(float(combined_score), 4),
                reasoning=reasoning,
            )
        )

    return recommendations


def skill_gap_candidate_pool_k(requested_top_k: int) -> int:
    """How many ranked candidates to generate before the proximity threshold filter.

    ``analyze_skill_gaps`` ranks by combined score (proximity + job unlock). A small
    ``top_k`` can surface a high-unlock skill whose proximity is below
    ``MATCH_RESPONSE_SKILL_MIN_SCORE``; widening the pool first lets the filter
    return up to ``requested_top_k`` rows that pass the threshold.
    """
    return max(requested_top_k * 10, MATCH_TOP_K_SKILL_GAPS, 20)


def filter_skill_gap_recommendations(
    skill_gaps: Sequence[SkillGapRecommendation], *, top_k: int | None = None
) -> list[SkillGapRecommendation]:
    """Keep only skill-gap rows whose proximity passes response threshold."""
    limit = MATCH_TOP_K_SKILL_GAPS if top_k is None else max(1, int(top_k))
    return [
        g for g in skill_gaps if g.proximity_score >= MATCH_RESPONSE_SKILL_MIN_SCORE
    ][:limit]


def skill_gaps_for(
    user: UserProfile,
    items: Sequence[MatchItem],
    top_k: int,
    *,
    scorer: SkillScorer,
    rescale_target: float,
) -> list[SkillGapRecommendation]:
    """Skill-gap analysis over ``items``, threshold-filtered to ``top_k`` rows (engine-agnostic)."""
    gaps = analyze_skill_gaps(
        user,
        items,
        scorer.engine,
        scorer.skill_labels,
        top_k=skill_gap_candidate_pool_k(top_k),
        resolve_id=scorer._resolve_label,
        rescale_target=rescale_target,
    )
    return filter_skill_gap_recommendations(gaps, top_k=top_k)
