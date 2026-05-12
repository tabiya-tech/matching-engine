"""
Skill Gap Analysis

Identifies skills that would be most useful for a user to learn based on:
1. Proximity to their existing skills (in embedding space)
2. Number of jobs that would be unlocked or improved by learning the skill
"""

import time
import numpy as np
from typing import List, Dict, Optional, Set

from app.config import SKILL_RESCALE_TARGET

def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def analyze_skill_gaps(
    user_profile: dict,
    all_jobs: List[dict],
    engine,  # SimilarityEngine
    skill_labels: Dict[str, str],
    top_k: int = 5,
    resolve_id=None,
    timing_out: Optional[Dict] = None,
) -> List[Dict]:
    """
    Analyzes skill gaps for a user by finding skills that:
    1. Are close to their existing skills in embedding space
    2. Would unlock or improve matches with jobs in the database
    
    Args:
        user_profile: User profile dict with skills_vector.top_skills
        all_jobs: List of all job postings (demand.jsonl)
        engine: SimilarityEngine instance with embedding matrix
        skill_labels: Dict mapping skill IDs to preferred labels
        top_k: Number of recommendations to return
        resolve_id: Optional callable to translate external IDs (ESCO)
                     to the embedding model's internal ID space.
    
    Returns:
        List of dicts with skill_id, skill_label, proximity_score, 
        job_unlock_count, and reasoning
    """
    t_total = time.perf_counter()
    uid = str(user_profile.get("user_id", "?"))
    n_jobs = len(all_jobs)
    # `resolve_id` is the SkillScorer's label-primary resolver. The historical
    # parameter name predates the UUID→label switch; it now takes a preferredLabel.
    _resolve = resolve_id if resolve_id is not None else (lambda x: x)

    # 1. Extract user's existing skills (resolved to internal IDs via label).
    user_skill_ids: Set[str] = set()
    user_skill_labels: Dict[str, str] = {}
    for s in user_profile.get("skills_vector", {}).get("top_skills", []):
        label = s.get("preferredLabel")
        if not label:
            continue
        resolved = _resolve(label)
        if resolved is None:
            continue
        user_skill_ids.add(resolved)
        user_skill_labels[resolved] = label
    
    if not user_skill_ids:
        if timing_out is not None:
            timing_out.update(
                {
                    "total_ms": _ms(t_total),
                    "n_recommendations": 0,
                    "n_candidates": 0,
                    "n_jobs": n_jobs,
                    "top_k": top_k,
                    "skip_reason": "no_resolvable_user_skills",
                }
            )
        return []

    # 2. Collect all candidate skills from all jobs, resolved to internal IDs
    candidate_skills: Set[str] = set()
    # Track original counts per resolved ID across jobs
    _ess_sets: List[Set[str]] = []
    _opt_sets: List[Set[str]] = []
    for job in all_jobs:
        ess_resolved = {s["id"] for s in job.get("essential_skills", []) if isinstance(s, dict) and s.get("id")}
        opt_resolved = {s["id"] for s in job.get("optional_skills", []) if isinstance(s, dict) and s.get("id")}
        _ess_sets.append(ess_resolved)
        _opt_sets.append(opt_resolved)
        candidate_skills.update(ess_resolved | opt_resolved)
    
    candidate_skills -= user_skill_ids
    
    if not candidate_skills:
        if timing_out is not None:
            timing_out.update(
                {
                    "total_ms": _ms(t_total),
                    "n_recommendations": 0,
                    "n_candidates": 0,
                    "n_user_skills": len(user_skill_ids),
                    "n_jobs": n_jobs,
                    "top_k": top_k,
                    "skip_reason": "no_candidates_after_user_set",
                }
            )
        return []

    # 3. Compute proximity score for each candidate. Apply per-cosine rescaling so
    # proximity_score lives in the same calibrated frame as the matching service's
    # essential_fit / final_score (whitened cosines compress otherwise). When rescaling
    # is disabled (target<=0), rescaled == raw and behaviour is identical.
    user_mat = engine._rows(list(user_skill_ids))
    user_skill_ids_list = list(user_skill_ids)

    _target = SKILL_RESCALE_TARGET
    _rescale_enabled = _target > 0.0
    def _rescale_value(v: float) -> float:
        return min(1.0, v / _target) if _rescale_enabled else v

    proximity_scores: Dict[str, float] = {}
    closest_user_skills: Dict[str, tuple] = {}
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
    job_unlock_counts: Dict[str, int] = {}
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
    
    combined_scores = {}
    for cand_id in candidate_skills:
        prox = proximity_scores.get(cand_id, 0.0)
        unlock = job_unlock_counts.get(cand_id, 0) / max(max_unlock, 1)
        
        # Weighted average: 40% proximity, 60% job unlock potential
        combined = 0.4 * prox + 0.6 * unlock
        combined_scores[cand_id] = combined
    
    # 6. Sort by combined score and return top_k
    sorted_candidates = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    recommendations = []
    for cand_id, combined_score in sorted_candidates:
        prox = proximity_scores.get(cand_id, 0.0)
        unlock_count = job_unlock_counts.get(cand_id, 0)
        skill_label = skill_labels.get(str(cand_id), str(cand_id))
        closest_user_skill_id, closest_sim = closest_user_skills.get(cand_id, (None, 0.0))
        closest_user_skill_label = user_skill_labels.get(closest_user_skill_id) if closest_user_skill_id else None
        
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
        
        recommendations.append({
            "skill_id": str(cand_id),
            "skill_label": skill_label,
            "proximity_score": round(float(prox), 4),
            "job_unlock_count": int(unlock_count),
            "combined_score": round(float(combined_score), 4),
            "reasoning": reasoning
        })
    
    total_ms = _ms(t_total)
    if timing_out is not None:
        timing_out.update(
            {
                "total_ms": total_ms,
                "n_recommendations": len(recommendations),
                "n_candidates": len(candidate_skills),
                "n_jobs": n_jobs,
                "top_k": top_k,
                "skip_reason": None,
            }
        )
    return recommendations
