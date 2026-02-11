"""
Skill Gap Analysis

Identifies skills that would be most useful for a user to learn based on:
1. Proximity to their existing skills (in embedding space)
2. Number of jobs that would be unlocked or improved by learning the skill
"""

import numpy as np
from typing import List, Dict, Optional, Set


def analyze_skill_gaps(
    user_profile: dict,
    all_jobs: List[dict],
    engine,  # SimilarityEngine
    skill_labels: Dict[str, str],
    top_k: int = 5
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
    
    Returns:
        List of dicts with skill_id, skill_label, proximity_score, 
        job_unlock_count, and reasoning
    """
    
    # 1. Extract user's existing skills
    user_skill_ids = {
        s.get("originUUID")
        for s in user_profile.get("skills_vector", {}).get("top_skills", [])
        if s.get("originUUID")
    }
    user_skill_labels = {
        s.get("originUUID"): s.get("preferredLabel")
        for s in user_profile.get("skills_vector", {}).get("top_skills", [])
        if s.get("originUUID")
    }
    
    if not user_skill_ids:
        return []
    
    # 2. Collect all candidate skills from all jobs (excluding user's skills)
    candidate_skills = set()
    for job in all_jobs:
        ess_skills = set(job.get("essential_skills_origin_uuids", []))
        opt_skills = set(job.get("optional_skills_origin_uuids", []))
        candidate_skills.update(ess_skills | opt_skills)
    
    # Remove user's existing skills
    candidate_skills = candidate_skills - user_skill_ids
    
    if not candidate_skills:
        return []
    
    # 3. Compute proximity score for each candidate
    #    (similarity to user's closest existing skill)
    user_mat = engine._rows(list(user_skill_ids))
    user_skill_ids_list = list(user_skill_ids)
    
    proximity_scores = {}
    closest_user_skills = {}  # Maps candidate_id to (user_skill_id, similarity)
    for cand_id in candidate_skills:
        cand_mat = engine._rows([cand_id])
        if cand_mat.size == 0 or user_mat.size == 0:
            proximity_scores[cand_id] = 0.0
            closest_user_skills[cand_id] = (None, 0.0)
        else:
            S = cand_mat @ user_mat.T  # Shape: (1, num_user_skills)
            argmax_idx = int(np.argmax(S))
            max_sim = float(np.max(S))
            closest_user_skill_id = user_skill_ids_list[argmax_idx]
            proximity_scores[cand_id] = max(0.0, max_sim)
            closest_user_skills[cand_id] = (closest_user_skill_id, max(0.0, max_sim))
    
    # 4. Compute job unlock potential for each candidate
    job_unlock_counts = {}
    for cand_id in candidate_skills:
        unlock_count = 0
        for job in all_jobs:
            ess_skills = set(job.get("essential_skills_origin_uuids", []))
            opt_skills = set(job.get("optional_skills_origin_uuids", []))
            
            # Count if skill is essential (would improve eligibility/score significantly)
            if cand_id in ess_skills:
                unlock_count += 2  # Weight essential skills more
            # Count if skill is optional (would improve score)
            elif cand_id in opt_skills:
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
    
    return recommendations
