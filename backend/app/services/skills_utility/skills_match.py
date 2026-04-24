#!/usr/bin/env python3
"""
CORE UTILITY CALCULATION ENGINE (U)

DESCRIPTION:
This script implements a Content-Based Utility Model designed to rank jobseekers 
against opportunities based on four core dimensions:

1. Essential Skill Proximity ($U_{ess}$): Uses the Best-Mean Cosine Similarity. 
   For every essential skill a job requires, the engine finds the youth's most 
   mathematically similar skill from the Node2Vec embedding space and averages 
   these "best matches".

2. Optional Skill Proximity ($U_{opt}$): Uses Mean-Vector Cosine Similarity. 
   It calculates the "centroid" (average vector) of all a seeker's skills and 
   compares it to the centroid of the job's optional skills to measure thematic 
   alignment.

3. Skill Group Recall ($U_{grp}$): A simple overlap metric that rewards 
   jobseekers for possessing skills within the specific competency groups 
   defined by the job.

4. Location Utility ($U_{loc}$): A geographic heuristic that assigns a score 
   of 1.0 for a city match, 0.7 for a province match, and 0.3 otherwise.

SCORING LOGIC:
The final score is a weighted average of these components, which is then 
subjected to a Soft Penalty. This penalty reduces the score based on the 
"Gap Share"—the percentage of essential skills that fall below a similarity 
threshold of 0.35.
"""

import numpy as np
from typing import Set, Dict, Optional
from dataclasses import dataclass

from app.config import (
    GATE_SIMILARITY_THRESHOLD,
    SKILL_ESSENTIAL_GEO_FLOOR,
    SKILL_MIN_ESSENTIAL_MATCH_SHARE,
    SKILL_U_GAP_PENALTY,
    SKILL_U_TAU_ELIG,
    SKILL_U_W_ESS,
    SKILL_U_W_GRP,
    SKILL_U_W_LOC,
    SKILL_U_W_OPT,
)

# =============================================================================
# 1) DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class Jobseeker:
    compass_id: str
    skills_origin_uuids: Set[str]
    skill_groups_origin_uuids: Set[str]
    city: Optional[str] = None
    province: Optional[str] = None

@dataclass(frozen=True)
class Opportunity:
    opportunity_id: str
    essential_skills_origin_uuids: Set[str]
    optional_skills_origin_uuids: Set[str]
    skill_groups_origin_uuids: Set[str]
    city: Optional[str] = None
    province: Optional[str] = None

# =============================================================================
# 2) COMPONENT HELPERS
# =============================================================================

def location_near_enough(js: Jobseeker, op: Opportunity) -> float:
    """Calculates geographic proximity score based on city/province."""
    s_city = (js.city or "").strip().casefold()
    s_prov = (js.province or "").strip().casefold()
    o_city = (op.city or "").strip().casefold()
    o_prov = (op.province or "").strip().casefold()
    
    if s_city and o_city and s_city == o_city:
        return 1.0  # Exact city match
    if s_prov and o_prov and s_prov == o_prov:
        return 0.7  # Province match
    return 0.3      # Baseline/No match

# =============================================================================
# 3) SIMILARITY ENGINE
# =============================================================================

class SimilarityEngine:
    """Handles the high-dimensional vector math for skill proximity."""
    def __init__(self, W: np.ndarray, skill_to_row: Dict[str, int]):
        self.W = W  # Row-normalised embedding matrix
        self.skill_to_row = skill_to_row

    def _rows(self, skill_ids) -> np.ndarray:
        """Retrieves vectors for specific skill IDs from the matrix."""
        idx = [self.skill_to_row[str(s)] for s in skill_ids if str(s) in self.skill_to_row]
        if not idx:
            return np.empty((0, self.W.shape[1]))
        return self.W[np.array(idx), :]

    def _rows_with_ids(self, skill_ids):
        """Retrieves vectors and aligned IDs for a list of skill IDs."""
        valid_ids = [str(s) for s in skill_ids if str(s) in self.skill_to_row]
        if not valid_ids:
            return np.empty((0, self.W.shape[1])), []
        idx = [self.skill_to_row[s] for s in valid_ids]
        return self.W[np.array(idx), :], valid_ids

    def _mean_unit(self, M: np.ndarray) -> Optional[np.ndarray]:
        """Calculates the unit-normalised mean vector of a set."""
        if M.size == 0: return None
        v = M.mean(axis=0)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else None


def _shared_pair_kernel(
    js: Jobseeker,
    op: Opportunity,
    engine: SimilarityEngine,
    geo_floor: Optional[float] = None,
):
    """Single essential similarity matmul and shared optional/location; used by U and p_hat.

    Returns a dict of intermediates so compute_U and feasibility can share one
    :math:`S = E @ J^\\top` and one match-detail list.
    """
    if geo_floor is None:
        geo_floor = SKILL_ESSENTIAL_GEO_FLOOR

    loc_score = location_near_enough(js, op)

    js_ids = list(js.skills_origin_uuids)
    ess_ids = list(op.essential_skills_origin_uuids)
    opt_ids = list(op.optional_skills_origin_uuids)

    js_mat, js_ids_valid = engine._rows_with_ids(js_ids)
    ess_mat, ess_ids_valid = engine._rows_with_ids(ess_ids)

    S = None
    rowmax = None
    argmax = None
    ess_sim = 0.0
    ess_geo = 0.0

    if ess_mat.size > 0 and js_mat.size > 0:
        S = ess_mat @ js_mat.T
        np.maximum(S, 0.0, out=S)
        rowmax = S.max(axis=1)
        argmax = S.argmax(axis=1)
        ess_sim = float(rowmax.mean())
        floored = np.maximum(rowmax, geo_floor)
        ess_geo = float(np.exp(np.mean(np.log(floored))))
    elif ess_ids_valid and js_mat.size == 0:
        rowmax = np.zeros(len(ess_ids_valid), dtype=np.float64)

    js_mean = engine._mean_unit(js_mat)
    opt_vec = engine._mean_unit(engine._rows(opt_ids))
    opt_sim = max(0.0, float(js_mean @ opt_vec)) if js_mean is not None and opt_vec is not None else 0.0

    if op.skill_groups_origin_uuids:
        inter = len(op.skill_groups_origin_uuids & js.skill_groups_origin_uuids)
        grp_sim = inter / len(op.skill_groups_origin_uuids)
    else:
        grp_sim = 0.0

    return {
        "loc_score": loc_score,
        "js_ids": js_ids,
        "ess_ids": ess_ids,
        "opt_ids": opt_ids,
        "js_mat": js_mat,
        "ess_mat": ess_mat,
        "js_ids_valid": js_ids_valid,
        "ess_ids_valid": ess_ids_valid,
        "S": S,
        "rowmax": rowmax,
        "argmax": argmax,
        "ess_sim": ess_sim,
        "ess_geo": ess_geo,
        "opt_sim": opt_sim,
        "grp_sim": grp_sim,
    }


def _build_essential_match_list(
    ess_mat, js_mat, S, rowmax, argmax, ess_ids_valid, js_ids_valid,
    skill_labels, user_skill_labels, threshold: float
):
    essential_skill_matches = []
    if ess_mat.size > 0 and js_mat.size > 0 and rowmax is not None and argmax is not None:
        for i, ess_id in enumerate(ess_ids_valid):
            best_idx = int(argmax[i])
            best_js_id = js_ids_valid[best_idx] if js_ids_valid else None
            sim = float(rowmax[i])
            essential_skill_matches.append({
                "job_skill_id": ess_id,
                "job_skill_label": skill_labels.get(ess_id),
                "best_user_skill_id": best_js_id,
                "best_user_skill_label": user_skill_labels.get(best_js_id),
                "similarity": round(sim, 4),
                "meets_threshold": sim >= threshold,
            })
    elif ess_ids_valid:
        for ess_id in ess_ids_valid:
            essential_skill_matches.append({
                "job_skill_id": ess_id,
                "job_skill_label": skill_labels.get(ess_id),
                "best_user_skill_id": None,
                "best_user_skill_label": None,
                "similarity": 0.0,
                "meets_threshold": False,
            })
    return essential_skill_matches


def _optional_and_group_match_lists(
    op: Opportunity, js: Jobseeker, opt_ids, js_ids, skill_labels, user_skill_labels, skill_group_labels,
):
    optional_exact_matches = [
        {
            "skill_id": opt_id,
            "skill_label": skill_labels.get(opt_id) or user_skill_labels.get(opt_id),
        }
        for opt_id in sorted(set(opt_ids) & set(js_ids))
    ]
    skill_group_matches = []
    for gid in sorted(op.skill_groups_origin_uuids & js.skill_groups_origin_uuids):
        skill_group_matches.append({
            "skill_group_id": gid,
            "skill_group_label": skill_group_labels.get(gid)
        })
    return optional_exact_matches, skill_group_matches


# =============================================================================
# 4) CORE UTILITY CALCULATION
# =============================================================================

def compute_U_complete(
    js: Jobseeker,
    op: Opportunity,
    engine: SimilarityEngine,
    skill_labels: Optional[Dict[str, str]] = None,
    user_skill_labels: Optional[Dict[str, str]] = None,
    skill_group_labels: Optional[Dict[str, str]] = None
):
    """Calculates the final Utility score with components and penalties.

    Uses a single essential similarity matmul via :func:`_shared_pair_kernel`.
    """
    W_LOC, W_ESS, W_OPT, W_GRP, W_GAP_PEN = (
        SKILL_U_W_LOC, SKILL_U_W_ESS, SKILL_U_W_OPT, SKILL_U_W_GRP, SKILL_U_GAP_PENALTY,
    )
    TAU_ELIG = SKILL_U_TAU_ELIG
    MIN_ESS_SHARE = SKILL_MIN_ESSENTIAL_MATCH_SHARE

    skill_labels = skill_labels or {}
    user_skill_labels = user_skill_labels or {}
    skill_group_labels = skill_group_labels or {}

    k = _shared_pair_kernel(js, op, engine)
    loc_score = k["loc_score"]
    ess_sim = k["ess_sim"]
    opt_sim = k["opt_sim"]
    grp_sim = k["grp_sim"]
    ess_rowmax = k["rowmax"]
    ess_mat, js_mat = k["ess_mat"], k["js_mat"]
    js_ids, opt_ids = k["js_ids"], k["opt_ids"]
    ess_ids_valid, js_ids_valid = k["ess_ids_valid"], k["js_ids_valid"]
    S, argmax = k["S"], k["argmax"]

    core_score = (
        W_LOC * loc_score + W_ESS * ess_sim + W_OPT * opt_sim + W_GRP * grp_sim
    ) / (W_LOC + W_ESS + W_OPT + W_GRP)

    if ess_rowmax is None or (hasattr(ess_rowmax, "size") and ess_rowmax.size == 0):
        gap_share, eligible = 0.0, True
    else:
        meets = (ess_rowmax >= TAU_ELIG)
        gap_share = float((~meets).mean())
        eligible = (float(meets.mean()) >= MIN_ESS_SHARE)

    u_final = max(0.0, core_score - (W_GAP_PEN * gap_share))

    essential_skill_matches = _build_essential_match_list(
        ess_mat, js_mat, S, ess_rowmax, argmax, ess_ids_valid, js_ids_valid,
        skill_labels, user_skill_labels, TAU_ELIG,
    )
    optional_exact_matches, skill_group_matches = _optional_and_group_match_lists(
        op, js, opt_ids, js_ids, skill_labels, user_skill_labels, skill_group_labels,
    )

    return {
        "U_final": round(u_final, 4),
        "is_eligible": eligible,
        "components": {
            "loc": round(loc_score, 2),
            "ess": round(ess_sim, 4),
            "opt": round(opt_sim, 4),
            "grp": round(grp_sim, 4)
        },
        "penalty": round(W_GAP_PEN * gap_share, 4),
        "match_details": {
            "essential_skill_matches": essential_skill_matches,
            "optional_exact_matches": optional_exact_matches,
            "skill_group_matches": skill_group_matches
        }
    }


# =============================================================================
# 5) FEASIBILITY SIGNALS  (recruiter-side, feeds into p_hat)
# =============================================================================

def compute_feasibility_signals(
    js: Jobseeker,
    op: Opportunity,
    engine: SimilarityEngine,
    gate_threshold: Optional[float] = None,
    skill_labels: Optional[Dict[str, str]] = None,
    user_skill_labels: Optional[Dict[str, str]] = None,
    skill_group_labels: Optional[Dict[str, str]] = None,
):
    """Recruiter-side feasibility; shares :func:`_shared_pair_kernel` with U."""
    if gate_threshold is None:
        gate_threshold = GATE_SIMILARITY_THRESHOLD

    skill_labels = skill_labels or {}
    user_skill_labels = user_skill_labels or {}
    skill_group_labels = skill_group_labels or {}

    k = _shared_pair_kernel(js, op, engine)
    ess_ids = k["ess_ids"]
    opt_ids = k["opt_ids"]
    js_ids = k["js_ids"]
    ess_rowmax = k["rowmax"]
    ess_geo = k["ess_geo"]
    opt_sim = k["opt_sim"]
    grp_recall = k["grp_sim"]
    ess_mat, js_mat = k["ess_mat"], k["js_mat"]
    ess_ids_valid, js_ids_valid = k["ess_ids_valid"], k["js_ids_valid"]
    S, argmax = k["S"], k["argmax"]

    has_essential_reqs = len(ess_ids) > 0
    if not has_essential_reqs:
        gap_share = 1.0
        gate_passed = False
        ess_geo = 0.0
    elif ess_rowmax is None or ess_rowmax.size == 0:
        gap_share = 1.0
        gate_passed = False
        ess_geo = 0.0
    else:
        meets = ess_rowmax >= gate_threshold
        gap_share = float((~meets).mean())
        gate_passed = bool(gap_share <= 0.5)

    essential_skill_matches = _build_essential_match_list(
        ess_mat, js_mat, S, ess_rowmax, argmax, ess_ids_valid, js_ids_valid,
        skill_labels, user_skill_labels, gate_threshold,
    )
    optional_exact_matches, skill_group_matches = _optional_and_group_match_lists(
        op, js, opt_ids, js_ids, skill_labels, user_skill_labels, skill_group_labels,
    )

    return {
        "gate_passed": gate_passed,
        "essential_fit": round(ess_geo, 4),
        "optional_sim": round(opt_sim, 4),
        "skill_group_recall": round(grp_recall, 4),
        "gap_share": round(gap_share, 4),
        "has_essential_skills": len(ess_ids) > 0,
        "has_optional_skills": len(opt_ids) > 0,
        "has_skill_groups": len(op.skill_groups_origin_uuids) > 0,
        "match_details": {
            "essential_skill_matches": essential_skill_matches,
            "optional_exact_matches": optional_exact_matches,
            "skill_group_matches": skill_group_matches,
        },
    }


def compute_utility_and_feasibility_pair(
    js: Jobseeker,
    op: Opportunity,
    engine: SimilarityEngine,
    skill_labels: Optional[Dict[str, str]] = None,
    user_skill_labels: Optional[Dict[str, str]] = None,
    skill_group_labels: Optional[Dict[str, str]] = None,
    gate_threshold: Optional[float] = None,
) -> tuple:
    """One essential matmul and one optional/group pass for the multiplicative pipeline."""
    if gate_threshold is None:
        gate_threshold = GATE_SIMILARITY_THRESHOLD
    W_LOC, W_ESS, W_OPT, W_GRP, W_GAP_PEN = (
        SKILL_U_W_LOC, SKILL_U_W_ESS, SKILL_U_W_OPT, SKILL_U_W_GRP, SKILL_U_GAP_PENALTY,
    )
    TAU_ELIG = SKILL_U_TAU_ELIG
    MIN_ESS_SHARE = SKILL_MIN_ESSENTIAL_MATCH_SHARE

    skill_labels = skill_labels or {}
    user_skill_labels = user_skill_labels or {}
    skill_group_labels = skill_group_labels or {}

    k = _shared_pair_kernel(js, op, engine)
    loc_score = k["loc_score"]
    ess_sim = k["ess_sim"]
    ess_geo = k["ess_geo"]
    opt_sim = k["opt_sim"]
    grp_sim = k["grp_sim"]
    ess_rowmax = k["rowmax"]
    ess_mat, js_mat = k["ess_mat"], k["js_mat"]
    ess_ids, opt_ids, js_ids = k["ess_ids"], k["opt_ids"], k["js_ids"]
    ess_ids_valid, js_ids_valid = k["ess_ids_valid"], k["js_ids_valid"]
    S, argmax = k["S"], k["argmax"]

    core_score = (
        W_LOC * loc_score + W_ESS * ess_sim + W_OPT * opt_sim + W_GRP * grp_sim
    ) / (W_LOC + W_ESS + W_OPT + W_GRP)

    if ess_rowmax is None or (hasattr(ess_rowmax, "size") and ess_rowmax.size == 0):
        u_gap, eligible = 0.0, True
    else:
        meets = (ess_rowmax >= TAU_ELIG)
        u_gap = float((~meets).mean())
        eligible = (float(meets.mean()) >= MIN_ESS_SHARE)

    u_final = max(0.0, core_score - (W_GAP_PEN * u_gap))

    if abs(TAU_ELIG - gate_threshold) >= 1e-9:
        u_ess = _build_essential_match_list(
            ess_mat, js_mat, S, ess_rowmax, argmax, ess_ids_valid, js_ids_valid,
            skill_labels, user_skill_labels, TAU_ELIG,
        )
        f_ess = _build_essential_match_list(
            ess_mat, js_mat, S, ess_rowmax, argmax, ess_ids_valid, js_ids_valid,
            skill_labels, user_skill_labels, gate_threshold,
        )
    else:
        u_ess = f_ess = _build_essential_match_list(
            ess_mat, js_mat, S, ess_rowmax, argmax, ess_ids_valid, js_ids_valid,
            skill_labels, user_skill_labels, TAU_ELIG,
        )

    opt_m, grp_m = _optional_and_group_match_lists(
        op, js, opt_ids, js_ids, skill_labels, user_skill_labels, skill_group_labels,
    )

    utility = {
        "U_final": round(u_final, 4),
        "is_eligible": eligible,
        "components": {
            "loc": round(loc_score, 2),
            "ess": round(ess_sim, 4),
            "opt": round(opt_sim, 4),
            "grp": round(grp_sim, 4),
        },
        "penalty": round(W_GAP_PEN * u_gap, 4),
        "match_details": {
            "essential_skill_matches": u_ess,
            "optional_exact_matches": opt_m,
            "skill_group_matches": grp_m,
        },
    }

    if not (len(ess_ids) > 0):
        f_gap = 1.0
        gate_passed = False
        f_ess_geo = 0.0
    elif ess_rowmax is None or ess_rowmax.size == 0:
        f_gap = 1.0
        gate_passed = False
        f_ess_geo = 0.0
    else:
        meets = ess_rowmax >= gate_threshold
        f_gap = float((~meets).mean())
        gate_passed = bool(f_gap <= 0.5)
        f_ess_geo = ess_geo

    feasibility = {
        "gate_passed": gate_passed,
        "essential_fit": round(f_ess_geo, 4),
        "optional_sim": round(opt_sim, 4),
        "skill_group_recall": round(grp_sim, 4),
        "gap_share": round(f_gap, 4),
        "has_essential_skills": len(ess_ids) > 0,
        "has_optional_skills": len(opt_ids) > 0,
        "has_skill_groups": len(op.skill_groups_origin_uuids) > 0,
        "match_details": {
            "essential_skill_matches": f_ess,
            "optional_exact_matches": opt_m,
            "skill_group_matches": grp_m,
        },
    }
    return utility, feasibility