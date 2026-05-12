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
    SKILL_ESSENTIAL_DAMPING_ALPHA,
    SKILL_ESSENTIAL_GEO_FLOOR,
    SKILL_MIN_ESSENTIAL_MATCH_SHARE,
    SKILL_RESCALE_TARGET,
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
    essential_skill_ids: Set[str]
    optional_skill_ids: Set[str]
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
    ess_ids = list(op.essential_skill_ids)
    opt_ids = list(op.optional_skill_ids)

    js_mat, js_ids_valid = engine._rows_with_ids(js_ids)
    ess_mat, ess_ids_valid = engine._rows_with_ids(ess_ids)

    # Rescaling is opt-in: SKILL_RESCALE_TARGET > 0 enables per-rowmax rescaling.
    # When disabled (raw Gemini / Node2Vec artefacts have no target in metadata),
    # rowmax_rescaled is just rowmax — every downstream consumer using the rescaled
    # field gets the raw value transparently, so the system behaves as if rescaling
    # weren't there. Whitened artefacts persist target_max_p999 in metadata and
    # SkillScorer hydrates SKILL_RESCALE_TARGET at startup.
    target = SKILL_RESCALE_TARGET
    rescale_enabled = target > 0.0

    def _rescale(arr):
        if not rescale_enabled:
            return arr
        return np.minimum(1.0, arr / target)

    S = None
    rowmax = None                  # raw rowmaxes (identity at 1.0); preserved for diagnostic / response
    rowmax_rescaled = None         # rescaled rowmaxes; used by every comparator in the new wiring
    argmax = None
    ess_sim = 0.0                  # mean of rescaled rowmaxes (used by U_complete)
    ess_geo = 0.0                  # score-weighted GM of rescaled rowmaxes (used by p_hat / feasibility)
    ess_sim_raw = 0.0              # diagnostic: mean of un-rescaled rowmaxes
    ess_geo_raw = 0.0              # diagnostic: score-weighted GM of un-rescaled rowmaxes
    identity_coverage = 0.0        # diagnostic: fraction of essentials matched exactly by user

    if ess_mat.size > 0 and js_mat.size > 0:
        S = ess_mat @ js_mat.T
        np.maximum(S, 0.0, out=S)
        rowmax = S.max(axis=1)
        argmax = S.argmax(axis=1)

        # Diagnostic: identity coverage = fraction of essentials where user has the exact
        # skill ID. Recorded for audit only; the GM math uses rescaled rowmaxes uniformly
        # so identity contributes naturally at its proportion in the data.
        ess_arr = np.array(ess_ids_valid)
        js_arr = np.array(js_ids_valid)
        identity_mask = ess_arr[:, None] == js_arr[None, :]
        identity_coverage = float(identity_mask.any(axis=1).sum()) / float(len(ess_ids_valid))

        # Diagnostic: the un-rescaled GM (pre-rescaling), for comparison with rescaled.
        ess_sim_raw = float(rowmax.mean())
        floored_raw = np.maximum(rowmax, geo_floor)
        log_floored_raw = np.log(floored_raw)
        if SKILL_ESSENTIAL_DAMPING_ALPHA == 0.0:
            ess_geo_raw = float(np.exp(log_floored_raw.mean()))
        else:
            w_raw = floored_raw ** SKILL_ESSENTIAL_DAMPING_ALPHA
            denom_raw = float(w_raw.sum())
            ess_geo_raw = float(np.exp((w_raw * log_floored_raw).sum() / denom_raw)) if denom_raw > 0 else float(geo_floor)

        # Per-rowmax rescaling. target is the empirical "saturation point" of non-identity
        # cosines (e.g. p99.9 over random pairs from the embedding). Identity rowmaxes (=1.0)
        # clip to 1.0 unchanged; non-identity rowmaxes in [0, target] stretch into [0, 1].
        # Score-weighted GM operates on the uniform-scale rescaled distribution so identity
        # and strong-related sit at the top together (bimodality flattens at the GM input).
        rowmax_rescaled = _rescale(rowmax)
        ess_sim = float(rowmax_rescaled.mean())
        floored = np.maximum(rowmax_rescaled, geo_floor)
        log_floored = np.log(floored)
        if SKILL_ESSENTIAL_DAMPING_ALPHA == 0.0:
            ess_geo = float(np.exp(log_floored.mean()))
        else:
            w = floored ** SKILL_ESSENTIAL_DAMPING_ALPHA
            denom = float(w.sum())
            ess_geo = float(np.exp((w * log_floored).sum() / denom)) if denom > 0 else float(geo_floor)
    elif ess_ids_valid and js_mat.size == 0:
        rowmax = np.zeros(len(ess_ids_valid), dtype=np.float64)
        rowmax_rescaled = rowmax

    # Optional skills: same rowmax-style approach as essentials (not mean-of-centroids).
    # Mean-of-centroids on whitened embeddings degenerates because whitening spreads
    # vectors so any centroid approaches zero magnitude and renormalises into direction
    # noise. Per-optional rowmax → rescale → mean keeps optional in the same calibrated
    # frame as essential. Each optional skill contributes its best user-skill cosine.
    opt_mat, opt_ids_valid = engine._rows_with_ids(opt_ids)
    if opt_mat.size > 0 and js_mat.size > 0:
        S_opt = opt_mat @ js_mat.T
        np.maximum(S_opt, 0.0, out=S_opt)
        rowmax_opt_raw = S_opt.max(axis=1)
        rowmax_opt_rescaled = _rescale(rowmax_opt_raw)
        opt_sim = float(rowmax_opt_rescaled.mean())
    else:
        opt_sim = 0.0

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
        "rowmax": rowmax,                  # raw rowmax (preserved for diagnostic / response)
        "rowmax_rescaled": rowmax_rescaled,  # rescaled rowmax (used by all gate / GM / mean comparators)
        "argmax": argmax,
        "ess_sim": ess_sim,
        "ess_geo": ess_geo,
        "opt_sim": opt_sim,
        "grp_sim": grp_sim,
        "identity_coverage": identity_coverage,
        "ess_sim_raw": ess_sim_raw,
        "ess_geo_raw": ess_geo_raw,
    }


def _build_essential_match_list(
    ess_mat, js_mat, S, rowmax, rowmax_rescaled, argmax, ess_ids_valid, js_ids_valid,
    skill_labels, user_skill_labels, threshold: float
):
    """Match-list payload for the response. `similarity` is in rescaled space (the
    same frame as final_score, p_hat, essential_fit), so a human reader sees one
    consistent scale. `similarity_raw` is the un-rescaled whitened cosine, kept for
    audit / debugging. `meets_threshold` compares the rescaled value, since the
    threshold lives in the same rescaled frame as all the gate decisions.
    """
    essential_skill_matches = []
    if ess_mat.size > 0 and js_mat.size > 0 and rowmax is not None and argmax is not None:
        # rowmax_rescaled is None only if the kernel skipped rescaling; in that case
        # fall back to raw (rescaling disabled => values are already in the right frame).
        rescaled = rowmax_rescaled if rowmax_rescaled is not None else rowmax
        for i, ess_id in enumerate(ess_ids_valid):
            best_idx = int(argmax[i])
            best_js_id = js_ids_valid[best_idx] if js_ids_valid else None
            sim_rescaled = float(rescaled[i])
            sim_raw = float(rowmax[i])
            essential_skill_matches.append({
                "job_skill_id": ess_id,
                "job_skill_label": skill_labels.get(ess_id),
                "best_user_skill_id": best_js_id,
                "best_user_skill_label": user_skill_labels.get(best_js_id),
                "similarity": round(sim_rescaled, 4),
                "similarity_raw": round(sim_raw, 4),
                "meets_threshold": sim_rescaled >= threshold,
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
    ess_rowmax_raw = k["rowmax"]                # raw (preserved for diagnostic)
    ess_rowmax_rescaled = k["rowmax_rescaled"]  # rescaled (used for gap_share / eligibility)
    ess_mat, js_mat = k["ess_mat"], k["js_mat"]
    js_ids, opt_ids = k["js_ids"], k["opt_ids"]
    ess_ids_valid, js_ids_valid = k["ess_ids_valid"], k["js_ids_valid"]
    S, argmax = k["S"], k["argmax"]

    core_score = (
        W_LOC * loc_score + W_ESS * ess_sim + W_OPT * opt_sim + W_GRP * grp_sim
    ) / (W_LOC + W_ESS + W_OPT + W_GRP)

    # gap_share / eligibility evaluated against rescaled rowmax so the TAU_ELIG threshold
    # lives in the same calibrated frame as ess_sim and final_score. When rescaling is
    # disabled (no artefact metadata), rowmax_rescaled == rowmax, so behaviour is identical
    # to before.
    rowmax_for_gates = ess_rowmax_rescaled if ess_rowmax_rescaled is not None else ess_rowmax_raw
    if rowmax_for_gates is None or (hasattr(rowmax_for_gates, "size") and rowmax_for_gates.size == 0):
        gap_share, eligible = 0.0, True
    else:
        meets = (rowmax_for_gates >= TAU_ELIG)
        gap_share = float((~meets).mean())
        eligible = (float(meets.mean()) >= MIN_ESS_SHARE)

    u_final = max(0.0, core_score - (W_GAP_PEN * gap_share))

    essential_skill_matches = _build_essential_match_list(
        ess_mat, js_mat, S, ess_rowmax_raw, ess_rowmax_rescaled, argmax,
        ess_ids_valid, js_ids_valid,
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
    ess_rowmax_raw = k["rowmax"]                # raw (preserved for diagnostic)
    ess_rowmax_rescaled = k["rowmax_rescaled"]  # rescaled (used for gate / gap_share)
    ess_geo = k["ess_geo"]
    opt_sim = k["opt_sim"]
    grp_recall = k["grp_sim"]
    ess_mat, js_mat = k["ess_mat"], k["js_mat"]
    ess_ids_valid, js_ids_valid = k["ess_ids_valid"], k["js_ids_valid"]
    S, argmax = k["S"], k["argmax"]

    # Gate evaluated against rescaled rowmax so GATE_SIMILARITY_THRESHOLD lives in the
    # same calibrated frame as essential_fit and final_score. When rescaling is disabled
    # (no artefact metadata), rowmax_rescaled == rowmax, so behaviour matches the
    # pre-rescaling system.
    rowmax_for_gates = ess_rowmax_rescaled if ess_rowmax_rescaled is not None else ess_rowmax_raw
    has_essential_reqs = len(ess_ids) > 0
    if not has_essential_reqs:
        gap_share = 1.0
        gate_passed = False
        ess_geo = 0.0
    elif rowmax_for_gates is None or rowmax_for_gates.size == 0:
        gap_share = 1.0
        gate_passed = False
        ess_geo = 0.0
    else:
        meets = rowmax_for_gates >= gate_threshold
        gap_share = float((~meets).mean())
        gate_passed = bool(gap_share <= 0.5)

    essential_skill_matches = _build_essential_match_list(
        ess_mat, js_mat, S, ess_rowmax_raw, ess_rowmax_rescaled, argmax,
        ess_ids_valid, js_ids_valid,
        skill_labels, user_skill_labels, gate_threshold,
    )
    optional_exact_matches, skill_group_matches = _optional_and_group_match_lists(
        op, js, opt_ids, js_ids, skill_labels, user_skill_labels, skill_group_labels,
    )

    return {
        "gate_passed": gate_passed,
        "essential_fit": round(ess_geo, 4),
        "essential_fit_raw": round(k.get("ess_geo_raw", 0.0), 4),
        "identity_coverage": round(k.get("identity_coverage", 0.0), 4),
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
    ess_rowmax_raw = k["rowmax"]                # raw (preserved for diagnostic)
    ess_rowmax_rescaled = k["rowmax_rescaled"]  # rescaled (used for both u-side and f-side gates)
    ess_mat, js_mat = k["ess_mat"], k["js_mat"]
    ess_ids, opt_ids, js_ids = k["ess_ids"], k["opt_ids"], k["js_ids"]
    ess_ids_valid, js_ids_valid = k["ess_ids_valid"], k["js_ids_valid"]
    S, argmax = k["S"], k["argmax"]

    core_score = (
        W_LOC * loc_score + W_ESS * ess_sim + W_OPT * opt_sim + W_GRP * grp_sim
    ) / (W_LOC + W_ESS + W_OPT + W_GRP)

    # u-side gap: TAU_ELIG against rescaled rowmax (calibrated frame).
    rowmax_for_gates = ess_rowmax_rescaled if ess_rowmax_rescaled is not None else ess_rowmax_raw
    if rowmax_for_gates is None or (hasattr(rowmax_for_gates, "size") and rowmax_for_gates.size == 0):
        u_gap, eligible = 0.0, True
    else:
        meets = (rowmax_for_gates >= TAU_ELIG)
        u_gap = float((~meets).mean())
        eligible = (float(meets.mean()) >= MIN_ESS_SHARE)

    u_final = max(0.0, core_score - (W_GAP_PEN * u_gap))

    if abs(TAU_ELIG - gate_threshold) >= 1e-9:
        u_ess = _build_essential_match_list(
            ess_mat, js_mat, S, ess_rowmax_raw, ess_rowmax_rescaled, argmax,
            ess_ids_valid, js_ids_valid,
            skill_labels, user_skill_labels, TAU_ELIG,
        )
        f_ess = _build_essential_match_list(
            ess_mat, js_mat, S, ess_rowmax_raw, ess_rowmax_rescaled, argmax,
            ess_ids_valid, js_ids_valid,
            skill_labels, user_skill_labels, gate_threshold,
        )
    else:
        u_ess = f_ess = _build_essential_match_list(
            ess_mat, js_mat, S, ess_rowmax_raw, ess_rowmax_rescaled, argmax,
            ess_ids_valid, js_ids_valid,
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

    # f-side gate: gate_threshold against rescaled rowmax.
    if not (len(ess_ids) > 0):
        f_gap = 1.0
        gate_passed = False
        f_ess_geo = 0.0
    elif rowmax_for_gates is None or rowmax_for_gates.size == 0:
        f_gap = 1.0
        gate_passed = False
        f_ess_geo = 0.0
    else:
        meets = rowmax_for_gates >= gate_threshold
        f_gap = float((~meets).mean())
        gate_passed = bool(f_gap <= 0.5)
        f_ess_geo = ess_geo

    feasibility = {
        "gate_passed": gate_passed,
        "essential_fit": round(f_ess_geo, 4),
        "essential_fit_raw": round(k.get("ess_geo_raw", 0.0), 4),
        "identity_coverage": round(k.get("identity_coverage", 0.0), 4),
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