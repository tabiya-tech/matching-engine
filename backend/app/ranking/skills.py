"""Skill embeddings: label resolution across language packs, the per-skill cosine matcher, and the utility / feasibility kernel behind SkillScorer."""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import Field

from app.config import (
    GATE_SIMILARITY_THRESHOLD,
    SKILL_ESSENTIAL_DAMPING_ALPHA,
    SKILL_ESSENTIAL_GEO_FLOOR,
    SKILL_MIN_ESSENTIAL_MATCH_SHARE,
    SKILL_U_GAP_PENALTY,
    SKILL_U_TAU_ELIG,
    SKILL_U_W_ESS,
    SKILL_U_W_GRP,
    SKILL_U_W_LOC,
    SKILL_U_W_OPT,
    taxonomy_pack_paths,
)
from app.domain.base import DomainModel
from app.domain.skills import SkillReference
from app.domain.user import UserProfile
from app.languages import CANONICAL_LANGUAGE, enabled_languages
from app.ranking.types import MatchItem, SkillMatchDetail, SkillPairScore
from app.schemas import (
    MatchedSkill,
    MatchedSkills,
    OptionalSkillMatch,
    SkillGroupMatch,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# skills_match
#
# CORE UTILITY CALCULATION ENGINE (U)
#
# DESCRIPTION:
# This script implements a Content-Based Utility Model designed to rank jobseekers
# against opportunities based on four core dimensions:
#
# 1. Essential Skill Proximity ($U_{ess}$): Uses the Best-Mean Cosine Similarity.
#    For every essential skill a job requires, the engine finds the youth's most
#    mathematically similar skill from the Node2Vec embedding space and averages
#    these "best matches".
#
# 2. Optional Skill Proximity ($U_{opt}$): Uses Mean-Vector Cosine Similarity.
#    It calculates the "centroid" (average vector) of all a seeker's skills and
#    compares it to the centroid of the job's optional skills to measure thematic
#    alignment.
#
# 3. Skill Group Recall ($U_{grp}$): A simple overlap metric that rewards
#    jobseekers for possessing skills within the specific competency groups
#    defined by the job.
#
# 4. Location Utility ($U_{loc}$): A geographic heuristic that assigns a score
#    of 1.0 for a city match, 0.7 for a province match, and 0.3 otherwise.
#
# SCORING LOGIC:
# The final score is a weighted average of these components, which is then
# subjected to a Soft Penalty. This penalty reduces the score based on the
# "Gap Share"—the percentage of essential skills that fall below a similarity
# threshold of 0.35.
# --------------------------------------------------------------------------------------------------

#!/usr/bin/env python3


# =============================================================================
# 1) DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class Jobseeker:
    """A user as the legacy skill-utility kernel sees it: resolved skill and group ids, location."""

    compass_id: str
    skills_origin_uuids: set[str]
    skill_groups_origin_uuids: set[str]
    city: str | None = None
    province: str | None = None


@dataclass(frozen=True)
class Opportunity:
    """An item as the legacy skill-utility kernel sees it: resolved skill and group ids, location."""

    opportunity_id: str
    essential_skill_ids: set[str]
    optional_skill_ids: set[str]
    skill_groups_origin_uuids: set[str]
    city: str | None = None
    province: str | None = None


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
    return 0.3  # Baseline/No match


# =============================================================================
# 3) SIMILARITY ENGINE
# =============================================================================


class SimilarityEngine:
    """Handles the high-dimensional vector math for skill proximity."""

    def __init__(
        self, W: np.ndarray, skill_to_row: dict[str, int], rescale_target: float = 0.0
    ):
        self.W = W  # Row-normalised embedding matrix
        self.skill_to_row = skill_to_row
        # Per-rowmax rescale target (> 0 enables rescaling; see _shared_pair_kernel).
        self.rescale_target = float(rescale_target)

    def _rows(self, skill_ids: Iterable[str]) -> np.ndarray:
        """Retrieves vectors for specific skill IDs from the matrix."""
        idx = [
            self.skill_to_row[str(s)] for s in skill_ids if str(s) in self.skill_to_row
        ]
        if not idx:
            return np.empty((0, self.W.shape[1]))
        return self.W[np.array(idx), :]

    def _rows_with_ids(self, skill_ids: Iterable[str]) -> tuple[np.ndarray, list[str]]:
        """Retrieves vectors and aligned IDs for a list of skill IDs."""
        valid_ids = [str(s) for s in skill_ids if str(s) in self.skill_to_row]
        if not valid_ids:
            return np.empty((0, self.W.shape[1])), []
        idx = [self.skill_to_row[s] for s in valid_ids]
        return self.W[np.array(idx), :], valid_ids

    def _mean_unit(self, M: np.ndarray) -> np.ndarray | None:
        """Calculates the unit-normalised mean vector of a set."""
        if M.size == 0:
            return None
        v = M.mean(axis=0)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else None


@dataclass(frozen=True)
class _PairKernel:
    """Intermediates of one (jobseeker, opportunity) pass, shared by U and feasibility."""

    loc_score: float
    js_ids: list[str]
    ess_ids: list[str]
    opt_ids: list[str]
    js_mat: np.ndarray
    ess_mat: np.ndarray
    js_ids_valid: list[str]
    ess_ids_valid: list[str]
    S: np.ndarray | None
    rowmax: np.ndarray | None  # raw rowmax (preserved for diagnostic / response)
    rowmax_rescaled: (
        np.ndarray | None
    )  # rescaled rowmax (used by all gate / GM / mean comparators)
    argmax: np.ndarray | None
    ess_sim: float
    ess_geo: float
    opt_sim: float
    grp_sim: float
    identity_coverage: float
    ess_sim_raw: float
    ess_geo_raw: float


def _shared_pair_kernel(
    js: Jobseeker,
    op: Opportunity,
    engine: SimilarityEngine,
    geo_floor: float | None = None,
) -> _PairKernel:
    """Single essential similarity matmul and shared optional/location; used by U and p_hat.

    Returns the intermediates so compute_U and feasibility can share one
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

    # Rescaling is opt-in: engine.rescale_target > 0 enables per-rowmax rescaling.
    # When disabled (raw Gemini / Node2Vec artefacts have no target in metadata),
    # rowmax_rescaled is just rowmax — every downstream consumer using the rescaled
    # field gets the raw value transparently, so the system behaves as if rescaling
    # weren't there. Whitened artefacts persist target_max_p999 in metadata, which the
    # engine is built with unless the operator overrides the target in the environment.
    target = engine.rescale_target
    rescale_enabled = target > 0.0

    def _rescale(arr: np.ndarray) -> np.ndarray:
        if not rescale_enabled:
            return arr
        return np.minimum(1.0, arr / target)

    S = None
    rowmax = None  # raw rowmaxes (identity at 1.0); preserved for diagnostic / response
    rowmax_rescaled = (
        None  # rescaled rowmaxes; used by every comparator in the new wiring
    )
    argmax = None
    ess_sim = 0.0  # mean of rescaled rowmaxes (used by U_complete)
    ess_geo = (
        0.0  # score-weighted GM of rescaled rowmaxes (used by p_hat / feasibility)
    )
    ess_sim_raw = 0.0  # diagnostic: mean of un-rescaled rowmaxes
    ess_geo_raw = 0.0  # diagnostic: score-weighted GM of un-rescaled rowmaxes
    identity_coverage = (
        0.0  # diagnostic: fraction of essentials matched exactly by user
    )

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
        identity_coverage = float(identity_mask.any(axis=1).sum()) / float(
            len(ess_ids_valid)
        )

        # Diagnostic: the un-rescaled GM (pre-rescaling), for comparison with rescaled.
        ess_sim_raw = float(rowmax.mean())
        floored_raw = np.maximum(rowmax, geo_floor)
        log_floored_raw = np.log(floored_raw)
        if SKILL_ESSENTIAL_DAMPING_ALPHA == 0.0:
            ess_geo_raw = float(np.exp(log_floored_raw.mean()))
        else:
            w_raw = floored_raw**SKILL_ESSENTIAL_DAMPING_ALPHA
            denom_raw = float(w_raw.sum())
            ess_geo_raw = (
                float(np.exp((w_raw * log_floored_raw).sum() / denom_raw))
                if denom_raw > 0
                else float(geo_floor)
            )

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
            w = floored**SKILL_ESSENTIAL_DAMPING_ALPHA
            denom = float(w.sum())
            ess_geo = (
                float(np.exp((w * log_floored).sum() / denom))
                if denom > 0
                else float(geo_floor)
            )
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

    return _PairKernel(
        loc_score=loc_score,
        js_ids=js_ids,
        ess_ids=ess_ids,
        opt_ids=opt_ids,
        js_mat=js_mat,
        ess_mat=ess_mat,
        js_ids_valid=js_ids_valid,
        ess_ids_valid=ess_ids_valid,
        S=S,
        rowmax=rowmax,
        rowmax_rescaled=rowmax_rescaled,
        argmax=argmax,
        ess_sim=ess_sim,
        ess_geo=ess_geo,
        opt_sim=opt_sim,
        grp_sim=grp_sim,
        identity_coverage=identity_coverage,
        ess_sim_raw=ess_sim_raw,
        ess_geo_raw=ess_geo_raw,
    )


def _build_essential_match_list(
    k: _PairKernel,
    skill_labels: dict[str, str],
    user_skill_labels: dict[str, str],
    threshold: float,
) -> list[MatchedSkill]:
    """Match-list payload for the response. ``similarity`` is in rescaled space (the
    same frame as final_score, p_hat, essential_fit), so a human reader sees one
    consistent scale. ``meets_threshold`` compares the rescaled value, since the
    threshold lives in the same rescaled frame as all the gate decisions.
    """
    essential_skill_matches: list[MatchedSkill] = []
    if (
        k.ess_mat.size > 0
        and k.js_mat.size > 0
        and k.rowmax is not None
        and k.argmax is not None
    ):
        # rowmax_rescaled is None only if the kernel skipped rescaling; in that case
        # fall back to raw (rescaling disabled => values are already in the right frame).
        rescaled = k.rowmax_rescaled if k.rowmax_rescaled is not None else k.rowmax
        for i, ess_id in enumerate(k.ess_ids_valid):
            best_idx = int(k.argmax[i])
            best_js_id = k.js_ids_valid[best_idx] if k.js_ids_valid else None
            sim_rescaled = float(rescaled[i])
            essential_skill_matches.append(
                MatchedSkill(
                    job_skill_id=ess_id,
                    job_skill_label=skill_labels.get(ess_id),
                    best_user_skill_id=best_js_id,
                    best_user_skill_label=user_skill_labels.get(best_js_id)
                    if best_js_id
                    else None,
                    similarity=round(sim_rescaled, 4),
                    meets_threshold=sim_rescaled >= threshold,
                )
            )
    elif k.ess_ids_valid:
        for ess_id in k.ess_ids_valid:
            essential_skill_matches.append(
                MatchedSkill(
                    job_skill_id=ess_id,
                    job_skill_label=skill_labels.get(ess_id),
                    similarity=0.0,
                    meets_threshold=False,
                )
            )
    return essential_skill_matches


def _optional_and_group_match_lists(
    op: Opportunity,
    js: Jobseeker,
    k: _PairKernel,
    skill_labels: dict[str, str],
    user_skill_labels: dict[str, str],
    skill_group_labels: dict[str, str],
) -> tuple[list[OptionalSkillMatch], list[SkillGroupMatch]]:
    optional_exact_matches = [
        OptionalSkillMatch(
            skill_id=opt_id,
            skill_label=skill_labels.get(opt_id) or user_skill_labels.get(opt_id),
        )
        for opt_id in sorted(set(k.opt_ids) & set(k.js_ids))
    ]
    skill_group_matches = [
        SkillGroupMatch(
            skill_group_id=gid, skill_group_label=skill_group_labels.get(gid)
        )
        for gid in sorted(op.skill_groups_origin_uuids & js.skill_groups_origin_uuids)
    ]
    return optional_exact_matches, skill_group_matches


class SkillUtilityComponents(DomainModel):
    """Components of the legacy skill utility ``U`` (each in [0, 1])."""

    loc: float = Field(
        description="Location proximity (1.0 city, 0.7 province, 0.3 otherwise)."
    )
    ess: float = Field(description="Mean best (rescaled) cosine over essential skills.")
    opt: float = Field(description="Mean best (rescaled) cosine over optional skills.")
    grp: float = Field(description="Skill-group recall.")


class SkillUtility(DomainModel):
    """Legacy skill utility ``U`` of a (jobseeker, opportunity) pair (``compute_U_complete``)."""

    U_final: float = Field(
        description="Weighted component blend minus the gap penalty, in [0, 1]."
    )
    is_eligible: bool = Field(
        description="True when the share of essentials above ``SKILL_U_TAU_ELIG`` reaches the eligibility share."
    )
    components: SkillUtilityComponents = Field(description="The blended components.")
    penalty: float = Field(description="Gap penalty subtracted from the blend.")
    match_details: MatchedSkills = Field(description="Per-skill match rows.")


class Feasibility(DomainModel):
    """Recruiter-side feasibility signals of a pair, feeding the legacy ``p_hat``."""

    gate_passed: bool = Field(
        description="True when at most half the essentials fall below the gate threshold."
    )
    essential_fit: float = Field(
        description="Score-weighted geometric mean of rescaled essential rowmaxes (0 when gated)."
    )
    essential_fit_raw: float = Field(
        description="As ``essential_fit``, un-rescaled (diagnostic)."
    )
    identity_coverage: float = Field(
        description="Share of essentials the user holds exactly (diagnostic)."
    )
    optional_sim: float = Field(description="Mean best cosine over optional skills.")
    skill_group_recall: float = Field(description="Skill-group recall.")
    gap_share: float = Field(
        description="Share of essentials below the gate threshold."
    )
    has_essential_skills: bool = Field(
        description="The opportunity lists essential skills."
    )
    has_optional_skills: bool = Field(
        description="The opportunity lists optional skills."
    )
    has_skill_groups: bool = Field(description="The opportunity lists skill groups.")
    match_details: MatchedSkills = Field(description="Per-skill match rows.")


def _utility(
    k: _PairKernel,
    *,
    essential_matches: list[MatchedSkill],
    optional_matches: list[OptionalSkillMatch],
    group_matches: list[SkillGroupMatch],
) -> SkillUtility:
    W_LOC, W_ESS, W_OPT, W_GRP, W_GAP_PEN = (
        SKILL_U_W_LOC,
        SKILL_U_W_ESS,
        SKILL_U_W_OPT,
        SKILL_U_W_GRP,
        SKILL_U_GAP_PENALTY,
    )
    core_score = (
        W_LOC * k.loc_score + W_ESS * k.ess_sim + W_OPT * k.opt_sim + W_GRP * k.grp_sim
    ) / (W_LOC + W_ESS + W_OPT + W_GRP)

    # gap_share / eligibility evaluated against rescaled rowmax so the TAU_ELIG threshold
    # lives in the same calibrated frame as ess_sim and final_score. When rescaling is
    # disabled (no artefact metadata), rowmax_rescaled == rowmax, so behaviour is identical
    # to before.
    rowmax_for_gates = k.rowmax_rescaled if k.rowmax_rescaled is not None else k.rowmax
    if rowmax_for_gates is None or rowmax_for_gates.size == 0:
        gap_share, eligible = 0.0, True
    else:
        meets = rowmax_for_gates >= SKILL_U_TAU_ELIG
        gap_share = float((~meets).mean())
        eligible = bool(float(meets.mean()) >= SKILL_MIN_ESSENTIAL_MATCH_SHARE)

    u_final = max(0.0, core_score - (W_GAP_PEN * gap_share))
    return SkillUtility(
        U_final=round(u_final, 4),
        is_eligible=eligible,
        components=SkillUtilityComponents(
            loc=round(k.loc_score, 2),
            ess=round(k.ess_sim, 4),
            opt=round(k.opt_sim, 4),
            grp=round(k.grp_sim, 4),
        ),
        penalty=round(W_GAP_PEN * gap_share, 4),
        match_details=MatchedSkills(
            essential_skill_matches=essential_matches,
            optional_exact_matches=optional_matches,
            skill_group_matches=group_matches,
        ),
    )


def _feasibility(
    op: Opportunity,
    k: _PairKernel,
    *,
    gate_threshold: float,
    essential_matches: list[MatchedSkill],
    optional_matches: list[OptionalSkillMatch],
    group_matches: list[SkillGroupMatch],
) -> Feasibility:
    # Gate evaluated against rescaled rowmax so GATE_SIMILARITY_THRESHOLD lives in the
    # same calibrated frame as essential_fit and final_score. When rescaling is disabled
    # (no artefact metadata), rowmax_rescaled == rowmax, so behaviour matches the
    # pre-rescaling system.
    rowmax_for_gates = k.rowmax_rescaled if k.rowmax_rescaled is not None else k.rowmax
    ess_geo = k.ess_geo
    if not k.ess_ids:
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

    return Feasibility(
        gate_passed=gate_passed,
        essential_fit=round(ess_geo, 4),
        essential_fit_raw=round(k.ess_geo_raw, 4),
        identity_coverage=round(k.identity_coverage, 4),
        optional_sim=round(k.opt_sim, 4),
        skill_group_recall=round(k.grp_sim, 4),
        gap_share=round(gap_share, 4),
        has_essential_skills=len(k.ess_ids) > 0,
        has_optional_skills=len(k.opt_ids) > 0,
        has_skill_groups=len(op.skill_groups_origin_uuids) > 0,
        match_details=MatchedSkills(
            essential_skill_matches=essential_matches,
            optional_exact_matches=optional_matches,
            skill_group_matches=group_matches,
        ),
    )


# =============================================================================
# 4) CORE UTILITY CALCULATION
# =============================================================================


def compute_U_complete(
    js: Jobseeker,
    op: Opportunity,
    engine: SimilarityEngine,
    skill_labels: dict[str, str] | None = None,
    user_skill_labels: dict[str, str] | None = None,
    skill_group_labels: dict[str, str] | None = None,
) -> SkillUtility:
    """Calculates the final Utility score with components and penalties.

    Uses a single essential similarity matmul via :func:`_shared_pair_kernel`.
    """
    skill_labels = skill_labels or {}
    user_skill_labels = user_skill_labels or {}
    k = _shared_pair_kernel(js, op, engine)
    opt_m, grp_m = _optional_and_group_match_lists(
        op, js, k, skill_labels, user_skill_labels, skill_group_labels or {}
    )
    return _utility(
        k,
        essential_matches=_build_essential_match_list(
            k, skill_labels, user_skill_labels, SKILL_U_TAU_ELIG
        ),
        optional_matches=opt_m,
        group_matches=grp_m,
    )


# =============================================================================
# 5) FEASIBILITY SIGNALS  (recruiter-side, feeds into p_hat)
# =============================================================================


def compute_feasibility_signals(
    js: Jobseeker,
    op: Opportunity,
    engine: SimilarityEngine,
    gate_threshold: float | None = None,
    skill_labels: dict[str, str] | None = None,
    user_skill_labels: dict[str, str] | None = None,
    skill_group_labels: dict[str, str] | None = None,
) -> Feasibility:
    """Recruiter-side feasibility; shares :func:`_shared_pair_kernel` with U."""
    if gate_threshold is None:
        gate_threshold = GATE_SIMILARITY_THRESHOLD
    skill_labels = skill_labels or {}
    user_skill_labels = user_skill_labels or {}
    k = _shared_pair_kernel(js, op, engine)
    opt_m, grp_m = _optional_and_group_match_lists(
        op, js, k, skill_labels, user_skill_labels, skill_group_labels or {}
    )
    return _feasibility(
        op,
        k,
        gate_threshold=gate_threshold,
        essential_matches=_build_essential_match_list(
            k, skill_labels, user_skill_labels, gate_threshold
        ),
        optional_matches=opt_m,
        group_matches=grp_m,
    )


def compute_utility_and_feasibility_pair(
    js: Jobseeker,
    op: Opportunity,
    engine: SimilarityEngine,
    skill_labels: dict[str, str] | None = None,
    user_skill_labels: dict[str, str] | None = None,
    skill_group_labels: dict[str, str] | None = None,
    gate_threshold: float | None = None,
) -> tuple[SkillUtility, Feasibility]:
    """One essential matmul and one optional/group pass for the multiplicative pipeline."""
    if gate_threshold is None:
        gate_threshold = GATE_SIMILARITY_THRESHOLD
    skill_labels = skill_labels or {}
    user_skill_labels = user_skill_labels or {}
    k = _shared_pair_kernel(js, op, engine)
    u_ess = _build_essential_match_list(
        k, skill_labels, user_skill_labels, SKILL_U_TAU_ELIG
    )
    f_ess = (
        _build_essential_match_list(k, skill_labels, user_skill_labels, gate_threshold)
        if abs(SKILL_U_TAU_ELIG - gate_threshold) >= 1e-9
        else u_ess
    )
    opt_m, grp_m = _optional_and_group_match_lists(
        op, js, k, skill_labels, user_skill_labels, skill_group_labels or {}
    )
    utility = _utility(
        k, essential_matches=u_ess, optional_matches=opt_m, group_matches=grp_m
    )
    feasibility = _feasibility(
        op,
        k,
        gate_threshold=gate_threshold,
        essential_matches=f_ess,
        optional_matches=opt_m,
        group_matches=grp_m,
    )
    return utility, feasibility


# --------------------------------------------------------------------------------------------------
# skill_label_packs
#
# Cross-language skill label resolution.
#
# Matching resolves skills by **label**: the user profile carries `preferredLabel` strings and
# the job posting carries `{id, label}` pairs, and both must land on a row of the embedding
# artefact. Labels are the trust anchor rather than UUIDs, which carry modelId-drift risk.
#
# That works across languages because a taxonomy translation describes the *same* skills.
# Skill ``ID`` values are per-taxonomy-model and share nothing across locales, but
# ``UUIDHISTORY``'s **oldest** entry is identical across them (13,896/13,896 skills between
# the English model and AR-es). So this module loads every enabled language's label pack and
# maps all of them onto one id space — the canonical language's, the one
# ``skill_to_row.json`` and the embedding artefacts are keyed on.
#
# The consequence worth stating plainly: a Spanish job posting matched against a Spanish user
# profile scores through exactly the same vectors as an English one, and neither side has to
# say which language it is in. Nothing about the embeddings is language-specific; only the
# label text used to reach them is.
# --------------------------------------------------------------------------------------------------

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def canon(label: str) -> str:
    """Canonical form for label-based resolution: lowercase, whitespace-collapsed."""
    return " ".join(str(label or "").strip().lower().split())


def oldest_uuid(uuid_history: str) -> str:
    """``UUIDHISTORY``'s oldest entry — the locale-stable identity of a taxonomy entity."""
    entries = [u.strip() for u in (uuid_history or "").split("\n") if u.strip()]
    return entries[-1] if entries else ""


class SkillLabelPacks:
    """Label → canonical-id lookups built from one or more languages' taxonomy packs.

    ``embedding_ids`` is the set of ids present in the embedding artefact: labels that would
    resolve outside it are dropped, exactly as the single-language loader did. ``read_rows`` reads
    one ``skills.csv`` into dict rows (see ``ArtifactsRepository.read_csv_rows``).
    """

    def __init__(
        self,
        embedding_ids: set[str],
        *,
        read_rows: Callable[[Path], list[dict[str, str]]],
        languages: Iterable[str] | None = None,
        canonical_language: str = CANONICAL_LANGUAGE,
    ):
        self._read_rows = read_rows
        self.canonical_language = canonical_language
        self.languages: tuple[str, ...] = tuple(languages or enabled_languages())
        if canonical_language not in self.languages:
            # The canonical pack defines the id space every other pack is joined onto.
            self.languages = (canonical_language, *self.languages)
        self._embedding_ids = embedding_ids

        # canonical_id -> preferredLabel, in the canonical language (display default).
        self.skill_labels: dict[str, str] = {}
        # language -> {canonical_id -> preferredLabel in that language}
        self.labels_by_language: dict[str, dict[str, str]] = {}
        # canonical preferredLabel (any language) -> canonical_id
        self.preferred_to_id: dict[str, str] = {}
        # canonical altLabel (any language) -> canonical_id
        self.altlabel_to_id: dict[str, str] = {}
        # per-locale taxonomy skill id -> canonical_id (for ids arriving on documents)
        self.id_to_canonical: dict[str, str] = {}
        # every ESCO origin / historical UUID seen in any pack -> canonical_id. A
        # drift-tolerant last-resort resolver for callers that only have a UUID.
        self.uuid_to_id: dict[str, str] = {}

        self.preferred_collisions = 0
        self.loaded_languages: list[str] = []
        self.unjoined_by_language: Counter = Counter()

        self._uuid_to_canonical: dict[str, str] = {}
        self._rows_cache: dict[Path, list[dict[str, str]]] = {}
        self._skills_paths = self._resolve_skills_paths()
        for language in self.languages:
            self._load_language(language)
        self._rows_cache.clear()

        if not self.preferred_to_id:
            logger.error(
                "SkillLabelPacks: NO labels resolved from %d language pack(s) %s — every skill on "
                "every request will fail to resolve, so matched_skills comes back empty and "
                "essential-coverage is 0 for every posting. Check that the packs and the embedding "
                "artefact (skill_to_row.json) are from the same taxonomy release.",
                len(self.loaded_languages),
                ",".join(self.loaded_languages),
            )

        logger.info(
            "SkillLabelPacks: %d language(s) %s | %d preferredLabel keys, %d altLabel keys "
            "(preferred-collisions: %d) | %d non-canonical ids mapped",
            len(self.loaded_languages),
            ",".join(self.loaded_languages),
            len(self.preferred_to_id),
            len(self.altlabel_to_id),
            self.preferred_collisions,
            len(self.id_to_canonical),
        )
        for language, count in self.unjoined_by_language.items():
            logger.warning(
                "SkillLabelPacks: %d skill(s) in the %r pack have no counterpart in the %r "
                "id space and were skipped — labels resolving to them cannot be scored. "
                "The two packs are probably from different taxonomy releases.",
                count,
                language,
                self.canonical_language,
            )

    # ── loading ──────────────────────────────────────────────────────────────

    def _resolve_skills_paths(self) -> dict[str, Path]:
        """Per-language ``skills.csv`` to load, recovering from an unusable pin.

        ``SKILLS_CSV_PATH`` pins *every* language to one file. Pointing it at a non-canonical
        pack therefore loads that pack as the canonical one, and since skill ids are per-locale
        none of them exist in the embedding artefact: the join drops all 13,896 rows, no label
        resolves, and the service keeps serving requests with an empty resolver (every posting
        gets essential-coverage 0). The pin cannot be honoured in that state, so prefer the
        per-language layout and say loudly what happened.
        """
        pinned = {
            lang: Path(taxonomy_pack_paths(lang)["skills"]) for lang in self.languages
        }
        canonical = pinned[self.canonical_language]
        if self._pack_shares_id_space(canonical):
            return pinned

        unpinned = {
            lang: Path(taxonomy_pack_paths(lang, ignore_pins=True)["skills"])
            for lang in self.languages
        }
        if unpinned[self.canonical_language] == canonical:
            return pinned  # not a pin: the layout's own canonical pack is the mismatched one
        if not self._pack_shares_id_space(unpinned[self.canonical_language]):
            logger.error(
                "SkillLabelPacks: neither the pinned canonical pack %s nor %s shares ids with the "
                "embedding artefact; skill resolution will fail for every request.",
                canonical,
                unpinned[self.canonical_language],
            )
            return pinned
        logger.error(
            "SkillLabelPacks: the %r pack is pinned to %s (SKILLS_CSV_PATH / SKILL_GROUPS_CSV_PATH / "
            "SKILL_HIERARCHY_CSV_PATH pin EVERY language to one file), whose skill ids are not in the "
            "embedding artefact — loading it would leave the resolver empty. Ignoring the pin and "
            "using the per-language packs instead. Clear those env vars: the taxonomy now lives under "
            "resources/skill_taxonomy/<lang>/ and is selected by TARGET_LANGUAGE.",
            self.canonical_language,
            canonical,
        )
        return unpinned

    def _pack_shares_id_space(self, path: Path) -> bool:
        """True if any ``ID`` in this pack is a row of the embedding artefact."""
        try:
            rows = self._read_rows_cached(path)
        except (OSError, UnicodeDecodeError, csv.Error):
            return False
        return any(
            str(row.get("ID") or "").strip() in self._embedding_ids for row in rows
        )

    def _read_rows_cached(self, path: Path) -> list[dict[str, str]]:
        """``_read_rows`` memoised for this instance — the canonical pack is read twice
        (id-space preflight, then loading) and these files are tens of megabytes."""
        rows = self._rows_cache.get(path)
        if rows is None:
            rows = self._read_rows(path)
            self._rows_cache[path] = rows
        return rows

    def _load_language(self, language: str) -> None:
        path = self._skills_paths[language]
        try:
            rows = self._read_rows_cached(path)
        except FileNotFoundError:
            if language == self.canonical_language:
                logger.error(
                    "SkillLabelPacks: canonical skills.csv not found at %s — skill "
                    "resolution will fail for every request",
                    path,
                )
            else:
                logger.warning(
                    "SkillLabelPacks: no %r taxonomy pack at %s — skills submitted in that "
                    "language will not resolve",
                    language,
                    path,
                )
            return

        is_canonical = language == self.canonical_language
        labels: dict[str, str] = {}
        joined = 0
        for row in rows:
            local_id = str(row.get("ID") or "").strip()
            if not local_id:
                continue
            uuid = oldest_uuid(row.get("UUIDHISTORY") or "")

            if is_canonical:
                canonical_id = local_id
                if uuid:
                    self._uuid_to_canonical.setdefault(uuid, canonical_id)
            else:
                canonical_id = self._uuid_to_canonical.get(uuid, "")
                if not canonical_id:
                    self.unjoined_by_language[language] += 1
                    continue
                if local_id != canonical_id:
                    self.id_to_canonical.setdefault(local_id, canonical_id)

            if canonical_id not in self._embedding_ids:
                continue
            joined += 1

            label = (row.get("PREFERREDLABEL") or "").strip()
            if label:
                labels[canonical_id] = label
                if is_canonical:
                    self.skill_labels[canonical_id] = label
                self._register_preferred(canon(label), canonical_id)

            for alt in (row.get("ALTLABELS") or "").split("\n"):
                key = canon(alt)
                if not key or key in self.preferred_to_id:
                    # Never let an altLabel shadow a preferredLabel hit; first writer
                    # wins among altLabels.
                    continue
                self.altlabel_to_id.setdefault(key, canonical_id)

            # Current ESCO origin UUID and every historical UUID → the same canonical id.
            # Locales share most of their UUID history, so this map is consistent across
            # packs rather than competing between them.
            for column in ("ORIGINURI", "UUIDHISTORY"):
                for match in _UUID_RE.findall((row.get(column) or "").lower()):
                    self.uuid_to_id.setdefault(match, canonical_id)

        self.labels_by_language[language] = labels
        self.loaded_languages.append(language)
        log = logger.info if joined else logger.error
        log(
            "SkillLabelPacks: %r pack → %d skills usable (%s)%s",
            language,
            joined,
            path.name if is_canonical else f"joined onto {self.canonical_language}",
            ""
            if joined
            else f" — NOTHING in {path} lands in the embedding id space, so no label submitted "
            f"in {language!r} can be scored; the pack and the embedding artefact are from "
            "different taxonomy releases",
        )

    def _register_preferred(self, key: str, canonical_id: str) -> None:
        if not key:
            return
        existing = self.preferred_to_id.get(key)
        if existing is None:
            self.preferred_to_id[key] = canonical_id
        elif existing != canonical_id:
            # Two skills share a label text (across or within languages). First writer
            # wins, as before; counted so the ambiguity is visible in the logs.
            self.preferred_collisions += 1

    # ── resolution ───────────────────────────────────────────────────────────

    def resolve_label(self, label: str | None) -> str | None:
        """Canonical id for a label in any loaded language, or None."""
        key = canon(label or "")
        if not key:
            return None
        found = self.preferred_to_id.get(key)
        if found is not None:
            return found
        return self.altlabel_to_id.get(key)

    def resolve_id(self, skill_id: str | None) -> str | None:
        """Canonical id for a taxonomy skill id from any language, or None.

        A fallback for documents whose label is missing: an id from a non-canonical
        taxonomy model still identifies the right skill.
        """
        sid = str(skill_id or "").strip()
        if not sid:
            return None
        if sid in self._embedding_ids:
            return sid
        return self.id_to_canonical.get(sid)

    def resolve_uuid(self, uuid: str | None) -> str | None:
        """Canonical id for an ESCO origin/historical UUID, or None."""
        key = str(uuid or "").strip().lower()
        if not key:
            return None
        return self.uuid_to_id.get(key)

    def display_labels(self, language: str | None = None) -> dict[str, str]:
        """canonical_id → preferredLabel in ``language``, falling back to the canonical one."""
        if not language or language == self.canonical_language:
            return self.skill_labels
        pack = self.labels_by_language.get(language)
        if not pack:
            return self.skill_labels
        # A language pack can be missing a skill the canonical one has; fill from canonical
        # so a response never shows a blank label.
        return {**self.skill_labels, **pack}


# --------------------------------------------------------------------------------------------------
# skill_matcher
# --------------------------------------------------------------------------------------------------


def _canon(label: str) -> str:
    return " ".join(label.strip().lower().split())


# ESCO origin UUID (embedded in the ORIGINURI URL) or any historical UUID (UUIDHISTORY column).


def compact_cosine_matched_skill_lines(
    per_job_skill: Sequence[SkillMatchDetail],
    *,
    limit: int = 24,
) -> list[str]:
    """Display lines ``job skill <- user skill (cosine)`` for the strongest ``limit`` matches."""
    if not per_job_skill or limit <= 0:
        return []
    ranked = sorted(per_job_skill, key=lambda x: x.cosine_similarity, reverse=True)[
        :limit
    ]
    lines: list[str] = []
    for row in ranked:
        jl = (row.job_skill_label or row.job_skill_id or "").strip()
        ul = (row.best_user_skill_label or row.best_user_skill_id or "").strip()
        c = row.cosine_similarity
        if jl and ul and _canon(jl) == _canon(ul):
            lines.append(f"{jl} ({c:.2f})")
        elif jl and ul:
            lines.append(
                f"{jl} \u2190 {ul} ({c:.2f})"
            )  # left arrow: job \u2190 user match
        elif jl:
            lines.append(f"{jl} ({c:.2f})")
        elif ul:
            lines.append(f"{ul} ({c:.2f})")
    return lines


class CosineSkillMatcher:
    """Row-normalised embedding matrix + label-based skill lookup."""

    def __init__(
        self,
        *,
        weights: np.ndarray,
        skill_to_row: dict[str, int],
        packs: SkillLabelPacks,
        display_language: str,
        rescale_target: float,
        model_label: str,
    ) -> None:
        W = weights
        if W.dtype != np.float32:
            W = W.astype(np.float32)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        self.W = W / np.where(norms > 0, norms, 1.0)
        # Whitened artifacts carry a rescale target (``target_max_p999``) so de-anisotropised cosines
        # map toward [0,1]; raw artifacts don't (target 0 => rescale is a no-op = historical behaviour).
        self._rescale_target = float(rescale_target)

        self.skill_to_row = skill_to_row

        self._embedding_ids = set(self.skill_to_row.keys())
        # Label maps come from every enabled language's taxonomy pack, all mapped onto this
        # one id space, so a Spanish label reaches the same embedding row as its English
        # counterpart (see ranking/skill_label_packs.py).
        self._packs = packs
        # DISPLAY labels follow the deployment's TARGET_LANGUAGE. Resolution is id-based and
        # language-neutral, but these strings are echoed straight back in the response
        # (``job_skill_label`` / ``best_user_skill_label``, the justification sentence), so a
        # Spanish deployment has to answer in Spanish rather than in the canonical English the
        # id space happens to be keyed on. Falls back to canonical per skill when a pack lacks one.
        self.skill_labels: dict[str, str] = self._packs.display_labels(display_language)
        self._preferred_to_id: dict[str, str] = self._packs.preferred_to_id
        self._altlabel_to_id: dict[str, str] = self._packs.altlabel_to_id
        # ESCO origin/history UUID -> canonical skill id (drift-tolerant fallback resolver).
        self._originuuid_to_id: dict[str, str] = self._packs.uuid_to_id
        self._preferred_collisions = self._packs.preferred_collisions

        self._missed_labels: Counter[str] = Counter()

        logger.info(
            "CosineSkillMatcher: %d embedding rows (model=%s, dim=%d); "
            "%d preferredLabel keys, %d altLabel keys (preferred-collisions: %d)",
            len(self._embedding_ids),
            model_label,
            self.W.shape[1],
            len(self._preferred_to_id),
            len(self._altlabel_to_id),
            self._preferred_collisions,
        )

    def _resolve_label(self, label: str | None) -> str | None:
        if not label:
            return None
        canon = _canon(label)
        if not canon:
            return None
        sid = self._preferred_to_id.get(canon)
        if sid is not None:
            return sid
        sid = self._altlabel_to_id.get(canon)
        if sid is not None:
            return sid
        self._missed_labels[label] += 1
        return None

    def _resolve_origin_uuid(self, origin_uuid: str | None) -> str | None:
        """Drift-tolerant fallback resolver: map an ESCO origin UUID (the current ORIGINURI
        value or ANY historical UUIDHISTORY value) to the SAME canonical skill id that label
        resolution returns. Labels stay the primary trust anchor (see ``_resolve_label``); this
        only recovers matches a renamed/missing label would otherwise lose. ``None`` if unknown.
        """
        if not origin_uuid:
            return None
        key = _canon(str(origin_uuid))
        return self._originuuid_to_id.get(key) if key else None

    def get_resolution_stats(self) -> dict[str, int | list[tuple[str, int]]]:
        """Per-process label-miss telemetry."""
        return {
            "total_misses": sum(self._missed_labels.values()),
            "distinct_missed_labels": len(self._missed_labels),
            "top_misses": self._missed_labels.most_common(20),
        }

    def _rows_with_ids(self, skill_ids: Sequence[str]) -> tuple[np.ndarray, list[str]]:
        """Return row matrix and list of IDs that exist in ``skill_to_row`` (stable order)."""
        valid_ids: list[str] = []
        for s in skill_ids:
            sid = str(s)
            if sid in self.skill_to_row:
                valid_ids.append(sid)
        if not valid_ids:
            return np.empty((0, self.W.shape[1]), dtype=np.float32), []
        idx = np.array([self.skill_to_row[s] for s in valid_ids], dtype=np.int64)
        return self.W[idx, :].astype(np.float32, copy=False), valid_ids

    @staticmethod
    def _ordered_unique_skill_ids(pairs: Iterable[tuple[str, str]]) -> list[str]:
        """pairs: (internal_id, label) — preserve first-seen order, de-dupe by id."""
        seen: set[str] = set()
        out: list[str] = []
        for sid, _ in pairs:
            if sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
        return out

    def _user_skill_pairs(self, user: UserProfile) -> list[tuple[str, str]]:
        """(resolved id, label) per user skill: by label, else by ESCO origin/history UUID."""
        out: list[tuple[str, str]] = []
        for s in user.skills_vector.top_skills:
            lab = s.preferredLabel
            sid = self._resolve_label(lab)
            if sid is None:  # fallback: ESCO origin/history UUID -> same canonical id
                sid = self._resolve_origin_uuid(s.originUUID)
            if sid is not None:
                out.append((sid, lab if lab else (self.skill_labels.get(sid) or "")))
        return out

    def _job_skill_pairs(self, item: MatchItem) -> list[tuple[str, str]]:
        """(resolved id, label) per essential then optional item skill: by label, else by id."""
        out: list[tuple[str, str]] = []
        for s in [*item.essential_skills, *item.optional_skills]:
            sid = self._resolve_label(s.label)
            if sid is None:  # fallback: ESCO origin/history UUID -> same canonical id
                sid = self._resolve_origin_uuid(s.id)
            if sid is not None:
                out.append(
                    (sid, s.label if s.label else (self.skill_labels.get(sid) or ""))
                )
        return out

    def essential_ids(self, item: MatchItem) -> frozenset[str]:
        """Label-resolved ids of the item's essential skills (the ``job_skill_id`` id space)."""
        out: set[str] = set()
        for s in item.essential_skills:
            rid = self._resolve_label(s.label) if s.label else None
            if rid:
                out.add(rid)
        return frozenset(out)

    def score_pair(self, user: UserProfile, item: MatchItem) -> SkillPairScore:
        """Cosine similarity only: mean over item skills of (max over user skills).

        :param user: The user
        :param item: The job or occupation
        :return: ``mean_best_cosine`` in ``[0, 1]`` (zero if no overlap to score) and the per-skill rows
        """
        user_pairs = self._user_skill_pairs(user)
        job_pairs = self._job_skill_pairs(item)
        user_ids = self._ordered_unique_skill_ids(user_pairs)
        job_ids = self._ordered_unique_skill_ids(job_pairs)
        user_labels = {sid: lab for sid, lab in user_pairs}
        job_labels = {sid: lab for sid, lab in job_pairs}

        u_mat, u_valid = self._rows_with_ids(user_ids)
        j_mat, j_valid = self._rows_with_ids(job_ids)

        if j_mat.size == 0 or u_mat.size == 0:
            return SkillPairScore(
                mean_best_cosine=0.0,
                min_best_cosine=0.0,
                n_user_skills_embedded=int(u_mat.shape[0]),
                n_job_skills_embedded=int(j_mat.shape[0]),
                per_job_skill=[],
            )

        sims = j_mat @ u_mat.T
        np.maximum(sims, 0.0, out=sims)
        argmax = sims.argmax(axis=1)
        rowmax = sims.max(axis=1)

        per: list[SkillMatchDetail] = []
        for i, jid in enumerate(j_valid):
            uid = u_valid[int(argmax[i])]
            per.append(
                SkillMatchDetail(
                    job_skill_id=jid,
                    job_skill_label=self.skill_labels.get(jid) or job_labels.get(jid),
                    best_user_skill_id=uid,
                    best_user_skill_label=self.skill_labels.get(uid)
                    or user_labels.get(uid),
                    cosine_similarity=round(float(rowmax[i]), 4),
                )
            )

        return SkillPairScore(
            mean_best_cosine=round(float(rowmax.mean()), 4),
            min_best_cosine=round(float(rowmax.min()), 4),
            n_user_skills_embedded=int(u_mat.shape[0]),
            n_job_skills_embedded=int(j_mat.shape[0]),
            per_job_skill=per,
        )

    def _rescale(self, sims: np.ndarray) -> np.ndarray:
        """Map whitened cosines toward [0,1] via the artifact's target (no-op when target<=0)."""
        t = self._rescale_target
        return np.minimum(1.0, sims / t) if t and t > 0 else sims

    def score_pair_v4(
        self, user: UserProfile, item: MatchItem
    ) -> list[SkillMatchDetail]:
        """v4 GATE matching: rescaled (whitened) similarity + one-to-one greedy assignment + exact-id
        flag. Each item skill is matched to AT MOST ONE distinct user skill, so extra user skills can't
        inflate coverage. Rows mirror ``score_pair`` plus ``cosine_similarity_raw`` and ``exact`` (user
        holds this exact skill id). Leaves ``score_pair`` (stage-1 detail, v2/v3) untouched.

        :param user: The user
        :param item: The job or occupation
        :return: One row per embedded item skill; unassigned skills have no user skill and similarity 0
        """
        user_pairs = self._user_skill_pairs(user)
        job_pairs = self._job_skill_pairs(item)
        user_ids = self._ordered_unique_skill_ids(user_pairs)
        job_ids = self._ordered_unique_skill_ids(job_pairs)
        user_labels = {sid: lab for sid, lab in user_pairs}
        job_labels = {sid: lab for sid, lab in job_pairs}
        user_id_set = set(user_ids)

        u_mat, u_valid = self._rows_with_ids(user_ids)
        j_mat, j_valid = self._rows_with_ids(job_ids)

        def _row(
            jid: str, uid: str | None, sim_resc: float, sim_raw: float
        ) -> SkillMatchDetail:
            return SkillMatchDetail(
                job_skill_id=jid,
                job_skill_label=self.skill_labels.get(jid) or job_labels.get(jid),
                best_user_skill_id=uid,
                best_user_skill_label=(
                    self.skill_labels.get(uid) or user_labels.get(uid)
                )
                if uid
                else None,
                cosine_similarity=round(float(sim_resc), 4),
                cosine_similarity_raw=round(float(sim_raw), 4),
                exact=jid in user_id_set,
            )

        if j_mat.size == 0:
            return []
        if u_mat.size == 0:
            return [_row(jid, None, 0.0, 0.0) for jid in j_valid]

        raw = j_mat @ u_mat.T
        np.maximum(raw, 0.0, out=raw)
        resc = self._rescale(raw)
        n_j, n_u = resc.shape
        # greedy one-to-one assignment maximising rescaled similarity (each user skill used once)
        order = np.argsort(resc, axis=None)[::-1]
        assigned = [-1] * n_j
        used_u: set[int] = set()
        cap = min(n_j, n_u)
        n_assigned = 0
        for flat in order:
            j, u = divmod(int(flat), n_u)
            if assigned[j] != -1 or u in used_u:
                continue
            assigned[j] = u
            used_u.add(u)
            n_assigned += 1
            if n_assigned >= cap:
                break
        return [
            _row(jid, None, 0.0, 0.0)
            if assigned[j] < 0
            else _row(
                jid, u_valid[assigned[j]], resc[j, assigned[j]], raw[j, assigned[j]]
            )
            for j, jid in enumerate(j_valid)
        ]

    def resolved_user_skill_labels_ordered(self, user: UserProfile) -> list[str]:
        """Display labels for embedded user skills (order preserved, de-duplicated by id)."""
        seen: set[str] = set()
        out: list[str] = []
        for sid, lab in self._user_skill_pairs(user):
            if sid in seen:
                continue
            seen.add(sid)
            out.append(lab or self.skill_labels.get(sid) or sid)
        return out

    def rank_jobs(
        self,
        user: UserProfile,
        items: Sequence[MatchItem],
        *,
        top_k: int = 10,
    ) -> list[tuple[MatchItem, SkillPairScore]]:
        """Score every item; the top ``top_k`` by (``mean_best_cosine``, ``min_best_cosine``) desc.

        :param user: The user
        :param items: Jobs or occupations
        :param top_k: How many to keep
        :return: ``(item, score)`` pairs, best first
        """
        scored = [(item, self.score_pair(user, item)) for item in items]
        scored.sort(key=lambda x: (-x[1].mean_best_cosine, -x[1].min_best_cosine))
        return scored[: max(0, top_k)]


# --------------------------------------------------------------------------------------------------
# skill_scorer
# --------------------------------------------------------------------------------------------------


class SkillScorer:
    """Node2Vec/Gemini skill utility + feasibility scorer over one skill-embedding matrix.

    Built from already-loaded data (see ``app.server_dependencies.model_dependencies``):
    ``weights`` is the stored ``embedding.weight``; ``skill_group_rows`` / ``hierarchy_rows`` are the
    taxonomy CSV rows, or None when the file is absent.
    """

    def __init__(
        self,
        *,
        weights: np.ndarray,
        skill_to_row: dict[str, int],
        packs: SkillLabelPacks,
        display_language: str,
        rescale_target: float,
        skill_group_rows: list[dict[str, str]] | None,
        hierarchy_rows: list[dict[str, str]] | None,
        model_label: str,
    ):
        W = weights
        # Promote lower-precision weights (e.g. fp16, used by the Gemini artefact
        # to stay under GitHub's per-file size limit) to fp32 before any
        # downstream math; SimilarityEngine and the cosine kernels assume fp32.
        if W.dtype != np.float32:
            W = W.astype(np.float32)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        W = W / np.where(norms > 0, norms, 1.0)

        self._embedding_ids = set(skill_to_row.keys())
        self.engine = SimilarityEngine(W, skill_to_row, rescale_target)

        # Label-primary resolution maps, built from every enabled language's taxonomy pack.
        # UUIDs are NOT used for resolution — they carry modelId-drift risk (a Compass-side
        # UUID can resolve to a different internal skill than the user's declared label).
        # Labels are stable across taxonomy versions; they're our trust anchor.
        #
        # Loading several languages into one resolver is what makes a Spanish posting match
        # a Spanish profile with nothing on the request: every pack is mapped onto this one
        # id space, so the embeddings are shared (see ranking/skill_label_packs.py).
        self._packs = packs
        # internal_id -> preferredLabel in the DEPLOYMENT's language: these strings are echoed in
        # responses (skill-gap labels, matched-skill labels), while resolution stays id-based and
        # language-neutral. Per-skill fallback to the canonical label when a pack lacks one.
        self.skill_labels = self._packs.display_labels(display_language)
        self._preferred_to_id = self._packs.preferred_to_id
        self._altlabel_to_id = self._packs.altlabel_to_id
        self._preferred_collisions = self._packs.preferred_collisions

        # Miss telemetry (per-process, lifetime-cumulative).
        self._missed_labels: Counter = Counter()

        logger.info(
            "SkillScorer: %d embedding IDs (model=%s, dim=%d); "
            "%d preferredLabel keys, %d altLabel keys (preferred-collisions: %d)",
            len(self._embedding_ids),
            model_label,
            W.shape[1],
            len(self._preferred_to_id),
            len(self._altlabel_to_id),
            self._preferred_collisions,
        )

        self.skill_group_labels = {}
        if skill_group_rows is not None:
            for row in skill_group_rows:
                group_id = row.get("ID")
                label = row.get("PREFERREDLABEL")
                if group_id and label:
                    self.skill_group_labels[str(group_id)] = label

        self._skill_to_groups: dict[str, set[str]] = {}
        if hierarchy_rows is not None:
            for row in hierarchy_rows:
                if (
                    row.get("PARENTOBJECTTYPE") == "skillgroup"
                    and row.get("CHILDOBJECTTYPE") == "skill"
                ):
                    child = str(row["CHILDID"])
                    parent = str(row["PARENTID"])
                    self._skill_to_groups.setdefault(child, set()).add(parent)
            logger.info(
                "SkillScorer: built skill→group lookup for %d skills",
                len(self._skill_to_groups),
            )
        else:
            logger.warning(
                "skill_hierarchy.csv not found — skill group derivation disabled"
            )

    def _derive_groups(self, skill_ids: set[str]) -> set[str]:
        """Look up parent skill-group IDs for a set of individual skill IDs."""
        groups = set()
        for sid in skill_ids:
            groups.update(self._skill_to_groups.get(sid, set()))
        return groups

    def _resolve_label(self, label: str) -> str | None:
        """Resolve a skill label to its internal embedding ID. Strict label-only.

        Lookup chain: preferredLabel → altLabel, in any loaded language → miss.
        UUIDs are deliberately not consulted — they carry modelId-drift risk.
        """
        if not label:
            return None
        sid = self._packs.resolve_label(label)
        if sid is not None:
            return sid
        self._missed_labels[label] += 1
        return None

    def display_labels(self, language: str | None = None) -> dict[str, str]:
        """internal_id → preferredLabel in ``language`` (canonical language when absent)."""
        return self._packs.display_labels(language)

    def get_resolution_stats(self) -> dict[str, int | list[tuple[str, int]]]:
        """Per-process miss telemetry. Useful for batch scripts at end-of-run."""
        return {
            "total_misses": sum(self._missed_labels.values()),
            "distinct_missed_labels": len(self._missed_labels),
            "top_misses": self._missed_labels.most_common(20),
        }

    def _build_objects(
        self, user: UserProfile, item: MatchItem
    ) -> tuple[Jobseeker, Opportunity, dict[str, str]]:
        """Build the Jobseeker / Opportunity pair and the user's resolved-id -> label map.

        Both sides resolve via labels (the trust anchor); item skills fall back to their taxonomy
        id. Skill groups are taken as given, or derived from the resolved skills when absent.

        :param user: The user
        :param item: The job or occupation
        :return: ``(jobseeker, opportunity, user_skill_labels)``
        :raises ValueError: If the user has no ``user_id``
        """
        if not user.user_id:
            raise ValueError("user must include user_id")

        user_skill_labels: dict[str, str] = {}
        resolved_user_ids: set[str] = set()
        for s in user.skills_vector.top_skills:
            label = s.preferredLabel
            resolved = self._resolve_label(label or "")
            if resolved is None:
                continue
            resolved_user_ids.add(resolved)
            if label:
                user_skill_labels[resolved] = label

        user_groups = {g for g in user.skill_groups_origin_uuids if g}
        if not user_groups:
            user_groups = self._derive_groups(resolved_user_ids)

        js = Jobseeker(
            compass_id=user.user_id,
            skills_origin_uuids=resolved_user_ids,
            skill_groups_origin_uuids=user_groups,
            city=user.city,
            province=user.province,
        )

        def _resolve_job_skills(items: Iterable[SkillReference]) -> set[str]:
            out: set[str] = set()
            for s in items:
                resolved = self._resolve_label(s.label)
                if resolved is None:
                    # Fallback for a label this build has no pack for: the job's `id` is a
                    # taxonomy skill ID written by the reranker, which the label packs
                    # crosswalk across locales. This is not the UUID path the docstring
                    # warns about — it is an exact per-model identifier, used only after
                    # the label misses.
                    resolved = self._packs.resolve_id(s.id)
                if resolved is not None:
                    out.add(resolved)
            return out

        resolved_ess = _resolve_job_skills(item.essential_skills)
        resolved_opt = _resolve_job_skills(item.optional_skills)

        job_groups = {g for g in item.skill_groups_origin_uuids if g}
        if not job_groups:
            job_groups = self._derive_groups(resolved_ess | resolved_opt)

        op = Opportunity(
            opportunity_id=item.uuid,
            essential_skill_ids=resolved_ess,
            optional_skill_ids=resolved_opt,
            skill_groups_origin_uuids=job_groups,
            city=item.city or item.location,
            province=item.province or item.location,
        )

        return js, op, user_skill_labels

    def calculate_score(self, user: UserProfile, item: MatchItem) -> SkillUtility:
        """Legacy skill utility ``U`` of the pair."""
        js, op, user_skill_labels = self._build_objects(user, item)
        return compute_U_complete(
            js,
            op,
            self.engine,
            skill_labels=self.skill_labels,
            user_skill_labels=user_skill_labels,
            skill_group_labels=self.skill_group_labels,
        )

    def calculate_feasibility(self, user: UserProfile, item: MatchItem) -> Feasibility:
        """Recruiter-side feasibility signals for the success-propensity proxy.

        Uses the same embedding engine as calculate_score but aggregates essential-skill
        similarities via geometric mean (stricter on gaps).
        """
        js, op, user_skill_labels = self._build_objects(user, item)
        return compute_feasibility_signals(
            js,
            op,
            self.engine,
            gate_threshold=GATE_SIMILARITY_THRESHOLD,
            skill_labels=self.skill_labels,
            user_skill_labels=user_skill_labels,
            skill_group_labels=self.skill_group_labels,
        )

    def score_utility_and_feasibility(
        self, user: UserProfile, item: MatchItem
    ) -> tuple[SkillUtility, Feasibility]:
        """Single embedding pass for multiplicative mode (U + feasibility for p_hat)."""
        js, op, user_skill_labels = self._build_objects(user, item)
        return compute_utility_and_feasibility_pair(
            js,
            op,
            self.engine,
            skill_labels=self.skill_labels,
            user_skill_labels=user_skill_labels,
            skill_group_labels=self.skill_group_labels,
            gate_threshold=GATE_SIMILARITY_THRESHOLD,
        )
