"""Per-candidate scoring after retrieval: skill coverage, Phase-2 overrides, demand, success propensity, preference enrichment (final score) and occupation slate selection."""

from __future__ import annotations

import inspect
import logging
from typing import Any

import numpy as np

from app.config import (
    DEMAND_SCORE_MAPPING,
    FINAL_SCORE_COMBINER,
    SUCCESS_PROPENSITY_CONFIG,
    V4_FULL_BADGE_PARTIAL,
    V4_FULL_BADGE_STRONG,
    V4_FULL_COVERAGE_FLOOR,
)
from app.ranking.preference import combine_final_score
from app.ranking.retrieval import (
    ConcatWhitener,
    job_matches_user_location,
    job_skill_labels_for_concat,
)
from app.ranking.skills import CosineSkillMatcher

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# demand
# --------------------------------------------------------------------------------------------------

# src/demand_scorer.py


class DemandScorer:
    def __init__(self):
        self.mapping = DEMAND_SCORE_MAPPING

    def calculate_score(self, job_posting: dict) -> dict:
        """
        Computes the S_demand score.

        Logic:
        1. Checks job_posting['attributes']['expected_demand'] -> Maps text to float.
        2. Fallback to 0.5 if missing.
        """

        # 1. Safely extract the attributes dictionary
        attributes = job_posting.get("attributes", {})

        # 2. Get the specific key
        demand_label = attributes.get("expected_demand")

        # 3. Map label to value
        if demand_label and demand_label in self.mapping:
            return {
                "score": self.mapping[demand_label],
                "label": demand_label,
                "present": True,
            }

        # 4. Fallback / Default
        return {
            "score": 0.5,
            "label": demand_label,
            "present": False,
        }


# --------------------------------------------------------------------------------------------------
# coverage
#
# Per-item skill coverage: matched-skill split, essential coverage, coverage badge and eligibility.
# --------------------------------------------------------------------------------------------------


def build_matched_skills(
    per_job_skill: Any,
    essential_ids: set,
    *,
    sim_threshold: float,
) -> dict[str, list[dict]]:
    """Build a MatchedSkills-shaped dict from CosineSkillMatcher ``per_job_skill`` rows.

    ``essential_ids`` MUST be the matcher-RESOLVED ids of the item's essential skills (resolved via
    the same label->id function the matcher uses to populate ``job_skill_id``), so the split is in
    the same id space. Every ``per_job_skill`` row is a resolved essential ∪ optional skill: rows in
    ``essential_ids`` -> essential (with per-skill cosine + meets_threshold); the rest are optional
    (kept as OptionalSkillMatch when cosine clears the threshold). Skill groups are not computed by
    the Gemini engine (empty list).
    """
    essential: list[dict] = []
    optional: list[dict] = []
    for r in per_job_skill or []:
        if not isinstance(r, dict):
            continue
        jid = str(r.get("job_skill_id") or "")
        if not jid:
            continue
        sim = float(r.get("cosine_similarity") or 0.0)
        exact = bool(r.get("exact"))
        meets = exact or (
            sim >= sim_threshold
        )  # exact-id overlap always counts as "has it"
        tier = "exact" if exact else ("embedding" if sim >= sim_threshold else "none")
        if jid in essential_ids:
            essential.append(
                {
                    "job_skill_id": jid,
                    "job_skill_label": r.get("job_skill_label"),
                    "best_user_skill_id": r.get("best_user_skill_id"),
                    "best_user_skill_label": r.get("best_user_skill_label"),
                    "similarity": round(sim, 4),
                    "meets_threshold": meets,
                    "match_tier": tier,
                }
            )
        elif meets:
            optional.append({"skill_id": jid, "skill_label": r.get("job_skill_label")})
    return {
        "essential_skill_matches": essential,
        "optional_exact_matches": optional,
        "skill_group_matches": [],
    }


def essential_coverage(essential_matches: list[dict], n_essential_total: int) -> float:
    """Share of a job's essential skills the user meets, in [0,1]. Denominator is the job's TOTAL
    essential count, so unresolved essentials (absent from ``essential_matches``) count as not-met."""
    n = max(int(n_essential_total or 0), len(essential_matches or []))
    if n == 0:
        return 1.0
    met = sum(1 for m in (essential_matches or []) if m.get("meets_threshold"))
    return met / n


def unparsed_ranking_coverage(parsed_covs: list[float], *, override: float) -> float:
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
    essential_matches: list[dict], *, n_essential_total: int, min_ess_share: float
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
    user: dict[str, Any],
    item: dict[str, Any],
    *,
    whitened_gate: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Return (per_job_skill, matcher-resolved essential id set) for matched_skills.

    Both sides go through CosineSkillMatcher._resolve_label, so the essential id set is in
    the same (label-resolved) id space as per_job_skill[].job_skill_id — the split is robust
    to id/label mismatches.
    """
    try:
        _score = matcher.score_pair_v4 if whitened_gate else matcher.score_pair
        per = _score(user, item).get("per_job_skill", []) or []
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("score_pair failed for %s: %s", item.get("uuid"), e)
        per = []
    ess_ids = set()
    for s in item.get("essential_skills") or []:
        lab = s.get("label")
        rid = matcher._resolve_label(lab) if lab else None
        if rid:
            ess_ids.add(rid)
    return per, ess_ids


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
        user: dict[str, Any],
        v3_row: dict[str, Any] | None,
        item_index: dict[str, dict[str, Any]],
        concat_by_uuid: dict[str, tuple],
        u_white_vec: np.ndarray | None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
        """
        - p_hat override = whitened+rescaled concat cosine(user, item) in [0,1]
        - coverage = essential-coverage in [0,1] (drives the achievability demotion)
        - detail cache {uuid: (per_job_skill, essential_ids)} reused by the formatters (no re-score).
        """
        p_over: dict[str, float] = {}
        cov_over: dict[str, float] = {}
        det_cache: dict[str, Any] = {}
        parsed_covs: list[float] = []  # coverages of items WITH parsed essential skills
        unparsed_uuids: list[
            str
        ] = []  # items with no essential skills (back-filled below)
        target = self._whitener.rescale_target()
        ce = (v3_row or {}).get("concat_gemini_ce_recommendations") or []
        for r in ce:
            if not isinstance(r, dict):
                continue
            uuid = str(r.get("job_uuid") or "")
            item = item_index.get(uuid)
            if not item or uuid in det_cache:
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
            per, ess_ids = skill_detail(
                self._matcher, user, item, whitened_gate=self._whitened_gate
            )
            det_cache[uuid] = (per, ess_ids)
            ms = build_matched_skills(per, ess_ids, sim_threshold=self._sim_threshold)
            n_ess = len(item.get("essential_skills") or [])
            if n_ess:
                cov = essential_coverage(ms["essential_skill_matches"], n_ess)
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
        return p_over, cov_over, det_cache


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


class SuccessPropensityScorer:
    def __init__(self):
        self.cfg = SUCCESS_PROPENSITY_CONFIG
        self.demand_mapping = DEMAND_SCORE_MAPPING

    # ------------------------------------------------------------------
    # M_ij: market opportunity
    # ------------------------------------------------------------------
    def _market_opportunity(self, job_posting: dict) -> float:
        """Compute market-opportunity signal from demand label.

        Currently uses the same demand-label mapping as the legacy DemandScorer.
        When richer market data become available (tightness, penetration,
        absorptive capacity, freshness), this method is the place to extend.
        """
        attributes = job_posting.get("attributes", {})
        demand_label = attributes.get("expected_demand")
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
        return sum(w * t for w, t in zip(weights, terms)) / sum(weights)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def calculate_score(
        self,
        user_profile: dict,
        job_posting: dict,
        feasibility: dict,
    ) -> dict:
        """Compute p_hat from pre-computed feasibility signals.

        Parameters
        ----------
        user_profile : dict
            User profile (currently unused here, but kept for future
            confidence / shrinkage extensions).
        job_posting : dict
            Job or occupation record.
        feasibility : dict
            Output of ``compute_feasibility_signals()`` — contains gate,
            essential fit, optional sim, skill group recall, etc.

        Returns
        -------
        dict with keys:
            p_hat            – float in [0, 1]
            components       – dict of G, E, R, M values
            demand_label     – str, the raw demand label (for display)
        """
        alpha = self.cfg["alpha_essential"]
        beta = self.cfg["beta_readiness"]
        gamma = self.cfg["gamma_market"]

        # --- components ---
        g_ij = 1.0 if feasibility["gate_passed"] else 0.0
        e_ij = feasibility["essential_fit"]
        r_ij = self._recruiter_readiness(
            feasibility["optional_sim"],
            feasibility["skill_group_recall"],
            has_optional_skills=feasibility.get("has_optional_skills", False),
            has_skill_groups=feasibility.get("has_skill_groups", False),
        )

        attributes = job_posting.get("attributes", {})
        demand_label = attributes.get("expected_demand")
        has_demand = demand_label and demand_label in self.demand_mapping

        if has_demand:
            m_ij = self.demand_mapping[demand_label]
        else:
            m_ij = None

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

        return {
            "p_hat": round(p_hat, 4),
            "components": {
                "gate": round(g_ij, 4),
                "essential_fit": round(e_ij, 4),
                "recruiter_readiness": round(r_ij, 4),
                "market_opportunity": round(m_ij, 4),
            },
            "demand_label": demand_label or "Unknown",
        }


# --------------------------------------------------------------------------------------------------
# enrichment
#
# Ranker: combine u_hat (preferences) and p_hat (skill retrieval score).
# --------------------------------------------------------------------------------------------------

# Engine-agnostic demand scorer (reads item attributes["expected_demand"]); torch-free.
_DEMAND_SCORER = DemandScorer()


def work_activity_match_for_dashboard(details: Any) -> dict[str, Any]:
    """Extract Part B (BWS × job work activities) for one job row."""
    for d in details or []:
        if not isinstance(d, dict) or d.get("attribute") != "work_activity_bws":
            continue
        wa_rows = []
        for w in d.get("wa_details") or []:
            if not isinstance(w, dict):
                continue
            wa_rows.append(
                {
                    "wa_code": w.get("wa_code"),
                    "wa_label": w.get("wa_label") or w.get("wa_code"),
                    "user_bws": w.get("user_bws"),
                    "wa_importance": w.get("wa_importance"),
                    "wa_level": w.get("wa_level"),
                    "wa_contribution": w.get("wa_contribution"),
                }
            )
        wa_rows.sort(
            key=lambda r: (
                -abs(float(r.get("wa_contribution") or 0)),
                str(r.get("wa_code") or ""),
            )
        )
        return {
            "S_wa": d.get("wa_score_sum"),
            "n_work_activities": d.get("n_work_activities") or len(wa_rows),
            "wa_aggregation": d.get("wa_aggregation") or "mean",
            "rows": wa_rows[:25],
        }
    return {"S_wa": None, "n_work_activities": 0, "wa_aggregation": "mean", "rows": []}


def preference_details_for_dashboard(details: Any) -> list[dict[str, Any]]:
    """Compact attribute-level preference rows for HTML (user weight vs job level)."""
    out: list[dict[str, Any]] = []
    for d in details or []:
        if not isinstance(d, dict):
            continue
        if d.get("layer") != "dce_attributes":
            continue
        attr = d.get("attribute")
        if not attr or attr in ("dce_utility",):
            continue
        if d.get("on_job") is False:
            continue
        out.append(
            {
                "attribute": attr,
                "attr_label": d.get("attr_label") or attr,
                "user_weight": d.get("user_weight"),
                "job_value": d.get("job_value"),
                "job_level_resolved": d.get("job_level_resolved"),
                "job_level_label": d.get("job_level_label"),
                "encoded_value": d.get("encoded_value"),
                "contribution": d.get("contribution"),
                "orientation": d.get("orientation"),
            }
        )
    out.sort(key=lambda r: (-abs(float(r.get("contribution") or 0)), r["attr_label"]))
    return out


def p_hat_from_skill_rec(rec: dict[str, Any]) -> tuple[float, str]:
    """
    p_hat = raw cosine similarity from match_v3 stage-1 (not CE min–max score).

    ``cross_encoder_score`` is only used for CE *ordering*; it is per-user min–max
    in [0, 1] so the top row is often 1.0 and must not be used as p_hat.
    """
    cos = rec.get("concat_cosine_similarity")
    if cos is not None:
        return float(cos), "concat_cosine_similarity"
    mbc = rec.get("mean_best_cosine")
    if mbc is not None:
        return float(mbc), "mean_best_cosine"
    return 0.0, "mean_best_cosine"


def compute_final_score(
    pref_result: dict[str, Any],
    p_hat: float,
    *,
    p_hat_source: str,
    combiner: str,
) -> tuple[float, dict[str, Any]]:
    """Compute final score from u_hat and p_hat (configurable)."""
    u_hat = float(pref_result.get("u_hat", 0.5))
    final = combine_final_score(u_hat, p_hat, combiner=combiner)  # type: ignore[arg-type]

    breakdown = {
        "scoring_mode": "multiplicative",
        "final_score_combiner": combiner,
        "u_hat": round(u_hat, 4),
        "p_hat": round(p_hat, 4),
        "p_hat_source": p_hat_source,
        "final_score": round(final, 4),
        "preference_score_legacy": round(float(pref_result.get("score", 0.0)), 4),
    }
    return final, breakdown


def enrich_recommendations_with_preferences(
    user: dict[str, Any],
    recs: list[dict[str, Any]],
    jobs_by_uuid: dict[str, dict[str, Any]],
    *,
    preference_scorer,
    include_work_activities: bool = True,
    final_score_combiner: str | None = None,
    include_demand: bool = False,
    demand_gamma: float = 0.0,
    p_hat_by_uuid: dict[str, float] | None = None,
    coverage_by_uuid: dict[str, float] | None = None,
    coverage_gamma: float = 0.0,
    location_tier_by_uuid: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Stage 3 only: compute u_hat per job, p_hat from stage 1–2 cosine, re-rank by u_hat × p_hat.

    Input ``recs`` must stay in CE order for ``cross_encoder_recommendations`` export;
    this function returns a new list sorted by ``final_score``.

    ``include_demand`` (with ``demand_gamma`` > 0) applies an occupation-only labour-market tilt:
    ``final *= M ** demand_gamma`` where M = expected-demand score in [0,1] (neutral 1.0 when the
    item has no/unknown demand). Opportunities call with ``include_demand=False`` (the default),
    so their ranking is unaffected.
    """
    scored: list[tuple[float, float, int, dict[str, Any]]] = []

    for i, rec in enumerate(recs):
        uid = str(rec.get("job_uuid") or "")
        job = jobs_by_uuid.get(uid)
        if job is None:
            continue

        calc = preference_scorer.calculate_score
        if "include_work_activities" in inspect.signature(calc).parameters:
            pref = calc(user, job, include_work_activities=include_work_activities)
        else:
            pref = calc(user, job)
        p_hat, p_hat_source = p_hat_from_skill_rec(rec)
        # Phase-2 (V4_FULL_RANK_DEMOTE): override p_hat's skills-fit with the WHITENED+rescaled concat
        # cosine (de-anisotropised, discriminative). Still in [0,1]; final = u_hat x p_hat unchanged.
        if p_hat_by_uuid is not None and uid in p_hat_by_uuid:
            p_hat = max(0.0, min(1.0, float(p_hat_by_uuid[uid])))
            p_hat_source = "concat_cosine_whitened"
        combiner = (final_score_combiner or FINAL_SCORE_COMBINER).strip().lower()
        final, breakdown = compute_final_score(
            pref, p_hat, p_hat_source=p_hat_source, combiner=combiner
        )

        # Occupation-only demand tilt (opportunities pass include_demand=False): multiply the
        # final score by M**gamma — M = expected-demand score in [0,1], neutral 1.0 when the item
        # has no/unknown demand. Mirrors the legacy p_hat market factor; this re-ranks occupations.
        if include_demand and demand_gamma > 0:
            dres = _DEMAND_SCORER.calculate_score(job)
            if dres.get("present"):
                m = max(0.0, min(1.0, float(dres.get("score") or 0.0)))
                final = final * (m**demand_gamma)
                breakdown["final_score"] = round(final, 4)
                breakdown["demand_score"] = round(m, 4)
                breakdown["demand_label"] = dres.get("label")
                breakdown["demand_gamma"] = round(float(demand_gamma), 4)

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
                breakdown["final_score"] = round(final, 4)
                breakdown["essential_coverage"] = round(cov, 4)
                breakdown["coverage_gamma"] = round(float(coverage_gamma), 4)
                breakdown["coverage_floor"] = round(floor, 4)
                breakdown["coverage_factor"] = round(factor, 4)

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
                breakdown["final_score"] = round(final, 4)
                breakdown["location_tier_factor"] = round(tier, 4)

        row = dict(rec)
        row["rank_cross_encoder"] = rec.get("rank")
        row["u_hat"] = pref.get("u_hat")
        row["preference_score"] = pref.get("score")
        row["preference_details"] = pref.get("details", [])
        row["preference_match_rows"] = preference_details_for_dashboard(
            pref.get("details")
        )
        row["work_activity_match"] = work_activity_match_for_dashboard(
            pref.get("details")
        )
        row["S_attrs"] = pref.get("S_attrs")
        row["S_wa"] = pref.get("S_wa")
        row["preference_include_work_activities"] = include_work_activities
        row["p_hat"] = round(p_hat, 4)
        row["p_hat_source"] = p_hat_source
        row["score_breakdown"] = breakdown
        row["final_score"] = breakdown["final_score"]
        scored.append((final, p_hat, i, row))

    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    out: list[dict[str, Any]] = []
    for rank, (_fs, _ph, _i, row) in enumerate(scored, start=1):
        row["rank"] = rank
        out.append(row)
    return out


def v3_recommendation_to_rec(
    row: dict[str, Any],
    job_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Map one ``concat_gemini_ce_recommendations`` row to the batch runner shape.

    Scores use raw ``concat_cosine_similarity`` (match_v3). Skill display uses
    concat label lists only (no per-skill embedding pairs).
    """
    jid = str(row.get("job_uuid") or "").strip()
    cos = row.get("concat_cosine_similarity")
    try:
        cos_f = float(cos) if cos is not None else None
    except (TypeError, ValueError):
        cos_f = None

    job = job_index.get(jid)
    job_skills = job_skill_labels_for_concat(job) if job else []

    return {
        "rank": row.get("rank"),
        "job_uuid": jid,
        "job_title": row.get("opportunity_title") or row.get("job_title"),
        "employer": row.get("employer"),
        "location": row.get("location"),
        "concat_cosine_similarity": cos_f,
        "mean_best_cosine": round(cos_f, 4) if cos_f is not None else None,
        "cross_encoder_logit": row.get("cross_encoder_logit"),
        "cross_encoder_score": row.get("cross_encoder_score"),
        "rank_cosine": row.get("rank_cosine"),
        "job_concat_skills": job_skills,
    }


def enriched_recommendations(
    user,
    v3_row,
    item_index,
    pref_scorer,
    combiner,
    *,
    location_filter=True,
    location_user=None,
    include_demand: bool = False,
    demand_gamma: float = 0.0,
    p_hat_by_uuid: dict[str, float] | None = None,
    coverage_by_uuid: dict[str, float] | None = None,
    coverage_gamma: float = 0.0,
    location_tier_by_uuid: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """CE recs for one user -> preference-enriched, final-score-sorted recs (rich; with details).

    With ``location_filter`` (default), recs are first restricted to items matching a location via
    the same lenient rule ``/match`` uses (``job_matches_user_location``). ``location_user`` (a
    dict with city/province/location) overrides which location to filter by WITHOUT changing the
    ``user`` whose preferences drive ``u_hat`` — used for the random-county fallback. For
    occupations this keeps one row/code (a single county) with that county's attributes.

    Note: the post-secondary education gate is already applied upstream in stage-1 retrieval
    (it skips ineligible items before the top-k cutoff), so no education filtering is needed here.
    """
    ce_http = (v3_row or {}).get("concat_gemini_ce_recommendations") or []
    if location_filter and ce_http:
        loc = location_user or user
        ce_http = [
            r
            for r in ce_http
            if isinstance(r, dict)
            and job_matches_user_location(
                item_index.get(str(r.get("job_uuid") or "")) or {}, loc
            )
        ]
    ce_internal = [
        v3_recommendation_to_rec(r, item_index) for r in ce_http if isinstance(r, dict)
    ]
    if not ce_internal:
        return []
    return enrich_recommendations_with_preferences(
        user,
        ce_internal,
        item_index,
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
