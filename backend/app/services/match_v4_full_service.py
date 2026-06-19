"""`/match_v4` full response: occupations + opportunities + skill-gaps via the Gemini engine.

Runs the v4 engine (Gemini concat cosine -> cross-encoder rerank -> u_hat x p_hat) over BOTH jobs
and occupations using a single shared user embedding, reuses the Node2Vec skill-gap analysis, and
assembles one `MatchResponse`-shaped dict per user. Per-item detail is best-effort from v4 outputs
(see match_v4_formatting).
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import (
    FINAL_SCORE_COMBINER,
    LOCATION_HUB_CHAINS_PATH,
    LOCATION_TIER_ENABLED,
    LOCATION_TIER_W_NATIONAL,
    LOCATION_TIER_W_REGIONAL,
    MATCH_TOP_K_SKILL_GAPS,
    MATCH_V4_OCC_DEMAND_GAMMA,
    MATCH_V4_TOP_K_OCCUPATIONS,
    V4_FULL_COVERAGE_GAMMA,
    V4_FULL_MIN_ESS_SHARE,
    V4_FULL_RANK_DEMOTE,
    V4_FULL_SIM_THRESHOLD,
    V4_FULL_UNPARSED_COVERAGE,
    V4_FULL_WHITENED_GATE,
)
from app.services.location_tiers import load_hub_chains
from app.services import match_v4_formatting as fmt
from app.services.gemini_ce_preference_matching.match_v3_bridge import (
    v3_recommendation_to_rec,
)
from app.services.gemini_ce_preference_matching.scoring import (
    enrich_recommendations_with_preferences,
)
from app.services.match_concat_gemini_ce_service import (
    _get_matcher,
    _get_v4_matcher,
    _job_stage1_embedding_vector,
    concat_rescale_target,
    embed_user_unit_vectors,
    run_match_concat_gemini_ce,
    whiten_concat_rows,
)
from app.services.preference_score_v1 import get_preference_scorer

__all__ = ["run_match_v4_full"]

logger = logging.getLogger(__name__)


def _index_by_uuid(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        uid = str(it.get("uuid") or it.get("_id") or "")
        if uid:
            out[uid] = it
    return out


def _user_matches_any_county(user: Dict[str, Any], counties: List[str]) -> bool:
    """True if the user's location matches at least one of the given occupation counties."""
    from app.services.matching_service import _job_matches_user_location

    for c in counties:
        if _job_matches_user_location({"city": c, "province": c, "location": c}, user):
            return True
    return False


def _enriched_recs(
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
    p_hat_by_uuid: Optional[Dict[str, float]] = None,
    coverage_by_uuid: Optional[Dict[str, float]] = None,
    coverage_gamma: float = 0.0,
    location_tier_by_uuid: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """CE recs for one user -> preference-enriched, final-score-sorted recs (rich; with details).

    With ``location_filter`` (default), recs are first restricted to items matching a location via
    the same lenient rule ``/match`` uses (``_job_matches_user_location``). ``location_user`` (a
    dict with city/province/location) overrides which location to filter by WITHOUT changing the
    ``user`` whose preferences drive ``u_hat`` — used for the random-county fallback. For
    occupations this keeps one row/code (a single county) with that county's attributes.

    Note: the post-secondary education gate is already applied upstream in
    ``run_match_concat_gemini_ce`` (it skips ineligible items before the top-k cutoff), so no
    education filtering is needed here.
    """
    ce_http = (v3_row or {}).get("concat_gemini_ce_recommendations") or []
    if location_filter and ce_http:
        from app.services.matching_service import _job_matches_user_location

        loc = location_user or user
        ce_http = [
            r
            for r in ce_http
            if isinstance(r, dict)
            and _job_matches_user_location(
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


def _location_tier_overrides(
    user: Dict[str, Any],
    v3_row: Optional[Dict[str, Any]],
    item_index: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """Per-uuid location-tier multiplier for a user's job shortlist (urban-pull Part B).

    local=1.0, regional hub=W_REGIONAL, national hub=W_NATIONAL, remote=1.0, off-chain=0.0. Returns ``{}``
    (a no-op) when the feature is disabled or the hub-chain data is unavailable. Runs independent of the
    Phase-2 coverage demotion (no whitening artifact needed)."""
    if not LOCATION_TIER_ENABLED:
        return {}
    hc = load_hub_chains(LOCATION_HUB_CHAINS_PATH)
    if hc is None:
        return {}
    county = user.get("province") or user.get("city") or ""
    tiers: Dict[str, float] = {}
    for r in (v3_row or {}).get("concat_gemini_ce_recommendations") or []:
        if not isinstance(r, dict):
            continue
        uuid = str(r.get("job_uuid") or "")
        if not uuid or uuid in tiers:
            continue
        item = item_index.get(uuid)
        if not item:
            continue
        tiers[uuid] = hc.tier_factor_for_job(
            item,
            county,
            w_regional=LOCATION_TIER_W_REGIONAL,
            w_national=LOCATION_TIER_W_NATIONAL,
        )
    return tiers


def _skill_gaps_for(
    user: Dict[str, Any], jobs: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """Reuse the existing Node2Vec skill-gap analysis (engine-agnostic). Lazy import (torch)."""
    from app.services.matching_service import (
        _filter_skill_gap_recommendations,
        _skill_gap_candidate_pool_k,
        scorer_skill,
    )
    from app.services.skill_gap_analysis import analyze_skill_gaps

    gaps = analyze_skill_gaps(
        user,
        jobs,
        scorer_skill.engine,
        scorer_skill.skill_labels,
        top_k=_skill_gap_candidate_pool_k(top_k),
        resolve_id=scorer_skill._resolve_label,
        timing_out=None,
    )
    return _filter_skill_gap_recommendations(gaps, top_k=top_k)


def run_match_v4_full(
    users: List[Dict[str, Any]],
    jobs: List[Dict[str, Any]],
    occupations: List[Dict[str, Any]],
    *,
    retrieve_top_k: int,
    final_top_k: int,
    final_score_combiner: Optional[str] = None,
    skill_gap_top_k: int = MATCH_TOP_K_SKILL_GAPS,
    mongo_timing: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return one MatchResponse-shaped dict per user (occupations + opportunities + skill-gaps)."""

    combiner = (final_score_combiner or FINAL_SCORE_COMBINER).strip().lower()
    if combiner not in ("product", "geometric_mean"):
        raise ValueError("final_score_combiner must be 'product' or 'geometric_mean'")
    if not users:
        return []

    u_norm = embed_user_unit_vectors(users)  # embed users ONCE, reuse for both corpora
    pref_scorer = get_preference_scorer()
    # Per-skill GATE matcher: whitened (default) or — via the kill-switch — the legacy raw matcher.
    matcher = _get_v4_matcher() if V4_FULL_WHITENED_GATE else _get_matcher()

    job_index = _index_by_uuid(jobs)
    occ_index = _index_by_uuid(occupations)

    # Phase-2 (V4_FULL_RANK_DEMOTE) ranking inputs. Snapshot the concat embeddings NOW — retrieval
    # (run_match_concat_gemini_ce) pops job_embedding off these dicts in place — and whiten the user
    # vectors once. When the toggle is off these stay empty and ranking is the Phase-1 behaviour.
    # Safety: if the toggle is on but the concat-whitening artifact is unavailable/incompatible
    # (target==0), DON'T half-apply Phase 2 (raw p_hat x coverage is count-biased — see notes); fall
    # back to pure Phase-1 (annotation-only, ranking unchanged) and log loudly.
    demote_active = V4_FULL_RANK_DEMOTE and concat_rescale_target() > 0
    if V4_FULL_RANK_DEMOTE and not demote_active:
        logger.error(
            "V4_FULL_RANK_DEMOTE is on but the concat-whitening artifact is unavailable; "
            "falling back to Phase-1 (no demotion, raw p_hat). Build/ship the artifact to enable Phase 2."
        )
    job_concat: Dict[str, np.ndarray] = {}
    occ_concat: Dict[str, np.ndarray] = {}
    u_white_by_uid: Dict[str, np.ndarray] = {}
    if demote_active:
        for j in jobs:
            v = _job_stage1_embedding_vector(j)
            if v is not None:
                job_concat[str(j.get("uuid") or "")] = v
        for o in occupations:
            v = _job_stage1_embedding_vector(o)
            if v is not None:
                occ_concat[str(o.get("uuid") or "")] = v
        u_white = whiten_concat_rows(u_norm)
        u_white_by_uid = {
            str(u.get("user_id") or ""): u_white[i] for i, u in enumerate(users)
        }

    job_v3 = run_match_concat_gemini_ce(
        users,
        jobs,
        retrieve_top_k=retrieve_top_k,
        final_top_k=final_top_k,
        mongo_timing=mongo_timing,
        user_unit_vectors=u_norm,
        # Urban-pull: weight the stage-1 cosine by location tier so relevant local jobs survive the
        # retrieve_top_k cutoff (occupations keep their own county scoping, so not applied there).
        apply_location_tier=LOCATION_TIER_ENABLED,
    )
    # Occupations are flattened into 4 identical-embedding county-rows per code (the fixed sample
    # counties Kilifi/Kitui/Mombasa/Nairobi). The per-user location filter (below) keeps only the
    # user's own county row per code, so the shortlist/CE slate must be wide enough that ~top_k
    # distinct codes survive: size at top_k * 4 (counties) * 2 (buffer). De-dup by code remains a
    # safety net.
    occ_breadth = max(retrieve_top_k, final_top_k, MATCH_V4_TOP_K_OCCUPATIONS * 8)
    occ_v3 = run_match_concat_gemini_ce(
        users,
        occupations,
        retrieve_top_k=occ_breadth,
        final_top_k=occ_breadth,
        user_unit_vectors=u_norm,
    )
    job_v3_by_uid = {str(r.get("user_id") or ""): r for r in job_v3}
    occ_v3_by_uid = {str(r.get("user_id") or ""): r for r in occ_v3}

    # Available occupation counties (Kilifi/Kitui/Mombasa/Nairobi). Safety net: if a user's province
    # matches none of them, fall back to a random available county so occupations still return.
    occ_counties = sorted(
        {str(o.get("province")) for o in occupations if o.get("province")}
    )

    def _skill_detail(user, item):
        """Return (per_job_skill, matcher-resolved essential id set) for matched_skills.

        Both sides go through CosineSkillMatcher._resolve_label, so the essential id set is in
        the same (label-resolved) id space as per_job_skill[].job_skill_id — the split is robust
        to id/label mismatches.
        """
        try:
            _score = (
                matcher.score_pair_v4 if V4_FULL_WHITENED_GATE else matcher.score_pair
            )
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

    def _rank_overrides(user, v3_row, item_index, concat_by_uuid, u_white_vec):
        """Phase-2 per-candidate ranking inputs for one user's shortlist:
        - p_hat override = whitened+rescaled concat cosine(user, item) in [0,1]
        - coverage = essential-coverage in [0,1] (drives the achievability demotion)
        - detail cache {uuid: (per_job_skill, essential_ids)} reused by the formatters (no re-score).
        """
        p_over: Dict[str, float] = {}
        cov_over: Dict[str, float] = {}
        det_cache: Dict[str, Any] = {}
        parsed_covs: List[float] = []  # coverages of items WITH parsed essential skills
        unparsed_uuids: List[str] = []  # items with no essential skills (back-filled below)
        target = concat_rescale_target()
        ce = (v3_row or {}).get("concat_gemini_ce_recommendations") or []
        for r in ce:
            if not isinstance(r, dict):
                continue
            uuid = str(r.get("job_uuid") or "")
            item = item_index.get(uuid)
            if not item or uuid in det_cache:
                continue
            jv = concat_by_uuid.get(uuid)
            if jv is not None and u_white_vec is not None and target > 0:
                jw = whiten_concat_rows(jv.reshape(1, -1))[0]
                cos = float(np.dot(u_white_vec, jw))
                p_over[uuid] = min(1.0, max(0.0, cos) / target)
            per, ess_ids = _skill_detail(user, item)
            det_cache[uuid] = (per, ess_ids)
            ms = fmt.build_matched_skills(
                per, ess_ids, sim_threshold=V4_FULL_SIM_THRESHOLD
            )
            n_ess = len(item.get("essential_skills") or [])
            if n_ess:
                cov = fmt.essential_coverage(ms["essential_skill_matches"], n_ess)
                cov_over[uuid] = cov
                parsed_covs.append(cov)
            else:
                unparsed_uuids.append(uuid)
        # Back-fill unparsed items with the typical (mean) parsed coverage of this shortlist (or the
        # configured override / neutral fallback) so they no longer escape the achievability demotion.
        if unparsed_uuids:
            fill = fmt.unparsed_ranking_coverage(
                parsed_covs, override=V4_FULL_UNPARSED_COVERAGE
            )
            for uuid in unparsed_uuids:
                cov_over[uuid] = fill
        return p_over, cov_over, det_cache

    out: List[Dict[str, Any]] = []
    cov_gamma = V4_FULL_COVERAGE_GAMMA if demote_active else 0.0
    for user in users:
        uid = str(user.get("user_id") or "")

        # Phase-2 ranking overrides (whitened p_hat + coverage demotion); empty dicts when toggle off,
        # in which case _enriched_recs falls back to the raw concat p_hat with no demotion (Phase 1).
        job_p, job_cov, job_det = ({}, {}, {})
        occ_p, occ_cov, occ_det = ({}, {}, {})
        if demote_active:
            uw = u_white_by_uid.get(uid)
            job_p, job_cov, job_det = _rank_overrides(
                user, job_v3_by_uid.get(uid), job_index, job_concat, uw
            )
            occ_p, occ_cov, occ_det = _rank_overrides(
                user, occ_v3_by_uid.get(uid), occ_index, occ_concat, uw
            )

        # Opportunities. Jobs keep the existing /match_v4 location scoping (Mongo prefilter via
        # get_all_jobs_with_timing(users=...)). Instead of a hard python location filter, urban-pull
        # applies a per-user SOFT location tier (local=1.0 > regional hub > national hub; off-chain=0)
        # as a final_score multiplier — local jobs preferred, hub jobs surface when better/needed, and
        # off-chain jobs (e.g. another batch user's locations) are dropped. Always-on (independent of
        # the Phase-2 coverage demotion); {} no-op when LOCATION_TIER_ENABLED is off.
        job_tiers = _location_tier_overrides(user, job_v3_by_uid.get(uid), job_index)
        opportunities: List[Dict[str, Any]] = []
        for rec in _enriched_recs(
            user,
            job_v3_by_uid.get(uid),
            job_index,
            pref_scorer,
            combiner,
            location_filter=False,
            p_hat_by_uuid=job_p,
            coverage_by_uuid=job_cov,
            coverage_gamma=cov_gamma,
            location_tier_by_uuid=job_tiers,
        ):
            item = job_index.get(str(rec.get("job_uuid") or ""))
            if not item:
                continue
            per, ess_ids = job_det.get(str(rec.get("job_uuid") or "")) or _skill_detail(
                user, item
            )
            opportunities.append(
                fmt.build_opportunity_row(
                    rec,
                    item,
                    per,
                    ess_ids,
                    rank=len(opportunities) + 1,
                    sim_threshold=V4_FULL_SIM_THRESHOLD,
                    min_ess_share=V4_FULL_MIN_ESS_SHARE,
                )
            )

        # Occupations: filter to the user's county; if the user's province matches no occupation
        # county, fall back to a random available county (location filter only — the user's real
        # preferences still drive u_hat). Then dedupe by code, keep best-ranked, take top-k.
        loc_user = None
        if occ_counties and not _user_matches_any_county(user, occ_counties):
            fallback = random.choice(occ_counties)
            loc_user = {"city": fallback, "province": fallback, "location": fallback}
            logger.warning(
                "User %r province=%r matches no occupation county %s; using random fallback county %r.",
                uid,
                user.get("province"),
                occ_counties,
                fallback,
            )
        occupations_out: List[Dict[str, Any]] = []
        seen_codes: set = set()
        for rec in _enriched_recs(
            user,
            occ_v3_by_uid.get(uid),
            occ_index,
            pref_scorer,
            combiner,
            location_user=loc_user,
            include_demand=True,
            demand_gamma=MATCH_V4_OCC_DEMAND_GAMMA,
            p_hat_by_uuid=occ_p,
            coverage_by_uuid=occ_cov,
            coverage_gamma=cov_gamma,
        ):
            item = occ_index.get(str(rec.get("job_uuid") or ""))
            if not item:
                continue
            code = str(item.get("originUuid") or item.get("uuid") or "")
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            per, ess_ids = occ_det.get(str(rec.get("job_uuid") or "")) or _skill_detail(
                user, item
            )
            occupations_out.append(
                fmt.build_occupation_row(
                    rec,
                    item,
                    per,
                    ess_ids,
                    rank=len(occupations_out) + 1,
                    sim_threshold=V4_FULL_SIM_THRESHOLD,
                    min_ess_share=V4_FULL_MIN_ESS_SHARE,
                )
            )
            if len(occupations_out) >= MATCH_V4_TOP_K_OCCUPATIONS:
                break

        out.append(
            {
                "user_id": uid,
                "occupation_recommendations": occupations_out,
                "opportunity_recommendations": opportunities,
                "skill_gap_recommendations": _skill_gaps_for(
                    user, jobs, skill_gap_top_k
                ),
            }
        )

    return out
