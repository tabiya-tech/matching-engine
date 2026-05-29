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

from app.config import (
    FINAL_SCORE_COMBINER,
    MATCH_TOP_K_SKILL_GAPS,
    MATCH_V4_TOP_K_OCCUPATIONS,
    V4_FULL_MIN_ESS_SHARE,
    V4_FULL_SIM_THRESHOLD,
)
from app.services import match_v4_formatting as fmt
from app.services.gemini_ce_preference_matching.match_v3_bridge import v3_recommendation_to_rec
from app.services.gemini_ce_preference_matching.scoring import (
    enrich_recommendations_with_preferences,
)
from app.services.match_concat_gemini_ce_service import (
    _get_matcher,
    embed_user_unit_vectors,
    run_match_concat_gemini_ce,
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
    user, v3_row, item_index, pref_scorer, combiner, *, location_filter=True, location_user=None
) -> List[Dict[str, Any]]:
    """CE recs for one user -> preference-enriched, final-score-sorted recs (rich; with details).

    With ``location_filter`` (default), recs are first restricted to items matching a location via
    the same lenient rule ``/match`` uses (``_job_matches_user_location``). ``location_user`` (a
    dict with city/province/location) overrides which location to filter by WITHOUT changing the
    ``user`` whose preferences drive ``u_hat`` — used for the random-county fallback. For
    occupations this keeps one row/code (a single county) with that county's attributes.
    """
    ce_http = (v3_row or {}).get("concat_gemini_ce_recommendations") or []
    if location_filter and ce_http:
        from app.services.matching_service import _job_matches_user_location

        loc = location_user or user
        ce_http = [
            r for r in ce_http
            if isinstance(r, dict)
            and _job_matches_user_location(item_index.get(str(r.get("job_uuid") or "")) or {}, loc)
        ]
    ce_internal = [v3_recommendation_to_rec(r, item_index) for r in ce_http if isinstance(r, dict)]
    if not ce_internal:
        return []
    return enrich_recommendations_with_preferences(
        user,
        ce_internal,
        item_index,
        preference_scorer=pref_scorer,
        include_work_activities=True,
        final_score_combiner=combiner,
    )


def _skill_gaps_for(user: Dict[str, Any], jobs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Reuse the existing Node2Vec skill-gap analysis (engine-agnostic). Lazy import (torch)."""
    from app.services.matching_service import _filter_skill_gap_recommendations, scorer_skill
    from app.services.skill_gap_analysis import analyze_skill_gaps

    gaps = analyze_skill_gaps(
        user,
        jobs,
        scorer_skill.engine,
        scorer_skill.skill_labels,
        top_k=top_k,
        resolve_id=scorer_skill._resolve_label,
        timing_out=None,
    )
    return _filter_skill_gap_recommendations(gaps)


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
    matcher = _get_matcher()

    job_index = _index_by_uuid(jobs)
    occ_index = _index_by_uuid(occupations)

    job_v3 = run_match_concat_gemini_ce(
        users, jobs, retrieve_top_k=retrieve_top_k, final_top_k=final_top_k,
        mongo_timing=mongo_timing, user_unit_vectors=u_norm,
    )
    # Occupations are flattened into 4 identical-embedding county-rows per code (the fixed sample
    # counties Kilifi/Kitui/Mombasa/Nairobi). The per-user location filter (below) keeps only the
    # user's own county row per code, so the shortlist/CE slate must be wide enough that ~top_k
    # distinct codes survive: size at top_k * 4 (counties) * 2 (buffer). De-dup by code remains a
    # safety net.
    occ_breadth = max(retrieve_top_k, final_top_k, MATCH_V4_TOP_K_OCCUPATIONS * 8)
    occ_v3 = run_match_concat_gemini_ce(
        users, occupations, retrieve_top_k=occ_breadth, final_top_k=occ_breadth,
        user_unit_vectors=u_norm,
    )
    job_v3_by_uid = {str(r.get("user_id") or ""): r for r in job_v3}
    occ_v3_by_uid = {str(r.get("user_id") or ""): r for r in occ_v3}

    # Available occupation counties (Kilifi/Kitui/Mombasa/Nairobi). Safety net: if a user's province
    # matches none of them, fall back to a random available county so occupations still return.
    occ_counties = sorted({str(o.get("province")) for o in occupations if o.get("province")})

    def _skill_detail(user, item):
        """Return (per_job_skill, matcher-resolved essential id set) for matched_skills.

        Both sides go through CosineSkillMatcher._resolve_label, so the essential id set is in
        the same (label-resolved) id space as per_job_skill[].job_skill_id — the split is robust
        to id/label mismatches.
        """
        try:
            per = matcher.score_pair(user, item).get("per_job_skill", []) or []
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

    out: List[Dict[str, Any]] = []
    for user in users:
        uid = str(user.get("user_id") or "")

        # Opportunities. Jobs keep the existing /match_v4 location scoping (Mongo prefilter via
        # get_all_jobs_with_timing(users=...)); no extra python location filter so we don't risk
        # dropping jobs whose location format differs from the user's.
        opportunities: List[Dict[str, Any]] = []
        for rec in _enriched_recs(user, job_v3_by_uid.get(uid), job_index, pref_scorer, combiner, location_filter=False):
            item = job_index.get(str(rec.get("job_uuid") or ""))
            if not item:
                continue
            per, ess_ids = _skill_detail(user, item)
            opportunities.append(
                fmt.build_opportunity_row(
                    rec, item, per, ess_ids,
                    rank=len(opportunities) + 1,
                    sim_threshold=V4_FULL_SIM_THRESHOLD, min_ess_share=V4_FULL_MIN_ESS_SHARE,
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
                uid, user.get("province"), occ_counties, fallback,
            )
        occupations_out: List[Dict[str, Any]] = []
        seen_codes: set = set()
        for rec in _enriched_recs(
            user, occ_v3_by_uid.get(uid), occ_index, pref_scorer, combiner, location_user=loc_user
        ):
            item = occ_index.get(str(rec.get("job_uuid") or ""))
            if not item:
                continue
            code = str(item.get("originUuid") or item.get("uuid") or "")
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            per, ess_ids = _skill_detail(user, item)
            occupations_out.append(
                fmt.build_occupation_row(
                    rec, item, per, ess_ids,
                    rank=len(occupations_out) + 1,
                    sim_threshold=V4_FULL_SIM_THRESHOLD, min_ess_share=V4_FULL_MIN_ESS_SHARE,
                )
            )
            if len(occupations_out) >= MATCH_V4_TOP_K_OCCUPATIONS:
                break

        out.append(
            {
                "user_id": uid,
                "occupation_recommendations": occupations_out,
                "opportunity_recommendations": opportunities,
                "skill_gap_recommendations": _skill_gaps_for(user, jobs, skill_gap_top_k),
            }
        )

    return out
