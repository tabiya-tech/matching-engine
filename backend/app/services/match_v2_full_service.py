"""`/experiments/v2/match` full response: occupations + opportunities + skill-gaps via the v2 engine.

Keeps the v2 **matching formula** unchanged (BM25 × embedding-cosine pool min-max fusion,
``hybrid_match_users_with_jobs``) but assembles the same ``MatchResponse`` shape ``/match_v4``
returns: one ``MatchResponse``-shaped dict per user with opportunity, occupation and skill-gap
recommendations.

Differences from the v2 ``hybrid_recommendations`` envelope (by design):
* ``final_score`` = the hybrid ``fusion_score``; v4-only ``u_hat``/``p_hat``/preference fields are
  left empty (the v2 engine computes no preference / p_hat signal).
* Occupations are scored with the **same** hybrid engine over the occupation corpus, then filtered
  to the user's county (with v4's random-county fallback) and de-duped by occupation code.
* ``matched_skills`` is rebuilt from the shared ``CosineSkillMatcher`` (taxonomy/uuid mapping),
  identical to the v4 path.
* Skill gaps reuse the existing ``/match_v4`` analysis unchanged.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

from app.config import (
    MATCH_TOP_K_SKILL_GAPS,
    MATCH_V4_TOP_K_OCCUPATIONS,
    V4_FULL_MIN_ESS_SHARE,
    V4_FULL_SIM_THRESHOLD,
)
from app.services import match_v4_formatting as fmt
from app.services.hybrid_scoring.run_bm25_cosine_hybrid import (
    get_cosine_matcher_singleton,
    hybrid_match_users_with_jobs,
)
from app.services.match_v4_full_service import _skill_gaps_for, _user_matches_any_county
from app.services.matching_service import _job_matches_user_location

__all__ = ["run_match_v2_full"]

logger = logging.getLogger(__name__)


def _index_by_uuid(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        uid = str(it.get("uuid") or it.get("_id") or "")
        if uid:
            out[uid] = it
    return out


def _occ_to_hybrid_item(o: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow copy of an occupation with job-style text keys so BM25 sees real content.

    The hybrid full-text corpus reads ``opportunity_title`` / ``opportunity_description`` (job
    fields). Occupations store ``occupation_label`` / ``description``; aliasing them lets the SAME
    BM25 + cosine engine rank occupations exactly like opportunities. Skills/location are already
    present and used as-is. The original occupation dict is untouched (used later for formatting).
    """
    item = dict(o)
    item.setdefault(
        "opportunity_title", o.get("occupation_label") or o.get("preferredLabel") or ""
    )
    item.setdefault("opportunity_description", o.get("description") or "")
    return item


def _skill_detail(matcher, user: Dict[str, Any], item: Dict[str, Any]):
    """(per_job_skill, matcher-resolved essential id set) — same contract as the v4 path.

    Both sides go through ``CosineSkillMatcher._resolve_label`` so the essential id set is in the
    same (label-resolved) id space as ``per_job_skill[].job_skill_id``.
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


def _v2_rec(fused_row: Dict[str, Any]) -> Dict[str, Any]:
    """Map one hybrid fused row to the generic ``rec`` the v4 formatters consume.

    ``final_score`` is the hybrid fusion score; v4-only signals are left empty so the formatter
    emits null ``u_hat``/``p_hat`` and no preferences (the v2 engine produces none).
    """
    fs = fused_row.get("fusion_score")
    if fs is None:
        fs = fused_row.get("weighted_minmax_fusion") or 0.0
    return {
        "final_score": float(fs),
        "u_hat": None,
        "p_hat": None,
        "preference_details": [],
        "score_breakdown": {},
    }


def _fused_rows(envelope: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Map user_id -> that user's ``column_fused_weighted_minmax`` rows."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for bundle in envelope.get("results") or []:
        uid = str(bundle.get("user_id") or "")
        out[uid] = bundle.get("column_fused_weighted_minmax") or []
    return out


def run_match_v2_full(
    users: List[Dict[str, Any]],
    jobs: List[Dict[str, Any]],
    occupations: List[Dict[str, Any]],
    *,
    fusion_top_k: int,
    alpha_on_cosine: float,
    skill_gap_top_k: int = MATCH_TOP_K_SKILL_GAPS,
) -> List[Dict[str, Any]]:
    """Return one ``MatchResponse``-shaped dict per user using the v2 hybrid matching formula."""
    if not users:
        return []

    matcher = get_cosine_matcher_singleton()
    job_index = _index_by_uuid(jobs)
    occ_index = _index_by_uuid(occupations)

    # Opportunities — v2 hybrid over the active job corpus (engine + education gate unchanged).
    opp_by_uid: Dict[str, List[Dict[str, Any]]] = {}
    if jobs:
        opp_env = hybrid_match_users_with_jobs(
            users, jobs, col_display_k=fusion_top_k, alpha_on_cosine=alpha_on_cosine
        )
        opp_by_uid = _fused_rows(opp_env)

    # Occupations — SAME hybrid engine over the occupation corpus. Score wide, then apply the v4
    # county filter / fallback / dedupe-by-code so ~top_k distinct occupation codes survive.
    occ_by_uid: Dict[str, List[Dict[str, Any]]] = {}
    if occupations:
        occ_breadth = max(fusion_top_k, MATCH_V4_TOP_K_OCCUPATIONS * 8)
        occ_items = [_occ_to_hybrid_item(o) for o in occupations]
        occ_env = hybrid_match_users_with_jobs(
            users, occ_items, col_display_k=occ_breadth, alpha_on_cosine=alpha_on_cosine
        )
        occ_by_uid = _fused_rows(occ_env)

    occ_counties = sorted(
        {str(o.get("province")) for o in occupations if o.get("province")}
    )

    out: List[Dict[str, Any]] = []
    for user in users:
        uid = str(user.get("user_id") or "")

        opportunities: List[Dict[str, Any]] = []
        for row in opp_by_uid.get(uid, []):
            item = job_index.get(str(row.get("job_uuid") or ""))
            if not item:
                continue
            if not _job_matches_user_location(item, user):
                continue  # strict same-location: urban-pull pool broadening is v4-only
            per, ess_ids = _skill_detail(matcher, user, item)
            opportunities.append(
                fmt.build_opportunity_row(
                    _v2_rec(row),
                    item,
                    per,
                    ess_ids,
                    rank=len(opportunities) + 1,
                    sim_threshold=V4_FULL_SIM_THRESHOLD,
                    min_ess_share=V4_FULL_MIN_ESS_SHARE,
                )
            )

        # Occupation county scoping (mirrors /match_v4): keep the user's own county row per code;
        # if the user's province matches no occupation county, fall back to a random available one
        # (location only — the user still drives skill matching).
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
        loc = loc_user or user

        occupations_out: List[Dict[str, Any]] = []
        seen_codes: set = set()
        for row in occ_by_uid.get(uid, []):
            item = occ_index.get(str(row.get("job_uuid") or ""))
            if not item:
                continue
            if not _job_matches_user_location(item, loc):
                continue
            code = str(item.get("originUuid") or item.get("uuid") or "")
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            per, ess_ids = _skill_detail(matcher, user, item)
            occupations_out.append(
                fmt.build_occupation_row(
                    _v2_rec(row),
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
