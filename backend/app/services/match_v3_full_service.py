"""`/experiments/v3/match` full response: occupations + opportunities + skill-gaps via the v3 engine.

Keeps the v3 **matching logic** unchanged (Gemini concat-cosine shortlist → cross-encoder rerank,
``run_match_concat_gemini_ce``) but assembles the same ``MatchResponse`` shape ``/match_v4``
returns. This is ``run_match_v4_full`` *without* the preference (u_hat × p_hat) step.

By design:
* ``final_score`` = raw ``concat_cosine_similarity`` (the chosen v3 score); results are ordered by
  it so rank matches score. v4-only ``u_hat``/``p_hat``/preference fields are left empty (the v3
  engine computes no preference signal).
* Occupations are scored with the **same** v3 engine over the occupation corpus (precomputed concat
  embeddings), then county-scoped + random-fallback + de-duped by code, like ``/match_v4``. No
  occupation demand tilt (that is a v4-only re-rank).
* ``matched_skills`` is rebuilt from the shared ``CosineSkillMatcher`` (taxonomy/uuid mapping).
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
from app.services.match_concat_gemini_ce_service import (
    _get_matcher,
    embed_user_unit_vectors,
    run_match_concat_gemini_ce,
)
from app.services.match_v4_full_service import (
    _index_by_uuid,
    _skill_gaps_for,
    _user_matches_any_county,
)
from app.services.matching_service import _job_matches_user_location

__all__ = ["run_match_v3_full"]

logger = logging.getLogger(__name__)


def _skill_detail(matcher, user: Dict[str, Any], item: Dict[str, Any]):
    """(per_job_skill, matcher-resolved essential id set) — same contract as the v4 path."""
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


def _cosine(ce_row: Dict[str, Any]) -> float:
    try:
        v = ce_row.get("concat_cosine_similarity")
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _v3_rec(ce_row: Dict[str, Any]) -> Dict[str, Any]:
    """Map one CE recommendation row to the generic ``rec`` the v4 formatters consume.

    ``final_score`` is the concat cosine (now whitened — stage-1 ranks in the whitened concat space
    when the artifact is present); v4-only signals are left empty so the formatter emits null
    ``u_hat``/``p_hat`` and no preferences (the v3 engine produces none).
    """
    return {
        "final_score": _cosine(ce_row),
        "u_hat": None,
        "p_hat": None,
        "preference_details": [],
        "score_breakdown": {},
    }


def _ce_rows_by_uid(
    v3_results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in v3_results or []:
        out[str(r.get("user_id") or "")] = (
            r.get("concat_gemini_ce_recommendations") or []
        )
    return out


def run_match_v3_full(
    users: List[Dict[str, Any]],
    jobs: List[Dict[str, Any]],
    occupations: List[Dict[str, Any]],
    *,
    retrieve_top_k: int,
    final_top_k: int,
    skill_gap_top_k: int = MATCH_TOP_K_SKILL_GAPS,
) -> List[Dict[str, Any]]:
    """Return one ``MatchResponse``-shaped dict per user using the v3 matching logic.

    The deployment's ``TARGET_LANGUAGE`` selects the cross-encoder checkpoint (see
    ``run_match_concat_gemini_ce``); skill resolution itself is language-neutral.
    """
    if not users:
        return []

    u_norm = embed_user_unit_vectors(users)  # embed users ONCE, reuse for both corpora
    matcher = _get_matcher()
    job_index = _index_by_uuid(jobs)
    occ_index = _index_by_uuid(occupations)

    # Opportunities — v3 engine over the active job corpus (engine + education gate unchanged).
    job_v3 = (
        run_match_concat_gemini_ce(
            users,
            jobs,
            retrieve_top_k=retrieve_top_k,
            final_top_k=final_top_k,
            user_unit_vectors=u_norm,
        )
        if jobs
        else []
    )
    # Occupations — SAME v3 engine over the occupation corpus. Score wide, then apply the v4 county
    # filter / fallback / dedupe-by-code so ~top_k distinct occupation codes survive.
    occ_breadth = max(retrieve_top_k, final_top_k, MATCH_V4_TOP_K_OCCUPATIONS * 8)
    occ_v3 = (
        run_match_concat_gemini_ce(
            users,
            occupations,
            retrieve_top_k=occ_breadth,
            final_top_k=occ_breadth,
            user_unit_vectors=u_norm,
        )
        if occupations
        else []
    )

    job_by_uid = _ce_rows_by_uid(job_v3)
    occ_by_uid = _ce_rows_by_uid(occ_v3)
    occ_counties = sorted(
        {str(o.get("province")) for o in occupations if o.get("province")}
    )

    out: List[Dict[str, Any]] = []
    for user in users:
        uid = str(user.get("user_id") or "")

        # Opportunities ranked by concat cosine (whitened space when the artifact is present; the
        # chosen v3 score). _cosine reads concat_cosine_similarity from the stage-1 retrieval.
        opp_sorted = sorted(job_by_uid.get(uid, []), key=_cosine, reverse=True)
        opportunities: List[Dict[str, Any]] = []
        for ce_row in opp_sorted:
            item = job_index.get(str(ce_row.get("job_uuid") or ""))
            if not item:
                continue
            if not _job_matches_user_location(item, user):
                continue  # strict same-location: urban-pull pool broadening is v4-only
            per, ess_ids = _skill_detail(matcher, user, item)
            opportunities.append(
                fmt.build_opportunity_row(
                    _v3_rec(ce_row),
                    item,
                    per,
                    ess_ids,
                    rank=len(opportunities) + 1,
                    sim_threshold=V4_FULL_SIM_THRESHOLD,
                    min_ess_share=V4_FULL_MIN_ESS_SHARE,
                )
            )

        # Occupation county scoping (mirrors /match_v4): keep the user's own county row per code;
        # random fallback county if the user's province matches none.
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

        occ_sorted = sorted(occ_by_uid.get(uid, []), key=_cosine, reverse=True)
        occupations_out: List[Dict[str, Any]] = []
        seen_codes: set = set()
        for ce_row in occ_sorted:
            item = occ_index.get(str(ce_row.get("job_uuid") or ""))
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
                    _v3_rec(ce_row),
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
