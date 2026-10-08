"""Per-item skill detail and the v4 Phase-2 ranking inputs (whitened p_hat + coverage demotion)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.ranking.coverage import (
    build_matched_skills,
    essential_coverage,
    unparsed_ranking_coverage,
)
from app.ranking.skill_matcher import CosineSkillMatcher
from app.ranking.whitening import ConcatWhitener

logger = logging.getLogger(__name__)


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
