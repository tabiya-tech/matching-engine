"""Per-item skill coverage: matched-skill split, essential coverage, coverage badge and eligibility."""

from __future__ import annotations

from typing import Any

from app.config import V4_FULL_BADGE_PARTIAL, V4_FULL_BADGE_STRONG


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
