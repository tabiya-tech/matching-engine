"""Build query/passage strings for cross-encoder scoring (skills-only)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def build_user_query_text(
    resolved_skill_labels: Sequence[str],
    *,
    max_skills: int = 48,
) -> str:
    """Preferred skill labels only (same strings ``CosineSkillMatcher`` embedded for cosine).

    Caller should pass ``resolved_user_skill_labels_ordered`` / equivalent preferred labels.
    """

    skills = [
        str(s).strip()
        for s in (resolved_skill_labels or [])[: max(0, int(max_skills))]
        if s and str(s).strip()
    ]
    if not skills:
        return ""
    return "; ".join(skills)


def build_job_passage_from_cosine_rec(
    rec: Dict[str, Any],
    *,
    max_skills: int = 64,
) -> str:
    """Job skill labels only, taken from cosine row ``per_job_skill`` (``job_skill_label``)."""

    pj = rec.get("per_job_skill") or []
    cap = max(0, int(max_skills))
    seen: set[str] = set()
    labels: List[str] = []
    for row in pj:
        if cap and len(labels) >= cap:
            break
        if not isinstance(row, dict):
            continue
        jl = str(row.get("job_skill_label") or "").strip()
        if not jl:
            continue
        key = jl.casefold()
        if key in seen:
            continue
        seen.add(key)
        labels.append(jl)
    if not labels:
        return ""
    return "; ".join(labels)
