"""Single string per side for embedding: concatenate skill labels only."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

DEFAULT_SEPARATOR = " | "


def _dedupe_stable(labels: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for raw in labels:
        s = str(raw).strip()
        if not s:
            continue
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def build_concat_embedding_text(
    labels: Sequence[str],
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    """Join skill labels into one line for embedding models."""

    return separator.join(_dedupe_stable(list(labels)))


def user_skill_labels_for_concat(user: Dict[str, Any]) -> List[str]:
    """Collect user-side skill label strings from ``resolved_skills`` or ``skills_vector.top_skills``."""

    out: List[str] = []
    for s in user.get("resolved_skills") or []:
        if isinstance(s, dict) and s.get("label"):
            out.append(str(s["label"]))
    if out:
        return _dedupe_stable(out)
    for s in (user.get("skills_vector") or {}).get("top_skills") or []:
        if isinstance(s, dict):
            lab = s.get("preferredLabel") or s.get("label")
            if lab:
                out.append(str(lab))
    return _dedupe_stable(out)


def job_skill_labels_for_concat(job: Dict[str, Any]) -> List[str]:
    """Collect job essential ∪ optional skill labels."""

    out: List[str] = []

    def _take(items: Any) -> None:
        for s in items or []:
            if isinstance(s, dict) and s.get("label"):
                out.append(str(s["label"]))

    _take(job.get("essential_skills"))
    _take(job.get("optional_skills"))
    return _dedupe_stable(out)


def user_concat_embedding_text(
    user: Dict[str, Any],
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    return build_concat_embedding_text(user_skill_labels_for_concat(user), separator=separator)


def job_concat_embedding_text(
    job: Dict[str, Any],
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    return build_concat_embedding_text(job_skill_labels_for_concat(job), separator=separator)
