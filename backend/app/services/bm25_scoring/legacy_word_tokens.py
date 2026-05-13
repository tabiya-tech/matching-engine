"""LEGACY — word-token (v1) BM25 query / document construction.

Kept for reference only. The active pipeline uses **phrase tokenisation**
(``text_builders.py``) — every taxonomy skill becomes one underscore-joined
token. This file preserves the original word-mode behaviour:

- Every text field is lowercased and split on word boundaries.
- A skill label like ``"apply teaching strategies"`` emits three tokens
  (``apply``, ``teaching``, ``strategies``).
- Programme / institution / school year fields are appended verbatim.

The functions here are self-contained — no imports from ``text_builders`` or
``bm25library`` — so the file can safely be archived without disturbing the live
pipeline. Hybrid / cosine runners do not import this module.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

# Conservative English stopword list (matches the original tokeniser).
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "will", "with", "this", "these", "those", "their",
    "but", "not", "we", "you", "your", "i", "they", "them", "our", "us",
    "into", "than", "then", "so", "up", "do", "does", "did",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(
    text: str,
    *,
    drop_stopwords: bool = True,
    min_len: int = 2,
) -> List[str]:
    """Lowercase + ASCII alnum splitter; optional stopword + min-length filter."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    if min_len > 1:
        tokens = [t for t in tokens if len(t) >= min_len]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


# ---------------------------------------------------------------------------
# User query construction
# ---------------------------------------------------------------------------

def _user_skill_labels(user: dict) -> List[str]:
    """Pull skill labels from a raw or pre-resolved user record."""
    out: List[str] = []
    for s in (user.get("resolved_skills") or []):
        if isinstance(s, dict):
            lab = s.get("label")
            if lab:
                out.append(str(lab))
    if out:
        return out
    for s in ((user.get("skills_vector") or {}).get("top_skills") or []):
        if isinstance(s, dict):
            lab = s.get("preferredLabel") or s.get("label")
            if lab:
                out.append(str(lab))
    return out


def user_query_string(user: dict, *, include_programme: bool = True) -> str:
    """Concatenate textual fields used as the v1 BM25 query."""
    parts: List[str] = list(_user_skill_labels(user))
    if include_programme:
        for k in ("programme_name", "institution_name", "school_year"):
            v = user.get(k)
            if v:
                parts.append(str(v))
    return " ".join(parts)


def user_query_tokens(user: dict, *, include_programme: bool = True) -> List[str]:
    return tokenize(user_query_string(user, include_programme=include_programme))


# ---------------------------------------------------------------------------
# Job document construction
# ---------------------------------------------------------------------------

def _skill_labels(items) -> List[str]:
    out: List[str] = []
    for s in items or []:
        if isinstance(s, dict):
            lab = s.get("label")
            if lab:
                out.append(str(lab))
    return out


def job_skills_text(job: dict) -> str:
    """Essential + optional labels joined as one string."""
    return " ".join(
        _skill_labels(job.get("essential_skills"))
        + _skill_labels(job.get("optional_skills"))
    )


def job_full_text(job: dict, *, description_chars: int = 20000) -> str:
    """Title + employer + location + skill labels + description (combined doc)."""
    parts: List[str] = [
        str(job.get("opportunity_title") or ""),
        str(job.get("employer") or ""),
        str(job.get("location") or ""),
        job_skills_text(job),
    ]
    desc = job.get("opportunity_description") or ""
    if desc:
        parts.append(str(desc)[:description_chars])
    return " ".join(p for p in parts if p)


def build_corpora(
    jobs: Iterable[dict],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Return ``(skills_only_tokens_per_job, full_tokens_per_job)`` (word mode)."""
    skills_corpus: List[List[str]] = []
    full_corpus: List[List[str]] = []
    for j in jobs:
        skills_corpus.append(tokenize(job_skills_text(j)))
        full_corpus.append(tokenize(job_full_text(j)))
    return skills_corpus, full_corpus
