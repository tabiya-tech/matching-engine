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

from app.domain.job import Job
from app.domain.skills import SkillReference
from app.domain.user import UserProfile
from app.ranking.types import MatchItem

# Conservative English stopword list (matches the original tokeniser).
from app.config import stopwords as config_stopwords
from app.languages import CANONICAL_LANGUAGE, default_language, normalise_language

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "this",
        "these",
        "those",
        "their",
        "but",
        "not",
        "we",
        "you",
        "your",
        "i",
        "they",
        "them",
        "our",
        "us",
        "into",
        "than",
        "then",
        "so",
        "up",
        "do",
        "does",
        "did",
    }
)


def _stopwords_for(language: str | None) -> frozenset[str]:
    """Stopwords for a language.

    The canonical language keeps this module's own list verbatim, so English tokenisation
    (and every BM25 score computed from it) is unchanged. Other languages come from
    ``app/languages/<lang>_config.py``; an empty set there means "filter nothing", which is
    safer than filtering English words out of another language's text.
    """
    lang = normalise_language(language) if language else default_language()
    if lang == CANONICAL_LANGUAGE:
        return _STOPWORDS
    return config_stopwords(lang)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(
    text: str,
    *,
    drop_stopwords: bool = True,
    min_len: int = 2,
    language: str | None = None,
) -> List[str]:
    """Lowercase + ASCII alnum splitter; optional stopword + min-length filter."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    if min_len > 1:
        tokens = [t for t in tokens if len(t) >= min_len]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _stopwords_for(language)]
    return tokens


# ---------------------------------------------------------------------------
# User query construction
# ---------------------------------------------------------------------------


def _user_skill_labels(user: UserProfile) -> List[str]:
    """The user's skill labels."""
    return [s.preferredLabel for s in user.skills_vector.top_skills if s.preferredLabel]


def user_query_string(user: UserProfile, *, include_programme: bool = True) -> str:
    """The user's skill labels joined as the v1 BM25 query.

    ``include_programme`` has no effect: ``UserProfile`` has no programme fields.
    """
    return " ".join(_user_skill_labels(user))


def user_query_tokens(
    user: UserProfile, *, include_programme: bool = True
) -> List[str]:
    """Word tokens of :func:`user_query_string`."""
    return tokenize(user_query_string(user, include_programme=include_programme))


# ---------------------------------------------------------------------------
# Job document construction
# ---------------------------------------------------------------------------


def _skill_labels(items: Iterable[SkillReference]) -> List[str]:
    return [s.label for s in items if s.label]


def job_skills_text(job: MatchItem) -> str:
    """Essential + optional labels joined as one string."""
    return " ".join(
        _skill_labels(job.essential_skills) + _skill_labels(job.optional_skills)
    )


def job_full_text(job: MatchItem, *, description_chars: int = 20000) -> str:
    """Title + employer + location + skill labels + description (combined doc)."""
    parts: List[str] = [
        job.title,
        (job.employer if isinstance(job, Job) else None) or "",
        job.location,
        job_skills_text(job),
    ]
    desc = job.opportunity_description if isinstance(job, Job) else job.description
    if desc:
        parts.append(desc[:description_chars])
    return " ".join(p for p in parts if p)


def build_corpora(
    jobs: Iterable[MatchItem],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Return ``(skills_only_tokens_per_job, full_tokens_per_job)`` (word mode)."""
    skills_corpus: List[List[str]] = []
    full_corpus: List[List[str]] = []
    for j in jobs:
        skills_corpus.append(tokenize(job_skills_text(j)))
        full_corpus.append(tokenize(job_full_text(j)))
    return skills_corpus, full_corpus
