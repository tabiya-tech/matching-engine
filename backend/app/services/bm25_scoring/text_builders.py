"""Phrase-mode tokenisation for BM25 (one taxonomy skill = one token).

Every skill label is collapsed to a single underscore-joined token
(``"Apply Teaching Strategies"`` -> ``apply_teaching_strategies``). Free-form
text fields — job title, employer, location, opportunity description, and a
user's optional programme / institution / school year — are still word-split.

Token streams used by ``bm25library``:

User query (one list per user)
    - ``skills_vector.top_skills[*].preferredLabel`` -> one phrase token per skill

Skills-only job document
    - ``essential_skills[*].label`` + ``optional_skills[*].label`` -> phrase tokens

Full-text job document
    - ``opportunity_title``, ``employer``, ``location``, ``opportunity_description``
      -> word tokens
    - plus all phrase tokens from the skills lists

Bonus: :func:`matched_skill_phrases` returns the deterministic taxonomy
overlap between a user and a job (used by the dashboard's
"Matched taxonomy skills" section).

Word-mode (v1) lives in :mod:`legacy_word_tokens`.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

# Conservative English stopword list. Small on purpose: BM25's IDF already
# down-weights frequent tokens; the goal here is just to skip pure noise.
from app.config import stopwords as config_stopwords
from app.domain.job import Job
from app.domain.skills import SkillReference
from app.domain.user import UserProfile
from app.languages import CANONICAL_LANGUAGE, default_language, normalise_language
from app.ranking.types import MatchItem

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


# Token = alphanumeric run, length >= 2 (so single letters / punctuation drop out).
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


def normalize_skill_phrase(
    label: str,
    *,
    drop_stopwords: bool = True,
    min_word_len: int = 2,
) -> Optional[str]:
    """Collapse one skill label into a single underscore-joined token.

    Returns ``None`` if no usable words remain after filtering.
    """
    words = tokenize(label or "", drop_stopwords=drop_stopwords, min_len=min_word_len)
    if not words:
        return None
    return "_".join(words)


# ---------------------------------------------------------------------------
# User query construction
# ---------------------------------------------------------------------------


def _user_skill_labels(user: UserProfile) -> List[str]:
    """The user's skill labels (``skills_vector.top_skills[].preferredLabel``)."""
    return [s.preferredLabel for s in user.skills_vector.top_skills if s.preferredLabel]


def user_skill_phrase_tokens(user: UserProfile) -> List[str]:
    """Phrase tokens from the user's skill labels (one token per skill, unique)."""
    out: List[str] = []
    seen: set[str] = set()
    for lab in _user_skill_labels(user):
        phr = normalize_skill_phrase(lab)
        if phr and phr not in seen:
            seen.add(phr)
            out.append(phr)
    return out


def user_query_tokens(
    user: UserProfile, *, include_programme: bool = False
) -> List[str]:
    """User-side BM25 tokens: skill phrases.

    ``include_programme`` is accepted for CLI compatibility only: ``UserProfile`` carries no
    programme / institution / school-year fields, so there is nothing to add.
    """
    return list(user_skill_phrase_tokens(user))


# ---------------------------------------------------------------------------
# Job document construction
# ---------------------------------------------------------------------------


def _skill_labels(items: Iterable[SkillReference]) -> List[str]:
    return [s.label for s in items if s.label]


def job_skill_phrase_tokens(job: MatchItem) -> List[str]:
    """Phrase tokens for one job: one underscore-joined token per skill."""
    out: List[str] = []
    for lab in _skill_labels(job.essential_skills) + _skill_labels(job.optional_skills):
        phr = normalize_skill_phrase(lab)
        if phr:
            out.append(phr)
    return out


def _description(job: MatchItem) -> str:
    return job.opportunity_description if isinstance(job, Job) else job.description


def job_full_tokens(job: MatchItem, *, description_chars: int = 20000) -> List[str]:
    """Free-form words + skill phrase tokens (full-text BM25 document).

    Jobs contribute title, employer, location and description; occupations their label, county
    and description. Description is hard-capped to avoid a multi-paragraph posting drowning
    out the rest of the fields; the cap is well above any realistic job text.
    """
    parts: List[str] = [
        job.title,
        (job.employer if isinstance(job, Job) else None) or "",
        job.location,
    ]
    desc = _description(job)
    if desc:
        parts.append(desc[:description_chars])
    tokens = tokenize(" ".join(p for p in parts if p))
    tokens.extend(job_skill_phrase_tokens(job))
    return tokens


def build_corpora(
    jobs: Iterable[MatchItem],
) -> Tuple[List[List[str]], List[List[str]]]:
    """Return ``(skills_only_tokens_per_job, full_tokens_per_job)``."""
    skills_corpus: List[List[str]] = []
    full_corpus: List[List[str]] = []
    for j in jobs:
        skills_corpus.append(job_skill_phrase_tokens(j))
        full_corpus.append(job_full_tokens(j))
    return skills_corpus, full_corpus


# ---------------------------------------------------------------------------
# Deterministic taxonomy overlap (used by the dashboard)
# ---------------------------------------------------------------------------


def matched_skill_phrases(user: UserProfile, job: MatchItem) -> List[str]:
    """Intersect user vs job skills as normalised phrase tokens (sorted).

    Same normalisation as the phrase tokeniser, so the strings displayed by
    the dashboard are exactly the tokens BM25 sees for the skills index.
    Answers "which declared skills appear on this job?" independent of BM25
    weighting.
    """
    u = set(user_skill_phrase_tokens(user))
    jset = set(job_skill_phrase_tokens(job))
    return sorted(u & jset)


def _phrase_to_first_skill_label(labels: Iterable[str]) -> Dict[str, str]:
    """Map normalised phrase token -> first observed raw label."""
    out: Dict[str, str] = {}
    for lab in labels:
        if not lab.strip():
            continue
        phr = normalize_skill_phrase(lab)
        if phr and phr not in out:
            out[phr] = lab.strip()
    return out


class SkillPhraseOverlap(BaseModel):
    """One skill phrase shared by a user and a job, with each side's wording."""

    phrase_token: str = Field(description="Normalised phrase token.")
    user_label: str = Field(description="The user's label for the skill.")
    job_label: str = Field(description="The job's label for the skill.")


def matched_skill_overlap(
    user: UserProfile, job: MatchItem
) -> List[SkillPhraseOverlap]:
    """Overlap skills with one example label from the user vs from the job.

    Each phrase token appears at most once in the sorted list; when the user's
    wording and the job's wording differ, both strings are retained so UIs can
    show ``user_label`` vs ``job_label``.
    """
    u_map = _phrase_to_first_skill_label(_user_skill_labels(user))
    j_map = _phrase_to_first_skill_label(
        _skill_labels(job.essential_skills) + _skill_labels(job.optional_skills)
    )
    common = sorted(set(u_map.keys()) & set(j_map.keys()))
    return [
        SkillPhraseOverlap(phrase_token=ph, user_label=u_map[ph], job_label=j_map[ph])
        for ph in common
    ]
