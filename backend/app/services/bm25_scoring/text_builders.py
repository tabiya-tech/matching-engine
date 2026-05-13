"""Phrase-mode tokenisation for BM25 (one taxonomy skill = one token).

Every skill label is collapsed to a single underscore-joined token
(``"Apply Teaching Strategies"`` -> ``apply_teaching_strategies``). Free-form
text fields — job title, employer, location, opportunity description, and a
user's optional programme / institution / school year — are still word-split.

Token streams used by ``bm25library``:

User query (one list per user)
    - ``resolved_skills[*].label`` or ``skills_vector.top_skills[*].preferredLabel``
      -> one phrase token per skill
    - Optionally: ``programme_name``, ``institution_name``, ``school_year``
      as loose word tokens (**off by default** — tends to leak noise such as
      *skills*, *training*, *year* onto the dashboard / query).

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

# Conservative English stopword list. Small on purpose: BM25's IDF already
# down-weights frequent tokens; the goal here is just to skip pure noise.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "was", "were", "will", "with", "this", "these", "those", "their",
    "but", "not", "we", "you", "your", "i", "they", "them", "our", "us",
    "into", "than", "then", "so", "up", "do", "does", "did",
})

# Token = alphanumeric run, length >= 2 (so single letters / punctuation drop out).
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str, *, drop_stopwords: bool = True, min_len: int = 2) -> List[str]:
    """Lowercase + ASCII alnum splitter; optional stopword + min-length filter."""
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    if min_len > 1:
        tokens = [t for t in tokens if len(t) >= min_len]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
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


def user_skill_phrase_tokens(user: dict) -> List[str]:
    """Phrase tokens from the user's skill labels (one token per skill, unique)."""
    out: List[str] = []
    seen: set = set()
    for lab in _user_skill_labels(user):
        phr = normalize_skill_phrase(lab)
        if phr and phr not in seen:
            seen.add(phr)
            out.append(phr)
    return out


def user_query_tokens(user: dict, *, include_programme: bool = False) -> List[str]:
    """User-side BM25 tokens: skill phrases (+ optional programme word tokens)."""
    tokens = list(user_skill_phrase_tokens(user))
    if include_programme:
        for k in ("programme_name", "institution_name", "school_year"):
            v = user.get(k)
            if v:
                tokens.extend(tokenize(str(v)))
    return tokens


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


def job_skill_phrase_tokens(job: dict) -> List[str]:
    """Phrase tokens for one job: one underscore-joined token per skill."""
    out: List[str] = []
    for lab in (
        _skill_labels(job.get("essential_skills"))
        + _skill_labels(job.get("optional_skills"))
    ):
        phr = normalize_skill_phrase(lab)
        if phr:
            out.append(phr)
    return out


def job_full_tokens(job: dict, *, description_chars: int = 20000) -> List[str]:
    """Free-form words + skill phrase tokens (full-text BM25 document).

    Description is hard-capped to avoid a multi-paragraph posting drowning
    out the rest of the fields; the cap is well above any realistic job text.
    """
    parts: List[str] = [
        str(job.get("opportunity_title") or ""),
        str(job.get("employer") or ""),
        str(job.get("location") or ""),
    ]
    desc = job.get("opportunity_description") or ""
    if desc:
        parts.append(str(desc)[:description_chars])
    tokens = tokenize(" ".join(p for p in parts if p))
    tokens.extend(job_skill_phrase_tokens(job))
    return tokens


def build_corpora(jobs: Iterable[dict]) -> Tuple[List[List[str]], List[List[str]]]:
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

def matched_skill_phrases(user: dict, job: dict) -> List[str]:
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
        if not isinstance(lab, str) or not lab.strip():
            continue
        phr = normalize_skill_phrase(lab)
        if phr and phr not in out:
            out[phr] = lab.strip()
    return out


def matched_skill_overlap(user: dict, job: dict) -> List[Dict[str, str]]:
    """Overlap skills with one example label from the user vs from the job.

    Each phrase token appears at most once in the sorted list; when the user's
    wording and the job's wording differ, both strings are retained so UIs can
    show ``user_label`` vs ``job_label``.
    """
    u_map = _phrase_to_first_skill_label(_user_skill_labels(user))
    j_labs = _skill_labels(job.get("essential_skills")) + _skill_labels(
        job.get("optional_skills")
    )
    j_map = _phrase_to_first_skill_label(j_labs)
    common = sorted(set(u_map.keys()) & set(j_map.keys()))
    return [
        {"phrase_token": ph, "user_label": u_map[ph], "job_label": j_map[ph]}
        for ph in common
    ]
