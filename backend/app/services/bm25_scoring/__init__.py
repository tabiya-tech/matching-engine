"""BM25 package: **phrase** tokenisation + ``rank_bm25`` (PyPI ``rank-bm25``) only.

- **Retriever / dashboard:** ``python -m app.services.bm25_scoring.bm25library`` — builds BM25 indexes
  with :class:`rank_bm25.BM25Okapi` over skill phrases ± full job text (:mod:`text_builders`).
  There is **no alternate BM25 engine** here.
- **Hybrid (BM25 + cosine + pool fusion):** ``app.services.hybrid_scoring`` imports this package for
  corpora/indexes only; cosine scores come from :mod:`app.services.cosine_similarity.skill_score`.

``legacy_word_tokens`` is archived v1 word tokenisation — not wired into the active path.

This package re-exports tokenisation helpers only. The BM25 runnable lives in
:mod:`bm25library` and is intentionally **not** auto-imported here so that
``python -m app.services.bm25_scoring.bm25library`` works without the
:class:`RuntimeWarning` ``runpy`` emits when a runnable submodule is also
imported via its parent package.

Import the runner directly when you need it::

    from app.services.bm25_scoring.bm25library import run, recommend_for_user
"""

from .text_builders import (
    build_corpora,
    job_full_tokens,
    job_skill_phrase_tokens,
    matched_skill_overlap,
    matched_skill_phrases,
    normalize_skill_phrase,
    tokenize,
    user_query_tokens,
    user_skill_phrase_tokens,
)

__all__ = [
    "build_corpora",
    "job_full_tokens",
    "job_skill_phrase_tokens",
    "matched_skill_overlap",
    "matched_skill_phrases",
    "normalize_skill_phrase",
    "tokenize",
    "user_query_tokens",
    "user_skill_phrase_tokens",
]
