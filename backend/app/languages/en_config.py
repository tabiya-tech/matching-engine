"""English language config — the matching service's original behaviour, unchanged.

Taxonomy label pack: ``resources/skill_taxonomy/en/``. Its ids are the canonical internal
id space (``resources/models/skill_to_row.json`` and the embedding artefacts are keyed on
them), so every other language's pack is mapped onto this one.
"""

import os

_STOPWORDS = (
    # Conservative English stopword list — the same one the BM25 / hybrid tokenisers
    # have always used. Kept here so a language can supply its own.
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
)


def _env(key: str, default: str) -> str:
    return (os.getenv(f"{key}_EN") or os.getenv(key) or "").strip() or default


LANGUAGE_CONFIG = {
    "language": "en",
    "name": "English",
    "locales": ("en", "eng", "english", "en-gb", "en-us", "en-ke", "en-zm", "en-za"),
    # Sub-directory of resources/skill_taxonomy/ and resources/occupations/.
    "resources_subdir": "en",
    "taxonomy_locale": "en",
    # Cross-encoder used by /match_v3, /match_v4 and the cosine rerank scripts. The
    # passages are skill-label text, so the checkpoint has to understand the language.
    "cross_encoder_model": _env(
        "CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ),
    # BM25 / hybrid tokenisation.
    "stopwords": _STOPWORDS,
}
