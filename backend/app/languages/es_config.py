"""Spanish language config (Argentina / Empujar — taxonomy locale ``AR-es``).

Taxonomy label pack: ``resources/skill_taxonomy/es/``, built from the AR-es taxonomy export
with ``scripts/build_language_taxonomy.py``. Its skill ids are per-locale and differ from
English's, so the resolver maps them onto the canonical English ids by joining on
``UUIDHISTORY``'s oldest entry — identical across locales for all 13,896 skills. The
embedding artefacts are therefore reused as-is; there is no Spanish retrain.

Override via env: ``CROSS_ENCODER_MODEL_NAME_ES``.
"""

import os

_STOPWORDS = (
    # Spanish counterpart of the English list: closed-class words that carry no signal in
    # a skill label. Deliberately short — BM25's IDF already down-weights common terms.
    "a",
    "al",
    "ante",
    "con",
    "de",
    "del",
    "e",
    "el",
    "en",
    "es",
    "la",
    "las",
    "lo",
    "los",
    "o",
    "para",
    "por",
    "que",
    "se",
    "segun",
    "según",
    "ser",
    "sin",
    "sobre",
    "su",
    "sus",
    "un",
    "una",
    "unos",
    "unas",
    "y",
)


def _env(key: str, default: str) -> str:
    return (os.getenv(f"{key}_ES") or os.getenv(key) or "").strip() or default


LANGUAGE_CONFIG = {
    "language": "es",
    "name": "Spanish",
    "locales": (
        "es",
        "spa",
        "spanish",
        "espanol",
        "ar-es",  # taxonomy export locale (model_info.csv LOCALE)
        "es-ar",
        "es-419",
        "es-es",
        "es-mx",
        "argentina",
        "argentine",
    ),
    "resources_subdir": "es",
    "taxonomy_locale": "AR-es",
    # Multilingual reranker trained on translated MS MARCO — the same task as the English
    # ms-marco checkpoint, so scores stay comparable in scale.
    "cross_encoder_model": _env(
        "CROSS_ENCODER_MODEL_NAME", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    ),
    "stopwords": _STOPWORDS,
}
