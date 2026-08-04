"""Language registry — the one place to edit when adding a language.

Each deployment is configured for one language via ``TARGET_LANGUAGE``; requests carry no
language. Two different things are going on, and it matters which is which:

**Skill resolution is language-neutral.** Both sides of a match resolve skills by *label*
(``SkillScorer._resolve_label``) into the internal id space of the embedding artefact.
Every enabled language's label pack is loaded into that one resolver, mapped onto the same
canonical ids, so a Spanish job posting and a Spanish user profile resolve into the same
vectors as English ones — whatever the deployment's language is set to. That is possible
because ``UUIDHISTORY``'s oldest entry is stable across taxonomy locales, which is what the
packs are joined on.

**Text scoring and display are not.** The cross-encoder checkpoint, the BM25 stopwords and
the labels echoed back in the response do depend on the language. Those follow
``TARGET_LANGUAGE``, else English.

Each registered language has an ``app/languages/{lang}_config.py`` exporting
``LANGUAGE_CONFIG``; same shape as ``scraper/config/{country}_config.py``.

Adding a language:

1. Add its code to :data:`LANGUAGE_REGISTRY`.
2. Add ``app/languages/{code}_config.py`` exporting ``LANGUAGE_CONFIG``.
3. Build its taxonomy label pack:
   ``python -m scripts.build_language_taxonomy --taxonomy-dir <export> --language {code}``
"""

from __future__ import annotations

import importlib
import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# ── Language registry ─────────────────────────────────────────────────────────
# The ONLY place you need to edit when adding a new language. Each entry must have a
# matching `app/languages/{code}_config.py` exporting LANGUAGE_CONFIG.
LANGUAGE_REGISTRY: tuple[str, ...] = ("en", "es")

# The language whose taxonomy ids define the internal id space — the ids in
# `skill_to_row.json` and in the embedding artefacts. Other languages' packs are mapped
# onto it. Changing this means rebuilding every embedding artefact.
CANONICAL_LANGUAGE = "en"

DEFAULT_LANGUAGE = "en"


def _key(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


@lru_cache(maxsize=None)
def get_language_config(language: str) -> dict[str, Any]:
    """``LANGUAGE_CONFIG`` for a registered language code.

    :raises ValueError: if the language is not in :data:`LANGUAGE_REGISTRY`.
    """
    code = _key(language)
    if code not in LANGUAGE_REGISTRY:
        raise ValueError(
            f"Unsupported language {language!r}. Registered: {', '.join(LANGUAGE_REGISTRY)}"
        )
    mod = importlib.import_module(f"app.languages.{code}_config")
    # Config modules read env at import time; reloading on a cache miss means
    # `get_language_config.cache_clear()` re-reads the environment (what tests expect).
    mod = importlib.reload(mod)
    return dict(mod.LANGUAGE_CONFIG)


@lru_cache(maxsize=None)
def _locale_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for code in LANGUAGE_REGISTRY:
        index[code] = code
        for alias in get_language_config(code).get("locales", ()):
            index[_key(alias)] = code
    return index


def normalise_language(value: str | None, *, fallback: str | None = None) -> str:
    """Resolve any locale spelling to a registered language code.

    Accepts a bare code (``es``), a taxonomy locale (``AR-es``), a POSIX locale
    (``es_AR.UTF-8``), a country slug (``argentina``) or an English language name
    (``spanish``). Unknown values fall back with a warning rather than raising: a typo
    in a deployment's env should degrade to English, not take the service down.
    """
    fb = fallback or default_language()
    if value is None:
        return fb
    key = _key(value)
    if not key:
        return fb
    index = _locale_index()
    if key in index:
        return index[key]
    for part in (key.split(".")[0], *reversed(key.split(".")[0].split("-"))):
        if part in index:
            return index[part]
    logger.warning(
        "Unknown language %r; using %r (registered: %s)",
        value,
        fb,
        ", ".join(LANGUAGE_REGISTRY),
    )
    return fb


def default_language() -> str:
    """``TARGET_LANGUAGE``, else ``MATCHING_LANGUAGE``, else ``en``."""
    raw = (
        os.getenv("TARGET_LANGUAGE")
        or os.getenv("MATCHING_LANGUAGE")
        or DEFAULT_LANGUAGE
    )
    key = _key(raw)
    index = _locale_index()
    if key in index:
        return index[key]
    logger.warning(
        "TARGET_LANGUAGE=%r is not registered; using %r", raw, DEFAULT_LANGUAGE
    )
    return DEFAULT_LANGUAGE


def enabled_languages() -> tuple[str, ...]:
    """Label packs loaded into the skill resolver — ``ENABLED_LANGUAGES``, default all.

    Unlike the model-loading services, loading every pack here is cheap (a CSV parse, no
    checkpoint) and it is what makes a single deployment match Spanish and English
    postings side by side. The canonical language is always included: it defines the id
    space the other packs are mapped onto.
    """
    raw = (os.getenv("ENABLED_LANGUAGES") or "").strip()
    if not raw or _key(raw) in ("all", "*"):
        return LANGUAGE_REGISTRY
    codes: list[str] = [CANONICAL_LANGUAGE]
    index = _locale_index()
    for part in raw.split(","):
        key = _key(part)
        if not key:
            continue
        code = index.get(key)
        if code is None:
            logger.warning("ENABLED_LANGUAGES: ignoring unregistered language %r", part)
            continue
        if code not in codes:
            codes.append(code)
    return tuple(codes)


def language_setting(language: str, key: str, default: Any = None) -> Any:
    """One value out of a language's ``LANGUAGE_CONFIG``."""
    return get_language_config(language).get(key, default)


__all__ = [
    "CANONICAL_LANGUAGE",
    "DEFAULT_LANGUAGE",
    "LANGUAGE_REGISTRY",
    "default_language",
    "enabled_languages",
    "get_language_config",
    "language_setting",
    "normalise_language",
]
