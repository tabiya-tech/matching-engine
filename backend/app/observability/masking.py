"""Client-side masking of trace payloads.

Ported from the llm-reranker. The matching traces carry no free text by default; this is the
safety net for the one payload that can be switched on (``recordEmbeddingInput``: the text sent
to Gemini). It redacts the mechanically detectable identifiers and truncates oversized payloads
*before* anything leaves the process.

It is heuristic — a cost and blast-radius control, not a guarantee that no PII is
exported. A deployment that cannot export that text at all should leave
``recordEmbeddingInput`` off, or point ``MATCHING_LANGFUSE_HOST`` at a self-hosted instance.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.observability.config import TracingConfig

logger = logging.getLogger(__name__)

REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"
REDACTED_NUMBER = "[REDACTED_NUMBER]"
TRUNCATION_SUFFIX = "...[TRUNCATED]"

# Deeper structures than this are collapsed to a string; guards against pathological payloads.
_MAX_DEPTH = 12

_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# A phone-shaped run: an optional country prefix followed by digits, spaces, dots, dashes
# and parentheses. The digit-count check in ``_replace_phone`` rejects the many short
# numeric runs (years, salary ranges, list indices) that this pattern also matches.
_PHONE_PATTERN = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_PHONE_MIN_DIGITS = 9
_PHONE_MAX_DIGITS = 15

# A bare run of digits long enough to be an identifier rather than a quantity.
_LONG_NUMBER_PATTERN = re.compile(r"(?<!\d)\d{9,}(?!\d)")


def redact(text: str) -> str:
    """Replace e-mail addresses, phone numbers and long digit runs in ``text``."""
    redacted = _EMAIL_PATTERN.sub(REDACTED_EMAIL, text)
    redacted = _PHONE_PATTERN.sub(_replace_phone, redacted)
    return _LONG_NUMBER_PATTERN.sub(REDACTED_NUMBER, redacted)


def _replace_phone(match: re.Match) -> str:
    """Replace a phone-shaped match only when it holds a plausible number of digits."""
    digit_count = sum(1 for char in match.group(0) if char.isdigit())
    if _PHONE_MIN_DIGITS <= digit_count <= _PHONE_MAX_DIGITS:
        return REDACTED_PHONE
    return match.group(0)


def truncate(text: str, max_chars: int) -> str:
    """Cut ``text`` to at most ``max_chars`` characters, marking that it was cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_SUFFIX


def mask_value(value: Any, *, redact_pii: bool, max_chars: int, _depth: int = 0) -> Any:
    """Recursively mask a trace payload: strings redacted and truncated, containers walked."""
    if _depth > _MAX_DEPTH:
        return truncate(str(value), max_chars)
    if isinstance(value, str):
        return truncate(redact(value) if redact_pii else value, max_chars)
    if isinstance(value, dict):
        return {
            key: mask_value(
                item, redact_pii=redact_pii, max_chars=max_chars, _depth=_depth + 1
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            mask_value(
                item, redact_pii=redact_pii, max_chars=max_chars, _depth=_depth + 1
            )
            for item in value
        ]
    return value


def build_mask_function(config: TracingConfig):
    """The ``mask`` callable the Langfuse client applies to every payload before export.

    It never raises: a payload that cannot be masked is dropped rather than exported
    unmasked, and rather than break the traced call.
    """

    def mask(*, data: Any, **_kwargs: Any) -> Any:
        try:
            return mask_value(
                data, redact_pii=config.mask_pii, max_chars=config.max_payload_chars
            )
        except Exception as e:
            logger.warning("Failed to mask a trace payload; dropping it. Error: %s", e)
            return "[MASKING_FAILED]"

    return mask
