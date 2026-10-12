"""Lenient value parsing for upstream documents, used only by ``SourceRecord`` models.

Upstream producers (scraper, classifier, llm-reranker, taxonomy exports) are not uniform about
scalar types: an ISCO code may be stored as ``2512`` or ``"2512"``, a closing date as a string or a
``datetime``, a flag as ``true`` or ``1``. These helpers normalise such values once, at the
boundary, so the strict domain models never see them.
"""

from datetime import date, datetime
from typing import Annotated, Any

import numpy as np
from bson import ObjectId
from pydantic import BeforeValidator

_SCALARS = (str, bool, int, float, date, datetime, ObjectId)


def to_optional_text(value: Any) -> str | None:
    """Text form of a stored scalar, or None when absent.

    Numbers, dates and ObjectIds are converted with ``str()``; a list (some producers store a
    location as ``["Nairobi", "Kiambu"]``) is joined with ``", "``. Mappings are rejected.

    :param value: The stored value
    :return: The value as text; None for None or an empty list
    :raises ValueError: If the value is a mapping or another non-scalar
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, _SCALARS):
        return str(value)
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value if v is not None and str(v).strip()]
        return ", ".join(parts) if parts else None
    raise ValueError(f"expected a scalar, got {type(value).__name__}")


def to_text_list(value: Any) -> list[str]:
    """List of strings from a stored list or single scalar; None and blank items are dropped.

    :param value: The stored value (list, scalar or None)
    :return: The non-empty items as text
    """
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    out: list[str] = []
    for item in items:
        text = to_optional_text(item)
        if text is not None and text.strip():
            out.append(text)
    return out


def parse_flag(value: Any) -> bool | None:
    """Binary flag stored as a bool, a 0/1 number or a yes/no string.

    :param value: The stored value
    :return: True / False for a recognised value, None for an absent or unrecognised one
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "y", "t"):
            return True
        if s in ("0", "false", "no", "n", "f", ""):
            return False
    return None


def parse_float(value: Any) -> float | None:
    """Number stored as a number or a numeric string.

    :param value: The stored value
    :return: The number; None for an absent, blank or non-numeric value
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_int(value: Any) -> int | None:
    """Whole number stored as an int, an integral float or a numeric string.

    :param value: The stored value
    :return: The integer; None for an absent or non-numeric value
    """
    number = parse_float(value)
    return int(number) if number is not None else None


def to_float32_vector(value: Any) -> np.ndarray | None:
    """1-D float32 array from a stored list of numbers; None when absent, empty or non-numeric.

    :param value: The stored value (normally a list of floats)
    :return: The vector, or None
    """
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    return vector if vector.ndim == 1 else None


OptionalText = Annotated[str | None, BeforeValidator(to_optional_text)]
"""A text field that also accepts numbers, dates, ObjectIds and lists of strings (see ``to_optional_text``)."""

TextList = Annotated[list[str], BeforeValidator(to_text_list)]
"""A list-of-strings field that also accepts a single scalar or None (see ``to_text_list``)."""

OptionalFlag = Annotated[bool | None, BeforeValidator(parse_flag)]
"""A boolean field that also accepts 0/1 and yes/no strings (see ``parse_flag``)."""

OptionalLevel = Annotated[int | None, BeforeValidator(parse_int)]
"""An integer field that also accepts integral floats and numeric strings (see ``parse_int``)."""

OptionalNumber = Annotated[float | None, BeforeValidator(parse_float)]
"""A float field that also accepts numeric strings; blanks and non-numbers become None."""

OptionalVector = Annotated[np.ndarray | None, BeforeValidator(to_float32_vector)]
"""A float32 vector field built from a list of numbers; anything else becomes None.

Models using it need ``arbitrary_types_allowed``."""
