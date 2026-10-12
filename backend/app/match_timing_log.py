"""Timing / metrics records for the matching pipeline, as structured log fields."""

from __future__ import annotations

import logging
from typing import Any, Optional

_mlog = logging.getLogger("match_timing")


def log_match_step(
    component: str,
    step: str,
    *,
    user_id: Optional[str] = None,
    **fields: Any,
) -> None:
    """
    Log one timing / metrics record; ``fields`` become top-level keys of the JSON log entry.

    Only counts, timings and the pseudonymous ``user_id`` belong here — never profile data.
    """
    record_fields: dict[str, Any] = {"component": component, "step": step}
    if user_id is not None:
        record_fields["user_id"] = user_id
    record_fields.update(fields)
    _mlog.info("%s: %s", component, step, extra={"fields": record_fields})
