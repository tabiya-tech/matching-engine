"""Readable, aligned timing blocks for the matching pipeline (no JSON noise)."""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional

_LAB = 24
_W = 70

_mlog = logging.getLogger("match_timing")
_configured = False


def _ensure_handler() -> None:
    global _configured
    if _configured or _mlog.handlers:
        return
    _configured = True
    _mlog.setLevel(logging.INFO)
    _mlog.propagate = False
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(message)s"))
    _mlog.addHandler(h)


def _fmt_value(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,d}"
    return str(v)


def log_match_step(
    component: str,
    step: str,
    *,
    user_id: Optional[str] = None,
    **fields: Any,
) -> None:
    """
    Log one timing / metrics block. kwargs order is preserved (3.7+).
    """
    _ensure_handler()
    lines: list[str] = [
        "",
        f"  {'=' * _W}",
        f"  > {component:16}  {step}",
        f"  {'-' * _W}",
    ]
    if user_id is not None:
        lines.append(f"  {'user_id':<{_LAB}}  {user_id}")
    for k, v in fields.items():
        lines.append(f"  {k:<{_LAB}}  {_fmt_value(v)}")
    lines.append(f"  {'=' * _W}")
    _mlog.info("\n".join(lines))


def init_match_timing_log() -> None:
    """Call from app startup so the stderr handler is registered before first /match (optional)."""
    _ensure_handler()
