"""Per-request matching settings (works with ``asyncio.to_thread``)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from app.matching_runtime import MatchingRuntimeSettings, get_default_runtime_settings

_var: ContextVar[Optional[MatchingRuntimeSettings]] = ContextVar(
    "matching_runtime_settings", default=None
)


def get_matching_runtime() -> MatchingRuntimeSettings:
    v = _var.get()
    if v is not None:
        return v
    return get_default_runtime_settings()


@contextmanager
def matching_runtime_scope(settings: MatchingRuntimeSettings) -> Iterator[None]:
    token = _var.set(settings)
    try:
        yield
    finally:
        _var.reset(token)
