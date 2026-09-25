"""The root trace every matching request runs under — the matching counterpart of the reranker's
``job_trace``.

:class:`MatchTracingMiddleware` opens one trace per ``POST`` to a matching route (``/match``,
``/match_v4``, ``/experiments/<v>/match``, and any future ``/match_*``). The trace is named after
the route and tagged ``route:<path>``; the request id and route propagate to every stage span, so
observation-level dashboards (per-stage latency percentiles) can filter by route. At the end the
root carries the HTTP status and the request's embedding totals, plus outcome tags — ``error``,
``client_error``, ``embedding_retried``, ``embedding_failed`` — so failed or retried requests are
one filter away in the Langfuse UI.

A request is found by its id: the response's ``X-Request-ID`` (a valid client-supplied one is
echoed back) and ``X-Trace-ID`` headers; a generated request id *is* the Langfuse trace id. Or by
the caller's pseudonymous ``user_id``, set by :func:`set_request_users`.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from app.observability import tracing

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "x-request-id"
TRACE_ID_HEADER = "x-trace-id"

# Client-supplied request ids are echoed into headers and trace metadata: keep them short and plain.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")
_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def is_match_path(path: str) -> bool:
    """True for every matching route, current and future versions alike."""
    return (
        path == "/match"
        or path.startswith("/match_")
        or (path.startswith("/experiments/") and path.endswith("/match"))
    )


@dataclass
class _RequestState:
    root: Any
    request_id: str
    trace_id: str
    # Engines run in worker threads (asyncio.to_thread copies this context): guard the counters.
    lock: threading.Lock = field(default_factory=threading.Lock)
    embedding_calls: int = 0
    embedding_retries: int = 0
    embedding_failures: int = 0
    embedding_tokens: int = 0


_request_ctx_var: ContextVar[_RequestState | None] = ContextVar(
    "match_request_trace", default=None
)


def note_embedding_call(*, retries: int, failed: bool, tokens: int | None) -> None:
    """Roll one Gemini call into the request's totals (for the root metadata and tags)."""
    state = _request_ctx_var.get()
    if state is None:
        return
    with state.lock:
        state.embedding_calls += 1
        state.embedding_retries += max(0, retries)
        state.embedding_failures += int(failed)
        state.embedding_tokens += tokens or 0


def set_request_users(users: Sequence[dict[str, Any]], **params: Any) -> None:
    """Attach the request's pseudonymous user id(s) and query parameters to the trace.

    Only ``user_id`` is read from each user — never location, skills or preferences. A single-user
    request sets the Langfuse ``user_id`` (searchable in the UI); batches list every id in the
    ``user_ids`` metadata instead, since a trace has one owner.
    """
    state = _request_ctx_var.get()
    if state is None:
        return
    try:
        from langfuse import LangfuseOtelSpanAttributes

        ids = [str(u.get("user_id") or "") for u in users]
        if len(ids) == 1 and ids[0]:
            # The root span's own attribute: propagate_attributes would only reach spans opened
            # after this point, and the trace's user is a property of the root.
            state.root._otel_span.set_attribute(
                LangfuseOtelSpanAttributes.TRACE_USER_ID, ids[0][:200]
            )
        metadata: dict[str, Any] = {"n_users": len(ids), "user_ids": ",".join(ids)}
        metadata.update({k: v for k, v in params.items() if v is not None})
        tracing.update_observation(state.root, metadata=metadata)
    except Exception as e:
        logger.warning(
            "Failed to attach the request's users to its trace. Error: %s", e
        )


def _request_id_from_headers(headers: Sequence[Any]) -> str:
    for key, value in headers:
        if key.lower() == REQUEST_ID_HEADER.encode():
            candidate = value.decode("latin-1").strip()
            if _REQUEST_ID_RE.match(candidate):
                return candidate
            break
    return uuid.uuid4().hex


def _trace_id_for(client: Any, request_id: str) -> str:
    """A generated request id is the trace id; a client-supplied one maps to it deterministically."""
    if _TRACE_ID_RE.match(request_id):
        return request_id
    return client.create_trace_id(seed=request_id)


@contextmanager
def _request_trace(route: str, request_id: str) -> Iterator[_RequestState | None]:
    client = tracing.get_tracing_client()
    if not tracing.is_tracing_enabled():
        yield None
        return
    if not tracing.should_sample(request_id):
        with tracing.suppress_tracing():
            yield None
        return
    try:
        from langfuse import propagate_attributes

        trace_id = _trace_id_for(client, request_id)
        manager = client.start_as_current_observation(
            name=route,
            as_type="span",
            trace_context={"trace_id": trace_id},
            metadata={"request_id": request_id, "route": route},
        )
    except Exception as e:
        logger.warning(
            "Failed to start the %r trace. Continuing untraced. Error: %s", route, e
        )
        yield None
        return

    with manager as root:
        state = _RequestState(root=root, request_id=request_id, trace_id=trace_id)
        token = _request_ctx_var.set(state)
        try:
            with propagate_attributes(
                trace_name=route,
                tags=["matching", f"route:{route}"],
                metadata={"request_id": request_id, "route": route},
            ):
                yield state
        finally:
            _request_ctx_var.reset(token)


def _finish_request(
    state: _RequestState, status: int | None, exc: BaseException | None
) -> None:
    """Outcome level, metadata and tags on the root span — still current, so tags merge onto it."""
    try:
        from langfuse import propagate_attributes

        tags: list[str] = []
        level = None
        status_message = None
        if exc is not None or status is None or status >= 500:
            tags.append("error")
            level = "ERROR"
            status_message = (
                tracing.error_label(exc) if exc is not None else f"HTTP {status}"
            )
        elif status >= 400:
            tags.append("client_error")
            level = "WARNING"
            status_message = f"HTTP {status}"
        with state.lock:
            if state.embedding_retries:
                tags.append("embedding_retried")
            if state.embedding_failures:
                tags.append("embedding_failed")
            metadata = {
                "http_status": status,
                "embedding_calls": state.embedding_calls,
                "embedding_retries": state.embedding_retries,
                "embedding_failures": state.embedding_failures,
                "embedding_tokens": state.embedding_tokens,
            }
        tracing.update_observation(
            state.root, metadata=metadata, level=level, status_message=status_message
        )
        if tags:
            with propagate_attributes(tags=tags):
                pass
    except Exception as e:
        logger.warning(
            "Failed to record the request outcome on its trace. Error: %s", e
        )


class MatchTracingMiddleware:
    """ASGI middleware: one Langfuse trace per ``POST`` to a matching route.

    Pure ASGI rather than ``BaseHTTPMiddleware`` so the endpoint runs in this task and inherits
    the trace context. Adds ``X-Request-ID`` and ``X-Trace-ID`` to traced responses.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or not is_match_path(scope.get("path") or "")
            or not tracing.is_tracing_enabled()
        ):
            await self.app(scope, receive, send)
            return

        route = scope["path"]
        request_id = _request_id_from_headers(scope.get("headers") or [])
        with _request_trace(route, request_id) as state:
            if state is None:
                await self.app(scope, receive, send)
                return
            status: dict[str, int | None] = {"code": None}

            async def send_with_ids(message: dict[str, Any]) -> None:
                if message.get("type") == "http.response.start":
                    status["code"] = message.get("status")
                    headers = list(message.get("headers") or [])
                    headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                    headers.append((TRACE_ID_HEADER.encode(), state.trace_id.encode()))
                    message = {**message, "headers": headers}
                await send(message)

            try:
                await self.app(scope, receive, send_with_ids)
            except BaseException as e:
                _finish_request(state, status["code"], e)
                raise
            _finish_request(state, status["code"], None)
