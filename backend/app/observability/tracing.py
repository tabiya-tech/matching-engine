"""The tracing layer, built on Langfuse.

Ported from the llm-reranker (``app/observability``), itself ported from the compass-connect
backend. Everything in this module is safe to call unconditionally: when tracing is disabled —
the default, and always the case under pytest — every helper degrades to a cheap no-op that
yields ``None``, so call sites never guard with ``if tracing_enabled``.

Nothing here is allowed to break the request it is observing. The Langfuse client batches and
exports on its own threads, and every entry point is wrapped so that a tracing failure is logged
and swallowed rather than propagated onto the matching path.

Nesting follows the current context. ``asyncio.to_thread`` copies it, so the engines — which run
in worker threads — stay attached to the request trace. A bare ``ThreadPoolExecutor`` does not:
run such work through ``contextvars.copy_context().run``.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from opentelemetry import trace as otel_trace_api

from app.observability.config import TracingConfig
from app.observability.masking import build_mask_function

logger = logging.getLogger(__name__)

_client: Any = None
_config: TracingConfig = TracingConfig()

# Set while inside a ``suppress_tracing()`` block, i.e. a request sampling skipped.
_suppressed_ctx_var: ContextVar[bool] = ContextVar("tracing_suppressed", default=False)

# The innermost observation opened by ``traced_observation``, so code below it can annotate it
# without the observation being threaded through every signature.
_current_observation_ctx_var: ContextVar[Any] = ContextVar(
    "current_observation", default=None
)

_HASH_BYTES = 8
_HASH_SPACE = 2 ** (_HASH_BYTES * 8)


def init_tracing(config: TracingConfig) -> None:
    """Initialise the tracing layer. Call once, at process startup.

    Missing credentials or a bad host are logged and otherwise ignored: the service
    starts and runs untraced rather than failing to boot.
    """
    global _client, _config
    _config = config

    if not config.enabled:
        logger.info(
            "Matching tracing is disabled (MATCHING_ENABLE_TRACING). No traces will be exported."
        )
        _client = None
        return

    if not config.public_key or not config.secret_key:
        logger.warning(
            "Matching tracing is enabled but the Langfuse keys are not set "
            "(MATCHING_LANGFUSE_PUBLIC_KEY / MATCHING_LANGFUSE_SECRET_KEY). Tracing will stay off."
        )
        _client = None
        return

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host,
            environment=config.environment,
            release=config.release,
            debug=config.debug,
            timeout=config.timeout,
            flush_at=config.flush_at,
            flush_interval=config.flush_interval,
            mask=build_mask_function(config),
            tracing_enabled=True,
        )
    except Exception as e:
        logger.error(
            "Failed to initialize the Langfuse client. Tracing will stay off. Error: %s",
            e,
            exc_info=True,
        )
        _client = None
        return

    logger.info(
        "Matching tracing initialized | host=%s | environment=%s | release=%s | "
        "sample_rate=%s | record_embedding_input=%s | mask_pii=%s",
        config.host,
        config.environment,
        config.release,
        config.sample_rate,
        config.record_embedding_input,
        config.mask_pii,
    )


def shutdown_tracing() -> None:
    """Flush buffered observations and shut the client down.

    Must run on exit: Cloud Run tears the instance down shortly after SIGTERM, and without an
    explicit flush the last requests' spans are lost.
    """
    global _client
    if _client is None:
        return
    try:
        _client.flush()
        _client.shutdown()
        logger.info("Matching tracing flushed and shut down.")
    except Exception as e:
        logger.warning("Failed to shut the Langfuse client down cleanly. Error: %s", e)
    finally:
        _client = None


def is_tracing_enabled() -> bool:
    """Whether spans created right now would actually be recorded."""
    return _client is not None and not _suppressed_ctx_var.get()


def get_tracing_config() -> TracingConfig:
    """The configuration the tracing layer was initialised with (defaults if never)."""
    return _config


def get_tracing_client() -> Any:
    """The underlying Langfuse client, for the rare call site that needs the raw SDK."""
    return _client


def should_sample(key: str) -> bool:
    """Whether the request identified by ``key`` is traced. Deterministic per key."""
    rate = _config.sample_rate
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:_HASH_BYTES], "big") / _HASH_SPACE < rate


@contextmanager
def suppress_tracing() -> Iterator[None]:
    """Suppress span creation for the block, including in contexts copied from inside it."""
    token = _suppressed_ctx_var.set(True)
    try:
        yield
    finally:
        _suppressed_ctx_var.reset(token)


def has_active_span() -> bool:
    """Whether an OpenTelemetry span is active in the current context.

    Asked of OpenTelemetry directly: the Langfuse client's ``…_current_span`` helpers
    log a warning when there is none, which is noise on every untraced call.
    """
    return otel_trace_api.get_current_span() is not otel_trace_api.INVALID_SPAN


@contextmanager
def traced_observation(
    *,
    name: str,
    as_type: str = "span",
    input: Any = None,
    metadata: dict | None = None,
    model: str | None = None,
    model_parameters: dict | None = None,
) -> Iterator[Any]:
    """Open an observation nested under whatever is currently active.

    Only opens one inside a traced request (an active span): the engines are also run by
    offline scripts, which must not emit orphan traces. An exception escaping the block marks
    the observation ``ERROR`` with the exception's class name.

    :param as_type: The Langfuse observation type — "span", "embedding", ….
    :param input: The input payload; masked and truncated before export.
    :return: The Langfuse observation, or ``None`` when not tracing — handle ``None``.
    """
    if not is_tracing_enabled() or not has_active_span():
        yield None
        return

    try:
        manager = _client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
            metadata=metadata or None,
            model=model,
            model_parameters=_scalar_only(model_parameters),
        )
    except Exception as e:
        logger.warning(
            "Failed to start the %r observation. Continuing untraced. Error: %s",
            name,
            e,
        )
        yield None
        return

    with manager as observation:
        token = _current_observation_ctx_var.set(observation)
        try:
            yield observation
        except Exception as e:
            update_observation(
                observation, level="ERROR", status_message=error_label(e)
            )
            raise
        finally:
            _current_observation_ctx_var.reset(token)


@contextmanager
def stage(name: str, **metadata: Any) -> Iterator[Any]:
    """A pipeline-stage span (``retrieval``, ``embedding``, ``shortlist``, …).

    ``metadata`` must be counts, flags and ids only — never user profile content.
    """
    with traced_observation(
        name=name, as_type="span", metadata=metadata or None
    ) as observation:
        yield observation


def update_observation(observation: Any, **fields: Any) -> None:
    """Update an observation, tolerating both a ``None`` observation and a failing SDK call."""
    if observation is None:
        return
    try:
        observation.update(**fields)
    except Exception as e:
        logger.warning("Failed to update an observation. Error: %s", e)


def update_current_observation(**fields: Any) -> None:
    """Update the innermost observation opened by :func:`traced_observation`, if any."""
    update_observation(_current_observation_ctx_var.get(), **fields)


def report_usage(usage_details: dict[str, Any] | None) -> None:
    """Attach token counts to the embedding currently being recorded.

    Langfuse prices the observation from these counts and its model definition. Non-integer and
    negative counts are dropped rather than sent, since Langfuse rejects them.
    """
    if not usage_details or _current_observation_ctx_var.get() is None:
        return
    clean = {
        str(k): int(v)
        for k, v in usage_details.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
    }
    if clean:
        update_current_observation(usage_details=clean)


def current_trace_id() -> str | None:
    """The id of the trace in progress, for correlating logs with traces."""
    if _client is None or not has_active_span():
        return None
    try:
        return _client.get_current_trace_id()
    except Exception:
        return None


def error_label(exc: BaseException | None) -> str:
    """Error class, plus the HTTP status the client exposes if any. Never the message:
    upstream error text can echo the request content."""
    if exc is None:
        return "error"
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    name = type(exc).__name__
    return f"{name} ({code})" if isinstance(code, int) else name


def _scalar_only(parameters: dict | None) -> dict | None:
    """Keep only the values Langfuse accepts as model parameters."""
    if not parameters:
        return None
    kept = {
        key: value
        for key, value in parameters.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    return kept or None
