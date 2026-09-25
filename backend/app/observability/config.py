"""Configuration for the matching tracing layer.

Mirrors the llm-reranker (and, before it, the compass-connect backend): a handful of dedicated
environment variables for the credentials plus one JSON blob (``MATCHING_TRACING_CONFIG``) for the
tunables, so a deployment can be re-tuned without a code change.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, fields, replace
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LANGFUSE_HOST = "https://cloud.langfuse.com"

# Langfuse restricts the environment name to lowercase alphanumerics, hyphens and
# underscores, and reserves the "langfuse" prefix for its own use.
_ENVIRONMENT_ALLOWED_CHARS = re.compile(r"[^a-z0-9_-]")
_ENVIRONMENT_MAX_LENGTH = 40

_TRUE = ("1", "true", "yes", "on")


@dataclass(frozen=True)
class TracingConfig:
    """The resolved configuration of the tracing layer.

    An instance with ``enabled=False`` (the default) makes every helper in
    :mod:`app.observability.tracing` a no-op, so tracing can be shipped dark.
    """

    # Master on/off switch. Everything else is ignored when this is False.
    enabled: bool = False
    # The Langfuse project keys. The secret key belongs in a secret store, never plain env.
    public_key: str = ""
    secret_key: str = ""
    # The Langfuse ingestion host (cloud, or a self-hosted instance).
    host: str = DEFAULT_LANGFUSE_HOST
    # The Langfuse environment: the deployment (Pulumi stack — dev, shp, zmb, …). Per-deployment
    # latency and embedding spend are read by filtering on it.
    environment: str | None = None
    # The deployed build, so a trace can be tied back to the image that produced it.
    release: str | None = None
    # Turn on the Langfuse SDK's own debug logging.
    debug: bool = False

    # Fraction of matching requests traced. Deterministic per request id, so a retried request
    # carrying the same X-Request-ID gets the same decision.
    sample_rate: float = 1.0

    # Export the text sent to Gemini on each embedding observation. Off by default: it is the
    # jobseeker's skill profile, and traces must carry no personal data beyond pseudonymous ids.
    # When on, the text is still masked (below) before it leaves the process.
    record_embedding_input: bool = False
    # Redact e-mail addresses, phone numbers and long digit runs before export.
    mask_pii: bool = True
    # Truncate any single string in a payload to this many characters.
    max_payload_chars: int = 10_000

    # SDK buffering. Spans leave on a background thread; these only bound memory and delay.
    flush_at: int = 512
    flush_interval: float = 5.0
    # Hard timeout, in seconds, on the exporter's HTTP calls, so a slow Langfuse never
    # couples its latency onto the matching path.
    timeout: int = 10


# Field → (type, min, max) for the values the JSON blob may override.
_TUNABLES: dict[str, tuple[type, float | None, float | None]] = {
    "debug": (bool, None, None),
    "sample_rate": (float, 0.0, 1.0),
    "record_embedding_input": (bool, None, None),
    "mask_pii": (bool, None, None),
    "max_payload_chars": (int, 1, None),
    "flush_at": (int, 1, None),
    "flush_interval": (float, 0.001, None),
    "timeout": (int, 1, None),
}


def sanitize_environment_name(name: str | None) -> str | None:
    """Coerce an environment name into the shape Langfuse accepts, or ``None``."""
    if not name:
        return None
    sanitized = _ENVIRONMENT_ALLOWED_CHARS.sub("-", name.strip().lower()).strip("-")
    if not sanitized:
        return None
    if sanitized.startswith("langfuse"):
        # "langfuse" is a reserved prefix; keep the name meaningful rather than dropping it.
        sanitized = f"env-{sanitized}"
    return sanitized[:_ENVIRONMENT_MAX_LENGTH]


def parse_tracing_config(
    *,
    enabled: bool,
    public_key: str | None,
    secret_key: str | None,
    host: str | None,
    environment: str | None = None,
    release: str | None = None,
    raw_config: str | None = None,
) -> TracingConfig:
    """Build a :class:`TracingConfig` from already-read environment values.

    The JSON blob uses camelCase keys (``{"sampleRate": 0.2}``). Unknown keys and
    out-of-range values are logged and ignored, so a bad value can never stop the
    service from starting — tracing is optional.
    """
    config = TracingConfig(
        enabled=enabled,
        public_key=(public_key or "").strip(),
        secret_key=(secret_key or "").strip(),
        host=(host or "").strip() or DEFAULT_LANGFUSE_HOST,
        environment=sanitize_environment_name(environment),
        release=(release or "").strip() or None,
    )
    if not raw_config or not raw_config.strip():
        return config
    try:
        parsed = json.loads(raw_config)
    except json.JSONDecodeError as e:
        logger.warning(
            "Invalid MATCHING_TRACING_CONFIG JSON. Using defaults. Error: %s", e
        )
        return config
    if not isinstance(parsed, dict):
        logger.warning("MATCHING_TRACING_CONFIG must be a JSON object. Using defaults.")
        return config
    return replace(config, **_overrides_from_json(parsed))


def tracing_config_from_env() -> TracingConfig:
    """Read the tracing configuration from the process environment.

    Tracing is off unless ``MATCHING_ENABLE_TRACING`` opts in, so a deployment that has
    not been given Langfuse credentials behaves exactly as before. The keys fall back to
    the Langfuse SDK's own variable names, so a host that already exports those needs
    nothing new.
    """
    enabled = (os.getenv("MATCHING_ENABLE_TRACING") or "").strip().lower() in _TRUE
    return parse_tracing_config(
        enabled=enabled,
        public_key=_first_env("MATCHING_LANGFUSE_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY"),
        secret_key=_first_env("MATCHING_LANGFUSE_SECRET_KEY", "LANGFUSE_SECRET_KEY"),
        host=_first_env("MATCHING_LANGFUSE_HOST", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
        environment=_first_env("MATCHING_TRACING_ENVIRONMENT"),
        # Cloud Run sets K_REVISION on a service.
        release=_first_env("MATCHING_TRACING_RELEASE", "K_REVISION"),
        raw_config=os.getenv("MATCHING_TRACING_CONFIG", ""),
    )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return None


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _overrides_from_json(source: dict[str, Any]) -> dict[str, Any]:
    """camelCase JSON keys → validated ``TracingConfig`` field values.

    Credentials, host, environment and release come from dedicated variables and are
    never taken from the blob.
    """
    known = {f.name for f in fields(TracingConfig)}
    out: dict[str, Any] = {}
    for key, value in source.items():
        name = _CAMEL_CASE_BOUNDARY.sub("_", str(key)).lower()
        if name not in _TUNABLES:
            if name not in known:
                logger.warning("Ignoring unknown MATCHING_TRACING_CONFIG key %r.", key)
            else:
                logger.warning(
                    "Ignoring MATCHING_TRACING_CONFIG key %r — set it with its own variable.",
                    key,
                )
            continue
        typ, lo, hi = _TUNABLES[name]
        coerced = _coerce(value, typ)
        if (
            coerced is None
            or (lo is not None and coerced < lo)
            or (hi is not None and coerced > hi)
        ):
            logger.warning(
                "Ignoring invalid MATCHING_TRACING_CONFIG value %r=%r.", key, value
            )
            continue
        out[name] = coerced
    return out


def _coerce(value: Any, typ: type) -> Any:
    if typ is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in (
            *_TRUE,
            "0",
            "false",
            "no",
            "off",
        ):
            return value.strip().lower() in _TRUE
        return None
    if isinstance(value, bool):
        return None
    try:
        return typ(value)
    except (TypeError, ValueError):
        return None
