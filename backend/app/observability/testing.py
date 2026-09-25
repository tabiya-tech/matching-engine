"""Test helpers for asserting on the spans the tracing layer produces.

Points the Langfuse client at an in-memory OpenTelemetry exporter, so tests assert on
the spans that *would* have been exported with no network access at all.

Langfuse keys its internal resources by public key, so a fresh client per test would
silently hand back the first client's exporter. The client is built once per process.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import app.observability.tracing as tracing_module
from app.observability.config import TracingConfig
from app.observability.masking import build_mask_function

# Placeholder credentials. Nothing is ever sent anywhere: the client exports to memory.
TEST_TRACING_CONFIG = TracingConfig(
    enabled=True,
    public_key="pk-lf-matching-test",
    secret_key="sk-lf-matching-test",
    host="http://localhost:1",
)

_client = None
_exporter: InMemorySpanExporter | None = None


class RecordedSpans:
    """A thin accessor over the in-memory exporter, so assertions read as span lookups."""

    def __init__(self, client, exporter: InMemorySpanExporter):
        self._client = client
        self._exporter = exporter

    def all(self) -> list:
        """All spans recorded so far. The processor batches, so the buffer is drained first."""
        self._client.flush()
        return list(self._exporter.get_finished_spans())

    def names(self) -> list[str]:
        return [span.name for span in self.all()]

    def by_name(self, name: str):
        """The first recorded span with ``name``, or ``None``."""
        for span in self.all():
            if span.name == name:
                return span
        return None

    @staticmethod
    def json_attribute(span, key: str) -> Any:
        """A span attribute Langfuse stores as JSON (usage, input, output), decoded.

        Plain-string payloads are stored as-is, not JSON-encoded, and come back unchanged.
        """
        raw = span.attributes.get(key)
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except ValueError:
            return raw


def _get_or_create_client():
    global _client, _exporter
    if _client is None:
        from langfuse import Langfuse

        _exporter = InMemorySpanExporter()
        _client = Langfuse(
            public_key=TEST_TRACING_CONFIG.public_key,
            secret_key=TEST_TRACING_CONFIG.secret_key,
            host=TEST_TRACING_CONFIG.host,
            span_exporter=_exporter,
            flush_interval=600,
            # The production mask, so tests see exactly what would be exported.
            mask=build_mask_function(TEST_TRACING_CONFIG),
        )
    return _client, _exporter


@contextmanager
def in_memory_tracing(config: TracingConfig | None = None) -> Iterator[RecordedSpans]:
    """Turn tracing on for the block, exporting to memory."""
    client, exporter = _get_or_create_client()
    exporter.clear()

    previous_client = tracing_module._client
    previous_config = tracing_module._config
    tracing_module._client = client
    tracing_module._config = config or TEST_TRACING_CONFIG
    try:
        yield RecordedSpans(client, exporter)
    finally:
        tracing_module._client = previous_client
        tracing_module._config = previous_config
