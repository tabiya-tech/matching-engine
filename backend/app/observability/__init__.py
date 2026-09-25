"""Request tracing for the matching service, on Langfuse.

Ported from the llm-reranker (itself ported from the compass-connect backend). ``tracing`` holds
the lifecycle and the span helpers, ``config`` the settings, ``masking`` the pre-export redaction.
``request_trace`` is the matching-specific root trace every ``/match*`` request runs under.

What gets recorded, per matching request: one trace named after the route; a span per pipeline
stage — ``retrieval``, ``embedding``, ``shortlist``, ``rerank``, ``preference_scoring``,
``formatting``, ``skill_gaps`` (``corpus`` metadata tells the jobs and occupations passes apart);
and an ``embedding`` for each Gemini ``embed_content`` call (model, dimensionality, token usage,
attempts, retries, per-attempt latency, failure). Langfuse prices each embedding from its token
usage and model. No request or response body is recorded.
"""

from app.observability.config import (
    TracingConfig,
    parse_tracing_config,
    tracing_config_from_env,
)
from app.observability.request_trace import (
    MatchTracingMiddleware,
    is_match_path,
    note_embedding_call,
    set_request_users,
)
from app.observability.tracing import (
    current_trace_id,
    get_tracing_client,
    get_tracing_config,
    init_tracing,
    is_tracing_enabled,
    report_usage,
    shutdown_tracing,
    stage,
    suppress_tracing,
    traced_observation,
    update_current_observation,
    update_observation,
)

__all__ = [
    "MatchTracingMiddleware",
    "TracingConfig",
    "current_trace_id",
    "get_tracing_client",
    "get_tracing_config",
    "init_tracing",
    "is_match_path",
    "is_tracing_enabled",
    "note_embedding_call",
    "parse_tracing_config",
    "report_usage",
    "set_request_users",
    "shutdown_tracing",
    "stage",
    "suppress_tracing",
    "traced_observation",
    "tracing_config_from_env",
    "update_current_observation",
    "update_observation",
]
