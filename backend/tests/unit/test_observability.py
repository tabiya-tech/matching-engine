"""Tests for the Langfuse tracing layer (``app.observability``).

Covers what the integration promises:

1. off by default, and a no-op when off — nothing to guard at call sites, no traced headers;
2. every ``POST`` to a matching route is one trace named after the route, with the request id,
   the pseudonymous ``user_id`` and outcome tags; other routes and methods are untraced;
3. stage spans nest under the request trace, including from ``asyncio.to_thread`` workers;
4. every Gemini ``embed_content`` call is an ``embedding`` carrying model, parameters, shape,
   token usage, attempts and retries — but not the vectors, and not the input text by default;
5. failed and retried requests are tagged; sampling behaves as configured;
6. the real app's match routes record their stages.

No network: spans go to an in-memory exporter; the embedding API is stubbed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app import observability
from app.observability import (
    MatchTracingMiddleware,
    TracingConfig,
    is_match_path,
    parse_tracing_config,
    set_request_users,
    stage,
    tracing_config_from_env,
)
from app.observability.testing import (
    TEST_TRACING_CONFIG,
    RecordedSpans,
    in_memory_tracing,
)

TYPE = "langfuse.observation.type"
LEVEL = "langfuse.observation.level"
INPUT = "langfuse.observation.input"
OUTPUT = "langfuse.observation.output"
MODEL = "langfuse.observation.model.name"
USAGE = "langfuse.observation.usage_details"
META = "langfuse.observation.metadata"
TAGS = "langfuse.trace.tags"
TRACE_NAME = "langfuse.trace.name"
USER_ID = "user.id"

_ENV_KEYS = (
    "MATCHING_ENABLE_TRACING",
    "MATCHING_LANGFUSE_PUBLIC_KEY",
    "MATCHING_LANGFUSE_SECRET_KEY",
    "MATCHING_LANGFUSE_HOST",
    "MATCHING_TRACING_ENVIRONMENT",
    "MATCHING_TRACING_RELEASE",
    "MATCHING_TRACING_CONFIG",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_BASE_URL",
    "K_REVISION",
)


@pytest.fixture()
def isolated_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


class _FakeEmbedModels:
    """Stub of ``genai.Client().models``: fails ``fail_times`` times, then embeds."""

    def __init__(self, dim: int, *, fail_times: int = 0, counted_tokens: int = 7):
        self.dim = dim
        self.fail_times = fail_times
        self.counted_tokens = counted_tokens
        self.calls = 0

    def embed_content(self, *, model, contents, config):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("503 unavailable")
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[0.1] * self.dim, statistics=None)
                for _ in contents
            ]
        )

    def count_tokens(self, *, model, contents):
        return SimpleNamespace(total_tokens=self.counted_tokens)


def _embed(texts, *, models, monkeypatch, dim=4):
    from app.clients import gemini_embedding_client as gemini_embeddings

    monkeypatch.setattr(
        "google.genai.Client", lambda **_: SimpleNamespace(models=models)
    )
    monkeypatch.setattr(gemini_embeddings.time, "sleep", lambda _s: None)
    return gemini_embeddings.embed_text_list(texts, api_key="k", embedding_dim=dim)


def _app(handler) -> FastAPI:
    """A minimal app with the tracing middleware and one route per path shape."""
    app = FastAPI()
    app.add_middleware(MatchTracingMiddleware)
    for path in ("/match", "/match_v4", "/experiments/v5/match", "/jobs/search"):
        app.post(path)(handler)
    return app


def _by_name(spans: RecordedSpans, name: str) -> list:
    return [s for s in spans.all() if s.name == name]


def _metadata(span, key: str):
    return RecordedSpans.json_attribute(span, f"{META}.{key}")


# ---------------------------------------------------------------------------
# 1. Configuration, and the no-op path
# ---------------------------------------------------------------------------


def test_tracing_is_off_by_default(isolated_env):
    config = tracing_config_from_env()
    assert config.enabled is False
    assert config.record_embedding_input is False


def test_env_enables_tracing_and_keys_fall_back_to_sdk_names(isolated_env):
    isolated_env.setenv("MATCHING_ENABLE_TRACING", "true")
    isolated_env.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    isolated_env.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
    isolated_env.setenv("MATCHING_TRACING_ENVIRONMENT", "ZMB")
    isolated_env.setenv("K_REVISION", "matching-00042")
    config = tracing_config_from_env()
    assert config.enabled is True
    assert (config.public_key, config.secret_key) == ("pk-lf-x", "sk-lf-x")
    assert config.environment == "zmb"
    assert config.release == "matching-00042"


def test_json_tunables_are_applied_and_bad_values_ignored():
    config = parse_tracing_config(
        enabled=True,
        public_key="pk",
        secret_key="sk",
        host=None,
        raw_config='{"sampleRate": 0.25, "recordEmbeddingInput": true, '
        '"maxPayloadChars": -1, "publicKey": "nope", "bogus": 1}',
    )
    assert config.sample_rate == 0.25
    assert config.record_embedding_input is True
    assert config.max_payload_chars == TracingConfig().max_payload_chars
    assert config.public_key == "pk"


def test_helpers_and_middleware_are_no_ops_when_tracing_is_off(monkeypatch):
    with stage("retrieval") as span:
        assert span is None
    set_request_users([{"user_id": "u1"}])

    async def handler():
        return {"ok": True}

    resp = TestClient(_app(handler)).post("/match_v4")
    assert resp.status_code == 200
    assert "x-request-id" not in resp.headers

    # The embedding path runs untraced (and never calls count_tokens) outside a request.
    models = _FakeEmbedModels(dim=4)
    models.count_tokens = None
    assert _embed(["a"], models=models, monkeypatch=monkeypatch).shape == (1, 4)


def test_match_paths_cover_every_version():
    for path in ("/match", "/match_v2", "/match_v4", "/experiments/v3/match"):
        assert is_match_path(path)
    for path in ("/jobs", "/health", "/experiments/v3/other", "/matches"):
        assert not is_match_path(path)


# ---------------------------------------------------------------------------
# 2–3. The request trace and its stages
# ---------------------------------------------------------------------------


def test_match_request_is_one_trace_with_user_and_request_id():
    async def handler():
        set_request_users([{"user_id": "u-42", "city": "Nairobi"}], final_top_k=30)
        with stage("retrieval", corpus="jobs"):
            pass
        return {"ok": True}

    with in_memory_tracing() as spans:
        resp = TestClient(_app(handler)).post(
            "/match_v4", headers={"X-Request-ID": "req-123"}
        )

        assert resp.status_code == 200
        assert resp.headers["x-request-id"] == "req-123"
        (root,) = _by_name(spans, "/match_v4")
        (retrieval,) = _by_name(spans, "retrieval")

    assert resp.headers["x-trace-id"] == f"{root.context.trace_id:032x}"
    assert retrieval.parent.span_id == root.context.span_id
    assert root.attributes[USER_ID] == "u-42"
    assert root.attributes[TRACE_NAME] == "/match_v4"
    assert "route:/match_v4" in root.attributes[TAGS]
    assert _metadata(root, "request_id") == "req-123"
    assert _metadata(root, "final_top_k") == 30
    assert _metadata(root, "http_status") == 200
    # Only the id is read from the user: no location or profile field reaches the trace.
    exported = " ".join(str(v) for s in spans.all() for v in s.attributes.values())
    assert "Nairobi" not in exported
    assert INPUT not in root.attributes and OUTPUT not in root.attributes


def test_generated_request_id_is_the_trace_id():
    async def handler():
        return {}

    with in_memory_tracing() as spans:
        resp = TestClient(_app(handler)).post("/match")
        (root,) = _by_name(spans, "/match")
    assert resp.headers["x-request-id"] == resp.headers["x-trace-id"]
    assert resp.headers["x-trace-id"] == f"{root.context.trace_id:032x}"


def test_batch_request_lists_user_ids_without_a_trace_owner():
    async def handler():
        set_request_users([{"user_id": "a"}, {"user_id": "b"}])
        return {}

    with in_memory_tracing() as spans:
        TestClient(_app(handler)).post("/match")
        (root,) = _by_name(spans, "/match")
    assert USER_ID not in root.attributes
    assert _metadata(root, "user_ids") == "a,b"


def test_stage_spans_in_worker_threads_nest_under_the_request():
    def engine():
        with stage("shortlist", corpus="jobs"):
            with stage("rerank", corpus="jobs"):
                pass

    async def handler():
        await asyncio.to_thread(engine)
        return {}

    with in_memory_tracing() as spans:
        TestClient(_app(handler)).post("/experiments/v5/match")
        (root,) = _by_name(spans, "/experiments/v5/match")
        (shortlist,) = _by_name(spans, "shortlist")
        (rerank,) = _by_name(spans, "rerank")
    assert shortlist.context.trace_id == root.context.trace_id
    assert shortlist.parent.span_id == root.context.span_id
    assert rerank.parent.span_id == shortlist.context.span_id
    assert _metadata(shortlist, "corpus") == "jobs"
    # The route propagates to every stage, so stage percentiles can be filtered by route.
    assert "route:/experiments/v5/match" in shortlist.attributes[TAGS]


def test_non_match_routes_and_methods_are_not_traced():
    async def handler():
        return {}

    with in_memory_tracing() as spans:
        client = TestClient(_app(handler))
        assert "x-request-id" not in client.post("/jobs/search").headers
        assert client.get("/match").status_code == 405
        assert spans.all() == []


# ---------------------------------------------------------------------------
# 4. Embeddings
# ---------------------------------------------------------------------------


def test_embed_call_is_an_embedding_with_usage_and_no_vectors_or_text(monkeypatch):
    models = _FakeEmbedModels(dim=8, counted_tokens=7)

    async def handler():
        await asyncio.to_thread(
            _embed,
            ["welding; metal fabrication"],
            models=models,
            monkeypatch=monkeypatch,
            dim=8,
        )
        return {}

    with in_memory_tracing() as spans:
        TestClient(_app(handler)).post("/match_v4")
        (emb,) = [s for s in spans.all() if s.attributes.get(TYPE) == "embedding"]
        (root,) = _by_name(spans, "/match_v4")

    assert emb.name == "embed_content"
    assert emb.parent.span_id == root.context.span_id
    assert emb.attributes[MODEL] == "gemini-embedding-001"
    assert RecordedSpans.json_attribute(emb, OUTPUT) == {"vectors": 1, "dim": 8}
    # Tokens from count_tokens: Langfuse prices the call from these and its model definition.
    assert RecordedSpans.json_attribute(emb, USAGE) == {"input": 7, "total": 7}
    assert _metadata(emb, "usage_source") == "count_tokens"
    assert _metadata(emb, "retries") == 0
    # The input is the jobseeker's skill profile: not exported unless a deployment opts in.
    assert INPUT not in emb.attributes
    assert _metadata(root, "embedding_tokens") == 7
    assert "embedding_retried" not in root.attributes[TAGS]


def test_embedding_input_is_exported_masked_when_opted_in(monkeypatch):
    from dataclasses import replace

    models = _FakeEmbedModels(dim=4)
    config = replace(TEST_TRACING_CONFIG, record_embedding_input=True)

    async def handler():
        _embed(["call me on +254 712 345 678"], models=models, monkeypatch=monkeypatch)
        return {}

    with in_memory_tracing(config) as spans:
        TestClient(_app(handler)).post("/match_v4")
        (emb,) = [s for s in spans.all() if s.attributes.get(TYPE) == "embedding"]
    assert RecordedSpans.json_attribute(emb, INPUT) == ["call me on [REDACTED_PHONE]"]


def test_retried_embedding_is_counted_and_tags_the_request(monkeypatch):
    models = _FakeEmbedModels(dim=4, fail_times=2)

    async def handler():
        _embed(["a"], models=models, monkeypatch=monkeypatch)
        return {}

    with in_memory_tracing() as spans:
        TestClient(_app(handler)).post("/match_v4")
        (emb,) = [s for s in spans.all() if s.attributes.get(TYPE) == "embedding"]
        (root,) = _by_name(spans, "/match_v4")

    assert _metadata(emb, "attempts") == 3
    assert _metadata(emb, "retries") == 2
    assert _metadata(emb, "errors") == ["RuntimeError", "RuntimeError"]
    assert len(_metadata(emb, "attempt_ms")) == 3
    assert emb.attributes[LEVEL] == "WARNING"
    assert "embedding_retried" in root.attributes[TAGS]
    assert _metadata(root, "embedding_retries") == 2


# ---------------------------------------------------------------------------
# 5. Failures and sampling
# ---------------------------------------------------------------------------


def test_failed_embedding_fails_the_request_and_tags_it(monkeypatch):
    models = _FakeEmbedModels(dim=4, fail_times=99)

    async def handler():
        try:
            with stage("embedding"):
                _embed(["a"], models=models, monkeypatch=monkeypatch)
        except RuntimeError:
            raise HTTPException(status_code=500, detail="Internal server error")
        return {}

    with in_memory_tracing() as spans:
        resp = TestClient(_app(handler)).post("/match_v4")
        (emb,) = [s for s in spans.all() if s.attributes.get(TYPE) == "embedding"]
        (embedding_stage,) = _by_name(spans, "embedding")
        (root,) = _by_name(spans, "/match_v4")

    assert resp.status_code == 500
    assert emb.attributes[LEVEL] == "ERROR"
    assert _metadata(emb, "failed") is True
    assert embedding_stage.attributes[LEVEL] == "ERROR"
    assert root.attributes[LEVEL] == "ERROR"
    assert {"error", "embedding_failed", "embedding_retried"} <= set(
        root.attributes[TAGS]
    )
    assert _metadata(root, "http_status") == 500


def test_client_error_is_tagged_as_warning():
    async def handler():
        raise HTTPException(status_code=400, detail="bad")

    with in_memory_tracing() as spans:
        TestClient(_app(handler)).post("/match")
        (root,) = _by_name(spans, "/match")
    assert root.attributes[LEVEL] == "WARNING"
    assert "client_error" in root.attributes[TAGS]


def test_unsampled_request_records_nothing_beneath_it(monkeypatch):
    from dataclasses import replace

    async def handler():
        with stage("retrieval") as span:
            assert span is None
        return {}

    with in_memory_tracing(replace(TEST_TRACING_CONFIG, sample_rate=0.0)) as spans:
        resp = TestClient(_app(handler)).post("/match_v4")
        assert resp.status_code == 200
        assert spans.all() == []


# ---------------------------------------------------------------------------
# 6. The real app
# ---------------------------------------------------------------------------


def test_match_route_is_traced(test_client):
    path = "/match"
    body = [{"user_id": "u1", "skills_vector": {"top_skills": []}}]
    headers = {"x-api-key": "k"}
    with in_memory_tracing() as spans:
        resp = test_client.post(path, json=body, headers=headers)
        assert resp.status_code == 200, resp.text
        (root,) = _by_name(spans, path)
        names = {s.name for s in spans.all()}
    assert resp.headers["x-trace-id"] == f"{root.context.trace_id:032x}"
    assert root.attributes[USER_ID] == "u1"
    assert "retrieval" in names
    assert observability.is_tracing_enabled() is False  # restored after the block


def test_v4_engine_records_shortlist_and_rerank_per_corpus():
    """The real stage-1 → CE engine emits its own spans (engine dependencies stubbed)."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    from app.clients.gemini_embedding_client import EMBEDDING_DIM
    from app.services.match_concat_gemini_ce_service import run_match_concat_gemini_ce

    def job(uuid, index):
        v = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v[index] = 1.0
        return {"uuid": uuid, "essential_skills": [], "job_embedding": v.tolist()}

    matcher = MagicMock()
    matcher.score_pair.return_value = {"per_job_skill": []}
    user = {"user_id": "u1", "skills_vector": {"top_skills": []}}
    u_vecs = np.zeros((1, EMBEDDING_DIM))
    u_vecs[0, 0] = 1.0
    rerank = MagicMock()
    rerank.build_pairs.return_value = []
    rerank.apply.side_effect = lambda recs, _scores, **_kw: list(recs)

    async def handler():
        with (
            patch(
                "app.services.match_concat_gemini_ce_service.get_skill_matcher",
                return_value=matcher,
            ),
            patch(
                "app.services.match_concat_gemini_ce_service.get_cross_encoder_client",
                return_value=MagicMock(model_name="ce-test"),
            ),
            patch("app.services.match_concat_gemini_ce_service._RERANK", rerank),
        ):
            for corpus in ("jobs", "occupations"):
                await asyncio.to_thread(
                    run_match_concat_gemini_ce,
                    [user],
                    [job("a", 0), job("b", 1)],
                    retrieve_top_k=5,
                    final_top_k=5,
                    user_unit_vectors=u_vecs,
                    corpus=corpus,
                )
        return {}

    with in_memory_tracing() as spans:
        TestClient(_app(handler)).post("/match_v4")
        shortlists = _by_name(spans, "shortlist")
        reranks = _by_name(spans, "rerank")
    assert [_metadata(s, "corpus") for s in shortlists] == ["jobs", "occupations"]
    assert [_metadata(s, "corpus") for s in reranks] == ["jobs", "occupations"]
    assert _metadata(shortlists[0], "n_candidates_with_embedding") == 2
    assert _metadata(reranks[0], "model") == "ce-test"
