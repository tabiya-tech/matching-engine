"""Gemini ``gemini-embedding-001`` text-embedding client.

``embed_text_list`` batches texts through ``google.genai`` with retries and records each call as a
Langfuse embedding when the request is traced. ``GeminiEmbeddingClient`` is the injectable wrapper
the matching service uses for user concat embeddings.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.observability import (
    get_tracing_config,
    note_embedding_call,
    report_usage,
    traced_observation,
    update_observation,
)
from app.observability.tracing import error_label

MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIM = 3072
TASK_TYPE = "SEMANTIC_SIMILARITY"
# embed_content rejects more than ~100 texts per API call (ClientError / 400).
GEMINI_EMBED_API_MAX_BATCH = 100
DEFAULT_BATCH_SIZE = 100
DEFAULT_SLEEP_S = 0.12
MAX_RETRIES = 5
# Traced calls only: how long to wait, once embed_content has returned, for the parallel
# count_tokens call (see _start_token_count) before giving up on usage for that call.
TOKEN_COUNT_WAIT_S = 0.2


_token_count_pool: Optional[ThreadPoolExecutor] = None
_token_count_pool_lock = threading.Lock()


def _start_token_count(
    client: Any, model: str, texts: Sequence[str]
) -> Optional[Future]:
    """Count the input tokens alongside the embed call, for the trace's usage (and so its cost).

    The Gemini Developer API's ``embed_content`` reports no token counts (only Vertex fills
    ``statistics.token_count``), and Langfuse prices an embedding from its usage. Running
    ``count_tokens`` concurrently keeps it off the request's critical path.
    """
    global _token_count_pool
    try:
        if _token_count_pool is None:
            with _token_count_pool_lock:
                if _token_count_pool is None:
                    _token_count_pool = ThreadPoolExecutor(
                        max_workers=4, thread_name_prefix="gemini-count-tokens"
                    )
        return _token_count_pool.submit(
            client.models.count_tokens, model=model, contents=list(texts)
        )
    except Exception:
        return None


def _embedding_usage(
    result: Any, token_count: Optional[Future]
) -> Tuple[Optional[Dict[str, int]], str]:
    """Input tokens for one call and where they came from; never an invented count.

    Prefers the per-vector ``statistics.token_count`` (Vertex), else the parallel
    ``count_tokens`` result if it is ready in time, else no usage.
    """
    total = 0
    seen = False
    for emb in getattr(result, "embeddings", None) or []:
        count = getattr(getattr(emb, "statistics", None), "token_count", None)
        if isinstance(count, (int, float)) and not isinstance(count, bool):
            total += int(count)
            seen = True
    if seen:
        return {"input": total, "total": total}, "embed_content"
    if token_count is not None:
        try:
            counted = token_count.result(timeout=TOKEN_COUNT_WAIT_S).total_tokens
            if isinstance(counted, int) and counted >= 0:
                return {"input": counted, "total": counted}, "count_tokens"
        except Exception:
            token_count.cancel()
    return None, "unavailable"


def _embed_chunk(
    client: Any,
    texts: Sequence[str],
    *,
    model: str,
    embedding_dim: int,
    task_type: str,
) -> np.ndarray:
    """One ``embed_content`` request (with retries), recorded as a Langfuse embedding when the
    request is traced."""
    from google.genai import types as genai_types

    payload = list(texts)
    last_err: Optional[BaseException] = None
    attempt_ms: List[float] = []
    errors: List[str] = []
    with traced_observation(
        name="embed_content",
        as_type="embedding",
        # The text is the jobseeker's skill profile: exported only when a deployment opts in.
        input=payload if get_tracing_config().record_embedding_input else None,
        model=model,
        model_parameters={
            "task_type": task_type,
            "output_dimensionality": embedding_dim,
        },
        metadata={
            "provider": "gemini",
            "texts": len(payload),
            "input_characters": sum(len(t) for t in payload),
        },
    ) as observation:
        token_count = (
            _start_token_count(client, model, payload)
            if observation is not None
            else None
        )
        for attempt in range(MAX_RETRIES):
            t_attempt = time.perf_counter()
            try:
                result = client.models.embed_content(
                    model=model,
                    contents=payload,
                    config=genai_types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=embedding_dim,
                    ),
                )
                rows = []
                for emb in result.embeddings:
                    rows.append(np.asarray(emb.values, dtype=np.float32))
                mat = np.stack(rows, axis=0)
                attempt_ms.append(round((time.perf_counter() - t_attempt) * 1000.0, 2))
                usage, usage_source = (
                    _embedding_usage(result, token_count)
                    if observation is not None
                    else (None, "unavailable")
                )
                # The vectors themselves are not exported: 3072 floats per text is ingest
                # cost with nothing to read. Their count and shape are what a trace needs.
                update_observation(
                    observation,
                    output={"vectors": int(mat.shape[0]), "dim": int(mat.shape[1])},
                    metadata={
                        "attempts": attempt + 1,
                        "retries": attempt,
                        "failed": False,
                        "attempt_ms": attempt_ms,
                        "errors": errors,
                        "usage_source": usage_source,
                    },
                    **(
                        {
                            "level": "WARNING",
                            "status_message": f"succeeded after {attempt} retries",
                        }
                        if attempt
                        else {}
                    ),
                )
                report_usage(usage)
                note_embedding_call(
                    retries=attempt,
                    failed=False,
                    tokens=usage["input"] if usage else None,
                )
                return mat
            except Exception as e:
                attempt_ms.append(round((time.perf_counter() - t_attempt) * 1000.0, 2))
                errors.append(error_label(e))
                last_err = e
                if attempt == MAX_RETRIES - 1:
                    break
                wait = 2**attempt
                detail = str(e).strip() or repr(e)
                if len(detail) > 280:
                    detail = detail[:277] + "…"
                print(
                    f"    batch embed attempt {attempt + 1}/{MAX_RETRIES} failed "
                    f"({type(e).__name__}): {detail}; sleep {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
        if token_count is not None:
            token_count.cancel()
        update_observation(
            observation,
            level="ERROR",
            status_message=error_label(last_err),
            metadata={
                "attempts": MAX_RETRIES,
                "retries": MAX_RETRIES - 1,
                "failed": True,
                "attempt_ms": attempt_ms,
                "errors": errors,
            },
        )
        note_embedding_call(retries=MAX_RETRIES - 1, failed=True, tokens=None)
    raise RuntimeError(
        f"Gemini embed_content failed after {MAX_RETRIES} attempts: {last_err}"
    )


def embed_text_list(
    texts: List[str],
    *,
    api_key: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sleep_s: float = DEFAULT_SLEEP_S,
    model: str = MODEL_NAME,
    embedding_dim: int = EMBEDDING_DIM,
    task_type: str = TASK_TYPE,
) -> np.ndarray:
    from google import genai

    if not texts:
        return np.zeros((0, embedding_dim), dtype=np.float32)
    cap = GEMINI_EMBED_API_MAX_BATCH
    if batch_size > cap:
        print(
            f"[gemini_embeddings] batch_size={batch_size} exceeds API max ({cap}); "
            f"using {cap} (see Gemini embed limits).",
            file=sys.stderr,
        )
        batch_size = cap
    client = genai.Client(api_key=api_key.strip())
    n = len(texts)
    safe = [t.strip() if t.strip() else " " for t in texts]
    out = np.zeros((n, embedding_dim), dtype=np.float32)
    t0 = time.perf_counter()
    n_batches = (n + batch_size - 1) // batch_size
    report_every = max(1, (n_batches + 9) // 10)
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n)
        chunk = safe[start:end]
        out[start:end] = _embed_chunk(
            client, chunk, model=model, embedding_dim=embedding_dim, task_type=task_type
        )
        nb = b + 1
        if nb == n_batches or nb % report_every == 0:
            print(f"    … {end}/{n} texts (batch {nb}/{n_batches})", file=sys.stderr)
        if b < n_batches - 1 and sleep_s > 0:
            time.sleep(sleep_s)
    elapsed = time.perf_counter() - t0
    print(
        f"[gemini_embeddings] embedded {n} texts in {elapsed:.1f}s ({n / elapsed:.1f}/s)",
        file=sys.stderr,
    )
    return out


class IGeminiEmbeddingClient(ABC):
    """Interface for embedding texts with Gemini."""

    model_name: str
    """The Gemini embedding model used for every call."""

    embedding_dim: int
    """The dimension of every returned vector."""

    @abstractmethod
    def ensure_configured(self) -> None:
        """
        Checks the client can make calls.

        :raises ValueError: If no API key is configured
        """
        raise NotImplementedError()

    @abstractmethod
    def embed_texts(
        self,
        texts: List[str],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sleep_s: float = DEFAULT_SLEEP_S,
    ) -> np.ndarray:
        """
        Embeds the given texts.

        :param texts: The texts to embed, one row per text
        :param batch_size: Texts per ``embed_content`` request (capped at the API maximum)
        :param sleep_s: Pause between batches
        :return: float32 matrix of shape ``[len(texts), embedding_dim]`` (not normalised)
        :raises ValueError: If no API key is configured
        :raises RuntimeError: If a batch still fails after all retries
        """
        raise NotImplementedError()


class GeminiEmbeddingClient(IGeminiEmbeddingClient):
    """Gemini embedding client keyed by ``GEMINI_API_KEY``.

    The key is read on every call unless one is given, so a key set after start-up is picked up.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model_name: str = MODEL_NAME,
        embedding_dim: int = EMBEDDING_DIM,
        task_type: str = TASK_TYPE,
    ) -> None:
        self._api_key = api_key
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._task_type = task_type
        self._logger = logging.getLogger(self.__class__.__name__)

    def _resolve_api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key.strip()
        return (os.environ.get("GEMINI_API_KEY") or "").strip()

    def ensure_configured(self) -> None:
        if not self._resolve_api_key():
            raise ValueError(
                "GEMINI_API_KEY is not set (required for user concat embeddings)"
            )

    def embed_texts(
        self,
        texts: List[str],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sleep_s: float = DEFAULT_SLEEP_S,
    ) -> np.ndarray:
        self.ensure_configured()
        return embed_text_list(
            texts,
            api_key=self._resolve_api_key(),
            batch_size=batch_size,
            sleep_s=sleep_s,
            model=self.model_name,
            embedding_dim=self.embedding_dim,
            task_type=self._task_type,
        )
