"""Provider for the singleton ConcatCrossEncoderEngine."""

import threading

from app.artifacts.get_artifacts_repository import get_artifacts_repository
from app.matching.concat_ce_engine import (
    ConcatCrossEncoderEngine,
    IConcatCrossEncoderEngine,
)
from app.ranking.rerank import CrossEncoderRerank
from app.ranking.retrieval import Stage1Retriever
from app.server_dependencies.model_dependencies import (
    get_concat_whitener,
    get_cross_encoder_client,
    get_gemini_embedding_client,
    get_skill_matcher,
)

_engine_singleton: IConcatCrossEncoderEngine | None = None
_engine_lock = threading.Lock()


def get_concat_ce_engine() -> IConcatCrossEncoderEngine:
    """Return the process-wide engine, instantiating it on first use (models load lazily)."""
    global _engine_singleton  # pylint: disable=global-statement

    if _engine_singleton is None:
        with _engine_lock:
            if _engine_singleton is None:
                embedding_client = get_gemini_embedding_client()
                _engine_singleton = ConcatCrossEncoderEngine(
                    embedding_client=embedding_client,
                    cross_encoder_provider=get_cross_encoder_client,
                    matcher_provider=get_skill_matcher,
                    whitener_provider=get_concat_whitener,
                    artifacts_repository=get_artifacts_repository(),
                    retriever=Stage1Retriever(
                        embedding_dim=embedding_client.embedding_dim
                    ),
                    rerank=CrossEncoderRerank(),
                )

    return _engine_singleton
