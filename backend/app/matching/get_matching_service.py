"""Dependency provider for the singleton MatchingService."""

import asyncio

from fastapi import Depends

from app.artifacts.repository import get_artifacts_repository
from app.config import V4_FULL_WHITENED_GATE
from app.jobs.get_jobs_repository import get_jobs_repository
from app.jobs.repository import IJobsRepository
from app.matching.service import IMatchingService, MatchingService
from app.occupations.get_occupations_repository import get_occupations_repository
from app.ranking.rerank import CrossEncoderRerank
from app.ranking.retrieval import Stage1Retriever
from app.server_dependencies.model_dependencies import (
    get_concat_whitener,
    get_cross_encoder_client,
    get_gemini_embedding_client,
    get_preference_scorer,
    get_skill_matcher,
    get_skill_scorer,
    get_v4_skill_matcher,
)

_matching_service_singleton: IMatchingService | None = None
_matching_service_lock = asyncio.Lock()


def create_matching_service(jobs_repository: IJobsRepository) -> IMatchingService:
    """A MatchingService wired to the process-wide repositories and model providers.

    Models are not loaded here; each loads on first use. Scripts call this directly.
    """
    embedding_client = get_gemini_embedding_client()
    return MatchingService(
        jobs_repository=jobs_repository,
        occupations_repository=get_occupations_repository(),
        artifacts_repository=get_artifacts_repository(),
        embedding_client=embedding_client,
        cross_encoder_provider=get_cross_encoder_client,
        retrieval_matcher_provider=get_skill_matcher,
        gate_matcher_provider=(
            get_v4_skill_matcher if V4_FULL_WHITENED_GATE else get_skill_matcher
        ),
        whitener_provider=get_concat_whitener,
        skill_scorer_provider=get_skill_scorer,
        preference_scorer_provider=get_preference_scorer,
        retriever=Stage1Retriever(embedding_dim=embedding_client.embedding_dim),
        rerank=CrossEncoderRerank(),
    )


async def get_matching_service(
    jobs_repository: IJobsRepository = Depends(get_jobs_repository),
) -> IMatchingService:
    """Return the process-wide MatchingService, instantiating it on first use."""
    global _matching_service_singleton  # pylint: disable=global-statement

    if _matching_service_singleton is None:
        async with _matching_service_lock:
            if _matching_service_singleton is None:
                _matching_service_singleton = create_matching_service(jobs_repository)

    return _matching_service_singleton
