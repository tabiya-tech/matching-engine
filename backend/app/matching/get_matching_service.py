"""Dependency provider for the MatchingService."""

from fastapi import Depends

from app.artifacts.repository import get_artifacts_repository
from app.config import V4_FULL_WHITENED_GATE
from app.jobs.get_jobs_repository import get_jobs_repository
from app.jobs.repository import IJobsRepository
from app.matching.service import IMatchingService, MatchingService
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


def create_matching_service(jobs_repository: IJobsRepository) -> IMatchingService:
    """A MatchingService wired to the given jobs repository and the process-wide model providers.

    Models are not loaded here; each loads on first use. Scripts call this directly.
    """
    embedding_client = get_gemini_embedding_client()
    return MatchingService(
        jobs_repository=jobs_repository,
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
    # Built per request, like the jobs repository it wraps: the repository holds a collection handle
    # that must not outlive the provider's client. Construction is cheap; the models behind the
    # providers load once per process.
    return create_matching_service(jobs_repository)
