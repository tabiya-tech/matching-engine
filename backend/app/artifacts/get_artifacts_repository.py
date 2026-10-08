"""Provider for the singleton ArtifactsRepository."""

from app.artifacts.repository import ArtifactsRepository, IArtifactsRepository

_artifacts_repository_singleton: IArtifactsRepository = ArtifactsRepository()


def get_artifacts_repository() -> IArtifactsRepository:
    """Return the process-wide ArtifactsRepository."""
    return _artifacts_repository_singleton
