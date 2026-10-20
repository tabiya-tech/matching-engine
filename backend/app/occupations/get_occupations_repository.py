from app.config import (
    OCCUPATION_CONCAT_EMBEDDINGS_PATH,
    OCCUPATIONS_SOURCE,
    occupation_json_path,
)
from app.occupations.mongo_repository import MongoOccupationsRepository
from app.occupations.repository import (
    FileOccupationsRepository,
    IOccupationsRepository,
)
from app.server_dependencies.db_dependencies import MatchingDBProvider


def file_occupations_repository() -> FileOccupationsRepository:
    """The committed occupation file + embeddings NPZ (``OCCUPATION_JSON_PATH``, read live)."""
    return FileOccupationsRepository(
        occupation_json_path(), OCCUPATION_CONCAT_EMBEDDINGS_PATH
    )


async def get_occupations_repository() -> IOccupationsRepository:
    """The occupation source selected by ``OCCUPATIONS_SOURCE``.

    ``mongo`` shares the provider's client with the jobs repository (same database); built per call
    for the same reason as ``get_jobs_repository``.
    """
    if OCCUPATIONS_SOURCE == "file":
        return file_occupations_repository()
    return MongoOccupationsRepository(await MatchingDBProvider.get_jobs_db())
