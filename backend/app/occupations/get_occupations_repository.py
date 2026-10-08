"""Provider for the singleton OccupationsRepository (it holds the process-wide corpus caches)."""

import threading

from app.occupations.repository import IOccupationsRepository, OccupationsRepository

_occupations_repository_singleton: IOccupationsRepository | None = None
_occupations_repository_lock = threading.Lock()


def get_occupations_repository() -> IOccupationsRepository:
    """Return the process-wide OccupationsRepository, instantiating it on first use."""
    global _occupations_repository_singleton  # pylint: disable=global-statement

    if _occupations_repository_singleton is None:
        with _occupations_repository_lock:
            if _occupations_repository_singleton is None:
                _occupations_repository_singleton = OccupationsRepository()

    return _occupations_repository_singleton
