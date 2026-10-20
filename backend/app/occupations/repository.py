"""Where the occupation catalogue comes from.

``IOccupationsRepository`` returns one ``OccupationRecord`` per occupation (the validated file entry
and its concat embedding); ``app.occupations.loader`` flattens and caches them. Two implementations:
``MongoOccupationsRepository`` (``app.occupations.mongo_repository``, the deployed source) and
``FileOccupationsRepository`` below (the committed JSON + NPZ, for tests and offline tools).
"""

import json
import logging
from abc import ABC, abstractmethod

import numpy as np
from pydantic import ConfigDict, Field

from app.domain.base import DomainModel
from app.occupations.occupation_file import OCCUPATION_FILE, OccupationFileEntry

logger = logging.getLogger(__name__)


class OccupationRecord(DomainModel):
    """One occupation as stored: its file entry and concat embedding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    entry: OccupationFileEntry = Field(description="The validated occupation entry.")
    embedding: np.ndarray | None = Field(
        default=None,
        repr=False,
        description="Raw L2-normalised concat skill embedding (float32); None when absent.",
    )
    whitened_embedding: np.ndarray | None = Field(
        default=None,
        repr=False,
        description="The same embedding whitened at ingest; None when not stored.",
    )
    whitening_artifact_sha256: str | None = Field(
        default=None,
        description="SHA-256 of the whitening artifact that made ``whitened_embedding``.",
    )


class IOccupationsRepository(ABC):
    """
    Interface for the Occupations Repository.
    Allows to swap the source (Mongo, files) and to fake it in tests.
    """

    @property
    @abstractmethod
    def source(self) -> str:
        """Short name of the source, for logs and timings (``mongo:<collection>``, ``file:<name>``)."""
        raise NotImplementedError

    @abstractmethod
    async def load_occupations(self) -> list[OccupationRecord]:
        """
        Load every occupation, in catalogue order.

        Entries that fail validation are skipped and logged, not raised.

        :return: The occupations; empty when the source holds none
        :raises Exception: If the source cannot be read (unreachable database, missing file)
        """
        raise NotImplementedError


def load_embeddings_npz(path: str) -> dict[str, np.ndarray]:
    """Occupation code -> float32 vector from an ``embed_occupations`` NPZ.

    :raises OSError: If the file cannot be read
    """
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as data:
        codes = [str(c) for c in data["codes"].tolist()]
        vectors = np.asarray(data["vectors"], dtype=np.float32)
    for code, vec in zip(codes, vectors, strict=True):
        out[code] = np.ascontiguousarray(vec, dtype=np.float32)
    return out


class FileOccupationsRepository(IOccupationsRepository):
    """Occupations from the committed occupation file and its concat-embeddings NPZ.

    The JSON must be readable (it is the catalogue); a missing or unreadable NPZ only means no
    embeddings, so the v4 engine skips the occupations, as before occupations moved to Mongo.
    """

    def __init__(self, occupation_json_path: str, embeddings_npz_path: str | None):
        self._json_path = occupation_json_path
        self._npz_path = embeddings_npz_path

    @property
    def source(self) -> str:
        return f"file:{self._json_path}"

    async def load_occupations(self) -> list[OccupationRecord]:
        with open(self._json_path, "r", encoding="utf-8") as f:
            entries = OCCUPATION_FILE.validate_python(json.load(f))
        embeddings = self._load_embeddings()
        return [
            OccupationRecord(
                entry=entry, embedding=embeddings.get(entry.occupation.code or "")
            )
            for entry in entries
        ]

    def _load_embeddings(self) -> dict[str, np.ndarray]:
        if not self._npz_path:
            return {}
        try:
            out = load_embeddings_npz(self._npz_path)
        except FileNotFoundError:
            logger.warning(
                "Occupation embeddings NPZ not found at %s; /match_v4 occupations will be skipped. "
                "Build it via `python -m tools.embeddings.embed_occupations`.",
                self._npz_path,
            )
            return {}
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "Failed to load occupation embeddings (%s): %s; occupations skipped.",
                self._npz_path,
                e,
            )
            return {}
        logger.info(
            "Loaded %d occupation concat embeddings from %s", len(out), self._npz_path
        )
        return out
