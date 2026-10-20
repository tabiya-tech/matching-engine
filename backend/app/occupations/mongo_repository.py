import logging
from collections.abc import Mapping
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import ValidationError

from app.clients.gemini_embedding_client import EMBEDDING_DIM
from app.occupations.occupation_document import (
    OCCUPATION_DOCUMENT_FIND_PROJECTION,
    OccupationDocument,
    vector_from_bin,
)
from app.occupations.occupation_file import OccupationFileEntry
from app.occupations.repository import IOccupationsRepository, OccupationRecord
from app.server_dependencies.database_collections import Collections


def record_from_document(
    doc: OccupationDocument, *, dim: int = EMBEDDING_DIM
) -> OccupationRecord:
    """The occupation record of a stored document (vectors of another length are dropped)."""
    # The plain file entry (already validated): the ingest fields and vector bytes stay behind.
    entry = OccupationFileEntry.model_construct(
        **{name: getattr(doc, name) for name in OccupationFileEntry.model_fields}
    )
    emb = doc.embedding
    whitened = vector_from_bin(emb.whitened_vector_bin, dim=dim) if emb else None
    whitening = emb.whitening if emb else None
    return OccupationRecord(
        entry=entry,
        embedding=vector_from_bin(emb.vector_bin, dim=dim) if emb else None,
        whitened_embedding=whitened,
        whitening_artifact_sha256=(
            whitening.artifact_sha256 if whitening and whitened is not None else None
        ),
    )


class MongoOccupationsRepository(IOccupationsRepository):
    """
    Mongo implementation of ``IOccupationsRepository``: reads the occupations written by
    ``scripts/ingest_occupations.py``.
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self._collection = db.get_collection(Collections.OCCUPATIONS)
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def source(self) -> str:
        return f"mongo:{Collections.OCCUPATIONS}"

    def _build_record(self, raw: Mapping[str, Any]) -> OccupationRecord | None:
        try:
            return record_from_document(OccupationDocument.model_validate(raw))
        except ValidationError as e:
            self._logger.warning(
                "Skipping invalid occupation document _id=%s: %s", raw.get("_id"), e
            )
            return None

    async def load_occupations(self) -> list[OccupationRecord]:
        cursor = self._collection.find({}, OCCUPATION_DOCUMENT_FIND_PROJECTION).sort(
            [("position", 1), ("_id", 1)]
        )
        records: list[OccupationRecord] = []
        async for raw in cursor:
            record = self._build_record(raw)
            if record is not None:
                records.append(record)
        return records
