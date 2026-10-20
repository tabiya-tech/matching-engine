from app.config import (
    MONGO_JOBS_COLLECTION,
    MONGO_OCCUPATION_GROUPS_COLLECTION,
    MONGO_OCCUPATIONS_COLLECTION,
)


class Collections:
    # Ranked/enriched job ads written by the llm-reranker. Unlike most collection names this one is
    # env-driven (MONGO_JOBS_COLLECTION) because each deployment stack reads a different collection.
    JOBS: str = MONGO_JOBS_COLLECTION
    # Occupations and occupation groups written by scripts/ingest_occupations.py (env-driven too, so a
    # stack can be pointed at a freshly ingested copy).
    OCCUPATIONS: str = MONGO_OCCUPATIONS_COLLECTION
    OCCUPATION_GROUPS: str = MONGO_OCCUPATION_GROUPS_COLLECTION
