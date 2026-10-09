from app.config import MONGO_JOBS_COLLECTION


class Collections:
    # Ranked/enriched job ads written by the llm-reranker. Unlike most collection names this one is
    # env-driven (MONGO_JOBS_COLLECTION) because each deployment stack reads a different collection.
    JOBS: str = MONGO_JOBS_COLLECTION
