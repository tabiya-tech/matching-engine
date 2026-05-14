"""BM25 × **cosine** skill embeddings × **hybrid pool fusion** (single package).

Canonical path: ``app.services.hybrid_scoring`` — run all commands from ``backend``.

- **`run_bm25_cosine_hybrid`** — BM25 + :class:`~app.services.cosine_similarity.skill_score.CosineSkillMatcher`
  ``mean_best_cosine`` ∩ pool + weighted min–max fusion **only**.
- **`hybrid_match_users_with_jobs`** (in ``run_bm25_cosine_hybrid``) — in-memory scoring for **FastAPI ``POST /match_v2``** (same fusion as CLI).

Example::

    python -m app.services.hybrid_scoring.run_bm25_cosine_hybrid \\
        --users app/services/index_based_matching/njila_match_input.resolved.jsonl \\
        --from-mongo \\
        --output ./path/to/results.json

    python -m app.services.hybrid_scoring.build_bm25_cosine_4col_dashboard \\
        --input ./path/to/results.json \\
        --output ./path/to/dashboard.html

Fusion blend: ``α·norm(cos)+(1−α)·norm(BM25)``. Set ``α`` with ``--alpha-on-cosine`` or env
``HYBRID_ALPHA_ON_COSINE`` / ``ALPHA_ON_COSINE`` (CLI wins). Lower ``α`` = more BM25.
"""
