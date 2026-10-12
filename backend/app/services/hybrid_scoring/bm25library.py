"""BM25 job recommender, ``rank_bm25`` phrase-token mode — alias of
:mod:`app.services.bm25_scoring.bm25library`, kept so
``python -m app.services.hybrid_scoring.bm25library`` keeps working.

This module used to be a verbatim copy; the hybrid runner imports the BM25 package directly.
"""

from app.services.bm25_scoring.bm25library import (  # noqa: F401
    Bm25Recommendation,
    Bm25RunResult,
    Bm25UserResult,
    build_okapi_indexes,
    combine,
    main,
    min_max_normalise,
    recommend_for_user,
    run,
    scores_all_okapi,
)

if __name__ == "__main__":
    raise SystemExit(main())
