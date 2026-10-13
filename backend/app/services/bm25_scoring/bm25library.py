"""BM25 job recommender powered by ``rank_bm25`` (phrase-token mode).

Tokenisation lives in :mod:`text_builders`; the word-token (v1) variant is in
:mod:`legacy_word_tokens`. For BM25 + cosine embeddings + hybrid pool fusion see
``app.services.hybrid_scoring.run_bm25_cosine_hybrid``.

Variants
========

``single``
    One BM25 index over title + employer + location + description (word
    tokens) + skill phrases. Score = raw BM25 against that single doc.

``hybrid`` (default)
    Two BM25 indexes -- skills-only (phrases) + full text -- min-max
    normalised per user and blended via ``skills_weight`` / ``text_weight``
    (default ``0.5`` / ``0.5``).

Each recommendation includes:

- ``matched_skills``: sorted phrase tokens appearing on **both** the user skill
  list and the job's essential ∪ optional skills (skills index vocabulary).
- ``matched_skills_detail``: the same overlaps with one ``user_label`` and one
  ``job_label`` per phrase (useful when wording differs before normalisation).

The BM25 query is the user's skill phrases only (``UserProfile`` carries no programme /
institution / school-year fields; ``--programme-context`` is accepted but has no effect).

CLI::

    cd backend
    python -m app.services.bm25_scoring.bm25library \\
        --users app/services/index_based_matching/njila_match_input.resolved.jsonl \\
        --from-mongo --variant hybrid \\
        --output ./path/to/bm25_catalog_results.json \\
        --top-k 10
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import List, Literal, Optional, Protocol, Tuple

import numpy as np
from pydantic import BaseModel, Field

from app.domain.user import UserProfile
from app.jobs.repository import JobsLoadTiming
from app.ranking.types import MatchItem
from app.services.offline_io import JobsSource, load_jobs, load_users

from .text_builders import (
    SkillPhraseOverlap,
    build_corpora,
    matched_skill_overlap,
    matched_skill_phrases,
    user_query_tokens,
)


class BM25Index(Protocol):
    """The part of ``rank_bm25.BM25Okapi`` this module uses."""

    corpus_size: int

    def get_scores(self, query: list[str]) -> np.ndarray: ...


try:
    from rank_bm25 import BM25Okapi
except ImportError as _e:  # pragma: no cover - optional clarity at import time
    BM25Okapi = None  # type: ignore[misc, assignment]
    _RANK_BM25_IMPORT_ERROR = _e
else:
    _RANK_BM25_IMPORT_ERROR = None


def _require_rank_bm25() -> None:
    if BM25Okapi is None:
        raise ImportError(
            "rank_bm25 is required for bm25library. Install with: pip install rank-bm25"
        ) from _RANK_BM25_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Score normalisation / blending (kept local — no dependency on a separate
# BM25 implementation; used only when variant == 'hybrid')
# ---------------------------------------------------------------------------


def min_max_normalise(scores: np.ndarray) -> np.ndarray:
    """Map ``scores`` into ``[0, 1]`` per call; constant/empty arrays -> zeros."""
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def combine(
    scores_list: Iterable[np.ndarray],
    weights: Iterable[float],
) -> np.ndarray:
    """Weighted sum after per-array min-max; weights need not sum to 1."""
    arrays = list(scores_list)
    if not arrays:
        return np.zeros(0, dtype=np.float64)
    weights = list(weights)
    if len(weights) != len(arrays):
        raise ValueError("weights and scores_list must have the same length")
    out = np.zeros_like(arrays[0], dtype=np.float64)
    for arr, w in zip(arrays, weights, strict=True):
        out += float(w) * min_max_normalise(arr)
    return out


# ---------------------------------------------------------------------------
# rank_bm25 wrappers
# ---------------------------------------------------------------------------


def dedupe_query_tokens(tokens: List[str]) -> List[str]:
    """Preserve order; each token contributes at most once."""
    return list(dict.fromkeys(tokens))


def scores_all_okapi(okapi: BM25Index, query_tokens: List[str]) -> np.ndarray:
    """BM25 vector for one query over all docs (shape ``(n_docs,)``)."""
    _require_rank_bm25()
    n_docs = int(getattr(okapi, "corpus_size", len(okapi.doc_len)))
    if not query_tokens:
        return np.zeros(n_docs, dtype=np.float64)
    q = dedupe_query_tokens(query_tokens)
    if not q:
        return np.zeros(n_docs, dtype=np.float64)
    return np.asarray(okapi.get_scores(q), dtype=np.float64)


def corpus_vocab_size(docs: List[List[str]]) -> int:
    """Distinct tokens across ``docs``."""
    return len({t for doc in docs for t in doc})


def avg_doc_length(docs: List[List[str]]) -> float:
    """Mean tokens per doc (0 for no docs)."""
    if not docs:
        return 0.0
    return float(sum(len(d) for d in docs)) / len(docs)


def build_okapi_indexes(
    skills_corpus: List[List[str]],
    full_corpus: List[List[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> Tuple[BM25Index, BM25Index]:
    """Return ``(skills_bm25, full_bm25)`` ``BM25Okapi`` instances."""
    _require_rank_bm25()
    return (
        BM25Okapi(skills_corpus, k1=k1, b=b),
        BM25Okapi(full_corpus, k1=k1, b=b),
    )


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class Bm25Recommendation(BaseModel):
    """One BM25-ranked job for a user."""

    rank: int = Field(description="1-based rank by BM25 score.")
    job_uuid: str = Field(description="Job id.")
    job_title: str = Field(description="Job title (occupation label for occupations).")
    employer: str | None = Field(default=None, description="Employer (jobs only).")
    location: str = Field(description="Location.")
    bm25_score: float = Field(description="Ranking score (blended for ``hybrid``).")
    bm25_skills_score_raw: float | None = Field(
        default=None, description="Raw skills-index score (``hybrid`` only)."
    )
    bm25_text_score_raw: float | None = Field(
        default=None, description="Raw full-text score (``hybrid`` only)."
    )
    bm25_score_raw: float | None = Field(
        default=None, description="Raw full-text score (``single`` only)."
    )
    matched_skills: list[str] = Field(description="Shared skill phrase tokens.")
    matched_skills_detail: list[SkillPhraseOverlap] = Field(
        description="Shared skills with each side's wording."
    )


class Bm25UserResult(BaseModel):
    """BM25 recommendations for one user."""

    user_id: str | None = Field(description="The user's id.")
    city: str = Field(description="The user's city.")
    province: str = Field(description="The user's province.")
    n_query_tokens: int = Field(description="Query tokens.")
    query_tokens: list[str] = Field(description="The BM25 query.")
    recommendations: list[Bm25Recommendation] = Field(description="Top-k jobs.")


class Bm25IndexStats(BaseModel):
    """Corpus statistics of the two BM25 indexes."""

    n_jobs: int = Field(description="Indexed jobs.")
    skills_vocab_size: int = Field(description="Distinct skill phrase tokens.")
    full_vocab_size: int = Field(description="Distinct full-text tokens.")
    skills_avg_doc_len: float = Field(description="Mean skills-doc length.")
    full_avg_doc_len: float = Field(description="Mean full-text doc length.")
    embedding_dim: int | None = Field(
        default=None, description="Embedding dimension (hybrid runner only)."
    )


class Bm25RunConfig(BaseModel):
    """Configuration of one BM25 run."""

    users_path: str = Field(description="Users file.")
    jobs_source: JobsSource = Field(description="``file`` or ``mongo``.")
    variant: Literal["single", "hybrid"] = Field(description="Index variant.")
    tokenization: str = Field(description="Tokenisation mode.")
    top_k: int = Field(description="Recommendations per user.")
    k1: float = Field(description="BM25 k1.")
    b: float = Field(description="BM25 b.")
    bm25_engine: str = Field(description="BM25 implementation.")
    include_programme_context: bool = Field(description="CLI flag (no effect).")
    skills_weight: float | None = Field(default=None, description="``hybrid`` only.")
    text_weight: float | None = Field(default=None, description="``hybrid`` only.")
    jobs_path: str | None = Field(default=None, description="Jobs file (file source).")
    mongo_filter_by_users: bool | None = Field(
        default=None, description="Location prefilter (mongo source)."
    )


class Bm25RunResult(BaseModel):
    """The JSON payload of one BM25 run."""

    config: Bm25RunConfig = Field(description="Run configuration.")
    index_stats: Bm25IndexStats = Field(description="Index statistics.")
    n_users: int = Field(description="Users scored.")
    n_jobs: int = Field(description="Jobs indexed.")
    results: list[Bm25UserResult] = Field(description="One entry per user.")
    mongo_timing: JobsLoadTiming | None = Field(
        default=None, description="Job-load timing (mongo source)."
    )


def index_stats(
    jobs: Sequence[MatchItem],
    skills_corpus: List[List[str]],
    full_corpus: List[List[str]],
    *,
    embedding_dim: int | None = None,
) -> Bm25IndexStats:
    """Statistics of the indexes built over ``jobs``."""
    return Bm25IndexStats(
        n_jobs=len(jobs),
        skills_vocab_size=corpus_vocab_size(skills_corpus),
        full_vocab_size=corpus_vocab_size(full_corpus),
        skills_avg_doc_len=round(avg_doc_length(skills_corpus), 2),
        full_avg_doc_len=round(avg_doc_length(full_corpus), 2),
        embedding_dim=embedding_dim,
    )


def bm25_recommendation(
    user: UserProfile,
    job: MatchItem,
    *,
    rank: int,
    score: float,
    skills_score: float | None,
    text_score: float,
    variant: Literal["single", "hybrid"],
    max_matched: int | None = None,
) -> Bm25Recommendation:
    """One recommendation row: scores plus the user/job skill overlap."""
    hybrid = variant == "hybrid" and skills_score is not None
    phrases = matched_skill_phrases(user, job)
    overlap = matched_skill_overlap(user, job)
    if max_matched is not None:
        phrases = phrases[:max_matched]
        overlap = overlap[:max_matched]
    return Bm25Recommendation(
        rank=rank,
        job_uuid=job.uuid,
        job_title=job.title,
        employer=getattr(job, "employer", None),
        location=job.location,
        bm25_score=round(score, 4),
        bm25_skills_score_raw=round(skills_score, 4)
        if hybrid and skills_score is not None
        else None,
        bm25_text_score_raw=round(text_score, 4) if hybrid else None,
        bm25_score_raw=None if hybrid else round(text_score, 4),
        matched_skills=phrases,
        matched_skills_detail=overlap,
    )


# ---------------------------------------------------------------------------
# Per-user recommendation
# ---------------------------------------------------------------------------


def recommend_for_user(
    user: UserProfile,
    jobs: Sequence[MatchItem],
    skills_okapi: BM25Index,
    full_okapi: BM25Index,
    *,
    variant: Literal["single", "hybrid"] = "hybrid",
    top_k: int = 10,
    skills_weight: float = 0.5,
    text_weight: float = 0.5,
    include_programme: bool = False,
) -> Tuple[List[Bm25Recommendation], List[str]]:
    """Rank jobs for one user; rows include ``matched_skills`` overlap."""
    q_tokens = user_query_tokens(user, include_programme=include_programme)

    if variant == "single":
        text_scores = scores_all_okapi(full_okapi, q_tokens)
        combined = text_scores
        skills_scores: Optional[np.ndarray] = None
    else:
        skills_scores = scores_all_okapi(skills_okapi, q_tokens)
        text_scores = scores_all_okapi(full_okapi, q_tokens)
        combined = combine([skills_scores, text_scores], [skills_weight, text_weight])

    order = np.argsort(-combined - 1e-9 * (text_scores if text_scores.size else 0.0))

    recs: List[Bm25Recommendation] = []
    for rank, idx in enumerate(order, 1):
        if rank > top_k:
            break
        i = int(idx)
        recs.append(
            bm25_recommendation(
                user,
                jobs[i],
                rank=rank,
                score=float(combined[i]),
                skills_score=float(skills_scores[i])
                if skills_scores is not None
                else None,
                text_score=float(text_scores[i]),
                variant=variant,
            )
        )

    return recs, q_tokens


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def run(
    users_path: Path,
    jobs_source: JobsSource,
    jobs_path: Optional[Path],
    output_path: Optional[Path],
    *,
    variant: Literal["single", "hybrid"] = "hybrid",
    top_k: int = 10,
    skills_weight: float = 0.5,
    text_weight: float = 0.5,
    k1: float = 1.5,
    b: float = 0.75,
    include_programme: bool = False,
    mongo_filter_by_users: bool = True,
) -> Bm25RunResult:
    """Load users + jobs, score with rank_bm25, optionally write a JSON payload."""
    _require_rank_bm25()
    users = load_users(users_path)
    jobs, mongo_timing = load_jobs(
        jobs_source, jobs_path, users, mongo_filter_by_users, log_prefix="[bm25]"
    )

    print(
        f"[bm25] loaded {len(users)} users, {len(jobs)} jobs "
        f"(engine=rank_bm25, tokenisation=phrase)",
        file=sys.stderr,
    )

    skills_corpus, full_corpus = build_corpora(jobs)
    skills_okapi, full_okapi = build_okapi_indexes(
        skills_corpus, full_corpus, k1=k1, b=b
    )
    print(
        f"[bm25] indexes built — skills vocab={corpus_vocab_size(skills_corpus)}, "
        f"full vocab={corpus_vocab_size(full_corpus)}, "
        f"avg|d|_skills={avg_doc_length(skills_corpus):.1f}, "
        f"avg|d|_full={avg_doc_length(full_corpus):.1f}",
        file=sys.stderr,
    )

    results: List[Bm25UserResult] = []
    for user in users:
        recs, q_tokens = recommend_for_user(
            user,
            jobs,
            skills_okapi,
            full_okapi,
            variant=variant,
            top_k=top_k,
            skills_weight=skills_weight,
            text_weight=text_weight,
            include_programme=include_programme,
        )
        results.append(
            Bm25UserResult(
                user_id=user.user_id,
                city=user.city,
                province=user.province,
                n_query_tokens=len(q_tokens),
                query_tokens=q_tokens,
                recommendations=recs,
            )
        )

    payload = Bm25RunResult(
        config=Bm25RunConfig(
            users_path=str(users_path),
            jobs_source=jobs_source,
            variant=variant,
            tokenization="phrase",
            top_k=top_k,
            k1=k1,
            b=b,
            bm25_engine="rank_bm25.BM25Okapi",
            include_programme_context=include_programme,
            skills_weight=skills_weight if variant == "hybrid" else None,
            text_weight=text_weight if variant == "hybrid" else None,
            jobs_path=(str(jobs_path) if jobs_path else None)
            if jobs_source == "file"
            else None,
            mongo_filter_by_users=mongo_filter_by_users
            if jobs_source == "mongo"
            else None,
        ),
        index_stats=index_stats(jobs, skills_corpus, full_corpus),
        n_users=len(users),
        n_jobs=len(jobs),
        results=results,
        mongo_timing=mongo_timing,
    )

    text = payload.model_dump_json(indent=2, exclude_none=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[bm25] wrote {output_path}", file=sys.stderr)
    else:
        sys.stdout.write(text + "\n")

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="BM25 job recommender via rank_bm25 (phrase tokenisation)."
    )
    p.add_argument(
        "--users", required=True, type=Path, help="JSON or JSONL list of user records."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--jobs",
        type=Path,
        help="Local jobs JSON list (ranked-job documents or flat job objects).",
    )
    src.add_argument(
        "--from-mongo",
        action="store_true",
        help="Load active jobs from MongoDB using settings in backend/.env.",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--variant",
        choices=("single", "hybrid"),
        default="hybrid",
        help=(
            "single = one BM25 index over the full-text doc; "
            "hybrid = skills-only + full-text blended via min-max norm."
        ),
    )
    p.add_argument(
        "--skills-weight",
        type=float,
        default=0.5,
        help="Hybrid only: weight on the skills-only BM25 score.",
    )
    p.add_argument(
        "--text-weight",
        type=float,
        default=0.5,
        help="Hybrid only: weight on the full-text BM25 score.",
    )
    p.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 term-frequency saturation parameter.",
    )
    p.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 length-normalisation parameter (0..1).",
    )
    p.add_argument(
        "--programme-context",
        action="store_true",
        help=(
            "Add programme_name / institution_name / school_year as loose word "
            "tokens to the query (disabled by default; often adds dashboard noise)."
        ),
    )
    p.add_argument(
        "--mongo-all-active",
        action="store_true",
        help=(
            "With --from-mongo: ignore JOBS_RETRIEVAL_FILTER and load every "
            "is_active=true job (see JOBS_RETRIEVAL_LIMIT)."
        ),
    )
    args = p.parse_args(argv)

    jobs_source: JobsSource = "mongo" if args.from_mongo else "file"
    jobs_path = None if jobs_source == "mongo" else args.jobs

    run(
        users_path=args.users,
        jobs_source=jobs_source,
        jobs_path=jobs_path,
        output_path=args.output,
        variant=args.variant,
        top_k=args.top_k,
        skills_weight=args.skills_weight,
        text_weight=args.text_weight,
        k1=args.k1,
        b=args.b,
        include_programme=bool(args.programme_context),
        mongo_filter_by_users=not args.mongo_all_active,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
