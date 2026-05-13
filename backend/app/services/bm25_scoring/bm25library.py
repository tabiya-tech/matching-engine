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

By default programme / institution / school-year **word** tokens are **not**
added to the BM25 query (they often leak generic words onto the dashboard).
Pass ``include_programme=True`` or CLI ``--programme-context`` to restore them.

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
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np

from .text_builders import (
    build_corpora,
    matched_skill_overlap,
    matched_skill_phrases,
    user_query_tokens,
)

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
    for arr, w in zip(arrays, weights):
        out += float(w) * min_max_normalise(arr)
    return out


# ---------------------------------------------------------------------------
# rank_bm25 wrappers
# ---------------------------------------------------------------------------

def dedupe_query_tokens(tokens: List[str]) -> List[str]:
    """Preserve order; each token contributes at most once."""
    return list(dict.fromkeys(tokens))


def scores_all_okapi(okapi: Any, query_tokens: List[str]) -> np.ndarray:
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
    return len({t for doc in docs for t in doc})


def avg_doc_length(docs: List[List[str]]) -> float:
    if not docs:
        return 0.0
    return float(sum(len(d) for d in docs)) / len(docs)


def build_okapi_indexes(
    skills_corpus: List[List[str]],
    full_corpus: List[List[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> Tuple[Any, Any]:
    """Return ``(skills_bm25, full_bm25)`` ``BM25Okapi`` instances."""
    _require_rank_bm25()
    return (
        BM25Okapi(skills_corpus, k1=k1, b=b),
        BM25Okapi(full_corpus, k1=k1, b=b),
    )


# ---------------------------------------------------------------------------
# Paths / env / I/O helpers (self-contained; no cross-service imports)
# ---------------------------------------------------------------------------

def _backend_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "backend":
            return parent
    raise RuntimeError(f"Could not locate 'backend/' from {here}")


def _ensure_backend_on_syspath() -> None:
    root = str(_backend_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_backend_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # pylint: disable=import-outside-toplevel
    except ImportError:
        return
    p = _backend_root() / ".env"
    if p.is_file():
        load_dotenv(p, override=False)


def _is_jsonl(path: Path) -> bool:
    return path.suffix.lower() == ".jsonl"


def _load_users(path: Path) -> List[dict]:
    if _is_jsonl(path):
        users: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{ln}: invalid JSON ({e})") from e
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"{path}:{ln}: expected an object, got {type(obj).__name__}"
                    )
                users.append(obj)
        return users
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list, got {type(data).__name__}")
    return data


def _load_jobs_file(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list, got {type(data).__name__}")
    return data


async def _async_fetch_jobs_mongo(
    users_for_filter: Optional[List[dict]],
) -> Tuple[List[dict], Dict[str, Any]]:
    _load_backend_dotenv()
    _ensure_backend_on_syspath()
    from app.database import get_all_jobs_with_timing  # pylint: disable=import-outside-toplevel

    return await get_all_jobs_with_timing(users_for_filter)


def fetch_jobs_from_mongo(
    users_for_filter: Optional[List[dict]],
) -> Tuple[List[dict], Dict[str, Any]]:
    """Load active jobs via the same Motor pipeline as the REST API."""
    return asyncio.run(_async_fetch_jobs_mongo(users_for_filter))


# ---------------------------------------------------------------------------
# Per-user recommendation
# ---------------------------------------------------------------------------

def recommend_for_user(
    user: dict,
    jobs: List[dict],
    skills_okapi: Any,
    full_okapi: Any,
    *,
    variant: Literal["single", "hybrid"] = "hybrid",
    top_k: int = 10,
    skills_weight: float = 0.5,
    text_weight: float = 0.5,
    include_programme: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Rank jobs for one user; rows include ``matched_skills`` overlap."""
    q_tokens = user_query_tokens(user, include_programme=include_programme)

    if variant == "single":
        text_scores = scores_all_okapi(full_okapi, q_tokens)
        combined = text_scores
        skills_scores: Optional[np.ndarray] = None
    else:
        skills_scores = scores_all_okapi(skills_okapi, q_tokens)
        text_scores = scores_all_okapi(full_okapi, q_tokens)
        combined = combine(
            [skills_scores, text_scores], [skills_weight, text_weight]
        )

    order = np.argsort(
        -combined - 1e-9 * (text_scores if text_scores.size else 0.0)
    )

    recs: List[Dict[str, Any]] = []
    for rank, idx in enumerate(order, 1):
        if rank > top_k:
            break
        j = jobs[int(idx)]
        rec: Dict[str, Any] = {
            "rank": rank,
            "job_uuid": j.get("uuid") or j.get("_id"),
            "job_title": j.get("opportunity_title"),
            "employer": j.get("employer"),
            "location": j.get("location"),
            "bm25_score": round(float(combined[int(idx)]), 4),
        }
        if variant == "hybrid" and skills_scores is not None:
            rec["bm25_skills_score_raw"] = round(float(skills_scores[int(idx)]), 4)
            rec["bm25_text_score_raw"] = round(float(text_scores[int(idx)]), 4)
        else:
            rec["bm25_score_raw"] = round(float(text_scores[int(idx)]), 4)
        rec["matched_skills"] = matched_skill_phrases(user, j)
        rec["matched_skills_detail"] = matched_skill_overlap(user, j)
        recs.append(rec)

    return recs, q_tokens


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------

def run(
    users_path: Path,
    jobs_source: Literal["file", "mongo"],
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
) -> Dict[str, Any]:
    """Load users + jobs, score with rank_bm25, optionally write a JSON payload."""
    _require_rank_bm25()
    users = _load_users(users_path)
    mongo_timing: Optional[Dict[str, Any]] = None

    if jobs_source == "file":
        if jobs_path is None:
            raise ValueError("jobs_path is required when jobs_source is 'file'")
        jobs = _load_jobs_file(jobs_path)
    else:
        users_arg = users if mongo_filter_by_users else None
        jobs, mongo_timing = fetch_jobs_from_mongo(users_arg)
        if not mongo_filter_by_users:
            print(
                "[bm25] mongo: loaded all active jobs (no per-user location filter)",
                file=sys.stderr,
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

    results: List[Dict[str, Any]] = []
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
        results.append({
            "user_id": user.get("user_id"),
            "city": user.get("city"),
            "province": user.get("province"),
            "n_query_tokens": len(q_tokens),
            "query_tokens": q_tokens,
            "recommendations": recs,
        })

    config: Dict[str, Any] = {
        "users_path": str(users_path),
        "jobs_source": jobs_source,
        "variant": variant,
        "tokenization": "phrase",
        "top_k": top_k,
        "k1": k1,
        "b": b,
        "bm25_engine": "rank_bm25.BM25Okapi",
        "include_programme_context": include_programme,
    }
    if variant == "hybrid":
        config["skills_weight"] = skills_weight
        config["text_weight"] = text_weight
    if jobs_source == "file":
        config["jobs_path"] = str(jobs_path) if jobs_path else None
    else:
        config["mongo_filter_by_users"] = mongo_filter_by_users

    payload: Dict[str, Any] = {
        "config": config,
        "index_stats": {
            "n_jobs": len(jobs),
            "skills_vocab_size": corpus_vocab_size(skills_corpus),
            "full_vocab_size": corpus_vocab_size(full_corpus),
            "skills_avg_doc_len": round(avg_doc_length(skills_corpus), 2),
            "full_avg_doc_len": round(avg_doc_length(full_corpus), 2),
        },
        "n_users": len(users),
        "n_jobs": len(jobs),
        "results": results,
    }
    if mongo_timing is not None:
        payload["mongo_timing"] = mongo_timing

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[bm25] wrote {output_path}", file=sys.stderr)
    else:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="BM25 job recommender via rank_bm25 (phrase tokenisation)."
    )
    p.add_argument("--users", required=True, type=Path,
                   help="JSON or JSONL list of user records.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jobs", type=Path,
                     help="Local jobs list JSON (same shape as API job dicts).")
    src.add_argument("--from-mongo", action="store_true",
                     help="Load active jobs from MongoDB using settings in backend/.env.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--variant", choices=("single", "hybrid"), default="hybrid",
        help=(
            "single = one BM25 index over the full-text doc; "
            "hybrid = skills-only + full-text blended via min-max norm."
        ),
    )
    p.add_argument("--skills-weight", type=float, default=0.5,
                   help="Hybrid only: weight on the skills-only BM25 score.")
    p.add_argument("--text-weight", type=float, default=0.5,
                   help="Hybrid only: weight on the full-text BM25 score.")
    p.add_argument("--k1", type=float, default=1.5,
                   help="BM25 term-frequency saturation parameter.")
    p.add_argument("--b", type=float, default=0.75,
                   help="BM25 length-normalisation parameter (0..1).")
    p.add_argument(
        "--programme-context", action="store_true",
        help=(
            "Add programme_name / institution_name / school_year as loose word "
            "tokens to the query (disabled by default; often adds dashboard noise)."
        ),
    )
    p.add_argument(
        "--mongo-all-active", action="store_true",
        help=(
            "With --from-mongo: ignore JOBS_RETRIEVAL_FILTER and load every "
            "is_active=true job (see JOBS_RETRIEVAL_LIMIT)."
        ),
    )
    args = p.parse_args(argv)

    jobs_source: Literal["file", "mongo"] = "mongo" if args.from_mongo else "file"
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
