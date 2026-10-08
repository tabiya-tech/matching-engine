"""Cross-encoder reranker for cosine-shortlisted job recommendations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

from app.clients.cross_encoder_client import ICrossEncoderClient

from .text_pairs import build_job_passage_from_cosine_rec, build_user_query_text

logger = logging.getLogger(__name__)


def _non_empty_skill_text(raw: str, *, side: str) -> str:
    s = (raw or "").strip()
    return s if s else f"({side}: no skill labels)"


def rerank_cosine_recommendations(
    resolved_skill_labels: Sequence[str],
    cosine_recs: Sequence[Dict[str, Any]],
    *,
    reranker: ICrossEncoderClient,
    final_top_k: int,
) -> List[Dict[str, Any]]:
    """Sort cosine-shortlisted rows by cross-encoder relevance; keep ``final_top_k`` results.

    Query and passages are **skills-only** strings (preferred user labels vs ``job_skill_label``
    from each row's ``per_job_skill``).

    The HF cross-encoder returns **ordinal relevance logits**, not cosine similarity—the
    reranker only sorts by descending logit order. Persisted outputs:

      * ``cross_encoder_logit``: raw model output (ranking key before tie-break).
      * ``cross_encoder_score``: nonnegative [0, 1], min-max of logits **within this user's
        final slate** (rank-preserving for distinct logits).

    Also sets ``rank_cosine`` from the cosine stage and final ``rank`` after CE reordering.
    """

    k = max(0, int(final_top_k))
    rows = list(cosine_recs)
    if k == 0 or not rows:
        return []

    query = _non_empty_skill_text(
        build_user_query_text(resolved_skill_labels),
        side="User",
    )
    passages = [
        _non_empty_skill_text(build_job_passage_from_cosine_rec(r), side="Job")
        for r in rows
    ]
    pairs = [(query, p) for p in passages]
    scores = reranker.predict_scores(pairs)
    n = len(rows)
    if len(scores) < n:
        scores = scores + [0.0] * (n - len(scores))
    else:
        scores = scores[:n]

    decorated: List[Tuple[float, int, Dict[str, Any]]] = []
    for i, r in enumerate(rows):
        ce = scores[i] if i < len(scores) else 0.0
        decorated.append((ce, i, dict(r)))

    decorated.sort(key=lambda t: (-t[0], t[1]))
    chosen = decorated[:k]
    logits = [float(ce) for ce, _i, _row in chosen]
    lo = min(logits) if logits else 0.0
    hi = max(logits) if logits else 0.0
    span = hi - lo

    out: List[Dict[str, Any]] = []
    for new_rank, (logit_raw, _i, row) in enumerate(chosen, start=1):
        row["rank_cosine"] = row.get("rank")
        row["cross_encoder_logit"] = round(float(logit_raw), 6)
        if span <= 0:
            ce_pos = 1.0 if logits else 0.0
        else:
            ce_pos = round((float(logit_raw) - lo) / span, 6)
        row["cross_encoder_score"] = ce_pos
        row["rank"] = new_rank
        out.append(row)
    return out
