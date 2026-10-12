"""Stage-2 cross-encoder rerank of a cosine shortlist (pure: the model call is the caller's), and the skills-only query/passage text it scores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.ranking.types import Candidate

# --------------------------------------------------------------------------------------------------
# text_pairs
#
# Build query/passage strings for cross-encoder scoring (skills-only).
# --------------------------------------------------------------------------------------------------


def build_user_query_text(
    resolved_skill_labels: Sequence[str],
    *,
    max_skills: int = 48,
) -> str:
    """Preferred skill labels only (same strings ``CosineSkillMatcher`` embedded for cosine).

    Caller should pass ``resolved_user_skill_labels_ordered`` / equivalent preferred labels.
    """

    skills = [
        str(s).strip()
        for s in (resolved_skill_labels or [])[: max(0, int(max_skills))]
        if s and str(s).strip()
    ]
    if not skills:
        return ""
    return "; ".join(skills)


def build_job_passage_from_candidate(
    candidate: Candidate,
    *,
    max_skills: int = 64,
) -> str:
    """Item skill labels only, taken from the candidate's ``per_job_skill`` (``job_skill_label``)."""

    cap = max(0, int(max_skills))
    seen: set[str] = set()
    labels: list[str] = []
    for row in candidate.per_job_skill:
        if cap and len(labels) >= cap:
            break
        jl = (row.job_skill_label or "").strip()
        if not jl:
            continue
        key = jl.casefold()
        if key in seen:
            continue
        seen.add(key)
        labels.append(jl)
    if not labels:
        return ""
    return "; ".join(labels)


# --------------------------------------------------------------------------------------------------
# rerank
#
# Stage-2 cross-encoder rerank of a cosine shortlist (pure: the model call is the caller's).
# --------------------------------------------------------------------------------------------------


def _non_empty_skill_text(raw: str, *, side: str) -> str:
    s = (raw or "").strip()
    return s if s else f"({side}: no skill labels)"


class ICrossEncoderRerank(ABC):
    """Interface for reranking a cosine shortlist by cross-encoder relevance."""

    @abstractmethod
    def build_pairs(
        self,
        resolved_skill_labels: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        final_top_k: int,
    ) -> list[tuple[str, str]]:
        """
        The [query, passage] pairs to score, one per shortlisted row (empty when nothing to rerank).

        :param resolved_skill_labels: The user's skill labels (the query)
        :param candidates: The stage-1 shortlist
        :param final_top_k: Slate size after rerank
        """
        raise NotImplementedError()

    @abstractmethod
    def apply(
        self,
        candidates: Sequence[Candidate],
        scores: list[float],
        *,
        final_top_k: int,
    ) -> list[Candidate]:
        """
        Orders the shortlist by the scores of ``build_pairs`` and keeps ``final_top_k`` rows.

        :param candidates: The same candidates passed to ``build_pairs``
        :param scores: One cross-encoder logit per pair
        :param final_top_k: Slate size after rerank
        :return: Reranked candidates with ``rank_cosine``, ``cross_encoder_logit``,
            ``cross_encoder_score`` and the new ``rank`` set
        """
        raise NotImplementedError()


class CrossEncoderRerank(ICrossEncoderRerank):
    """Sort cosine-shortlisted rows by cross-encoder relevance; keep ``final_top_k`` results.

    Query and passages are **skills-only** strings (preferred user labels vs ``job_skill_label``
    from each candidate's ``per_job_skill``).

    The HF cross-encoder returns **ordinal relevance logits**, not cosine similarity—the
    reranker only sorts by descending logit order. Persisted outputs:

      * ``cross_encoder_logit``: raw model output (ranking key before tie-break).
      * ``cross_encoder_score``: nonnegative [0, 1], min-max of logits **within this user's
        final slate** (rank-preserving for distinct logits).

    Also sets ``rank_cosine`` from the cosine stage and final ``rank`` after CE reordering.
    """

    def build_pairs(
        self,
        resolved_skill_labels: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        final_top_k: int,
    ) -> list[tuple[str, str]]:
        k = max(0, int(final_top_k))
        rows = list(candidates)
        if k == 0 or not rows:
            return []

        query = _non_empty_skill_text(
            build_user_query_text(resolved_skill_labels),
            side="User",
        )
        passages = [
            _non_empty_skill_text(build_job_passage_from_candidate(r), side="Job")
            for r in rows
        ]
        return [(query, p) for p in passages]

    def apply(
        self,
        candidates: Sequence[Candidate],
        scores: list[float],
        *,
        final_top_k: int,
    ) -> list[Candidate]:
        k = max(0, int(final_top_k))
        rows = list(candidates)
        if k == 0 or not rows:
            return []

        n = len(rows)
        if len(scores) < n:
            scores = scores + [0.0] * (n - len(scores))
        else:
            scores = scores[:n]

        decorated = sorted(
            ((float(scores[i]), i, r) for i, r in enumerate(rows)),
            key=lambda t: (-t[0], t[1]),
        )
        chosen = decorated[:k]
        logits = [ce for ce, _i, _row in chosen]
        lo = min(logits) if logits else 0.0
        hi = max(logits) if logits else 0.0
        span = hi - lo

        out: list[Candidate] = []
        for new_rank, (logit_raw, _i, row) in enumerate(chosen, start=1):
            if span <= 0:
                ce_pos = 1.0 if logits else 0.0
            else:
                ce_pos = round((logit_raw - lo) / span, 6)
            out.append(
                row.model_copy(
                    update={
                        "rank_cosine": row.rank,
                        "cross_encoder_logit": round(logit_raw, 6),
                        "cross_encoder_score": ce_pos,
                        "rank": new_rank,
                    }
                )
            )
        return out
