"""BM25 × embedding **cosine skill similarity** ∩ common pool → **hybrid** fusion JSON/HTML.

Pipeline
========

1. **BM25** — :mod:`bm25_scoring.bm25library` / ``rank_bm25.BM25Okapi`` (PyPI ``rank-bm25``) on phrase tokens.
2. **Cosine skills** — legacy :class:`~app.ranking.skills.CosineSkillMatcher`
   (``mean_best_cosine`` pool; essential ∪ optional; vectors only — no ``U_complete``, loc, groups).
3. **Common candidates** — top BM25 K ∩ top cosine K, union fallback below ``min_common``.
4. **Fusion (only method)** — on that pool:
   ``α · norm(mean_best_cosine) + (1−α) · norm(BM25)`` (min–max within candidates).
   **α** (weight on cosine): ``--alpha-on-cosine`` overrides, else read from env
   ``HYBRID_ALPHA_ON_COSINE`` or ``ALPHA_ON_COSINE``, else default ``0.5``.
   **More BM25 → lower α** (e.g. ``0.25`` ⇒ 75 % of the blend is BM25 after normalisation).

CLI (from ``backend``)::

    python -m app.services.hybrid_scoring.run_bm25_cosine_hybrid \\
        --users app/services/index_based_matching/njila_match_input.resolved.jsonl \\
        --from-mongo \\
        --output ./path/to/results.json

Optional MRR::

    python -m app.services.hybrid_scoring.run_bm25_cosine_hybrid \\
        --users app/services/index_based_matching/njila_match_input.resolved.jsonl \\
        --from-mongo \\
        --output ./path/to/out.json \\
        --mrr-relevance-json ./path/to/relevance.json

Requires ``rank-bm25``. Every row is a typed model; the JSON file is ``HybridRunResult``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import List, Literal, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field

from app.config import EMBEDDING_MODEL_PATH, SKILL_TO_ROW_PATH, SKILLS_CSV_PATH
from app.domain.user import UserProfile
from app.jobs.repository import JobsLoadTiming
from app.ranking.retrieval import (
    job_requires_post_secondary,
    user_lacks_post_secondary,
)
from app.ranking.skills import (
    CosineSkillMatcher,
    compact_cosine_matched_skill_lines,
)
from app.ranking.types import MatchItem
from app.services.bm25_scoring.bm25library import (
    BM25Index,
    Bm25IndexStats,
    Bm25Recommendation,
    bm25_recommendation,
    build_corpora,
    build_okapi_indexes,
    combine,
    index_stats,
    scores_all_okapi,
)
from app.services.bm25_scoring.text_builders import (
    matched_skill_phrases,
    user_query_tokens,
)
from app.services.offline_io import (
    JobsSource,
    load_backend_dotenv,
    load_jobs,
    load_users,
)

Bm25Variant = Literal["single", "hybrid"]


# --------------------------------------------------------------------------------------------------
# output models
# --------------------------------------------------------------------------------------------------


class CosineSummary(BaseModel):
    """Pooled cosine of one (user, job) pair, without the per-skill rows."""

    mean_best_cosine: float = Field(description="Mean best cosine over job skills.")
    min_best_cosine: float = Field(description="Minimum best cosine over job skills.")
    n_user_skills_embedded: int = Field(description="User skills with a vector.")
    n_job_skills_embedded: int = Field(description="Job skills with a vector.")


class _JobRef(BaseModel):
    job_uuid: str = Field(description="Job id.")
    job_title: str = Field(description="Job title (occupation label for occupations).")
    employer: str | None = Field(default=None, description="Employer (jobs only).")
    location: str = Field(description="Location.")


class HybridCosineRow(_JobRef):
    """One row of the cosine-skills column (top jobs by ``mean_best_cosine``)."""

    rank: int = Field(description="1-based rank by cosine.")
    mean_best_cosine: float = Field(description="Mean best cosine.")
    min_best_cosine: float = Field(description="Minimum best cosine.")
    rank_bm25_global: int = Field(description="Rank by BM25 over all jobs.")
    rank_cosine_global: int = Field(description="Rank by cosine over all jobs.")


class HybridCommonRow(_JobRef):
    """One job of the common candidate pool (BM25 top-K ∩ cosine top-K, or the union fallback)."""

    rank_within_bm25_pool: int = Field(
        description="Rank in the BM25 pool (999999 = absent)."
    )
    rank_within_cosine_pool: int = Field(
        description="Rank in the cosine pool (999999 = absent)."
    )
    mean_best_cosine: float = Field(description="Mean best cosine.")
    cosine_detail: CosineSummary = Field(description="Pooled cosine detail.")
    matched_skills: list[str] = Field(description="Shared skill phrase tokens (≤ 30).")
    in_pure_intersection: bool = Field(description="In both pools.")
    bm25_score: float = Field(description="BM25 ranking score.")
    bm25_skills_score_raw: float | None = Field(
        default=None, description="Raw skills-index score (``hybrid`` only)."
    )
    bm25_text_score_raw: float | None = Field(
        default=None, description="Raw full-text score (``hybrid`` only)."
    )
    bm25_score_raw: float | None = Field(
        default=None, description="Raw full-text score (``single`` only)."
    )


class HybridFusedRow(_JobRef):
    """One job of the fused column: ``α·norm(cosine) + (1−α)·norm(BM25)`` within the pool."""

    rank: int = Field(description="1-based rank by fusion score.")
    fusion_method: str = Field(description="Fusion method name.")
    fusion_score: float = Field(description="Fused score in [0, 1].")
    weighted_minmax_fusion: float = Field(description="Same as ``fusion_score``.")
    cos_norm_within_candidates: float = Field(description="Pool-normalised cosine.")
    bm25_norm_within_candidates: float = Field(description="Pool-normalised BM25.")
    mean_best_cosine_raw: float = Field(description="Raw mean best cosine.")
    bm25_score_raw: float = Field(description="Raw BM25 ranking score.")
    rank_bm25_global: int = Field(description="Rank by BM25 over all jobs.")
    rank_cosine_global: int = Field(description="Rank by cosine over all jobs.")
    was_in_bm25_cosine_pure_intersection: bool = Field(description="In both pools.")
    cosine_detail: CosineSummary = Field(description="Pooled cosine detail.")
    matched_skills: list[str] = Field(description="Shared skill phrase tokens (≤ 12).")
    matched_skills_cosine: list[str] = Field(
        description="``job ← user (cosine)`` lines for the strongest skill matches (≤ 40)."
    )


class ReciprocalRank(BaseModel):
    """Reciprocal rank of the first relevant job in one ordering."""

    reciprocal_rank: float = Field(
        description="1 / rank of the first hit; 0 without one."
    )
    first_hit_rank: int | None = Field(
        description="Rank of the first hit; None without one."
    )


class MrrFirstRelevant(BaseModel):
    """Per-user reciprocal ranks of the first labelled-relevant job."""

    n_relevant_in_labels: int = Field(
        description="Relevant jobs labelled for the user."
    )
    bm25_global: ReciprocalRank = Field(description="BM25 over all jobs.")
    cosine_skills_global: ReciprocalRank = Field(description="Cosine over all jobs.")
    fused_global_weighted_minmax: ReciprocalRank = Field(
        description="Fusion min-maxed over all jobs."
    )
    fused_candidate_pool_weighted_minmax: ReciprocalRank = Field(
        description="Fusion within the candidate pool."
    )


class HybridUserBundle(BaseModel):
    """Every hybrid column for one user."""

    user_id: str | None = Field(description="The user's id.")
    city: str = Field(description="The user's city.")
    province: str = Field(description="The user's province.")
    n_bm25_query_tokens: int = Field(description="BM25 query tokens.")
    query_tokens_bm25: list[str] = Field(description="The BM25 query (≤ 200).")
    n_user_skill_labels_display: int = Field(description="Displayed user skill labels.")
    n_resolved_user_skills_cosine: int = Field(description="User skills with a vector.")
    user_skill_labels_for_display: list[str] = Field(description="User skill labels.")
    resolved_user_skill_labels: list[str] = Field(description="Same labels.")
    bm25_pool_k_used: int = Field(description="BM25 pool size.")
    cosine_pool_k_used: int = Field(description="Cosine pool size.")
    n_bm25_cosine_intersection_pure: int = Field(description="Pool intersection size.")
    n_candidates_union_or_intersection: int = Field(description="Candidates fused.")
    fallback_union_used: bool = Field(description="Intersection too small; union used.")
    column_common_slice_max_applied: int = Field(description="Common rows kept.")
    column_bm25: list[Bm25Recommendation] = Field(description="Top jobs by BM25.")
    column_cosine_skills: list[HybridCosineRow] = Field(
        description="Top jobs by cosine."
    )
    column_common: list[HybridCommonRow] = Field(description="The candidate pool.")
    column_fused_weighted_minmax: list[HybridFusedRow] = Field(
        description="Top jobs by fusion score."
    )
    mrr_first_relevant: MrrFirstRelevant | None = Field(
        default=None, description="Reciprocal ranks, when relevance labels were given."
    )

    def without_jobs(self, uuids: Set[str]) -> HybridUserBundle:
        """A copy with the jobs in ``uuids`` removed from every column."""
        return self.model_copy(
            update={
                "column_bm25": [r for r in self.column_bm25 if r.job_uuid not in uuids],
                "column_cosine_skills": [
                    r for r in self.column_cosine_skills if r.job_uuid not in uuids
                ],
                "column_common": [
                    r for r in self.column_common if r.job_uuid not in uuids
                ],
                "column_fused_weighted_minmax": [
                    r
                    for r in self.column_fused_weighted_minmax
                    if r.job_uuid not in uuids
                ],
            }
        )


class HybridConfig(BaseModel):
    """Configuration of one hybrid run."""

    entrypoint: str = Field(description="Producing function.")
    scorer: str = Field(description="Scorer name.")
    skill_model: str = Field(description="Skill similarity model.")
    fusion: str = Field(description="Fusion method.")
    alpha_on_cosine_skill: float = Field(description="Fusion weight on cosine.")
    alpha_on_bm25: float = Field(description="Fusion weight on BM25 (1 − α).")
    variant_bm25: Bm25Variant = Field(description="BM25 index variant.")
    bm25_pool_k: int = Field(description="BM25 pool size.")
    cosine_pool_k: int = Field(description="Cosine pool size.")
    min_common_intersection: int = Field(
        description="Minimum intersection before union."
    )
    max_fallback_pool: int = Field(description="Union pool cap.")
    column_display_rows: int = Field(description="Rows per column.")
    column_common_max_rows_saved: int = Field(description="Common-pool rows kept.")
    k1: float = Field(description="BM25 k1.")
    b: float = Field(description="BM25 b.")
    include_programme_context: bool = Field(description="CLI flag (no effect).")
    embedding_model_path: str = Field(description="Embedding artifact.")
    skill_to_row_path: str = Field(description="Skill -> row map.")
    skills_csv_path: str = Field(description="Taxonomy skills CSV.")
    notes: str = Field(description="Free-text notes.")
    skills_weight: float | None = Field(default=None, description="``hybrid`` only.")
    text_weight: float | None = Field(default=None, description="``hybrid`` only.")
    users_path: str | None = Field(default=None, description="Users file (CLI).")
    jobs_source: JobsSource | None = Field(
        default=None, description="Job source (CLI)."
    )
    jobs_path: str | None = Field(
        default=None, description="Jobs file (CLI, file source)."
    )
    mongo_filter_by_users: bool | None = Field(
        default=None, description="Location prefilter (CLI)."
    )
    mrr_relevance_json: str | None = Field(
        default=None, description="MRR labels (CLI)."
    )


class ResolutionStats(BaseModel):
    """Label-miss telemetry of the skill matcher."""

    total_misses: int = Field(description="Unresolved label lookups.")
    distinct_missed_labels: int = Field(description="Distinct unresolved labels.")
    top_misses: list[tuple[str, int]] = Field(description="Most frequent misses.")


class MrrEvaluation(BaseModel):
    """Mean reciprocal ranks over the labelled users."""

    relevance_json: str = Field(description="Labels file.")
    queries_evaluated_nonempty_labels: int = Field(description="Users with labels.")
    queries_total_in_run: int = Field(description="Users in the run.")
    mean_reciprocal_rank: dict[str, float] = Field(description="Ordering -> MRR.")
    notes: str = Field(description="Free-text notes.")


class HybridRunResult(BaseModel):
    """The payload of one hybrid run (``hybrid_match_users_with_jobs`` / the CLI)."""

    config: HybridConfig = Field(description="Run configuration.")
    index_stats: Bm25IndexStats = Field(description="Index statistics.")
    n_users: int = Field(description="Users scored.")
    n_jobs: int = Field(description="Jobs indexed.")
    results: list[HybridUserBundle] = Field(description="One bundle per user.")
    skill_cosine_resolution_stats_end_of_run: ResolutionStats = Field(
        description="Matcher label-miss telemetry."
    )
    mongo_timing: JobsLoadTiming | None = Field(
        default=None, description="Job-load timing (CLI, mongo source)."
    )
    mrr_evaluation: MrrEvaluation | None = Field(
        default=None, description="MRR summary (CLI, with labels)."
    )


# --------------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------------


def _load_mrr_relevance(path: Path) -> dict[str, Set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object user_id → [uuid,...]")
    out: dict[str, Set[str]] = {}
    for k, v in data.items():
        uid = str(k)
        if v is None:
            out[uid] = set()
        elif isinstance(v, list):
            out[uid] = {str(x) for x in v if x is not None and str(x)}
        else:
            raise ValueError(f"{path}: value for {uid!r} must be a list of job UUIDs")
    return out


def _rr_first_hit(ordered_keys: Sequence[str], relevant: Set[str]) -> ReciprocalRank:
    if relevant:
        for pos, k in enumerate(ordered_keys, start=1):
            if k in relevant:
                return ReciprocalRank(
                    reciprocal_rank=1.0 / float(pos), first_hit_rank=pos
                )
    return ReciprocalRank(reciprocal_rank=0.0, first_hit_rank=None)


def _sorted_job_keys_from_argsort(
    jobs: Sequence[MatchItem], order: np.ndarray
) -> List[str]:
    return [jobs[int(i)].uuid for i in order.tolist()]


def _sidebar_skill_labels(user: UserProfile) -> List[str]:
    return [s.preferredLabel for s in user.skills_vector.top_skills if s.preferredLabel]


def _min_max_1d(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def bm25_vectors(
    user: UserProfile,
    jobs: Sequence[MatchItem],
    skills_okapi: BM25Index,
    full_okapi: BM25Index,
    *,
    variant_bm25: Bm25Variant,
    skills_weight: float,
    text_weight: float,
    include_programme: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, List[str]]:
    """``(combined, skills_scores | None, text_scores, query_tokens)`` for one user."""
    q_tokens = user_query_tokens(user, include_programme=include_programme)
    if variant_bm25 == "single":
        text_scores = scores_all_okapi(full_okapi, q_tokens)
        return text_scores, None, text_scores, q_tokens
    skills_scores = scores_all_okapi(skills_okapi, q_tokens)
    text_scores = scores_all_okapi(full_okapi, q_tokens)
    combined = combine([skills_scores, text_scores], [skills_weight, text_weight])
    return combined, skills_scores, text_scores, q_tokens


def cosine_scores_for_jobs(
    matcher: CosineSkillMatcher,
    user: UserProfile,
    jobs: Sequence[MatchItem],
) -> Tuple[np.ndarray, List[CosineSummary]]:
    """``mean_best_cosine`` per job + compact cosine detail aligned with ``jobs``."""

    vals = np.zeros(len(jobs), dtype=np.float64)
    details: List[CosineSummary] = []
    for i, job in enumerate(jobs):
        sp = matcher.score_pair(user, job)
        vals[i] = sp.mean_best_cosine
        details.append(
            CosineSummary(
                mean_best_cosine=round(float(vals[i]), 4),
                min_best_cosine=sp.min_best_cosine,
                n_user_skills_embedded=sp.n_user_skills_embedded,
                n_job_skills_embedded=sp.n_job_skills_embedded,
            )
        )
    return vals, details


def _ranks_high_to_low(scores: np.ndarray) -> np.ndarray:
    n = scores.shape[0]
    if n == 0:
        return scores
    order = np.lexsort((np.arange(n), -scores))
    ranks = np.empty(n, dtype=np.int32)
    for pos, idx in enumerate(order):
        ranks[idx] = pos + 1
    return ranks


def _job_ref(job: MatchItem) -> dict[str, str | None]:
    return {
        "job_uuid": job.uuid,
        "job_title": job.title,
        "employer": getattr(job, "employer", None),
        "location": job.location,
    }


def one_user_bundle(
    user: UserProfile,
    jobs: Sequence[MatchItem],
    skills_okapi: BM25Index,
    full_okapi: BM25Index,
    matcher: CosineSkillMatcher,
    *,
    variant_bm25: Bm25Variant,
    bm25_pool_k: int,
    cosine_pool_k: int,
    min_common: int,
    max_fallback_pool: int,
    col_display_k: int,
    alpha_on_cosine: float,
    skills_weight: float,
    text_weight: float,
    include_programme: bool,
    relevance_ids: Optional[Set[str]],
) -> HybridUserBundle:
    """Every hybrid column (BM25, cosine, common pool, fused) for one user."""
    combined, skills_v, text_v, q_tokens = bm25_vectors(
        user,
        jobs,
        skills_okapi,
        full_okapi,
        variant_bm25=variant_bm25,
        skills_weight=skills_weight,
        text_weight=text_weight,
        include_programme=include_programme,
    )

    cosine_v, cosine_details = cosine_scores_for_jobs(matcher, user, jobs)

    n = len(jobs)
    pool_b = max(1, min(bm25_pool_k, n))
    pool_c = max(1, min(cosine_pool_k, n))

    bm_ord = np.argsort(-combined - 1e-9 * text_v)
    cos_ord = np.argsort(-cosine_v)

    bm_top_idx = bm_ord[:pool_b].tolist()
    cos_top_idx = cos_ord[:pool_c].tolist()

    bm_keys = {jobs[i].uuid: pos for pos, i in enumerate(bm_top_idx, start=1)}
    cos_keys = {jobs[i].uuid: pos for pos, i in enumerate(cos_top_idx, start=1)}

    intersect_keys = set(bm_keys) & set(cos_keys)

    bm25_rank_full = _ranks_high_to_low(combined)
    cosine_rank_full = _ranks_high_to_low(cosine_v)

    fallback_union = False
    if len(intersect_keys) >= min_common:
        chosen_keys = intersect_keys
    else:
        fallback_union = True
        merged: List[str] = []
        seen: set[str] = set()
        for i in cos_top_idx + bm_top_idx:
            k = jobs[i].uuid
            if k and k not in seen:
                seen.add(k)
                merged.append(k)
        chosen_keys = set(merged[:max_fallback_pool])

    key_to_ji = {j.uuid: i for i, j in enumerate(jobs)}
    cand_ji = [key_to_ji[k] for k in chosen_keys if k in key_to_ji]
    cand_arr = np.array(cand_ji, dtype=np.int64)

    def _bm25_raw(ji: int) -> dict[str, float | None]:
        if variant_bm25 == "hybrid" and skills_v is not None:
            return {
                "bm25_skills_score_raw": round(float(skills_v[ji]), 4),
                "bm25_text_score_raw": round(float(text_v[ji]), 4),
            }
        return {"bm25_score_raw": round(float(text_v[ji]), 4)}

    fused_preview: List[HybridFusedRow] = []
    fus: Optional[np.ndarray] = None
    if cand_arr.size > 0:
        b_n = _min_max_1d(combined[cand_arr])
        cos_n = _min_max_1d(cosine_v[cand_arr])
        fus = (
            float(alpha_on_cosine) * cos_n + (1.0 - float(alpha_on_cosine)) * b_n
        ).astype(np.float64, copy=False)
        order_f = np.argsort(-fus)

        for fr, ix in enumerate(order_f.flat[:col_display_k], start=1):
            cand_slot = int(ix)
            ji = int(cand_arr[cand_slot])
            job = jobs[ji]
            fused = round(float(fus[cand_slot]), 6)
            fused_preview.append(
                HybridFusedRow(
                    **_job_ref(job),
                    rank=fr,
                    fusion_method="hybrid_pool_minmax",
                    fusion_score=fused,
                    weighted_minmax_fusion=fused,
                    cos_norm_within_candidates=round(float(cos_n[cand_slot]), 6),
                    bm25_norm_within_candidates=round(float(b_n[cand_slot]), 6),
                    mean_best_cosine_raw=round(float(cosine_v[ji]), 4),
                    bm25_score_raw=round(float(combined[ji]), 4),
                    rank_bm25_global=int(bm25_rank_full[ji]),
                    rank_cosine_global=int(cosine_rank_full[ji]),
                    was_in_bm25_cosine_pure_intersection=job.uuid in intersect_keys,
                    cosine_detail=cosine_details[ji],
                    matched_skills=matched_skill_phrases(user, job)[:12],
                    matched_skills_cosine=compact_cosine_matched_skill_lines(
                        matcher.score_pair(user, job).per_job_skill, limit=40
                    ),
                )
            )

    common_rows_sorted: List[HybridCommonRow] = []
    for ji in cand_ji:
        job = jobs[ji]
        common_rows_sorted.append(
            HybridCommonRow(
                **_job_ref(job),
                rank_within_bm25_pool=int(bm_keys.get(job.uuid, 999999)),
                rank_within_cosine_pool=int(cos_keys.get(job.uuid, 999999)),
                mean_best_cosine=round(float(cosine_v[ji]), 4),
                cosine_detail=cosine_details[ji],
                matched_skills=matched_skill_phrases(user, job)[:30],
                in_pure_intersection=job.uuid in intersect_keys,
                bm25_score=round(float(combined[ji]), 4),
                **_bm25_raw(ji),
            )
        )
    common_rows_sorted.sort(
        key=lambda r: (
            r.rank_within_bm25_pool + r.rank_within_cosine_pool,
            r.job_uuid,
        )
    )

    col_bm25 = [
        bm25_recommendation(
            user,
            jobs[ji],
            rank=pos,
            score=float(combined[ji]),
            skills_score=float(skills_v[ji]) if skills_v is not None else None,
            text_score=float(text_v[ji]),
            variant=variant_bm25,
            max_matched=40,
        )
        for pos, ji in enumerate(bm_top_idx[:col_display_k], start=1)
    ]

    col_cos = [
        HybridCosineRow(
            **_job_ref(jobs[ji]),
            rank=pos,
            mean_best_cosine=cosine_details[ji].mean_best_cosine,
            min_best_cosine=cosine_details[ji].min_best_cosine,
            rank_bm25_global=int(bm25_rank_full[ji]),
            rank_cosine_global=int(cosine_rank_full[ji]),
        )
        for pos, ji in enumerate(cos_top_idx[:col_display_k], start=1)
    ]

    resolved = matcher.resolved_user_skill_labels_ordered(user)
    labels = resolved[:200] or _sidebar_skill_labels(user)[:200]
    common_cap = min(max(col_display_k * 8, col_display_k), 200)

    mrr_first_relevant: Optional[MrrFirstRelevant] = None
    if relevance_ids:
        b_mm = _min_max_1d(combined)
        c_mm = _min_max_1d(cosine_v)
        fus_gl = float(alpha_on_cosine) * c_mm + (1.0 - float(alpha_on_cosine)) * b_mm
        fus_g_ord = np.argsort(-fus_gl - 1e-9 * combined)

        keys_fus_c: List[str] = []
        if cand_arr.size > 0 and fus is not None:
            keys_fus_c = [
                jobs[int(cand_arr[int(ix)])].uuid
                for ix in np.argsort(-fus).flatten().tolist()
            ]

        mrr_first_relevant = MrrFirstRelevant(
            n_relevant_in_labels=len(relevance_ids),
            bm25_global=_rr_first_hit(
                _sorted_job_keys_from_argsort(jobs, bm_ord), relevance_ids
            ),
            cosine_skills_global=_rr_first_hit(
                _sorted_job_keys_from_argsort(jobs, cos_ord), relevance_ids
            ),
            fused_global_weighted_minmax=_rr_first_hit(
                _sorted_job_keys_from_argsort(jobs, fus_g_ord), relevance_ids
            ),
            fused_candidate_pool_weighted_minmax=_rr_first_hit(
                keys_fus_c, relevance_ids
            ),
        )

    return HybridUserBundle(
        user_id=user.user_id,
        city=user.city,
        province=user.province,
        n_bm25_query_tokens=len(q_tokens),
        query_tokens_bm25=q_tokens[:200],
        n_user_skill_labels_display=len(labels),
        n_resolved_user_skills_cosine=len(resolved),
        user_skill_labels_for_display=labels,
        resolved_user_skill_labels=labels,
        bm25_pool_k_used=pool_b,
        cosine_pool_k_used=pool_c,
        n_bm25_cosine_intersection_pure=len(intersect_keys),
        n_candidates_union_or_intersection=len(cand_ji),
        fallback_union_used=fallback_union,
        column_common_slice_max_applied=min(common_cap, len(common_rows_sorted)),
        column_bm25=col_bm25,
        column_cosine_skills=col_cos,
        column_common=common_rows_sorted[:common_cap],
        column_fused_weighted_minmax=fused_preview,
        mrr_first_relevant=mrr_first_relevant,
    )


_cosine_matcher_singleton: Optional[CosineSkillMatcher] = None


def get_cosine_matcher_singleton() -> CosineSkillMatcher:
    """Reuse one embedding loader across CLI batches and ``POST /match_v2`` calls."""

    global _cosine_matcher_singleton
    if _cosine_matcher_singleton is None:
        from app.server_dependencies.model_dependencies import build_skill_matcher

        _cosine_matcher_singleton = build_skill_matcher()
    return _cosine_matcher_singleton


def hybrid_match_users_with_jobs(
    users: Sequence[UserProfile],
    jobs: Sequence[MatchItem],
    *,
    variant_bm25: Bm25Variant = "hybrid",
    bm25_pool_k: int = 120,
    cosine_pool_k: int = 120,
    min_common: int = 3,
    max_fallback_pool: int = 200,
    col_display_k: int = 15,
    alpha_on_cosine: float = 0.5,
    skills_weight: float = 0.5,
    text_weight: float = 0.5,
    k1: float = 1.5,
    b: float = 0.75,
    include_programme: bool = False,
    mrr_labels: Optional[Mapping[str, Set[str]]] = None,
) -> HybridRunResult:
    """Run BM25×cosine hybrid over in-memory ``users`` and ``jobs`` (or occupations).

    Same scoring as CLI :func:`run` but without file paths or Mongo fetches.
    Used by ``POST /match_v2`` and internally by :func:`run`.

    Preconditions: ``rank-bm25`` installed; ``jobs`` non-empty (needed to build indexes).

    :raises ValueError: If ``jobs`` is empty
    """

    from app.services.bm25_scoring.bm25library import _require_rank_bm25

    _require_rank_bm25()
    if not jobs:
        raise ValueError(
            "hybrid_match_users_with_jobs requires at least one job document"
        )

    skills_corpus, full_corpus = build_corpora(jobs)
    skills_okapi, full_okapi = build_okapi_indexes(
        skills_corpus, full_corpus, k1=k1, b=b
    )
    matcher = get_cosine_matcher_singleton()

    # Post-secondary education gate: indexes are built over all jobs (shared across users),
    # so we drop ineligible jobs from each user's output columns rather than pre-filtering jobs.
    ps_required_uuids = {j.uuid for j in jobs if job_requires_post_secondary(j)}
    results: List[HybridUserBundle] = []
    for u in users:
        rel = mrr_labels.get(u.user_id or "") if mrr_labels is not None else None
        bundle = one_user_bundle(
            u,
            jobs,
            skills_okapi,
            full_okapi,
            matcher,
            variant_bm25=variant_bm25,
            bm25_pool_k=bm25_pool_k,
            cosine_pool_k=cosine_pool_k,
            min_common=min_common,
            max_fallback_pool=max_fallback_pool,
            col_display_k=col_display_k,
            alpha_on_cosine=alpha_on_cosine,
            skills_weight=skills_weight,
            text_weight=text_weight,
            include_programme=include_programme,
            relevance_ids=rel,
        )
        if ps_required_uuids and user_lacks_post_secondary(u):
            bundle = bundle.without_jobs(ps_required_uuids)
        results.append(bundle)

    config = HybridConfig(
        entrypoint="hybrid_match_users_with_jobs",
        scorer="bm25_cosine_skills_four_col_hybrid_pool_minmax",
        skill_model="CosineSkillMatcher.mean_best_cosine (embedding cosine on skills)",
        fusion="weighted_minmax_on_candidate_pool_bm25_cosine_only",
        alpha_on_cosine_skill=alpha_on_cosine,
        alpha_on_bm25=round(1.0 - alpha_on_cosine, 4),
        variant_bm25=variant_bm25,
        bm25_pool_k=bm25_pool_k,
        cosine_pool_k=cosine_pool_k,
        min_common_intersection=min_common,
        max_fallback_pool=max_fallback_pool,
        column_display_rows=col_display_k,
        column_common_max_rows_saved=min(max(col_display_k * 8, col_display_k), 200),
        k1=k1,
        b=b,
        include_programme_context=include_programme,
        embedding_model_path=str(EMBEDDING_MODEL_PATH),
        skill_to_row_path=str(SKILL_TO_ROW_PATH),
        skills_csv_path=str(SKILLS_CSV_PATH),
        notes=(
            "CosineSkillMatcher pooled embedding cosine for skills; BM25 hybrid index; "
            "common ∩ pool; fused = α·norm(cosine)+(1−α)·norm(BM25) within pool only."
        ),
        skills_weight=skills_weight if variant_bm25 == "hybrid" else None,
        text_weight=text_weight if variant_bm25 == "hybrid" else None,
    )
    stats = matcher.get_resolution_stats()
    return HybridRunResult(
        config=config,
        index_stats=index_stats(
            jobs, skills_corpus, full_corpus, embedding_dim=int(matcher.W.shape[1])
        ),
        n_users=len(users),
        n_jobs=len(jobs),
        results=results,
        skill_cosine_resolution_stats_end_of_run=ResolutionStats.model_validate(stats),
    )


def run(
    users_path: Path,
    jobs_source: JobsSource,
    jobs_path: Optional[Path],
    output_path: Optional[Path],
    *,
    variant_bm25: Bm25Variant = "hybrid",
    bm25_pool_k: int = 120,
    cosine_pool_k: int = 120,
    min_common: int = 3,
    max_fallback_pool: int = 200,
    col_display_k: int = 15,
    alpha_on_cosine: float = 0.5,
    skills_weight: float = 0.5,
    text_weight: float = 0.5,
    k1: float = 1.5,
    b: float = 0.75,
    include_programme: bool = False,
    mongo_filter_by_users: bool = True,
    mrr_relevance_path: Optional[Path] = None,
) -> HybridRunResult:
    """CLI run: load users + jobs, score, add MRR when labelled, write the JSON payload.

    :raises FileNotFoundError: If ``mrr_relevance_path`` does not exist
    """
    from app.services.bm25_scoring.bm25library import _require_rank_bm25

    _require_rank_bm25()

    mrr_labels: dict[str, Set[str]] = {}
    if mrr_relevance_path is not None:
        mp = Path(mrr_relevance_path).expanduser()
        if not mp.is_file():
            raise FileNotFoundError(
                f"MRR relevance file not found: {mp} — omit --mrr-relevance-json to skip "
                "MRR, or pass an existing JSON path (mapping user_id → [job_uuid, …])."
            )
        mrr_labels = _load_mrr_relevance(mp)

    users = load_users(users_path)
    jobs, mongo_timing = load_jobs(
        jobs_source,
        jobs_path,
        users,
        mongo_filter_by_users,
        log_prefix="[bm25_cosine_hybrid]",
    )

    payload = hybrid_match_users_with_jobs(
        users,
        jobs,
        variant_bm25=variant_bm25,
        bm25_pool_k=bm25_pool_k,
        cosine_pool_k=cosine_pool_k,
        min_common=min_common,
        max_fallback_pool=max_fallback_pool,
        col_display_k=col_display_k,
        alpha_on_cosine=alpha_on_cosine,
        skills_weight=skills_weight,
        text_weight=text_weight,
        k1=k1,
        b=b,
        include_programme=include_programme,
        mrr_labels=mrr_labels if mrr_relevance_path is not None else None,
    )
    config = payload.config.model_copy(
        update={
            "users_path": str(users_path),
            "jobs_source": jobs_source,
            "mongo_filter_by_users": mongo_filter_by_users,
            "jobs_path": (str(jobs_path) if jobs_path else None)
            if jobs_source == "file"
            else None,
            "mrr_relevance_json": str(mrr_relevance_path)
            if mrr_relevance_path is not None
            else None,
        }
    )

    print(
        f"[bm25_cosine_hybrid] {len(users)} users × {len(jobs)} jobs "
        f"(BM25 variant={variant_bm25}, alpha_on_cosine={alpha_on_cosine})",
        file=sys.stderr,
    )

    mrr_evaluation: Optional[MrrEvaluation] = None
    if mrr_relevance_path is not None:
        keys_m = (
            "bm25_global",
            "cosine_skills_global",
            "fused_global_weighted_minmax",
            "fused_candidate_pool_weighted_minmax",
        )
        sums = {k: 0.0 for k in keys_m}
        n_labeled = 0
        for r in payload.results:
            block = r.mrr_first_relevant
            if block is None:
                continue
            n_labeled += 1
            for k in keys_m:
                rr: ReciprocalRank = getattr(block, k)
                sums[k] += rr.reciprocal_rank
        means = (
            {k: round(sums[k] / n_labeled, 6) for k in keys_m}
            if n_labeled
            else {k: 0.0 for k in keys_m}
        )
        mrr_evaluation = MrrEvaluation(
            relevance_json=str(mrr_relevance_path),
            queries_evaluated_nonempty_labels=n_labeled,
            queries_total_in_run=len(users),
            mean_reciprocal_rank=means,
            notes=(
                "MRR uses first labeled job_uuid: catalog BM25 rank, cosine-skill rank, "
                "global fused min-max hybrid, hybrid order within candidate pool."
            ),
        )
        print(
            f"[bm25_cosine_hybrid] MRR labeled_queries={n_labeled} means={means}",
            file=sys.stderr,
        )

    payload = payload.model_copy(
        update={
            "config": config,
            "mongo_timing": mongo_timing,
            "mrr_evaluation": mrr_evaluation,
        }
    )

    text = payload.model_dump_json(indent=2, exclude_none=True)
    if output_path is not None:
        out_path_fin = Path(str(output_path))
        out_path_fin.parent.mkdir(parents=True, exist_ok=True)
        out_path_fin.write_text(text, encoding="utf-8")
        print(f"[bm25_cosine_hybrid] wrote {out_path_fin}", file=sys.stderr)
    else:
        sys.stdout.write(text + "\n")

    return payload


def _alpha_on_cosine_from_env() -> Tuple[Optional[float], Optional[str]]:
    """Return (alpha, env_key_used) from first valid env variable."""

    for key in ("HYBRID_ALPHA_ON_COSINE", "ALPHA_ON_COSINE"):
        raw = os.environ.get(key)
        if raw is None or not str(raw).strip():
            continue
        try:
            return float(raw), key
        except ValueError:
            print(
                f"[bm25_cosine_hybrid] ignoring invalid {key}={raw!r} (expected float)",
                file=sys.stderr,
            )
    return None, None


def main(argv: Optional[List[str]] = None) -> int:
    load_backend_dotenv()

    p = argparse.ArgumentParser(
        description=(
            "BM25 × CosineSkillMatcher: common ∩ pool → hybrid min-max fusion (single fusion method)."
        ),
    )
    p.add_argument("--users", required=True, type=Path)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jobs", type=Path)
    src.add_argument("--from-mongo", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--variant-bm25", choices=("single", "hybrid"), default="hybrid")
    p.add_argument("--bm25-pool-k", type=int, default=120)
    p.add_argument(
        "--cosine-pool-k",
        type=int,
        default=None,
        help="Top-K jobs by mean_best_cosine (default: same as --legacy-pool-k or 120).",
    )
    p.add_argument(
        "--legacy-pool-k",
        type=int,
        default=120,
        help="Deprecated alias for --cosine-pool-k (backward compatibility).",
    )
    p.add_argument("--min-common", type=int, default=3)
    p.add_argument("--max-fallback-pool", type=int, default=200)
    p.add_argument("--col-display-k", type=int, default=15)
    ac = p.add_mutually_exclusive_group(required=False)
    ac.add_argument(
        "--alpha-on-cosine",
        type=float,
        default=None,
        dest="alpha_cos",
        help=(
            "Fusion weight on pool-normalised cosine (BM25 gets 1−this). "
            "Overrides env HYBRID_ALPHA_ON_COSINE / ALPHA_ON_COSINE; default 0.5. "
            "Lower = more BM25 (e.g. 0.2)."
        ),
    )
    ac.add_argument(
        "--alpha-on-legacy-u",
        type=float,
        default=None,
        dest="alpha_leg",
        help="Deprecated alias for --alpha-on-cosine (same semantics after switch to cosine).",
    )
    p.add_argument("--skills-weight", type=float, default=0.5)
    p.add_argument("--text-weight", type=float, default=0.5)
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--programme-context", action="store_true")
    p.add_argument("--mongo-all-active", action="store_true")
    p.add_argument("--mrr-relevance-json", type=Path, default=None)

    args = p.parse_args(argv)

    env_alpha, env_key = _alpha_on_cosine_from_env()
    alpha_c = args.alpha_cos
    if alpha_c is None and args.alpha_leg is not None:
        alpha_c = args.alpha_leg
    if alpha_c is None:
        alpha_c = env_alpha
    if alpha_c is None:
        alpha_c = 0.5
    if alpha_c < 0.0 or alpha_c > 1.0:
        print(
            f"[bm25_cosine_hybrid] alpha-on-cosine must be in [0,1], got {alpha_c}",
            file=sys.stderr,
        )
        return 2
    if (
        args.alpha_cos is None
        and args.alpha_leg is None
        and env_alpha is not None
        and env_key
    ):
        print(
            f"[bm25_cosine_hybrid] using alpha_on_cosine={alpha_c} from env {env_key}",
            file=sys.stderr,
        )

    cosm_k = args.cosine_pool_k
    if cosm_k is None:
        cosm_k = args.legacy_pool_k

    js: JobsSource = "mongo" if args.from_mongo else "file"
    jp = None if js == "mongo" else args.jobs

    if js == "file" and jp is not None and not jp.expanduser().is_file():
        print(
            "[bm25_cosine_hybrid] jobs file not found — use --from-mongo.",
            file=sys.stderr,
        )
        return 2

    try:
        run(
            users_path=args.users,
            jobs_source=js,
            jobs_path=jp,
            output_path=args.output,
            variant_bm25=args.variant_bm25,
            bm25_pool_k=args.bm25_pool_k,
            cosine_pool_k=cosm_k,
            min_common=args.min_common,
            max_fallback_pool=args.max_fallback_pool,
            col_display_k=args.col_display_k,
            alpha_on_cosine=alpha_c,
            skills_weight=args.skills_weight,
            text_weight=args.text_weight,
            k1=args.k1,
            b=args.b,
            include_programme=args.programme_context,
            mongo_filter_by_users=not args.mongo_all_active,
            mrr_relevance_path=args.mrr_relevance_json,
        )
    except FileNotFoundError as e:
        print(f"[bm25_cosine_hybrid] {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
