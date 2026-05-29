"""BM25 × embedding **cosine skill similarity** ∩ common pool → **hybrid** fusion JSON/HTML.

Pipeline
========

1. **BM25** — :mod:`bm25_scoring.bm25library` / ``rank_bm25.BM25Okapi`` (PyPI ``rank-bm25``) on phrase tokens.
2. **Cosine skills** — legacy :class:`~app.services.cosine_similarity.skill_score.CosineSkillMatcher`
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

Requires ``rank-bm25``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np

from app.config import EMBEDDING_MODEL_PATH, SKILLS_CSV_PATH, SKILL_TO_ROW_PATH
from app.services.bm25_scoring.bm25library import (
    avg_doc_length,
    build_corpora,
    build_okapi_indexes,
    combine,
    corpus_vocab_size,
    scores_all_okapi,
)
from app.services.bm25_scoring.text_builders import (
    matched_skill_overlap,
    matched_skill_phrases,
    user_query_tokens,
)
from app.services.cosine_similarity.run_cosine_matching import (
    load_jobs as load_jobs_cosine,
    _load_users,
)
from app.services.cosine_similarity.skill_score import CosineSkillMatcher, compact_cosine_matched_skill_lines
from app.services.education_eligibility import (
    job_requires_post_secondary,
    user_lacks_post_secondary,
)


def _load_mrr_relevance(path: Path) -> Dict[str, Set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object user_id → [uuid,...]")
    out: Dict[str, Set[str]] = {}
    for k, v in data.items():
        uid = str(k)
        if v is None:
            out[uid] = set()
        elif isinstance(v, list):
            out[uid] = {str(x) for x in v if x is not None and str(x)}
        else:
            raise ValueError(f"{path}: value for {uid!r} must be a list of job UUIDs")
    return out


def _rr_first_hit(
    ordered_keys: Sequence[str],
    relevant: Set[str],
) -> Tuple[float, Optional[int]]:
    if not relevant:
        return 0.0, None
    for pos, k in enumerate(ordered_keys, start=1):
        if k in relevant:
            return 1.0 / float(pos), pos
    return 0.0, None


def _job_key(job: Dict[str, Any]) -> str:
    return str(job.get("uuid") or job.get("_id") or "")


def _sorted_job_keys_from_argsort(jobs: List[dict], order: np.ndarray) -> List[str]:
    return [_job_key(jobs[int(i)]) for i in order.tolist()]


def _sidebar_skill_labels(user: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for s in user.get("resolved_skills") or []:
        if isinstance(s, dict) and s.get("label"):
            out.append(str(s["label"]))
    if out:
        return out
    for s in (user.get("skills_vector") or {}).get("top_skills") or []:
        if isinstance(s, dict):
            lab = s.get("preferredLabel") or s.get("label")
            if lab:
                out.append(str(lab))
    return out


def _min_max_1d(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def bm25_vectors(
    user: dict,
    jobs: List[dict],
    skills_okapi: Any,
    full_okapi: Any,
    *,
    variant_bm25: Literal["single", "hybrid"],
    skills_weight: float,
    text_weight: float,
    include_programme: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, List[str]]:
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
    user_raw: dict,
    jobs: Sequence[dict],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """``mean_best_cosine`` per job + compact cosine detail dicts aligned with ``jobs``."""

    n = len(jobs)
    vals = np.zeros(n, dtype=np.float64)
    details: List[Dict[str, Any]] = []
    for i, job in enumerate(jobs):
        sp = matcher.score_pair(user_raw, job)
        vals[i] = float(sp.get("mean_best_cosine") or 0.0)
        details.append({
            "mean_best_cosine": round(float(vals[i]), 4),
            "min_best_cosine": sp.get("min_best_cosine"),
            "n_user_skills_embedded": sp.get("n_user_skills_embedded"),
            "n_job_skills_embedded": sp.get("n_job_skills_embedded"),
        })
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


def _bm25_rec_fields(
    ji: int,
    combined: np.ndarray,
    skills_v: Optional[np.ndarray],
    text_v: np.ndarray,
    variant_bm25: Literal["single", "hybrid"],
    job: dict,
    user_raw: dict,
    *,
    rank: int,
) -> Dict[str, Any]:
    r: Dict[str, Any] = {
        "rank": rank,
        "job_uuid": job.get("uuid") or job.get("_id"),
        "job_title": job.get("opportunity_title"),
        "employer": job.get("employer"),
        "location": job.get("location"),
        "bm25_score": round(float(combined[ji]), 4),
    }
    if variant_bm25 == "hybrid" and skills_v is not None:
        r["bm25_skills_score_raw"] = round(float(skills_v[ji]), 4)
        r["bm25_text_score_raw"] = round(float(text_v[ji]), 4)
    else:
        r["bm25_score_raw"] = round(float(text_v[ji]), 4)
    r["matched_skills"] = matched_skill_phrases(user_raw, job)[:40]
    r["matched_skills_detail"] = (matched_skill_overlap(user_raw, job))[:40]
    return r


def _common_pool_row(
    job: dict,
    user_raw: dict,
    *,
    bm25_rank_pool: int,
    cosine_rank_pool: int,
    bm25_score: float,
    mean_cos: float,
    cosine_detail: Dict[str, Any],
    in_pure_intersection: bool,
    variant_bm25: Literal["single", "hybrid"],
    combined: np.ndarray,
    skills_v: Optional[np.ndarray],
    text_v: np.ndarray,
    ji: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "job_uuid": job.get("uuid") or job.get("_id"),
        "job_title": job.get("opportunity_title"),
        "employer": job.get("employer"),
        "location": job.get("location"),
        "rank_within_bm25_pool": bm25_rank_pool,
        "rank_within_legacy_pool": cosine_rank_pool,
        "rank_within_cosine_pool": cosine_rank_pool,
        "mean_best_cosine": round(float(mean_cos), 4),
        "legacy_U_final": round(float(mean_cos), 4),
        "cosine_detail": cosine_detail,
        "matched_skills": matched_skill_phrases(user_raw, job)[:30],
        "in_bm25_legacy_pure_intersection": in_pure_intersection,
        "in_pure_intersection": in_pure_intersection,
    }
    row["bm25_score"] = round(float(bm25_score), 4)
    if variant_bm25 == "hybrid" and skills_v is not None:
        row["bm25_skills_score_raw"] = round(float(skills_v[ji]), 4)
        row["bm25_text_score_raw"] = round(float(text_v[ji]), 4)
    else:
        row["bm25_score_raw"] = round(float(text_v[ji]), 4)
    return row


def one_user_bundle(
    user_raw: dict,
    jobs: List[dict],
    skills_okapi: Any,
    full_okapi: Any,
    matcher: CosineSkillMatcher,
    *,
    variant_bm25: Literal["single", "hybrid"],
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
) -> Dict[str, Any]:
    combined, skills_v, text_v, q_tokens = bm25_vectors(
        user_raw,
        jobs,
        skills_okapi,
        full_okapi,
        variant_bm25=variant_bm25,
        skills_weight=skills_weight,
        text_weight=text_weight,
        include_programme=include_programme,
    )

    cosine_v, cosine_details = cosine_scores_for_jobs(matcher, user_raw, jobs)

    n = len(jobs)
    pool_b = max(1, min(bm25_pool_k, n))
    pool_c = max(1, min(cosine_pool_k, n))

    if text_v is None:
        bm_ord = np.argsort(-combined)
    else:
        bm_ord = np.argsort(-combined - 1e-9 * text_v)
    cos_ord = np.argsort(-cosine_v)

    bm_top_idx = bm_ord[:pool_b].tolist()
    cos_top_idx = cos_ord[:pool_c].tolist()

    bm_keys = {_job_key(jobs[i]): pos for pos, i in enumerate(bm_top_idx, start=1)}
    cos_keys = {_job_key(jobs[i]): pos for pos, i in enumerate(cos_top_idx, start=1)}

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
        for i in cos_top_idx:
            k = _job_key(jobs[i])
            if k and k not in seen:
                seen.add(k)
                merged.append(k)
        for i in bm_top_idx:
            k = _job_key(jobs[i])
            if k and k not in seen:
                seen.add(k)
                merged.append(k)
        chosen_keys = set(merged[:max_fallback_pool])

    key_to_ji = {_job_key(j): i for i, j in enumerate(jobs)}
    cand_ji = [key_to_ji[k] for k in chosen_keys if k in key_to_ji]
    cand_arr = np.array(cand_ji, dtype=np.int64)

    fused_preview: List[Dict[str, Any]] = []
    fus: Optional[np.ndarray] = None
    if cand_arr.size > 0:
        b_sel = combined[cand_arr]
        cos_sel = cosine_v[cand_arr]
        b_n = _min_max_1d(b_sel)
        cos_n = _min_max_1d(cos_sel)
        fus = (
            float(alpha_on_cosine) * cos_n
            + (1.0 - float(alpha_on_cosine)) * b_n
        ).astype(np.float64, copy=False)
        order_f = np.argsort(-fus)

        for fr, ix in enumerate(order_f.flat[: col_display_k], start=1):
            cand_slot = int(ix)
            ji = int(cand_arr[cand_slot])
            job = jobs[ji]
            k = _job_key(job)
            det = cosine_details[ji]
            emb_sp = matcher.score_pair(user_raw, job)
            matched_cos_lines = compact_cosine_matched_skill_lines(
                emb_sp.get("per_job_skill") or [],
                limit=40,
            )
            fused_preview.append({
                "rank": fr,
                "fusion_method": "hybrid_pool_minmax",
                "fusion_score": round(float(fus[cand_slot]), 6),
                "weighted_minmax_fusion": round(float(fus[cand_slot]), 6),
                "legacy_U_norm_within_candidates": round(float(cos_n[cand_slot]), 6),
                "cos_norm_within_candidates": round(float(cos_n[cand_slot]), 6),
                "bm25_norm_within_candidates": round(float(b_n[cand_slot]), 6),
                "mean_best_cosine_raw": round(float(cosine_v[ji]), 4),
                "legacy_U_final_raw": round(float(cosine_v[ji]), 4),
                "bm25_score_raw": round(float(combined[ji]), 4),
                "rank_bm25_global": int(bm25_rank_full[ji]),
                "rank_legacy_global": int(cosine_rank_full[ji]),
                "rank_cosine_global": int(cosine_rank_full[ji]),
                "was_in_bm25_legacy_pure_intersection": k in intersect_keys,
                "was_in_bm25_cosine_pure_intersection": k in intersect_keys,
                "legacy_components": {},
                "cosine_detail": det,
                "job_uuid": job.get("uuid") or job.get("_id"),
                "job_title": job.get("opportunity_title"),
                "employer": job.get("employer"),
                "location": job.get("location"),
                "matched_skills": matched_skill_phrases(user_raw, job)[:12],
                "matched_skills_cosine": matched_cos_lines,
            })

    common_rows_sorted: List[Dict[str, Any]] = []
    for ji in cand_ji:
        k = _job_key(jobs[ji])
        in_pure = k in intersect_keys
        common_rows_sorted.append(
            _common_pool_row(
                jobs[ji],
                user_raw,
                bm25_rank_pool=int(bm_keys.get(k, 999999)),
                cosine_rank_pool=int(cos_keys.get(k, 999999)),
                bm25_score=float(combined[ji]),
                mean_cos=float(cosine_v[ji]),
                cosine_detail=cosine_details[ji],
                in_pure_intersection=in_pure,
                variant_bm25=variant_bm25,
                combined=combined,
                skills_v=skills_v,
                text_v=text_v,
                ji=ji,
            )
        )

    common_rows_sorted.sort(
        key=lambda r: (
            r["rank_within_bm25_pool"] + r["rank_within_legacy_pool"],
            str(r["job_uuid"]),
        )
    )

    col_bm25: List[Dict[str, Any]] = []
    for pos, ji in enumerate(bm_top_idx[:col_display_k], start=1):
        col_bm25.append(
            _bm25_rec_fields(
                ji, combined, skills_v, text_v, variant_bm25, jobs[ji], user_raw, rank=pos
            )
        )

    col_cos: List[Dict[str, Any]] = []
    for pos, ji in enumerate(cos_top_idx[:col_display_k], start=1):
        j = jobs[ji]
        det = cosine_details[ji]
        col_cos.append({
            "rank": pos,
            "job_uuid": j.get("uuid") or j.get("_id"),
            "job_title": j.get("opportunity_title"),
            "employer": j.get("employer"),
            "location": j.get("location"),
            "mean_best_cosine": det.get("mean_best_cosine"),
            "legacy_U_final": det.get("mean_best_cosine"),
            "min_best_cosine": det.get("min_best_cosine"),
            "legacy_components": {},
            "rank_bm25_global": int(bm25_rank_full[ji]),
            "rank_cosine_global": int(cosine_rank_full[ji]),
            "rank_legacy_global": int(cosine_rank_full[ji]),
        })

    labels = matcher.resolved_user_skill_labels_ordered(user_raw)[:200]
    if not labels:
        labels = _sidebar_skill_labels(user_raw)[:200]

    common_cap = min(max(col_display_k * 8, col_display_k), 200)

    n_resolved_cosine = len(matcher.resolved_user_skill_labels_ordered(user_raw))

    mrr_first_relevant: Optional[Dict[str, Any]] = None
    if relevance_ids is not None and len(relevance_ids) > 0:
        keys_bm25_g = _sorted_job_keys_from_argsort(jobs, bm_ord)
        keys_cos_g = _sorted_job_keys_from_argsort(jobs, cos_ord)

        bm25_g_rr, bm25_g_rnk = _rr_first_hit(keys_bm25_g, relevance_ids)
        cos_g_rr, cos_g_rnk = _rr_first_hit(keys_cos_g, relevance_ids)

        b_mm = _min_max_1d(combined)
        c_mm = _min_max_1d(cosine_v)
        fus_gl = float(alpha_on_cosine) * c_mm + (1.0 - float(alpha_on_cosine)) * b_mm
        fus_g_ord = np.argsort(-fus_gl - 1e-9 * combined)
        keys_fus_g = _sorted_job_keys_from_argsort(jobs, fus_g_ord)
        fus_g_rr, fus_g_rnk = _rr_first_hit(keys_fus_g, relevance_ids)

        cand_rr, cand_rnk = 0.0, None
        if cand_arr.size > 0 and fus is not None:
            cand_f_ord = np.argsort(-fus)
            keys_fus_c = [
                _job_key(jobs[int(cand_arr[int(ix)])])
                for ix in cand_f_ord.flatten().tolist()
            ]
            cand_rr, cand_rnk = _rr_first_hit(keys_fus_c, relevance_ids)

        mrr_first_relevant = {
            "n_relevant_in_labels": len(relevance_ids),
            "bm25_global": {"reciprocal_rank": bm25_g_rr, "first_hit_rank": bm25_g_rnk},
            "cosine_skills_global": {"reciprocal_rank": cos_g_rr, "first_hit_rank": cos_g_rnk},
            "legacy_u_global": {"reciprocal_rank": cos_g_rr, "first_hit_rank": cos_g_rnk},
            "fused_global_weighted_minmax": {
                "reciprocal_rank": fus_g_rr,
                "first_hit_rank": fus_g_rnk,
            },
            "fused_candidate_pool_weighted_minmax": {
                "reciprocal_rank": cand_rr,
                "first_hit_rank": cand_rnk,
            },
        }

    out: Dict[str, Any] = {
        "user_id": user_raw.get("user_id"),
        "city": user_raw.get("city"),
        "province": user_raw.get("province"),
        "n_bm25_query_tokens": len(q_tokens),
        "query_tokens_bm25": q_tokens[:200],
        "n_user_skill_labels_display": len(labels),
        "n_resolved_user_skills_cosine": n_resolved_cosine,
        "user_skill_labels_for_display": labels,
        "resolved_user_skill_labels": labels,
        "bm25_pool_k_used": pool_b,
        "cosine_pool_k_used": pool_c,
        "legacy_pool_k_used": pool_c,
        "n_bm25_legacy_intersection_pure": len(intersect_keys),
        "n_bm25_cosine_intersection_pure": len(intersect_keys),
        "n_candidates_union_or_intersection": len(cand_ji),
        "fallback_union_used": fallback_union,
        "column_common_slice_max_applied": min(common_cap, len(common_rows_sorted)),
        "column_bm25": col_bm25,
        "column_cosine_skills": col_cos,
        "column_legacy_u": col_cos,
        "column_common": common_rows_sorted[:common_cap],
        "column_fused_weighted_minmax": fused_preview,
    }
    if mrr_first_relevant is not None:
        out["mrr_first_relevant"] = mrr_first_relevant
    return out


_cosine_matcher_singleton: Optional[CosineSkillMatcher] = None


def get_cosine_matcher_singleton() -> CosineSkillMatcher:
    """Reuse one embedding loader across CLI batches and ``POST /match_v2`` calls."""

    global _cosine_matcher_singleton
    if _cosine_matcher_singleton is None:
        _cosine_matcher_singleton = CosineSkillMatcher()
    return _cosine_matcher_singleton


def hybrid_match_users_with_jobs(
    users: List[Dict[str, Any]],
    jobs: List[Dict[str, Any]],
    *,
    variant_bm25: Literal["single", "hybrid"] = "hybrid",
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
    mrr_labels: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Any]:
    """Run BM25×cosine hybrid over in-memory ``users`` and ``jobs``.

    Same scoring as CLI :func:`run` but without file paths or Mongo fetches.
    Used by ``POST /match_v2`` and internally by :func:`run`.

    Preconditions: ``rank-bm25`` installed; ``jobs`` non-empty (needed to build indexes).
    """

    from app.services.bm25_scoring.bm25library import _require_rank_bm25

    _require_rank_bm25()
    if not jobs:
        raise ValueError("hybrid_match_users_with_jobs requires at least one job document")

    skills_corpus, full_corpus = build_corpora(jobs)
    skills_okapi, full_okapi = build_okapi_indexes(
        skills_corpus, full_corpus, k1=k1, b=b
    )
    matcher = get_cosine_matcher_singleton()

    cfg_skill = "CosineSkillMatcher.mean_best_cosine (embedding cosine on skills)"
    # Post-secondary education gate: indexes are built over all jobs (shared across users),
    # so we drop ineligible jobs from each user's output columns rather than pre-filtering jobs.
    ps_required_uuids = {
        str(j.get("uuid") or "") for j in jobs if job_requires_post_secondary(j)
    }
    _job_row_columns = (
        "column_bm25",
        "column_cosine_skills",
        "column_legacy_u",
        "column_common",
        "column_fused_weighted_minmax",
    )
    results: List[Dict[str, Any]] = []
    for u in users:
        uid_k = str(u.get("user_id") or "")
        rel = mrr_labels.get(uid_k) if mrr_labels is not None else None
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
            for col_key in _job_row_columns:
                col = bundle.get(col_key)
                if isinstance(col, list):
                    bundle[col_key] = [
                        r
                        for r in col
                        if str(r.get("job_uuid") or r.get("uuid") or "")
                        not in ps_required_uuids
                    ]
        results.append(bundle)

    config: Dict[str, Any] = {
        "entrypoint": "hybrid_match_users_with_jobs",
        "scorer": "bm25_cosine_skills_four_col_hybrid_pool_minmax",
        "legacy_skill_model": cfg_skill,
        "fusion": "weighted_minmax_on_candidate_pool_bm25_cosine_only",
        "alpha_on_cosine_skill": alpha_on_cosine,
        "(1-alpha)_on_bm25": round(1.0 - alpha_on_cosine, 4),
        "variant_bm25": variant_bm25,
        "bm25_pool_k": bm25_pool_k,
        "cosine_pool_k": cosine_pool_k,
        "legacy_pool_k": cosine_pool_k,
        "min_common_intersection": min_common,
        "max_fallback_pool": max_fallback_pool,
        "column_display_rows": col_display_k,
        "column_common_max_rows_saved": min(max(col_display_k * 8, col_display_k), 200),
        "k1": k1,
        "b": b,
        "include_programme_context": include_programme,
        "embedding_model_path": str(EMBEDDING_MODEL_PATH),
        "skill_to_row_path": str(SKILL_TO_ROW_PATH),
        "skills_csv_path": str(SKILLS_CSV_PATH),
        "notes": (
            "CosineSkillMatcher pooled embedding cosine for skills; BM25 hybrid index; "
            "common ∩ pool; fused = α·norm(cosine)+(1−α)·norm(BM25) within pool only."
        ),
    }
    if variant_bm25 == "hybrid":
        config["skills_weight"] = skills_weight
        config["text_weight"] = text_weight

    embedding_dim = int(matcher.W.shape[1])
    payload: Dict[str, Any] = {
        "config": config,
        "index_stats": {
            "n_jobs": len(jobs),
            "skills_vocab_size": corpus_vocab_size(skills_corpus),
            "full_vocab_size": corpus_vocab_size(full_corpus),
            "skills_avg_doc_len": round(avg_doc_length(skills_corpus), 2),
            "full_avg_doc_len": round(avg_doc_length(full_corpus), 2),
            "embedding_dim": embedding_dim,
        },
        "n_users": len(users),
        "n_jobs": len(jobs),
        "results": results,
        "skill_cosine_resolution_stats_end_of_run": matcher.get_resolution_stats(),
    }
    return payload


def run(
    users_path: Path,
    jobs_source: Literal["file", "mongo"],
    jobs_path: Optional[Path],
    output_path: Optional[Path],
    *,
    variant_bm25: Literal["single", "hybrid"] = "hybrid",
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
) -> Dict[str, Any]:
    from app.services.bm25_scoring.bm25library import _require_rank_bm25

    _require_rank_bm25()

    mrr_labels: Dict[str, Set[str]] = {}
    if mrr_relevance_path is not None:
        mp = Path(mrr_relevance_path).expanduser()
        if not mp.is_file():
            raise FileNotFoundError(
                f"MRR relevance file not found: {mp} — omit --mrr-relevance-json to skip "
                "MRR, or pass an existing JSON path (mapping user_id → [job_uuid, …])."
            )
        mrr_labels = _load_mrr_relevance(mp)

    users = _load_users(users_path)
    jobs, mongo_timing = load_jobs_cosine(
        jobs_source,
        jobs_path,
        users,
        mongo_filter_by_users,
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

    cfg_skill = "CosineSkillMatcher.mean_best_cosine (embedding cosine on skills)"
    payload["config"].update({
        "users_path": str(users_path),
        "jobs_source": jobs_source,
        "mongo_filter_by_users": mongo_filter_by_users,
        "legacy_skill_model": cfg_skill,
    })
    if jobs_source == "file":
        payload["config"]["jobs_path"] = str(jobs_path) if jobs_path else None
    if mrr_relevance_path is not None:
        payload["config"]["mrr_relevance_json"] = str(mrr_relevance_path)

    results = payload["results"]

    print(
        f"[bm25_cosine_hybrid] {len(users)} users × {len(jobs)} jobs "
        f"(BM25 variant={variant_bm25}, alpha_on_cosine={alpha_on_cosine})",
        file=sys.stderr,
    )

    if mongo_timing is not None:
        payload["mongo_timing"] = mongo_timing

    if mrr_relevance_path is not None:
        keys_m = (
            "bm25_global",
            "cosine_skills_global",
            "fused_global_weighted_minmax",
            "fused_candidate_pool_weighted_minmax",
        )
        sums = {k: 0.0 for k in keys_m}
        n_labeled = 0
        for r in results:
            block = r.get("mrr_first_relevant")
            if not block:
                continue
            n_labeled += 1
            for k in keys_m:
                sums[k] += float(block[k]["reciprocal_rank"])
        means = (
            {k: round(sums[k] / n_labeled, 6) for k in keys_m}
            if n_labeled
            else {k: 0.0 for k in keys_m}
        )
        payload["mrr_evaluation"] = {
            "relevance_json": str(mrr_relevance_path),
            "queries_evaluated_nonempty_labels": n_labeled,
            "queries_total_in_run": len(users),
            "mean_reciprocal_rank": means,
            "notes": (
                "MRR uses first labeled job_uuid: catalog BM25 rank, cosine-skill rank, "
                "global fused min-max hybrid, hybrid order within candidate pool."
            ),
        }
        print(
            f"[bm25_cosine_hybrid] MRR labeled_queries={n_labeled} means={means}",
            file=sys.stderr,
        )

    out_path_fin = Path(str(output_path)) if output_path is not None else None
    if out_path_fin is not None:
        out_path_fin.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path_fin, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[bm25_cosine_hybrid] wrote {out_path_fin}", file=sys.stderr)
    else:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")

    return payload


def _backend_root() -> Path:
    here = Path(__file__).resolve()
    # hybrid_scoring -> services -> app -> backend
    return here.parents[3]


def _load_backend_dotenv() -> None:
    """Populate os.environ from ``backend/.env`` so HYBRID_ALPHA_ON_COSINE etc. apply."""

    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        return
    p = _backend_root() / ".env"
    if p.is_file():
        load_dotenv(p, override=False)


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
    _load_backend_dotenv()

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
    if args.alpha_cos is None and args.alpha_leg is None and env_alpha is not None and env_key:
        print(
            f"[bm25_cosine_hybrid] using alpha_on_cosine={alpha_c} from env {env_key}",
            file=sys.stderr,
        )

    cosm_k = args.cosine_pool_k
    if cosm_k is None:
        cosm_k = args.legacy_pool_k

    js: Literal["file", "mongo"] = "mongo" if args.from_mongo else "file"
    jp = None if js == "mongo" else args.jobs

    if js == "file" and jp is not None and not jp.expanduser().is_file():
        print("[bm25_cosine_hybrid] jobs file not found — use --from-mongo.", file=sys.stderr)
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
