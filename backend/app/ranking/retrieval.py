"""Stage-1 retrieval: whitened concat-cosine shortlist per user, with the education gate and the
optional location-tier (urban-pull) weighting applied before the ``retrieve_top_k`` cutoff."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.ranking.education import job_requires_post_secondary, user_lacks_post_secondary
from app.ranking.location import HubChains
from app.ranking.skill_matcher import CosineSkillMatcher
from app.ranking.vectors import (
    is_prewhitened,
    l2_normalize_rows,
    stage1_vector,
    strip_vectors,
)
from app.ranking.whitening import ConcatWhitener

logger = logging.getLogger(__name__)


@dataclass
class Stage1Corpus:
    """The items of one corpus that carry a stage-1 vector, aligned row-for-row."""

    rows: List[Dict[str, Any]]
    """Items with a stage-1 vector (their vector fields are stripped in place)."""

    n_loaded: int
    """Items passed in, with or without a vector."""

    matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """Raw stacked vectors (float64)."""

    normalized: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    """L2-normalised ``matrix`` (float64)."""

    ids: List[str] = field(default_factory=list)
    requires_post_secondary: List[bool] = field(default_factory=list)


def _sorted_indices_desc(sim_row: np.ndarray) -> np.ndarray:
    return np.argsort(-sim_row, kind="stable")


class IStage1Retriever(ABC):
    """Interface for the stage-1 shortlist."""

    @abstractmethod
    def prepare(self, items: List[Dict[str, Any]]) -> Stage1Corpus:
        """
        Collects the items that carry a stage-1 vector and strips the vectors off those dicts.

        :param items: Jobs or occupation rows
        :return: The aligned corpus (``rows`` empty when no item has a vector)
        """
        raise NotImplementedError()

    @abstractmethod
    def shortlist(
        self,
        users: List[Dict[str, Any]],
        corpus: Stage1Corpus,
        user_unit_vectors: np.ndarray,
        *,
        matcher: CosineSkillMatcher,
        whitener: ConcatWhitener,
        retrieve_top_k: int,
        hub_chains: Optional[HubChains] = None,
        w_regional: float = 1.0,
        w_national: float = 1.0,
    ) -> List[List[Dict[str, Any]]]:
        """
        Ranks the corpus per user and keeps the top ``retrieve_top_k`` eligible rows.

        :param users: The users, aligned with ``user_unit_vectors``
        :param corpus: A non-empty corpus from ``prepare``
        :param user_unit_vectors: L2-normalised user embeddings ``[n_users, dim]``
        :param matcher: Per-skill matcher whose ``score_pair`` detail is attached to each row
        :param whitener: Concat whitening (a no-op when its artifact is unavailable)
        :param retrieve_top_k: Shortlist size per user
        :param hub_chains: Location tiers to weight the ranking by, or None to skip tiering
        :return: One shortlist (cosine-ranked rows) per user
        """
        raise NotImplementedError()


class Stage1Retriever(IStage1Retriever):
    def __init__(self, *, embedding_dim: int):
        self._embedding_dim = embedding_dim

    def prepare(self, items: List[Dict[str, Any]]) -> Stage1Corpus:
        job_rows: List[Dict[str, Any]] = []
        vectors: List[np.ndarray] = []
        for j in items:
            v = stage1_vector(j, dim=self._embedding_dim)
            if v is None:
                continue
            job_rows.append(j)
            vectors.append(v)

        for j in job_rows:
            j.pop("concat_skill_embedding_gemini", None)
            j.pop("job_embedding", None)

        # Post-secondary education gate: aligned with job_rows, used to skip candidates per user.
        job_requires_ps = [job_requires_post_secondary(j) for j in job_rows]

        corpus = Stage1Corpus(
            rows=job_rows,
            n_loaded=len(items),
            requires_post_secondary=job_requires_ps,
        )
        if not job_rows:
            return corpus

        corpus.matrix = np.stack(vectors, axis=0).astype(np.float64)
        corpus.normalized = l2_normalize_rows(corpus.matrix.astype(np.float32)).astype(
            np.float64
        )
        corpus.ids = [str(j.get("uuid") or "") for j in job_rows]
        return corpus

    def shortlist(
        self,
        users: List[Dict[str, Any]],
        corpus: Stage1Corpus,
        user_unit_vectors: np.ndarray,
        *,
        matcher: CosineSkillMatcher,
        whitener: ConcatWhitener,
        retrieve_top_k: int,
        hub_chains: Optional[HubChains] = None,
        w_regional: float = 1.0,
        w_national: float = 1.0,
    ) -> List[List[Dict[str, Any]]]:
        rt = max(1, int(retrieve_top_k))
        job_rows = corpus.rows
        j_mat = corpus.matrix
        j_norm = corpus.normalized
        jid_list = corpus.ids
        job_requires_ps = corpus.requires_post_secondary
        u_norm = user_unit_vectors

        # Whitened-space stage-1 retrieval. The concat artifact (same one the DB used to whiten
        # job_embedding) is present in practice, so we rank in the de-anisotropised whitened space (the
        # meaningful signal; raw concat cosine sd ~0.02). Jobs already whitened on the DB side are used
        # as-is; RAW vectors (occupations, offline, not-yet-whitened jobs) are whitened in-process once
        # (numerically identical to the DB result — same artifact). When the artifact is unavailable
        # (target==0) we fall back to the legacy raw cosine and log loudly (DB-whitened jobs degrade).
        _whiten_target = whitener.rescale_target()
        if _whiten_target > 0:
            j_used = j_norm.copy()
            raw_idx = [k for k, jr in enumerate(job_rows) if not is_prewhitened(jr)]
            if raw_idx:
                j_used[raw_idx] = whitener.whiten_rows(j_mat[raw_idx])
            u_used = whitener.whiten_rows(u_norm)
        else:
            if any(is_prewhitened(jr) for jr in job_rows):
                logger.error(
                    "concat whitening artifact unavailable but DB job_embedding is whitened; stage-1 "
                    "cosine will be raw-user vs whitened-job (degraded). Ship concat_whitening_gemini.npz."
                )
            j_used, u_used = j_norm, u_norm

        shortlists: List[List[Dict[str, Any]]] = []
        for i, user in enumerate(users):
            sim_row = (u_used[i : i + 1] @ j_used.T).reshape(-1)
            # Location-tier weighting of the stage-1 ranking (urban-pull). Rank by cosine * tier so local
            # jobs are favoured for the shortlist; keep the RAW cosine for the stored similarity downstream.
            loc_tier_vec = None
            if hub_chains is not None:
                county = user.get("province") or user.get("city") or ""
                loc_tier_vec = np.array(
                    [
                        hub_chains.tier_factor_for_job(
                            jr, county, w_regional=w_regional, w_national=w_national
                        )
                        for jr in job_rows
                    ],
                    dtype=float,
                )
                rank_row = sim_row * loc_tier_vec
            else:
                rank_row = sim_row
            order = _sorted_indices_desc(rank_row)
            user_no_ps = user_lacks_post_secondary(user)

            cosine_recs: List[Dict[str, Any]] = []
            for ji in order:
                if user_no_ps and job_requires_ps[int(ji)]:
                    continue  # job requires post-secondary education the user does not have
                if loc_tier_vec is not None and loc_tier_vec[int(ji)] <= 0.0:
                    continue  # off-chain location for this user: excluded at retrieval
                jid = jid_list[int(ji)]
                job_obj = job_rows[int(ji)]
                job_plain = strip_vectors(job_obj)
                concat_sim = float(sim_row[int(ji)])
                detail = matcher.score_pair(user, job_plain)
                detail = dict(detail)
                detail["concat_cosine_similarity"] = round(concat_sim, 6)
                detail["mean_best_cosine"] = round(concat_sim, 4)
                detail["min_best_cosine"] = round(concat_sim, 4)

                cosine_recs.append(
                    {
                        "rank": len(cosine_recs) + 1,
                        "job_uuid": jid,
                        "job_title": job_plain.get("opportunity_title"),
                        "employer": job_plain.get("employer"),
                        "location": job_plain.get("location"),
                        **detail,
                    }
                )
                if len(cosine_recs) >= rt:
                    break

            for r_i, row in enumerate(cosine_recs, start=1):
                row["rank"] = r_i

            shortlists.append(cosine_recs)
        return shortlists
