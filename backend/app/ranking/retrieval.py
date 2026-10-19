"""Stage-1 candidate retrieval: concat-embedding text and vectors, whitening, the post-secondary education gate, location-tier weighting, and the whitened-cosine shortlist."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from app.artifacts.types import ConcatWhitening
from app.domain.user import UserProfile
from app.location.keys import UserLocationKeys
from app.ranking.skills import CosineSkillMatcher
from app.ranking.types import Candidate, MatchItem

logger = logging.getLogger(__name__)

ItemT = TypeVar("ItemT", bound=MatchItem)

# --------------------------------------------------------------------------------------------------
# vectors
#
# Stage-1 concat embedding vectors on jobs / occupations, and row normalisation.
# --------------------------------------------------------------------------------------------------


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Rows scaled to unit L2 norm (zero rows unchanged), as float32."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms > 0, norms, 1.0).astype(np.float32)
    return (mat / norms).astype(np.float32)


def stage1_vector(item: MatchItem, *, dim: int) -> np.ndarray | None:
    """The item's concat embedding as float32, or None when absent or not ``dim`` long."""
    vec = item.embedding
    if vec is None or vec.ndim != 1 or vec.size != dim:
        return None
    return vec.astype(np.float32, copy=False)


def is_prewhitened(item: MatchItem) -> bool:
    """True iff the item's stage-1 embedding is ALREADY whitened (DB-side for jobs, once at load
    for occupations); raw vectors are whitened in-process."""
    return item.embedding_whitened


def index_by_uuid(items: Sequence[ItemT]) -> dict[str, ItemT]:
    """Items keyed by ``uuid`` (items without one are dropped; the last duplicate wins)."""
    return {it.uuid: it for it in items if it.uuid}


# --------------------------------------------------------------------------------------------------
# whitening
#
# Concat-embedding whitening ((x - mu) @ W, re-normalised) and its p99 rescale target.
# --------------------------------------------------------------------------------------------------


class ConcatWhitener:
    """Whitening for the user/job concat embedding space.

    ``artifact`` comes from ``ArtifactsRepository.load_concat_whitening``; None when the artifact is
    unavailable — then rows are only L2-normalised and the rescale target is 0.0.
    """

    def __init__(self, artifact: ConcatWhitening | None):
        self._artifact = artifact

    def whiten_rows(self, vecs: np.ndarray) -> np.ndarray:
        """L2-normalise rows, apply the concat whitening ((.-mu)@W), re-normalise -> unit whitened rows.
        If the artifact is unavailable, returns the L2-normalised rows unchanged."""
        v = l2_normalize_rows(np.asarray(vecs, dtype=np.float64))
        cw = self._artifact
        if cw is None:
            return v
        return l2_normalize_rows((v - cw.mu) @ cw.W)

    def rescale_target(self) -> float:
        """p99 rescale target for the whitened concat cosine (0.0 if the artifact is unavailable)."""
        cw = self._artifact
        return cw.target if cw is not None else 0.0


# --------------------------------------------------------------------------------------------------
# concat_embedding_text
#
# Single string per side for embedding: concatenate skill labels only.
# --------------------------------------------------------------------------------------------------

DEFAULT_SEPARATOR = " | "


def _dedupe_stable(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in labels:
        s = str(raw).strip()
        if not s:
            continue
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def build_concat_embedding_text(
    labels: Sequence[str],
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    """Join skill labels into one line for embedding models."""

    return separator.join(_dedupe_stable(list(labels)))


def user_skill_labels_for_concat(user: UserProfile) -> list[str]:
    """The user's skill labels (``skills_vector.top_skills[].preferredLabel``), de-duplicated."""
    return _dedupe_stable(
        [s.preferredLabel for s in user.skills_vector.top_skills if s.preferredLabel]
    )


def job_skill_labels_for_concat(item: MatchItem) -> list[str]:
    """The item's essential ∪ optional skill labels, de-duplicated."""
    return _dedupe_stable(
        [s.label for s in [*item.essential_skills, *item.optional_skills] if s.label]
    )


def user_concat_embedding_text(
    user: UserProfile,
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    """The user's embedding text: skill labels joined by ``separator``."""
    return build_concat_embedding_text(
        user_skill_labels_for_concat(user), separator=separator
    )


def job_concat_embedding_text(
    item: MatchItem,
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    """The item's embedding text: skill labels joined by ``separator``."""
    return build_concat_embedding_text(
        job_skill_labels_for_concat(item), separator=separator
    )


# --------------------------------------------------------------------------------------------------
# education
#
# Post-secondary education eligibility gate (shared by all matching endpoints).
#
# A job may declare ``requires_post_secondary`` (boolean) under
# ``llm_job_attributes.attributes`` in Mongo; ``build_job_from_ranked`` normalises it onto
# ``Job.requires_post_secondary`` (occupations carry the same field).
#
# The user side comes from ``MatchRequest.any_post_secondary_educ`` (0/1, optional).
#
# Gate (applied uniformly across /match, /match_v2, /match_v3, /match_v4):
# a user is **ineligible** for a job only when the job requires post-secondary
# education *and* the user explicitly reported having none.
#
# Fail-open by design — only positive evidence excludes:
# * Job has no / false ``requires_post_secondary``  -> eligible (never hide a job
#   that simply lacks the field).
# * User did not supply ``any_post_secondary_educ`` (None) -> eligible; preserves
#   behaviour for clients that omit the optional field. Only an explicit ``0`` excludes.
# --------------------------------------------------------------------------------------------------


def job_requires_post_secondary(item: MatchItem) -> bool:
    """True only when the item explicitly requires post-secondary education.

    Absent / unknown (None) is "not required", so an item is never hidden for lacking the field.
    The flag is normalised from bool / 0/1 / strings when the document is read (``parse_flag``).
    """
    return item.requires_post_secondary is True


def user_lacks_post_secondary(user: UserProfile) -> bool:
    """True only when the user explicitly reported no post-secondary education (``0``).

    ``1``, a stray value, or an omitted field (None) is not treated as lacking.
    """
    return user.any_post_secondary_educ == 0


def is_education_eligible(user: UserProfile, item: MatchItem) -> bool:
    """Education gate: ineligible iff the item requires post-secondary and the user has none."""
    return not (job_requires_post_secondary(item) and user_lacks_post_secondary(user))


def filter_jobs_by_education(user: UserProfile, items: Sequence[ItemT]) -> list[ItemT]:
    """Drop items the user is education-ineligible for.

    No-op (returns a shallow copy) when the user has — or did not report lacking —
    post-secondary education, so the common path stays cheap.
    """
    if not user_lacks_post_secondary(user):
        return list(items)
    return [j for j in items if not job_requires_post_secondary(j)]


def zqf_annotation(
    user_zqf: int | None, job_zqf_min: int | None
) -> tuple[bool | None, int | None]:
    """(zqf_eligible, zqf_gap) or (None, None) when either side is missing."""
    if user_zqf is not None and job_zqf_min is not None:
        return (user_zqf >= job_zqf_min, abs(user_zqf - job_zqf_min))
    return (None, None)


# --------------------------------------------------------------------------------------------------
# retrieval
#
# Stage-1 retrieval: whitened concat-cosine shortlist per user, with the education gate and the
# optional location-tier (urban-pull) weighting applied before the ``retrieve_top_k`` cutoff.
# --------------------------------------------------------------------------------------------------


class Stage1Corpus(BaseModel):
    """The items of one corpus that carry a stage-1 vector, aligned row-for-row."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rows: SkipValidation[list[MatchItem]] = Field(
        description="Items with a stage-1 vector, in input order (the caller's own objects)."
    )
    n_loaded: int = Field(description="Items passed in, with or without a vector.")
    matrix: np.ndarray = Field(
        default_factory=lambda: np.zeros((0, 0)),
        description="Raw stacked vectors (float64).",
    )
    normalized: np.ndarray = Field(
        default_factory=lambda: np.zeros((0, 0)),
        description="L2-normalised ``matrix`` (float64).",
    )
    ids: list[str] = Field(
        default_factory=list, description="Item uuids, aligned with ``rows``."
    )
    requires_post_secondary: list[bool] = Field(
        default_factory=list, description="Education-gate flags, aligned with ``rows``."
    )


def _sorted_indices_desc(sim_row: np.ndarray) -> np.ndarray:
    return np.argsort(-sim_row, kind="stable")


class IStage1Retriever(ABC):
    """Interface for the stage-1 shortlist."""

    @abstractmethod
    def prepare(self, items: Sequence[MatchItem]) -> Stage1Corpus:
        """
        Collects the items that carry a stage-1 vector.

        :param items: Jobs or occupation rows
        :return: The aligned corpus (``rows`` empty when no item has a vector)
        """
        raise NotImplementedError()

    @abstractmethod
    def shortlist(
        self,
        users: Sequence[UserProfile],
        corpus: Stage1Corpus,
        user_unit_vectors: np.ndarray,
        *,
        matcher: CosineSkillMatcher,
        whitener: ConcatWhitener,
        retrieve_top_k: int,
        location_keys: Sequence[UserLocationKeys] | None = None,
    ) -> list[list[Candidate]]:
        """
        Ranks the corpus per user and keeps the top ``retrieve_top_k`` eligible rows.

        :param users: The users, aligned with ``user_unit_vectors``
        :param corpus: A non-empty corpus from ``prepare``
        :param user_unit_vectors: L2-normalised user embeddings ``[n_users, dim]``
        :param matcher: Per-skill matcher whose ``score_pair`` detail is attached to each candidate
        :param whitener: Concat whitening (a no-op when its artifact is unavailable)
        :param retrieve_top_k: Shortlist size per user
        :param location_keys: Per-user eligible location keys and tiers (aligned with ``users``) to
            weight the ranking by and drop ineligible items with, or None to skip location tiering
        :return: One cosine-ranked shortlist per user
        """
        raise NotImplementedError()


class Stage1Retriever(IStage1Retriever):
    """Whitened concat-cosine shortlist with the education gate and optional location tiers."""

    def __init__(self, *, embedding_dim: int):
        self._embedding_dim = embedding_dim

    def prepare(self, items: Sequence[MatchItem]) -> Stage1Corpus:
        rows: list[MatchItem] = []
        vectors: list[np.ndarray] = []
        for item in items:
            v = stage1_vector(item, dim=self._embedding_dim)
            if v is None:
                continue
            rows.append(item)
            vectors.append(v)

        corpus = Stage1Corpus(
            rows=rows,
            n_loaded=len(items),
            # Post-secondary education gate: aligned with rows, used to skip candidates per user.
            requires_post_secondary=[job_requires_post_secondary(j) for j in rows],
        )
        if not rows:
            return corpus

        corpus.matrix = np.stack(vectors, axis=0).astype(np.float64)
        corpus.normalized = l2_normalize_rows(corpus.matrix.astype(np.float32)).astype(
            np.float64
        )
        corpus.ids = [j.uuid for j in rows]
        return corpus

    def shortlist(
        self,
        users: Sequence[UserProfile],
        corpus: Stage1Corpus,
        user_unit_vectors: np.ndarray,
        *,
        matcher: CosineSkillMatcher,
        whitener: ConcatWhitener,
        retrieve_top_k: int,
        location_keys: Sequence[UserLocationKeys] | None = None,
    ) -> list[list[Candidate]]:
        rt = max(1, int(retrieve_top_k))
        rows = corpus.rows
        j_mat = corpus.matrix
        j_norm = corpus.normalized
        job_requires_ps = corpus.requires_post_secondary
        u_norm = user_unit_vectors

        # Whitened-space stage-1 retrieval. The concat artifact (same one the DB used to whiten
        # job_embedding) is present in practice, so we rank in the de-anisotropised whitened space (the
        # meaningful signal; raw concat cosine sd ~0.02). Jobs already whitened on the DB side are used
        # as-is; RAW vectors (occupations, offline, not-yet-whitened jobs) are whitened in-process once
        # (numerically identical to the DB result — same artifact). When the artifact is unavailable
        # (target==0) we fall back to the legacy raw cosine and log loudly (DB-whitened jobs degrade).
        if whitener.rescale_target() > 0:
            j_used = j_norm.copy()
            raw_idx = [k for k, jr in enumerate(rows) if not is_prewhitened(jr)]
            if raw_idx:
                j_used[raw_idx] = whitener.whiten_rows(j_mat[raw_idx])
            u_used = whitener.whiten_rows(u_norm)
        else:
            if any(is_prewhitened(jr) for jr in rows):
                logger.error(
                    "concat whitening artifact unavailable but DB job_embedding is whitened; stage-1 "
                    "cosine will be raw-user vs whitened-job (degraded). Ship concat_whitening_gemini.npz."
                )
            j_used, u_used = j_norm, u_norm

        shortlists: list[list[Candidate]] = []
        for i, user in enumerate(users):
            sim_row = (u_used[i : i + 1] @ j_used.T).reshape(-1)
            # Location-tier weighting of the stage-1 ranking (urban-pull). Rank by cosine * tier so local
            # jobs are favoured for the shortlist; keep the RAW cosine for the stored similarity downstream.
            # The pool holds every batch user's jobs, so items outside this user's keys (tier 0) are dropped.
            loc_tier_vec = None
            if location_keys is not None:
                user_keys = location_keys[i]
                loc_tier_vec = np.array(
                    [user_keys.tier_for(jr.location_keys) for jr in rows], dtype=float
                )
                rank_row = sim_row * loc_tier_vec
            else:
                rank_row = sim_row
            order = _sorted_indices_desc(rank_row)
            user_no_ps = user_lacks_post_secondary(user)

            candidates: list[Candidate] = []
            for ji in order:
                ji = int(ji)
                if user_no_ps and job_requires_ps[ji]:
                    continue  # job requires post-secondary education the user does not have
                if loc_tier_vec is not None and loc_tier_vec[ji] <= 0.0:
                    continue  # off-chain location for this user: excluded at retrieval
                item = rows[ji]
                candidates.append(
                    Candidate(
                        item=item,
                        rank=len(candidates) + 1,
                        concat_cosine_similarity=round(float(sim_row[ji]), 6),
                        per_job_skill=matcher.score_pair(user, item).per_job_skill,
                    )
                )
                if len(candidates) >= rt:
                    break
            shortlists.append(candidates)
        return shortlists
