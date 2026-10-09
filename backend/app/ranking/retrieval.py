"""Stage-1 candidate retrieval: concat-embedding text and vectors, whitening, the post-secondary education gate, location tiers, and the whitened-cosine shortlist."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from app.ranking.skills import CosineSkillMatcher

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# vectors
#
# Stage-1 concat embedding vectors on job / occupation dicts, and row normalisation.
# --------------------------------------------------------------------------------------------------


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms > 0, norms, 1.0).astype(np.float32)
    return (mat / norms).astype(np.float32)


def stage1_vector(job: dict[str, Any], *, dim: int) -> np.ndarray | None:
    """Prefer NPZ-sync BSON; fall back to ``job_embedding`` float list on the job doc."""

    sub = job.get("concat_skill_embedding_gemini")
    if isinstance(sub, dict):
        vb = sub.get("vector_bin")
        if vb is not None:
            raw = getattr(vb, "bytes", None) or bytes(vb)
            arr = np.frombuffer(raw, dtype=np.float32)
            if arr.size == dim:
                return arr

    je = job.get("job_embedding")
    # Accept a float list (Mongo job docs) or a numpy array (occupation embeddings attached
    # in-process by OccupationsRepository.attach_embeddings).
    if isinstance(je, np.ndarray):
        if je.ndim == 1 and je.size == dim:
            return je.astype(np.float32, copy=False)
    elif isinstance(je, list) and je:
        arr = np.asarray(je, dtype=np.float32)
        if arr.ndim == 1 and arr.size == dim:
            return arr
    return None


def is_prewhitened(job: dict[str, Any]) -> bool:
    """True iff this job's stage-1 embedding is ALREADY whitened on the DB side (set by
    build_job_dict_from_ranked from llm_reranker_meta.embedding.whitening.enabled). Occupations and
    offline jobs lack the flag -> raw (whitened in-process)."""
    return bool(job.get("job_embedding_whitened"))


def strip_vectors(job: dict[str, Any]) -> dict[str, Any]:
    out = dict(job)
    out.pop("concat_skill_embedding_gemini", None)
    out.pop("job_embedding", None)
    return out


def index_by_uuid(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for it in items:
        uid = str(it.get("uuid") or it.get("_id") or "")
        if uid:
            out[uid] = it
    return out


# --------------------------------------------------------------------------------------------------
# whitening
#
# Concat-embedding whitening ((x - mu) @ W, re-normalised) and its p99 rescale target.
# --------------------------------------------------------------------------------------------------


class ConcatWhitener:
    """Whitening for the user/job concat embedding space.

    ``artifact`` is ``{mu, W, target}`` (see ``ArtifactsRepository.load_concat_whitening``), or
    empty/None when the artifact is unavailable — then rows are only L2-normalised and the rescale
    target is 0.0.
    """

    def __init__(self, artifact: dict[str, Any] | None):
        self._artifact = artifact or None

    def whiten_rows(self, vecs: np.ndarray) -> np.ndarray:
        """L2-normalise rows, apply the concat whitening ((.-mu)@W), re-normalise -> unit whitened rows.
        If the artifact is unavailable, returns the L2-normalised rows unchanged."""
        v = l2_normalize_rows(np.asarray(vecs, dtype=np.float64))
        cw = self._artifact
        if cw is None:
            return v
        return l2_normalize_rows((v - cw["mu"]) @ cw["W"])

    def rescale_target(self) -> float:
        """p99 rescale target for the whitened concat cosine (0.0 if the artifact is unavailable)."""
        cw = self._artifact
        return cw["target"] if cw else 0.0


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


def user_skill_labels_for_concat(user: dict[str, Any]) -> list[str]:
    """Collect user-side skill label strings from ``resolved_skills`` or ``skills_vector.top_skills``."""

    out: list[str] = []
    for s in user.get("resolved_skills") or []:
        if isinstance(s, dict) and s.get("label"):
            out.append(str(s["label"]))
    if out:
        return _dedupe_stable(out)
    for s in (user.get("skills_vector") or {}).get("top_skills") or []:
        if isinstance(s, dict):
            lab = s.get("preferredLabel") or s.get("label")
            if lab:
                out.append(str(lab))
    return _dedupe_stable(out)


def job_skill_labels_for_concat(job: dict[str, Any]) -> list[str]:
    """Collect job essential ∪ optional skill labels."""

    out: list[str] = []

    def _take(items: Any) -> None:
        for s in items or []:
            if isinstance(s, dict) and s.get("label"):
                out.append(str(s["label"]))

    _take(job.get("essential_skills"))
    _take(job.get("optional_skills"))
    return _dedupe_stable(out)


def user_concat_embedding_text(
    user: dict[str, Any],
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    return build_concat_embedding_text(
        user_skill_labels_for_concat(user), separator=separator
    )


def job_concat_embedding_text(
    job: dict[str, Any],
    *,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    return build_concat_embedding_text(
        job_skill_labels_for_concat(job), separator=separator
    )


# --------------------------------------------------------------------------------------------------
# education
#
# Post-secondary education eligibility gate (shared by all matching endpoints).
#
# A job may declare ``requires_post_secondary`` (boolean) under
# ``llm_job_attributes.attributes`` in Mongo; it is surfaced as a top-level
# ``requires_post_secondary`` key on the flat job dict by
# :func:`app.jobs.repository.build_job_dict_from_ranked`.
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


def _coerce_flag(val: Any) -> bool | None:
    """Normalise a binary flag that may arrive as bool, int (0/1), float, or string.

    Both the job side (``requires_post_secondary``) and the user side
    (``any_post_secondary_educ``) may be stored as a boolean (``true``/``false``) OR as an
    integer (``1``/``0``), depending on the producer. This collapses both representations:

    * Returns ``True``  for ``True`` / ``1`` / ``"1"`` / ``"true"`` / ``"yes"``.
    * Returns ``False`` for ``False`` / ``0`` / ``"0"`` / ``"false"`` / ``"no"`` / ``""``.
    * Returns ``None`` for missing/unrecognised values (e.g. ``None`` or a stray ``2``),
      so callers can apply their own fail-open default rather than guessing.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if val == 1:
            return True
        if val == 0:
            return False
        return None
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("1", "true", "yes", "y", "t"):
            return True
        if s in ("0", "false", "no", "n", "f", ""):
            return False
    return None


def job_requires_post_secondary(job: dict[str, Any]) -> bool:
    """True only when the job explicitly requires post-secondary education.

    Accepts the flag as bool or int (``true``/``1`` => requires), at the top level or nested
    under ``attributes``. Anything else (absent / unrecognised) is treated as "not required"
    so a job is never hidden for lacking the field.
    """
    val = job.get("requires_post_secondary")
    if val is None:
        attrs = job.get("attributes")
        if isinstance(attrs, dict):
            val = attrs.get("requires_post_secondary")
    return _coerce_flag(val) is True


def user_lacks_post_secondary(user: dict[str, Any]) -> bool:
    """True only when the user explicitly reported no post-secondary education.

    Accepts the flag as bool or int (``false``/``0`` => lacks). A stray/unknown value or an
    absent field is NOT treated as lacking here — but note ``MatchRequest`` defaults
    ``any_post_secondary_educ`` to ``0``, so an omitted field becomes "lacks" after validation.
    """
    return _coerce_flag(user.get("any_post_secondary_educ")) is False


def is_education_eligible(user: dict[str, Any], job: dict[str, Any]) -> bool:
    """Education gate: ineligible iff the job requires post-secondary and the user has none."""
    return not (job_requires_post_secondary(job) and user_lacks_post_secondary(user))


def filter_jobs_by_education(
    user: dict[str, Any], jobs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Drop jobs the user is education-ineligible for.

    No-op (returns a shallow copy) when the user has — or did not report lacking —
    post-secondary education, so the common path stays cheap.
    """
    if not user_lacks_post_secondary(user):
        return list(jobs)
    return [j for j in jobs if not job_requires_post_secondary(j)]


def zqf_annotation(user_zqf, job_zqf_min):
    """(zqf_eligible, zqf_gap) or (None, None) when either side is missing."""
    if user_zqf is not None and isinstance(job_zqf_min, (int, float)):
        jmin = int(job_zqf_min)
        ulevel = int(user_zqf)
        return (ulevel >= jmin, abs(ulevel - jmin))
    return (None, None)


# --------------------------------------------------------------------------------------------------
# location
#
# Tiered urban-pull location matching.
#
# Maps a user's county to an ordered fallback **hub chain** (local -> regional hub -> national hub) and
# scores each job by which tier it falls in, so non-hub users (whose own county has few jobs) still get a
# full list while local jobs stay preferred. Hub counties do not pull outward.
#
# Two consumers:
#   * the Mongo prefilter (app.jobs.repository.JobsRepository.find_active) uses ``chain_for`` to widen a
#     user's candidate pool to include their hub regions;
#   * the v4 opportunity ranker uses ``tier_factor_for_job`` as a per-uuid [0,1] multiplier on final_score
#     (local=1.0, regional=W_REGIONAL, national=W_NATIONAL, off-chain=0.0).
#
# Region matching uses ``job_matches_user_location`` — the same lenient casefold-substring + always-remote
# rule every matching path uses.
#
# The hub data is a small JSON of exceptions (see resources/location/location_hub_chains.json, loaded by
# ``ArtifactsRepository.load_hub_chains``); every county not named there defaults to
# ``[self, national_hub]``. If the file is missing/malformed the loader returns None and callers fall back to
# today's strict behaviour (the feature becomes a no-op).
# --------------------------------------------------------------------------------------------------


def _cf(s: Any) -> str:
    """Casefold + strip, aligned with _norm / app.jobs.repository._norm_loc_value."""
    if s is None:
        return ""
    t = str(s).strip()
    return t.casefold() if t else ""


def _norm(v: str | None) -> str:
    return str(v).strip().casefold() if v is not None else ""


def job_matches_user_location(job: dict[str, Any], user: dict[str, Any]) -> bool:
    """Lenient location match.
    - Always matches 'Remote' jobs
    - Matches if city or province match (case-insensitive, substring)
    """
    user_city = _norm(user.get("city"))
    user_province = _norm(user.get("province"))

    job_city = _norm(job.get("city"))
    job_province = _norm(job.get("province"))
    job_loc = _norm(job.get("location"))

    #  Always include Remote jobs
    if "remote" in job_city or "remote" in job_province or "remote" in job_loc:
        return True

    if not user_city or not user_province:
        return False

    # Check City Match (Lenient)
    if job_city and (user_city in job_city or job_city in user_city):
        return True

    # Check Province Match (Lenient)
    if job_province and (
        user_province in job_province or job_province in user_province
    ):
        return True

    # Fallback to location string match
    if job_loc:
        return user_city in job_loc or user_province in job_loc

    return False


def user_matches_any_county(user: dict[str, Any], counties: list[str]) -> bool:
    """True if the user's location matches at least one of the given occupation counties."""
    for c in counties:
        if job_matches_user_location({"city": c, "province": c, "location": c}, user):
            return True
    return False


def _job_matches(job: dict[str, Any], region_user: dict[str, Any]) -> bool:
    return job_matches_user_location(job, region_user)


class HubChains:
    """Parsed county -> hub-chain map. Built from the exceptions JSON; chains derived lazily."""

    def __init__(
        self,
        national_hub: str,
        regional_hubs: dict[str, list[str]] | None,
        hub_self_only: list[str] | None,
    ) -> None:
        self.national: str = _cf(national_hub)
        self.regional_of: dict[str, str] = {}  # county_cf -> regional hub_cf
        for hub, counties in (regional_hubs or {}).items():
            h = _cf(hub)
            if not h:
                continue
            for c in counties or []:
                cc = _cf(c)
                if cc:
                    self.regional_of[cc] = h
        self.self_only: set = {_cf(h) for h in (hub_self_only or []) if _cf(h)}

    def chain_for(self, county_cf: str) -> list[str]:
        """Ordered fallback regions for a user county: [local, regional hub?, national hub?].

        Hubs in ``hub_self_only`` -> ``[self]`` (no outward pull). Unknown/empty counties -> the
        national hub only (safe default). Order encodes tier: index 0 = local.
        """
        c = _cf(county_cf)
        if not c:
            return [self.national] if self.national else []
        if c in self.self_only:
            return [c]
        chain = [c]
        reg = self.regional_of.get(c)
        if reg and reg not in chain:
            chain.append(reg)
        if self.national and self.national not in chain:
            chain.append(self.national)
        return chain

    def tier_factor_for_job(
        self,
        job: dict[str, Any],
        county_cf: str,
        *,
        w_regional: float,
        w_national: float,
    ) -> float:
        """Location multiplier for one job given the user's county.

        local (chain[0]) -> 1.0; national hub -> ``w_national``; any other in-chain (regional) hub ->
        ``w_regional``; remote jobs -> 1.0 (always allowed); anything off-chain -> 0.0. Roles are keyed
        by identity (local / national), NOT chain index, so a county whose chain skips the regional tier
        (e.g. Kitui -> [kitui, nairobi]) still scores Nairobi at the national weight.
        """
        chain = self.chain_for(county_cf)
        if not chain:
            return 1.0  # no chain (unknown national hub) -> don't penalise; degrade to neutral
        local = chain[0]
        for region in chain:
            ruser = {"city": region, "province": region, "location": region}
            if _job_matches(job, ruser):
                if region == local:
                    return 1.0
                if region == self.national:
                    return float(w_national)
                return float(w_regional)
        return 0.0


# --------------------------------------------------------------------------------------------------
# retrieval
#
# Stage-1 retrieval: whitened concat-cosine shortlist per user, with the education gate and the
# optional location-tier (urban-pull) weighting applied before the ``retrieve_top_k`` cutoff.
# --------------------------------------------------------------------------------------------------


class Stage1Corpus(BaseModel):
    """The items of one corpus that carry a stage-1 vector, aligned row-for-row."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rows: SkipValidation[list[dict[str, Any]]]
    """Items with a stage-1 vector (their vector fields are stripped in place). Not validated, so
    these are the caller's own dicts rather than copies."""

    n_loaded: int
    """Items passed in, with or without a vector."""

    matrix: np.ndarray = Field(default_factory=lambda: np.zeros((0, 0)))
    """Raw stacked vectors (float64)."""

    normalized: np.ndarray = Field(default_factory=lambda: np.zeros((0, 0)))
    """L2-normalised ``matrix`` (float64)."""

    ids: list[str] = Field(default_factory=list)
    requires_post_secondary: list[bool] = Field(default_factory=list)


def _sorted_indices_desc(sim_row: np.ndarray) -> np.ndarray:
    return np.argsort(-sim_row, kind="stable")


class IStage1Retriever(ABC):
    """Interface for the stage-1 shortlist."""

    @abstractmethod
    def prepare(self, items: list[dict[str, Any]]) -> Stage1Corpus:
        """
        Collects the items that carry a stage-1 vector and strips the vectors off those dicts.

        :param items: Jobs or occupation rows
        :return: The aligned corpus (``rows`` empty when no item has a vector)
        """
        raise NotImplementedError()

    @abstractmethod
    def shortlist(
        self,
        users: list[dict[str, Any]],
        corpus: Stage1Corpus,
        user_unit_vectors: np.ndarray,
        *,
        matcher: CosineSkillMatcher,
        whitener: ConcatWhitener,
        retrieve_top_k: int,
        hub_chains: HubChains | None = None,
        w_regional: float = 1.0,
        w_national: float = 1.0,
    ) -> list[list[dict[str, Any]]]:
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

    def prepare(self, items: list[dict[str, Any]]) -> Stage1Corpus:
        job_rows: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []
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
        users: list[dict[str, Any]],
        corpus: Stage1Corpus,
        user_unit_vectors: np.ndarray,
        *,
        matcher: CosineSkillMatcher,
        whitener: ConcatWhitener,
        retrieve_top_k: int,
        hub_chains: HubChains | None = None,
        w_regional: float = 1.0,
        w_national: float = 1.0,
    ) -> list[list[dict[str, Any]]]:
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

        shortlists: list[list[dict[str, Any]]] = []
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

            cosine_recs: list[dict[str, Any]] = []
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
