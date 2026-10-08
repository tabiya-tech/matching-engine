from collections import Counter
import logging
from typing import Dict, List, Optional

import numpy as np

from app.config import GATE_SIMILARITY_THRESHOLD
from app.ranking.skill_label_packs import SkillLabelPacks
from app.ranking.skills_match import (
    Jobseeker,
    Opportunity,
    SimilarityEngine,
    compute_U_complete,
    compute_feasibility_signals,
    compute_utility_and_feasibility_pair,
)

logger = logging.getLogger(__name__)


def _canon(label: str) -> str:
    """Canonical form for label-based skill resolution: lowercase, whitespace-collapsed."""
    return " ".join(label.strip().lower().split())


class SkillScorer:
    """Node2Vec/Gemini skill utility + feasibility scorer over one skill-embedding matrix.

    Built from already-loaded data (see ``app.server_dependencies.model_dependencies``):
    ``weights`` is the stored ``embedding.weight``; ``skill_group_rows`` / ``hierarchy_rows`` are the
    taxonomy CSV rows, or None when the file is absent.
    """

    def __init__(
        self,
        *,
        weights: np.ndarray,
        skill_to_row: Dict[str, int],
        packs: SkillLabelPacks,
        display_language: str,
        rescale_target: float,
        skill_group_rows: Optional[List[dict]],
        hierarchy_rows: Optional[List[dict]],
        model_label: str,
    ):
        W = weights
        # Promote lower-precision weights (e.g. fp16, used by the Gemini artefact
        # to stay under GitHub's per-file size limit) to fp32 before any
        # downstream math; SimilarityEngine and the cosine kernels assume fp32.
        if W.dtype != np.float32:
            W = W.astype(np.float32)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        W = W / np.where(norms > 0, norms, 1.0)

        self._embedding_ids = set(skill_to_row.keys())
        self.engine = SimilarityEngine(W, skill_to_row, rescale_target)

        # Label-primary resolution maps, built from every enabled language's taxonomy pack.
        # UUIDs are NOT used for resolution — they carry modelId-drift risk (a Compass-side
        # UUID can resolve to a different internal skill than the user's declared label).
        # Labels are stable across taxonomy versions; they're our trust anchor.
        #
        # Loading several languages into one resolver is what makes a Spanish posting match
        # a Spanish profile with nothing on the request: every pack is mapped onto this one
        # id space, so the embeddings are shared (see ranking/skill_label_packs.py).
        self._packs = packs
        # internal_id -> preferredLabel in the DEPLOYMENT's language: these strings are echoed in
        # responses (skill-gap labels, matched-skill labels), while resolution stays id-based and
        # language-neutral. Per-skill fallback to the canonical label when a pack lacks one.
        self.skill_labels = self._packs.display_labels(display_language)
        self._preferred_to_id = self._packs.preferred_to_id
        self._altlabel_to_id = self._packs.altlabel_to_id
        self._preferred_collisions = self._packs.preferred_collisions

        # Miss telemetry (per-process, lifetime-cumulative).
        self._missed_labels: Counter = Counter()

        logger.info(
            "SkillScorer: %d embedding IDs (model=%s, dim=%d); "
            "%d preferredLabel keys, %d altLabel keys (preferred-collisions: %d)",
            len(self._embedding_ids),
            model_label,
            W.shape[1],
            len(self._preferred_to_id),
            len(self._altlabel_to_id),
            self._preferred_collisions,
        )

        self.skill_group_labels = {}
        if skill_group_rows is not None:
            for row in skill_group_rows:
                group_id = row.get("ID")
                label = row.get("PREFERREDLABEL")
                if group_id and label:
                    self.skill_group_labels[str(group_id)] = label

        self._skill_to_groups: dict[str, set[str]] = {}
        if hierarchy_rows is not None:
            for row in hierarchy_rows:
                if (
                    row.get("PARENTOBJECTTYPE") == "skillgroup"
                    and row.get("CHILDOBJECTTYPE") == "skill"
                ):
                    child = str(row["CHILDID"])
                    parent = str(row["PARENTID"])
                    self._skill_to_groups.setdefault(child, set()).add(parent)
            logger.info(
                "SkillScorer: built skill→group lookup for %d skills",
                len(self._skill_to_groups),
            )
        else:
            logger.warning(
                "skill_hierarchy.csv not found — skill group derivation disabled"
            )

    def _derive_groups(self, skill_ids: set[str]) -> set[str]:
        """Look up parent skill-group IDs for a set of individual skill IDs."""
        groups = set()
        for sid in skill_ids:
            groups.update(self._skill_to_groups.get(sid, set()))
        return groups

    def _resolve_label(self, label: str) -> str | None:
        """Resolve a skill label to its internal embedding ID. Strict label-only.

        Lookup chain: preferredLabel → altLabel, in any loaded language → miss.
        UUIDs are deliberately not consulted — they carry modelId-drift risk.
        """
        if not label:
            return None
        sid = self._packs.resolve_label(label)
        if sid is not None:
            return sid
        self._missed_labels[label] += 1
        return None

    def display_labels(self, language: str | None = None) -> dict[str, str]:
        """internal_id → preferredLabel in ``language`` (canonical language when absent)."""
        return self._packs.display_labels(language)

    def get_resolution_stats(self) -> dict:
        """Per-process miss telemetry. Useful for batch scripts at end-of-run."""
        return {
            "total_misses": sum(self._missed_labels.values()),
            "distinct_missed_labels": len(self._missed_labels),
            "top_misses": self._missed_labels.most_common(20),
        }

    def _build_objects(self, user_profile: dict, job_posting: dict):
        """Build Jobseeker / Opportunity dataclasses + user skill labels.

        Both sides resolve via labels — user payload carries preferredLabel,
        job payload carries essential_skills/optional_skills as [{id, label}, ...]
        dicts (label is the trust anchor). Group IDs come pre-resolved from
        the upstream taxonomy export and are passed through unchanged.
        """
        user_id = user_profile.get("user_id") or user_profile.get("youth_id")
        if not user_id:
            raise ValueError("user_profile must include user_id (or legacy youth_id)")

        user_top_skills = (
            user_profile.get("skills_vector", {}).get("top_skills", []) or []
        )

        user_skill_labels: dict[str, str] = {}
        resolved_user_ids: set[str] = set()
        for s in user_top_skills:
            if not isinstance(s, dict):
                continue
            label = s.get("preferredLabel")
            resolved = self._resolve_label(label)
            if resolved is None:
                continue
            resolved_user_ids.add(resolved)
            if label:
                user_skill_labels[resolved] = label

        raw_user_groups = user_profile.get("skill_groups_origin_uuids", []) or []
        user_groups = {str(g) for g in raw_user_groups if g}
        if not user_groups:
            user_groups = self._derive_groups(resolved_user_ids)

        js = Jobseeker(
            compass_id=str(user_id),
            skills_origin_uuids=resolved_user_ids,
            skill_groups_origin_uuids=user_groups,
            city=user_profile.get("city"),
            province=user_profile.get("province"),
        )

        def _resolve_job_skills(items) -> set[str]:
            out: set[str] = set()
            for s in items or []:
                if not isinstance(s, dict):
                    continue
                resolved = self._resolve_label(s.get("label"))
                if resolved is None:
                    # Fallback for a label this build has no pack for: the job's `id` is a
                    # taxonomy skill ID written by the reranker, which the label packs
                    # crosswalk across locales. This is not the UUID path the docstring
                    # warns about — it is an exact per-model identifier, used only after
                    # the label misses.
                    resolved = self._packs.resolve_id(s.get("id"))
                if resolved is not None:
                    out.add(resolved)
            return out

        resolved_ess = _resolve_job_skills(job_posting.get("essential_skills"))
        resolved_opt = _resolve_job_skills(job_posting.get("optional_skills"))

        raw_job_groups = job_posting.get("skill_groups_origin_uuids", []) or []
        job_groups = {str(g) for g in raw_job_groups if g}
        if not job_groups:
            job_groups = self._derive_groups(resolved_ess | resolved_opt)

        op = Opportunity(
            opportunity_id=str(job_posting.get("uuid")),
            essential_skill_ids=resolved_ess,
            optional_skill_ids=resolved_opt,
            skill_groups_origin_uuids=job_groups,
            city=job_posting.get("city") or job_posting.get("location"),
            province=job_posting.get("province") or job_posting.get("location"),
        )

        return js, op, user_skill_labels

    def calculate_score(self, user_profile: dict, job_posting: dict) -> dict:
        """Returns the full dictionary of skill utility components (legacy U_final)."""
        js, op, user_skill_labels = self._build_objects(user_profile, job_posting)
        return compute_U_complete(
            js,
            op,
            self.engine,
            skill_labels=self.skill_labels,
            user_skill_labels=user_skill_labels,
            skill_group_labels=self.skill_group_labels,
        )

    def calculate_feasibility(self, user_profile: dict, job_posting: dict) -> dict:
        """Returns recruiter-side feasibility signals for the success-propensity proxy.

        Uses the same Node2Vec embedding engine as calculate_score but aggregates
        essential-skill similarities via geometric mean (stricter on gaps).
        """
        js, op, user_skill_labels = self._build_objects(user_profile, job_posting)
        return compute_feasibility_signals(
            js,
            op,
            self.engine,
            gate_threshold=GATE_SIMILARITY_THRESHOLD,
            skill_labels=self.skill_labels,
            user_skill_labels=user_skill_labels,
            skill_group_labels=self.skill_group_labels,
        )

    def score_utility_and_feasibility(
        self, user_profile: dict, job_posting: dict
    ) -> tuple[dict, dict]:
        """Single embedding pass for multiplicative mode (U + feasibility for p_hat)."""
        js, op, user_skill_labels = self._build_objects(user_profile, job_posting)
        u, f = compute_utility_and_feasibility_pair(
            js,
            op,
            self.engine,
            skill_labels=self.skill_labels,
            user_skill_labels=user_skill_labels,
            skill_group_labels=self.skill_group_labels,
            gate_threshold=GATE_SIMILARITY_THRESHOLD,
        )
        return u, f
