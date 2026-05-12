from collections import Counter
from pathlib import Path
import logging

import os

import torch
import json
import csv
import numpy as np

from app.config import (
    EMBEDDING_MODEL_PATH,
    GATE_SIMILARITY_THRESHOLD,
    SKILL_GROUPS_CSV_PATH,
    SKILLS_CSV_PATH,
    SKILL_HIERARCHY_CSV_PATH,
    SKILL_TO_ROW_PATH,
)
from app.services.skills_utility import skills_match as _skills_match
from app.services.skills_utility.skills_match import (
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
    def __init__(self):
        self.MODEL_PATH = Path(EMBEDDING_MODEL_PATH)
        self.MAPPING_PATH = Path(SKILL_TO_ROW_PATH)
        self.SKILLS_CSV_PATH = Path(SKILLS_CSV_PATH)
        self.SKILL_GROUPS_CSV_PATH = Path(SKILL_GROUPS_CSV_PATH)
        self.HIERARCHY_CSV_PATH = Path(SKILL_HIERARCHY_CSV_PATH)

        state = torch.load(self.MODEL_PATH, map_location="cpu")

        # Hydrate per-rowmax rescale target from artefact metadata, unless the operator
        # has explicitly set SKILL_RESCALE_TARGET in env (operator wins). Whitened artefacts
        # carry a `target_max_p999` value computed at build time; raw / non-whitened
        # artefacts won't have one, in which case we leave the existing default in place.
        # Mutate the binding inside skills_match (where the kernel reads it) — same pattern
        # used for SKILL_ESSENTIAL_DAMPING_ALPHA overrides in batch scripts.
        if "SKILL_RESCALE_TARGET" not in os.environ:
            target_from_artefact = (state.get("whitening") or {}).get("target_max_p999")
            if target_from_artefact is not None:
                _skills_match.SKILL_RESCALE_TARGET = float(target_from_artefact)
                logger.info(
                    "SkillScorer: SKILL_RESCALE_TARGET set from artefact metadata = %.4f",
                    float(target_from_artefact),
                )

        W = state["state_dict"]["embedding.weight"].numpy()
        # Promote lower-precision weights (e.g. fp16, used by the Gemini artefact
        # to stay under GitHub's per-file size limit) to fp32 before any
        # downstream math; SimilarityEngine and the cosine kernels assume fp32.
        if W.dtype != np.float32:
            W = W.astype(np.float32)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        W = W / np.where(norms > 0, norms, 1.0)

        with open(self.MAPPING_PATH, "r") as f:
            skill_to_row = json.load(f)

        self._embedding_ids = set(skill_to_row.keys())
        self.engine = SimilarityEngine(W, skill_to_row)

        # Label-primary resolution maps. UUIDs are NOT used for resolution — they
        # carry modelId-drift risk (a Compass-side UUID can resolve to a different
        # internal skill than the user's declared label). Labels are stable across
        # taxonomy versions; they're our trust anchor.
        self.skill_labels: dict[str, str] = {}            # internal_id -> preferredLabel (display)
        self._preferred_to_id: dict[str, str] = {}        # canonical preferredLabel -> internal_id
        self._altlabel_to_id: dict[str, str] = {}         # canonical altLabel -> internal_id
        self._preferred_collisions = 0
        try:
            with open(self.SKILLS_CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row.get("ID")
                    label = row.get("PREFERREDLABEL") or ""
                    if not sid or sid not in self._embedding_ids:
                        continue
                    if label:
                        self.skill_labels[str(sid)] = label
                        canon = _canon(label)
                        if canon and canon not in self._preferred_to_id:
                            self._preferred_to_id[canon] = str(sid)
                        elif canon and self._preferred_to_id.get(canon) != str(sid):
                            self._preferred_collisions += 1
                    for alt in (row.get("ALTLABELS") or "").split("\n"):
                        canon = _canon(alt)
                        if not canon:
                            continue
                        # Don't shadow a preferredLabel hit via altLabel; first writer wins on alt.
                        if canon in self._preferred_to_id:
                            continue
                        self._altlabel_to_id.setdefault(canon, str(sid))
        except FileNotFoundError:
            self.skill_labels = {}

        # Miss telemetry (per-process, lifetime-cumulative).
        self._missed_labels: Counter = Counter()

        logger.info(
            "SkillScorer: %d embedding IDs (model=%s, dim=%d); "
            "%d preferredLabel keys, %d altLabel keys (preferred-collisions: %d)",
            len(self._embedding_ids),
            state.get("model_name") or self.MODEL_PATH.name,
            W.shape[1],
            len(self._preferred_to_id),
            len(self._altlabel_to_id),
            self._preferred_collisions,
        )

        self.skill_group_labels = {}
        try:
            with open(self.SKILL_GROUPS_CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    group_id = row.get("ID")
                    label = row.get("PREFERREDLABEL")
                    if group_id and label:
                        self.skill_group_labels[str(group_id)] = label
        except FileNotFoundError:
            self.skill_group_labels = {}

        self._skill_to_groups: dict[str, set[str]] = {}
        try:
            with open(self.HIERARCHY_CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("PARENTOBJECTTYPE") == "skillgroup" and row.get("CHILDOBJECTTYPE") == "skill":
                        child = str(row["CHILDID"])
                        parent = str(row["PARENTID"])
                        self._skill_to_groups.setdefault(child, set()).add(parent)
            logger.info("SkillScorer: built skill→group lookup for %d skills", len(self._skill_to_groups))
        except FileNotFoundError:
            logger.warning("skill_hierarchy.csv not found — skill group derivation disabled")

    def _derive_groups(self, skill_ids: set[str]) -> set[str]:
        """Look up parent skill-group IDs for a set of individual skill IDs."""
        groups = set()
        for sid in skill_ids:
            groups.update(self._skill_to_groups.get(sid, set()))
        return groups

    def _resolve_label(self, label: str) -> str | None:
        """Resolve a skill label to its internal embedding ID. Strict label-only.

        Lookup chain: canonical preferredLabel → canonical altLabel → miss.
        UUIDs are deliberately not consulted — they carry modelId-drift risk.
        """
        if not label:
            return None
        canon = _canon(label)
        if not canon:
            return None
        sid = self._preferred_to_id.get(canon)
        if sid is not None:
            return sid
        sid = self._altlabel_to_id.get(canon)
        if sid is not None:
            return sid
        self._missed_labels[label] += 1
        return None

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

        user_top_skills = user_profile.get("skills_vector", {}).get("top_skills", []) or []

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

    def score_utility_and_feasibility(self, user_profile: dict, job_posting: dict) -> tuple[dict, dict]:
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