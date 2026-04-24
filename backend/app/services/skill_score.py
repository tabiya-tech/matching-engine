from pathlib import Path
import logging

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
from app.services.skills_utility.skills_match import (
    Jobseeker,
    Opportunity,
    SimilarityEngine,
    compute_U_complete,
    compute_feasibility_signals,
    compute_utility_and_feasibility_pair,
)

logger = logging.getLogger(__name__)


class SkillScorer:
    def __init__(self):
        self.MODEL_PATH = Path(EMBEDDING_MODEL_PATH)
        self.MAPPING_PATH = Path(SKILL_TO_ROW_PATH)
        self.SKILLS_CSV_PATH = Path(SKILLS_CSV_PATH)
        self.SKILL_GROUPS_CSV_PATH = Path(SKILL_GROUPS_CSV_PATH)
        self.HIERARCHY_CSV_PATH = Path(SKILL_HIERARCHY_CSV_PATH)

        state = torch.load(self.MODEL_PATH, map_location="cpu")
        W = state["state_dict"]["embedding.weight"].numpy()
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        W = W / np.where(norms > 0, norms, 1.0)

        with open(self.MAPPING_PATH, "r") as f:
            skill_to_row = json.load(f)

        self._embedding_ids = set(skill_to_row.keys())
        self.engine = SimilarityEngine(W, skill_to_row)

        self.skill_labels = {}
        self._esco_to_internal: dict[str, str] = {}
        try:
            with open(self.SKILLS_CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    skill_id = row.get("ID")
                    label = row.get("PREFERREDLABEL")
                    if skill_id and label:
                        self.skill_labels[str(skill_id)] = label
                    uuid_history = row.get("UUIDHISTORY", "")
                    origin_uri = row.get("ORIGINURI", "")
                    if skill_id:
                        for uuid in uuid_history.strip().split("\n"):
                            uuid = uuid.strip()
                            if uuid:
                                self._esco_to_internal[uuid] = str(skill_id)
                        if "/esco/skill/" in origin_uri:
                            esco_uuid = origin_uri.split("/esco/skill/")[-1]
                            self._esco_to_internal[esco_uuid] = str(skill_id)
        except FileNotFoundError:
            self.skill_labels = {}

        logger.info(
            "SkillScorer: %d embedding IDs, %d ESCO->internal mappings",
            len(self._embedding_ids), len(self._esco_to_internal),
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

    def _resolve_id(self, raw_id: str) -> str:
        """Translate a skill ID to the embedding model's internal ID space.

        Accepts ESCO standard UUIDs (e.g. 'd5c9e39f-...'), internal IDs
        (already in embedding), or any other format.  Returns the internal
        ID if a mapping exists, otherwise returns the input unchanged.
        """
        sid = str(raw_id)
        if sid in self._embedding_ids:
            return sid
        return self._esco_to_internal.get(sid, sid)

    def _resolve_ids(self, raw_ids) -> set[str]:
        """Batch-resolve a collection of skill IDs."""
        return {self._resolve_id(sid) for sid in raw_ids if sid}

    def _build_objects(self, user_profile: dict, job_posting: dict):
        """Build Jobseeker / Opportunity dataclasses + user skill labels.

        All skill IDs are resolved to the embedding model's internal ID
        space so that ESCO UUIDs, Tabiya IDs, or internal IDs all work
        transparently.
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
            raw = s.get("originUUID")
            if not raw:
                continue
            resolved = self._resolve_id(raw)
            resolved_user_ids.add(resolved)
            label = s.get("preferredLabel")
            if label:
                user_skill_labels[resolved] = label
                if resolved != raw:
                    user_skill_labels[raw] = label

        user_groups = self._resolve_ids(
            user_profile.get("skill_groups_origin_uuids", [])
        )
        if not user_groups:
            user_groups = self._derive_groups(resolved_user_ids)

        js = Jobseeker(
            compass_id=str(user_id),
            skills_origin_uuids=resolved_user_ids,
            skill_groups_origin_uuids=user_groups,
            city=user_profile.get("city"),
            province=user_profile.get("province"),
        )

        resolved_ess = self._resolve_ids(
            job_posting.get("essential_skills_origin_uuids", [])
        )
        resolved_opt = self._resolve_ids(
            job_posting.get("optional_skills_origin_uuids", [])
        )
        job_groups = self._resolve_ids(
            job_posting.get("skill_groups_origin_uuids", [])
        )
        if not job_groups:
            job_groups = self._derive_groups(resolved_ess | resolved_opt)

        op = Opportunity(
            opportunity_id=str(job_posting.get("uuid")),
            essential_skills_origin_uuids=resolved_ess,
            optional_skills_origin_uuids=resolved_opt,
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