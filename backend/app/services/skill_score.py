from pathlib import Path

import torch
import json
import csv
import numpy as np

from app.services.skills_utility.skills_match import (
    Jobseeker,
    Opportunity,
    SimilarityEngine,
    compute_U_complete,
)

current_dir = Path(__file__).parent


class SkillScorer:
    def __init__(self):
        self.MODEL_PATH = current_dir / "skills_utility/Output/skill_embedding_model.pt"
        self.MAPPING_PATH = current_dir / "skills_utility/Output/skill_to_row.json"
        self.SKILLS_CSV_PATH = current_dir / "skills_utility/skills.csv"
        self.SKILL_GROUPS_CSV_PATH = current_dir / "skills_utility/skill_groups.csv"

        state = torch.load(self.MODEL_PATH, map_location="cpu")
        W = state["state_dict"]["embedding.weight"].numpy()
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        W = W / np.where(norms > 0, norms, 1.0)

        with open(self.MAPPING_PATH, "r") as f:
            skill_to_row = json.load(f)

        self.engine = SimilarityEngine(W, skill_to_row)

        self.skill_labels = {}
        try:
            with open(self.SKILLS_CSV_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    skill_id = row.get("ID")
                    label = row.get("PREFERREDLABEL")
                    if skill_id and label:
                        self.skill_labels[str(skill_id)] = label
        except FileNotFoundError:
            self.skill_labels = {}

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

    def calculate_score(self, user_profile: dict, job_posting: dict) -> dict:
        """Returns the full dictionary of skill utility components."""

        user_id = user_profile.get("user_id") or user_profile.get("youth_id")
        if not user_id:
            raise ValueError("user_profile must include user_id (or legacy youth_id)")

        user_top_skills = user_profile.get("skills_vector", {}).get("top_skills", []) or []

        user_skill_labels = {
            s.get("originUUID"): s.get("preferredLabel")
            for s in user_top_skills
            if isinstance(s, dict) and s.get("originUUID")
        }

        js = Jobseeker(
            compass_id=str(user_id),
            skills_origin_uuids={
                s.get("originUUID") for s in user_top_skills if isinstance(s, dict) and s.get("originUUID")
            },
            skill_groups_origin_uuids=set(user_profile.get("skill_groups_origin_uuids", [])),
            city=user_profile.get("city"),
            province=user_profile.get("province"),
        )

        op = Opportunity(
            opportunity_id=str(job_posting.get("uuid")),
            essential_skills_origin_uuids=set(job_posting.get("essential_skills_origin_uuids", [])),
            optional_skills_origin_uuids=set(job_posting.get("optional_skills_origin_uuids", [])),
            skill_groups_origin_uuids=set(job_posting.get("skill_groups_origin_uuids", [])),
            city=job_posting.get("city") or job_posting.get("location"),
            province=job_posting.get("province") or job_posting.get("location"),
        )

        return compute_U_complete(
            js,
            op,
            self.engine,
            skill_labels=self.skill_labels,
            user_skill_labels=user_skill_labels,
            skill_group_labels=self.skill_group_labels,
        )