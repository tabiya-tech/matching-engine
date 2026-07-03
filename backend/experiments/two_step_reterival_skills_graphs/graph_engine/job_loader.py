from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class JobSkill:
    label: str
    relation_type: str  # "essential" or "optional"
    graph_node_id: Optional[str] = None
    matched: bool = False


@dataclass
class Job:
    job_id: str
    title: str
    employer: str
    city: str
    skills: list[JobSkill] = field(default_factory=list)

    @property
    def essential_skills(self) -> list[JobSkill]:
        return [s for s in self.skills if s.relation_type == "essential"]

    @property
    def optional_skills(self) -> list[JobSkill]:
        return [s for s in self.skills if s.relation_type == "optional"]


def load_ranked_jobs_v2(path: Path) -> list[Job]:
    with open(path, encoding="utf-8") as f:
        raw_jobs = json.load(f)

    jobs: list[Job] = []
    for raw in raw_jobs:
        meta = raw.get("classifier_metadata", {})
        llm_skills = raw.get("llm_classified_skills", {})
        skills: list[JobSkill] = []

        for kind in ("essential", "optional"):
            for s in llm_skills.get(kind, []):
                label = s.get("label", "")
                if label:
                    skills.append(JobSkill(label=label, relation_type=kind))

        jobs.append(
            Job(
                job_id=raw.get("job_id", {}).get("$oid", ""),
                title=meta.get("title", ""),
                employer=meta.get("employer", ""),
                city=meta.get("city", ""),
                skills=skills,
            )
        )
    return jobs
