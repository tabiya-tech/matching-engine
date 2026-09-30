from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UserSkill:
    preferred_label: str


@dataclass
class UserProfile:
    user_id: str
    city: str
    province: str
    skills: list[UserSkill] = field(default_factory=list)


def parse_user(raw: dict) -> UserProfile:
    skills_raw = raw.get("skills_vector", {}).get("top_skills", [])
    skills = [UserSkill(preferred_label=s["preferredLabel"]) for s in skills_raw]
    return UserProfile(
        user_id=raw.get("user_id", ""),
        city=raw.get("city", ""),
        province=raw.get("province", ""),
        skills=skills,
    )


def load_users_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_user_by_id(path: Path, user_id: str) -> UserProfile | None:
    for raw in load_users_jsonl(path):
        if raw.get("user_id") == user_id:
            return parse_user(raw)
    return None
