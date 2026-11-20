from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeType(str, Enum):
    SKILL = "skill"
    SKILL_GROUP = "skillgroup"


class EdgeType(str, Enum):
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"


@dataclass
class Skill:
    id: str
    preferred_label: str

    @classmethod
    def from_csv_row(cls, row: dict) -> Skill:
        return cls(
            id=row["ID"].strip(),
            preferred_label=row.get("PREFERREDLABEL", "").strip(),
        )


@dataclass
class SkillGroup:
    id: str
    preferred_label: str

    @classmethod
    def from_csv_row(cls, row: dict) -> SkillGroup:
        return cls(
            id=row["ID"].strip(),
            preferred_label=row.get("PREFERREDLABEL", "").strip(),
        )
