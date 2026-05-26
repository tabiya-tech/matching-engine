"""Map demand-side level tags → Vⱼ using ladder position and gain/cost orientation."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal, Optional

# Production / KenyaJobs_V2 level ids → schema ids in job_attributes_schema (1).json
LEVEL_ID_ALIASES: Dict[str, Dict[str, str]] = {
    "earnings_per_month": {
        "earn_15k": "earn_10_20k",
        "earn_30k": "earn_20_35k",
        "earn_50k": "earn_50_80k",
        "earn_70k": "earn_50_80k",
    },
    "social_interaction": {
        "soc_customers": "soc_people",
        "soc_peers": "soc_people",
    },
}

_EARN_K_RE = re.compile(r"^earn_(\d+)k$", re.IGNORECASE)

Orientation = Literal["gain", "cost"]

# Client: gain (+) → V' = ladder; cost (−) → V' = 1 − ladder
GAIN_ORIENTED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "earnings_per_month",  # Earnings
        "career_growth",  # Career Growth
        "social_interaction",  # Social Interaction (higher = more people)
        "task_content",  # Task Routine / task ladder (higher = more creative)
        "work_flexibility",  # default gain: higher flexibility = better
        "social_meaning",  # default gain: higher meaning = better
    }
)
COST_ORIENTED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "physical_demand",  # Physical Demand — light = good → invert
    }
)

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "preferences_utility"
    / "untitled folder"
    / "job_attributes_schema (1).json"
)


@lru_cache(maxsize=1)
def load_attribute_schema(path: Optional[str] = None) -> dict:
    p = Path(path) if path else _SCHEMA_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def attribute_orientation(attr_name: str) -> Orientation:
    if attr_name in COST_ORIENTED_ATTRIBUTES:
        return "cost"
    return "gain"


def _level_index(level_id: str, level_ids: list[str]) -> Optional[int]:
    if not level_id or level_id not in level_ids:
        return None
    return level_ids.index(level_id)


def _schema_spec(attr_name: str, schema: dict) -> Optional[dict]:
    meta = {a["name"]: a for a in schema.get("attributes", [])}
    return meta.get(attr_name)


def _earnings_level_from_kes(kes: float, level_ids: list[str]) -> Optional[str]:
    """Map a monthly KES amount to the schema earnings bucket id."""
    if kes < 10_000:
        return "earn_lt10k" if "earn_lt10k" in level_ids else None
    brackets = [
        (20_000, "earn_10_20k"),
        (35_000, "earn_20_35k"),
        (50_000, "earn_35_50k"),
        (80_000, "earn_50_80k"),
        (120_000, "earn_80_120k"),
        (180_000, "earn_180_180k"),
        (300_000, "earn_180_300k"),
    ]
    for bound, lid in brackets:
        if kes < bound:
            return lid if lid in level_ids else None
    return "earn_300k_plus" if "earn_300k_plus" in level_ids else None


def resolve_schema_level_id(
    attr_name: str,
    raw_level_id: Optional[str],
    schema: dict,
) -> Optional[str]:
    """
    Normalize job level tags to ids declared in the attribute schema.

    Returns None if the job did not supply a level or it cannot be mapped.
    """
    if raw_level_id is None:
        return None
    raw = str(raw_level_id).strip()
    if not raw or raw in ("—", "-", "â€", "â€\""):
        return None

    spec = _schema_spec(attr_name, schema)
    if not spec:
        return None
    level_ids = [lv["id"] for lv in spec.get("levels", [])]
    if raw in level_ids:
        return raw

    aliased = (LEVEL_ID_ALIASES.get(attr_name) or {}).get(raw)
    if aliased and aliased in level_ids:
        return aliased

    if attr_name == "earnings_per_month":
        m = _EARN_K_RE.match(raw)
        if m:
            kes = float(m.group(1)) * 1000.0
            mapped = _earnings_level_from_kes(kes, level_ids)
            if mapped:
                return mapped

    return None


def ladder_position(attr_name: str, level_id: Optional[str], schema: dict) -> float:
    """Ladder position ∈ [0, 1]: lowest schema bucket → 0, highest → 1."""
    resolved = resolve_schema_level_id(attr_name, level_id, schema)
    if resolved is None:
        return 0.0

    spec = _schema_spec(attr_name, schema)
    if not spec:
        return 0.0
    level_ids = [lv["id"] for lv in spec.get("levels", [])]
    if not level_ids:
        return 0.0
    if len(level_ids) == 1:
        return 1.0 if resolved == level_ids[0] else 0.0
    idx = _level_index(resolved, level_ids)
    if idx is None:
        return 0.0
    return float(idx) / float(len(level_ids) - 1)


def job_level_to_vj(attr_name: str, level_id: Optional[str], schema: dict) -> float:
    """
    Vⱼ for Part A (client directional mapping).

    Gain (+): V'ⱼ = ladder position
    Cost (−): V'ⱼ = 1.0 − ladder position  (e.g. phys_light → 1, phys_heavy → 0)
    """
    pos = ladder_position(attr_name, level_id, schema)
    if attribute_orientation(attr_name) == "cost":
        return 1.0 - pos
    return pos


def attribute_label(attr_name: str, schema: dict) -> str:
    meta = {a["name"]: a for a in schema.get("attributes", [])}
    spec = meta.get(attr_name)
    if spec and spec.get("label"):
        return str(spec["label"])
    return attr_name.replace("_", " ").title()


def level_label(attr_name: str, level_id: Optional[str], schema: dict) -> str:
    """Human-readable job level for dashboards (uses schema label after resolve)."""
    resolved = resolve_schema_level_id(attr_name, level_id, schema)
    if not resolved:
        return "—"
    spec = _schema_spec(attr_name, schema)
    if not spec:
        return str(level_id or "—")
    for lv in spec.get("levels", []):
        if lv.get("id") == resolved:
            return str(lv.get("label") or resolved)
    return str(resolved)


def job_level_to_vj_detail(
    attr_name: str, level_id: Optional[str], schema: dict
) -> Dict[str, float | str]:
    pos = ladder_position(attr_name, level_id, schema)
    orient = attribute_orientation(attr_name)
    vj = (1.0 - pos) if orient == "cost" else pos
    return {
        "orientation": orient,
        "ladder_position": round(pos, 4),
        "vj": round(vj, 4),
    }
