from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

_BACKEND_ROOT = Path(__file__).resolve().parent.parent  # .../backend
_RESOURCES = _BACKEND_ROOT / "resources"
_DEFAULT_OCC = _RESOURCES / "occupations" / "combined_occupation_database_with_wa.json"
_DEFAULT_MODEL_DIR = _RESOURCES / "models"


def _s(key: str, default: str) -> str:
    v = os.getenv(key)
    return (v or "").strip() or default


def _f(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    return float(v)


def _i(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    return int(v)


def _b(key: str, default: bool) -> bool:
    v = (os.getenv(key) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Mongo — job source (enriched collection from migration / reranker)
# ---------------------------------------------------------------------------
MONGO_DB_NAME: str = _s("MONGO_DB_NAME", "matching_service")
# Default: RankedJobsEnriched (see scripts/enrich_ranked_jobs_to_new_collection.py). Set to RankedJobs for legacy.
MONGO_JOBS_COLLECTION: str = _s("MONGO_JOBS_COLLECTION", "RankedJobsEnriched")
MONGO_MATCHING_CONFIG_DB_NAME: str = _s("MONGO_MATCHING_CONFIG_DB_NAME", MONGO_DB_NAME)
MONGO_MATCHING_CONFIG_COLLECTION: str = _s("MONGO_MATCHING_CONFIG_COLLECTION", "matching_configuration")
# Keep mongo routing document (_id: mongo_routing) in a stable config DB, not in jobs DB.
MONGO_ROUTING_CONFIG_DB_NAME: str = _s("MONGO_ROUTING_CONFIG_DB_NAME", "ma-testing-db")
MONGO_TEST_USERS_DB_NAME: str = _s("MONGO_TEST_USERS_DB_NAME", MONGO_DB_NAME)
MONGO_TEST_USERS_COLLECTION: str = _s("MONGO_TEST_USERS_COLLECTION", "test_users")

# HTTP /match: when set, load at most N jobs with is_active + (remote OR per-user location),
# as a superset of matching_service._job_matches_user_location. Set to 0 to disable
# the extra filter and load all active jobs (scripts, back-compat).
JOBS_RETRIEVAL_FILTER: bool = _b("JOBS_RETRIEVAL_FILTER", True)
JOBS_RETRIEVAL_LIMIT: int = _i("JOBS_RETRIEVAL_LIMIT", 10_000)
# Mongo find() inclusion projection (fields used by build_job_dict_from_ranked). Set 0 to load full documents.
JOBS_FIND_USE_PROJECTION: bool = _b("JOBS_FIND_USE_PROJECTION", True)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
SCORING_MODE: str = _s("SCORING_MODE", "multiplicative")
if SCORING_MODE not in ("multiplicative", "additive"):
    raise ValueError("SCORING_MODE must be 'multiplicative' or 'additive'")

# Runtime source for tunable matching config used by /match and /config:
# - mongodb: use Mongo overrides merged on code defaults
# - env:     use env/.env values from this module
# - auto:    use Mongo when available; fallback to env values on empty/failure
MATCHING_CONFIG_SOURCE: str = _s("MATCHING_CONFIG_SOURCE", "auto").lower()
if MATCHING_CONFIG_SOURCE not in ("mongodb", "env", "auto"):
    raise ValueError(
        "MATCHING_CONFIG_SOURCE must be one of: 'mongodb', 'env', 'auto'"
    )

# Additive (legacy) weights
GLOBAL_WEIGHTS: Dict[str, float] = {
    "w1_skills": _f("ADDITIVE_W1_SKILLS", 0.40),
    "w2_preference": _f("ADDITIVE_W2_PREFERENCE", 0.40),
    "w3_market": _f("ADDITIVE_W3_MARKET", 0.20),
}

# ---------------------------------------------------------------------------
# Match output sizes
# ---------------------------------------------------------------------------
MATCH_TOP_K_OPPORTUNITIES: int = _i("MATCH_TOP_K_OPPORTUNITIES", 5)
MATCH_TOP_K_OCCUPATIONS: int = _i("MATCH_TOP_K_OCCUPATIONS", 5)
MATCH_TOP_K_SKILL_GAPS: int = _i("MATCH_TOP_K_SKILL_GAPS", 5)

# When false, skip _job_matches_user_location in _match_items (opportunities and occupations).
# Mongo prefilter is separate: JOBS_RETRIEVAL_FILTER. Default true keeps current behaviour.
MATCH_APPLY_LOCATION_FILTER: bool = _b("MATCH_APPLY_LOCATION_FILTER", True)

# ---------------------------------------------------------------------------
# Success propensity  (p_hat = G * E^alpha * R^beta * M^gamma)
# ---------------------------------------------------------------------------
GATE_SIMILARITY_THRESHOLD: float = _f("GATE_SIMILARITY_THRESHOLD", 0.35)
# Response filtering threshold for scored skill outputs:
# - essential_skill_matches.similarity in opportunities/occupations
# - skill_gap_recommendations.proximity_score
MATCH_RESPONSE_SKILL_MIN_SCORE: float = _f(
    "MATCH_RESPONSE_SKILL_MIN_SCORE",
    GATE_SIMILARITY_THRESHOLD,
)

SUCCESS_PROPENSITY_CONFIG: Dict[str, Any] = {
    "alpha_essential": _f("PHAT_ALPHA_ESSENTIAL", 0.5),
    "beta_readiness": _f("PHAT_BETA_READINESS", 0.2),
    "gamma_market": _f("PHAT_GAMMA_MARKET", 0.3),
    "gate_threshold": GATE_SIMILARITY_THRESHOLD,
}

# ---------------------------------------------------------------------------
# Skill utility (U) / feasibility — aligned with skills_match
# ---------------------------------------------------------------------------
SKILL_U_W_LOC: float = _f("SKILL_U_W_LOC", 0.20)
SKILL_U_W_ESS: float = _f("SKILL_U_W_ESS", 0.50)
SKILL_U_W_OPT: float = _f("SKILL_U_W_OPT", 0.20)
SKILL_U_W_GRP: float = _f("SKILL_U_W_GRP", 0.10)
SKILL_U_GAP_PENALTY: float = _f("SKILL_U_GAP_PENALTY", 0.50)
SKILL_U_TAU_ELIG: float = _f("SKILL_U_TAU_ELIG", 0.35)
SKILL_MIN_ESSENTIAL_MATCH_SHARE: float = _f("SKILL_MIN_ESSENTIAL_MATCH_SHARE", 1.0)
SKILL_ESSENTIAL_GEO_FLOOR: float = _f("SKILL_ESSENTIAL_GEO_FLOOR", 1e-6)

# ---------------------------------------------------------------------------
# Demand label → numeric
# ---------------------------------------------------------------------------
DEMAND_SCORE_MAPPING: Dict[str, float] = {
    "Very Low Expected Demand": 0.10,
    "Low Expected Demand": 0.25,
    "Moderate Expected Demand": 0.5,
    "High Expected Demand": 0.75,
    "Very High Expected Demand": 1.0,
}

# ---------------------------------------------------------------------------
# Preference model (enabling attributes, betas) — use config file, not 30 env vars
# ---------------------------------------------------------------------------
PREFERENCE_BASE_CONSTANT: float = _f("PREFERENCE_BASE_CONSTANT", 0.5)
PREFERENCE_LEGACY_SCORE_SCALE: float = _f("PREFERENCE_LEGACY_SCORE_SCALE", 0.2)
PREFERENCE_SIGMOID_NUMERATOR: float = _f("PREFERENCE_SIGMOID_NUMERATOR", 4.0)

PREFERENCE_CONFIG: Dict[str, Any] = {
    "base_constant": PREFERENCE_BASE_CONSTANT,
    "attributes": {
        "earnings_per_month": {
            "enabled": _s("PREF_ENABLE_EARNINGS", "true").lower() in ("1", "true", "yes"),
            "type": "ordered_linear",
            "beta": 0.5,
            "mapping": {
                "earn_15k": 0.10,
                "earn_30k": 0.33,
                "earn_50k": 0.67,
                "earn_70k": 1.0,
            },
        },
        "task_content": {
            "enabled": _s("PREF_ENABLE_TASK_CONTENT", "false").lower() in ("1", "true", "yes"),
            "type": "dummy",
            "beta": 0.2,
            "active_level": "task_creative",
        },
        "physical_demand": {
            "enabled": _s("PREF_ENABLE_PHYSICAL_DEMAND", "true").lower() in ("1", "true", "yes"),
            "type": "dummy",
            "beta": -0.4,
            "active_level": "phys_heavy",
        },
        "work_flexibility": {
            "enabled": _s("PREF_ENABLE_WORK_FLEXIBILITY", "false").lower() in ("1", "true", "yes"),
            "type": "dummy",
            "beta": 0.4,
            "active_level": "flex_high",
        },
        "social_interaction": {
            "enabled": _s("PREF_ENABLE_SOCIAL", "true").lower() in ("1", "true", "yes"),
            "type": "dummy",
            "beta": 0.1,
            "active_level": "soc_people",
        },
        "career_growth": {
            "enabled": _s("PREF_ENABLE_CAREER_GROWTH", "true").lower() in ("1", "true", "yes"),
            "type": "dummy",
            "beta": 0.5,
            "active_level": "growth_high",
        },
        "social_meaning": {
            "enabled": _s("PREF_ENABLE_SOCIAL_MEANING", "false").lower() in ("1", "true", "yes"),
            "type": "dummy",
            "beta": 0.3,
            "active_level": "mean_high",
        },
    },
}

# ---------------------------------------------------------------------------
# Data files (server-side)
# ---------------------------------------------------------------------------
OCCUPATION_JSON_PATH: str = _s("OCCUPATION_JSON_PATH", str(_DEFAULT_OCC))

EMBEDDING_MODEL_PATH: str = _s("EMBEDDING_MODEL_PATH", str(_DEFAULT_MODEL_DIR / "skill_embedding_model.pt"))
SKILL_TO_ROW_PATH: str = _s("SKILL_TO_ROW_PATH", str(_DEFAULT_MODEL_DIR / "skill_to_row.json"))
_TAX = _RESOURCES / "skill_taxonomy"
SKILLS_CSV_PATH: str = _s("SKILLS_CSV_PATH", str(_TAX / "skills.csv"))
SKILL_GROUPS_CSV_PATH: str = _s("SKILL_GROUPS_CSV_PATH", str(_TAX / "skill_groups.csv"))
SKILL_HIERARCHY_CSV_PATH: str = _s("SKILL_HIERARCHY_CSV_PATH", str(_TAX / "skill_hierarchy.csv"))
