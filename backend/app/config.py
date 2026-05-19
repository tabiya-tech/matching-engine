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


def _resolve_under_backend(raw: str) -> str:
    """Resolve a path configured in .env relative to ``backend/``.

    * Absolute paths → ``Path.resolve()`` as-is.
    * Leading ``backend/`` is stripped (common when mixing repo-root-relative paths with
      a cwd of ``backend/``, which would otherwise look for ``backend/backend/...``).
    """

    p = Path(raw.strip()).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    parts = p.parts
    if parts and parts[0] == "backend":
        p = Path(*parts[1:])
    return str((_BACKEND_ROOT / p).resolve())



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
# Default: RankedJobsEnriched (see scripts/enrich_ranked_jobs_to_new_collection.py). Set to RankedJobs for legacy.
MONGO_JOBS_COLLECTION: str = _s("MONGO_JOBS_COLLECTION", "RankedJobsEnriched")

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

# Cosine batch runner cross-encoder rerank (see cross_encoder.reranker, run_cosine_matching).
CROSS_ENCODER_MODEL_NAME: str = _s(
    "CROSS_ENCODER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
CROSS_ENCODER_BATCH_SIZE: int = _i("CROSS_ENCODER_BATCH_SIZE", 16)
COSINE_CROSS_ENCODER_RETRIEVE_TOP_K: int = _i("COSINE_CROSS_ENCODER_RETRIEVE_TOP_K", 50)

# POST /match_v2 — BM25 × cosine hybrid pool fusion (see hybrid_scoring.run_bm25_cosine_hybrid).
MATCH_V2_HYBRID_TOP_K: int = _i("MATCH_V2_HYBRID_TOP_K", 20)
MATCH_V2_MAX_USERS_PER_REQUEST: int = _i("MATCH_V2_MAX_USERS_PER_REQUEST", 32)

# When false, skip _job_matches_user_location in _match_items (opportunities and occupations).
# Mongo prefilter is separate: JOBS_RETRIEVAL_FILTER. Default true keeps current behaviour.
MATCH_APPLY_LOCATION_FILTER: bool = _b("MATCH_APPLY_LOCATION_FILTER", True)

# ---------------------------------------------------------------------------
# Success propensity  (p_hat = G * E^alpha * R^beta * M^gamma)
# ---------------------------------------------------------------------------
GATE_SIMILARITY_THRESHOLD: float = _f("GATE_SIMILARITY_THRESHOLD", 0.10)
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
SKILL_U_GAP_PENALTY: float = _f("SKILL_U_GAP_PENALTY", 0.25)
SKILL_U_TAU_ELIG: float = _f("SKILL_U_TAU_ELIG", 0.35)
SKILL_MIN_ESSENTIAL_MATCH_SHARE: float = _f("SKILL_MIN_ESSENTIAL_MATCH_SHARE", 1.0)
SKILL_ESSENTIAL_GEO_FLOOR: float = _f("SKILL_ESSENTIAL_GEO_FLOOR", 1e-6)
# Score-weighted geometric mean for essential_fit: weight each row-max by (rowmax ** alpha).
# alpha=0 -> uniform weights -> recovers the naive GM (default, current behaviour).
# alpha>0 -> low scores contribute proportionally less; near-zero scores self-abstain via x^a*ln(x) -> 0.
SKILL_ESSENTIAL_DAMPING_ALPHA: float = _f("SKILL_ESSENTIAL_DAMPING_ALPHA", 0.0)

# ---------------------------------------------------------------------------
# Per-rowmax rescaling target for whitened cosines.
# ---------------------------------------------------------------------------
# Whitened cosines compress the discriminative band: identity = 1.0 (tautological),
# strong non-identity sits at ~0.2-0.3, random ~0.0. The score-weighted GM on raw
# rowmaxes inherits a bimodality (identity vs everything else) that compresses the
# downstream final_score range. Per-rowmax rescaling — divide each rowmax by SKILL_
# RESCALE_TARGET, clip at 1.0 — stretches the [0, target] band into [0, 1] so identity
# and strong non-identity both contribute at the top of the GM input distribution.
#
# The natural anchor for SKILL_RESCALE_TARGET is the upper edge of the non-identity
# distribution in the embedding (e.g. p99.9 over random pairs). The whitening artefact
# build script computes this and persists it in the .pt metadata as
# state["whitening"]["target_max_p999"]; SkillScorer.__init__ reads that value and
# sets this default at startup. Setting SKILL_RESCALE_TARGET in env overrides the
# artefact's value (useful for ad-hoc calibration without rebuilding).
#
# Default of 0.0 is the "disabled" sentinel — if no artefact provides target_max_p999
# AND no env override is set, rescaling is a no-op (rowmax_rescaled == rowmax). This
# makes raw Gemini and Node2Vec artefacts behave as if rescaling didn't exist, which
# is the right default because those artefacts have differently-shaped cosine
# distributions and don't benefit from this particular rescaling.
SKILL_RESCALE_TARGET: float = _f("SKILL_RESCALE_TARGET", 0.0)

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

EMBEDDING_MODEL_PATH: str = _resolve_under_backend(
    _s("EMBEDDING_MODEL_PATH", str(_DEFAULT_MODEL_DIR / "skill_embedding_model_gemini.pt"))
)
SKILL_TO_ROW_PATH: str = _resolve_under_backend(
    _s("SKILL_TO_ROW_PATH", str(_DEFAULT_MODEL_DIR / "skill_to_row.json"))
)
_TAX = _RESOURCES / "skill_taxonomy"
SKILLS_CSV_PATH: str = _s("SKILLS_CSV_PATH", str(_TAX / "skills.csv"))
SKILL_GROUPS_CSV_PATH: str = _s("SKILL_GROUPS_CSV_PATH", str(_TAX / "skill_groups.csv"))
SKILL_HIERARCHY_CSV_PATH: str = _s("SKILL_HIERARCHY_CSV_PATH", str(_TAX / "skill_hierarchy.csv"))
