from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

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

# GET /jobs cursor-paginated browse endpoint: default and hard-cap page sizes.
JOBS_PAGE_DEFAULT_LIMIT: int = _i("JOBS_PAGE_DEFAULT_LIMIT", 20)
JOBS_PAGE_MAX_LIMIT: int = _i("JOBS_PAGE_MAX_LIMIT", 100)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
SCORING_MODE: str = _s("SCORING_MODE", "multiplicative")
if SCORING_MODE not in ("multiplicative", "additive"):
    raise ValueError("SCORING_MODE must be 'multiplicative' or 'additive'")

# How to combine u_hat and p_hat in "multiplicative" pipelines.
# - product:        final = u_hat * p_hat
# - geometric_mean: final = sqrt(u_hat * p_hat)
FINAL_SCORE_COMBINER: str = _s("FINAL_SCORE_COMBINER", "product").strip().lower()
if FINAL_SCORE_COMBINER not in ("product", "geometric_mean"):
    raise ValueError("FINAL_SCORE_COMBINER must be 'product' or 'geometric_mean'")

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

# --- /match_v4 tiered urban-pull location matching (see services/location_tiers.py) ---
# Kenyan job supply is heavily Nairobi-concentrated, so strict same-county filtering leaves non-hub
# users (e.g. Kitui ~2 jobs, Kilifi ~10) with near-empty lists. When enabled, a user's pool is widened
# to their hub chain (local -> regional hub -> national hub) AND v4 opportunities are softly re-ranked
# by a per-job location tier: local=1.0, regional=W_REGIONAL, national=W_NATIONAL, off-chain=0.0. So
# local jobs are preferred but a clearly-better hub job can still surface and scarce-local lists fill
# from the hub. Hub counties (national/regional) do not pull outward. Set false to fully restore the
# old strict pool + scoring (instant rollback). v4/v5 opportunities only; v1/v3/occupations unchanged.
LOCATION_TIER_ENABLED: bool = _b("LOCATION_TIER_ENABLED", True)
# Soft tier weights (cosine + final_score multipliers). Calibrated on live data; tunable. 0.70/0.50
# encodes a strong local preference: a regional-hub job must be ~1.4x better fit (1/0.70), a national
# job ~2x better (1/0.50), to outrank a local job. Lower = stronger local preference.
LOCATION_TIER_W_REGIONAL: float = _f("LOCATION_TIER_W_REGIONAL", 0.70)
LOCATION_TIER_W_NATIONAL: float = _f("LOCATION_TIER_W_NATIONAL", 0.50)
# County -> hub-chain exceptions (small JSON; every other county defaults to [self, national_hub]).
LOCATION_HUB_CHAINS_PATH: str = _resolve_under_backend(
    _s("LOCATION_HUB_CHAINS_PATH", str(_RESOURCES / "location" / "location_hub_chains.json"))
)

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
    "Below Average Expected Demand": 0.40,
    "Moderate Expected Demand": 0.5,
    "Above Average Expected Demand": 0.60,
    "High Expected Demand": 0.75,
    "Very High Expected Demand": 1.0,
    "Extremely High Expected Demand": 1.0,
}

# ---------------------------------------------------------------------------
# Preference model (enabling attributes, betas) — use config file, not 30 env vars
# ---------------------------------------------------------------------------
PREFERENCE_BASE_CONSTANT: float = _f("PREFERENCE_BASE_CONSTANT", 0.5)
PREFERENCE_LEGACY_SCORE_SCALE: float = _f("PREFERENCE_LEGACY_SCORE_SCALE", 0.2)
PREFERENCE_SIGMOID_NUMERATOR: float = _f("PREFERENCE_SIGMOID_NUMERATOR", 4.0)

# Unified DCE+BWS additive-RUM scorer (default) vs the old beta-config scorer (A/B escape hatch).
PREFERENCE_SCORER_MODE: str = _s("PREFERENCE_SCORER_MODE", "unified").strip().lower()
# "hybrid_v1" is the former name for the v1 scorer that the unified scorer replaces; accept it
# as a deprecated alias so a stale env value doesn't crash startup.
if PREFERENCE_SCORER_MODE == "hybrid_v1":
    PREFERENCE_SCORER_MODE = "unified"
if PREFERENCE_SCORER_MODE not in ("unified", "legacy"):
    raise ValueError("PREFERENCE_SCORER_MODE must be 'unified' or 'legacy'")

HYBRID_PREF_SIGMOID_FACTOR: float = _f("HYBRID_PREF_SIGMOID_FACTOR", 2.646)
HYBRID_PREF_VIGNETTES_FOR_FULL_CONFIDENCE: int = _i(
    "HYBRID_PREF_VIGNETTES_FOR_FULL_CONFIDENCE", 10
)
_hybrid_schema = _s("HYBRID_PREF_SCHEMA_PATH", "").strip()
HYBRID_PREF_SCHEMA_PATH: str = _hybrid_schema if _hybrid_schema else ""

# BWS work-activity preference integration into u_hat (additive-RUM).
# u_hat = logistic(gamma * [alpha * V_task_hat + (1-alpha) * V_dce_hat]), each component in [-1,1].
# alpha = task-vs-DCE weight (0.5 = equal after scale-harmonisation); gamma = logistic gain.
BWS_ALPHA: float = min(1.0, max(0.0, _f("BWS_ALPHA", 0.5)))
BWS_GAIN_GAMMA: float = _f("BWS_GAIN_GAMMA", 4.0)
# "additive_rum" (new, default) | "legacy" (old importance x level x bws sum/mean) — escape hatch
# for A/B comparison and the alpha sensitivity sweep.
BWS_INTEGRATION_MODE: str = _s("BWS_INTEGRATION_MODE", "additive_rum").strip().lower()
if BWS_INTEGRATION_MODE not in ("additive_rum", "legacy"):
    raise ValueError("BWS_INTEGRATION_MODE must be 'additive_rum' or 'legacy'")

# DCE-attribute utility (unified scorer). The per-user value v_k = sigmoid(beta_k) is
# inverted via beta_hat = logit(clamp(v_k, eps, 1-eps)); eps bounds extreme values.
DCE_LOGIT_EPS: float = _f("DCE_LOGIT_EPS", 0.01)
# Optional per-attribute multiplicative scale on beta_hat (JSON object string), e.g.
# '{"earnings_per_month": 5.0}' to compensate the near-neutral continuous earnings term.
# Default: no scaling (every attribute = 1.0).
import json as _json  # noqa: E402

_dce_scale_raw = _s("DCE_ATTR_SCALE", "")
try:
    DCE_ATTR_SCALE: Dict[str, float] = (
        {str(k): float(v) for k, v in _json.loads(_dce_scale_raw).items()}
        if _dce_scale_raw
        else {}
    )
except (ValueError, TypeError, AttributeError):
    DCE_ATTR_SCALE = {}

# --- /match_v4 full-response (MatchResponse via the Gemini-embeddings engine) ---
# Per-skill threshold for MatchedSkill.meets_threshold / essential-coverage, in the WHITENED+rescaled
# space (see V4_FULL_EMBEDDING_MODEL_PATH). Calibrate properly (2026-06-11 skill-eligibility notes);
# ~0.45 separates same-field (~0.6) from unrelated (~0.27) on current data. NOTE: this is the
# rescaled-whitened scale, NOT the old raw-Gemini cosine — do not reuse the historical 0.6.
V4_FULL_SIM_THRESHOLD: float = _f("V4_FULL_SIM_THRESHOLD", 0.45)
# Min essential-coverage (share of a job's essential skills the user meets) for is_eligible=True.
# Recalibrated 2026-06-18: 0.5 -> 0.38. The old 0.5 bar flagged genuine in-field matches ineligible
# (electrician cov 0.39 for an "Electrician" posting; masseuse 0.44 for a "Massage Specialist") — and
# the mean coverage of parsed jobs is only ~0.39, i.e. the TYPICAL real match sat below the bar. 0.38
# admits both anchor cases; a live 38-user sweep moved aggregate eligible share 32% -> 43% (the
# per-skill SIM bar stays at 0.45 to avoid loosening what counts as a match). Env-tunable for rollback.
V4_FULL_MIN_ESS_SHARE: float = _f("V4_FULL_MIN_ESS_SHARE", 0.38)
# Graded skill-match badge bands on essential-coverage in [0,1] (strong / partial / weak).
V4_FULL_BADGE_STRONG: float = _f("V4_FULL_BADGE_STRONG", 0.7)
V4_FULL_BADGE_PARTIAL: float = _f("V4_FULL_BADGE_PARTIAL", 0.4)
# Kill-switch for /match_v4's per-skill GATE. True (default) = whitened matcher + one-to-one
# assignment + exact-id (the new meaningful gate). False = revert the per-skill detail to the legacy
# raw matcher (score_pair, max-over-user, no rescale) — the old saturated behaviour — for fast rollback.
V4_FULL_WHITENED_GATE: bool = _b("V4_FULL_WHITENED_GATE", True)
# --- /match_v4 Phase 2: ranking demotion (whitened-concat p_hat + essential-coverage factor) ---
# Master toggle. True (default) = p_hat's skills-fit is the WHITENED+rescaled concat cosine AND
# final_score is demoted by essential-coverage**gamma (achievability); final stays u_hat x p_hat.
# False = Phase-1 behaviour: p_hat is the (unrescaled) stage-1 concat cosine and the gate is
# annotation-only (ranking unchanged) — INSTANT ROLLBACK via env V4_FULL_RANK_DEMOTE=false, no redeploy.
# NOTE: stage-1 retrieval now ranks in the WHITENED concat space whenever the artifact is present
# (job_embedding is whitened on the DB side, or whitened in-process for raw corpora), independent of
# this toggle — see match_concat_gemini_ce_service. So Phase-1's p_hat is the whitened stage-1 cosine.
V4_FULL_RANK_DEMOTE: bool = _b("V4_FULL_RANK_DEMOTE", True)
# Demotion strength: p_hat *= essential_coverage ** gamma (0 -> no demotion; 1 -> linear). 1.0 = full
# achievability ordering (gamma sweep: corr(rank,cov) -0.42, cov@1 0.88, weak items at top ~0). Env-tunable.
V4_FULL_COVERAGE_GAMMA: float = _f("V4_FULL_COVERAGE_GAMMA", 1.0)
# Ranking coverage for postings with NO parsed essential skills (unparsed). essential_coverage()
# returns 1.0 for these, which previously gave them a demotion-free ride to the top of the v4 ranking
# even with zero genuine skill overlap (2026-06 nail-tech failure: a masseuse's top-10 was entirely
# unparsed white-collar jobs). Default (-1.0) = treat an unparsed job as a TYPICAL job: in the RANKING
# path, substitute the live mean of the parsed-job coverages in this user's shortlist (per pool). A
# value in [0,1] overrides with a fixed constant (1.0 reproduces the old free-passage behaviour ->
# instant rollback). DISPLAY essential_coverage / is_eligible are unchanged. Pilot mean parsed
# coverage ~0.39 (opportunities) / ~0.43 (occupations).
V4_FULL_UNPARSED_COVERAGE: float = _f("V4_FULL_UNPARSED_COVERAGE", -1.0)
# Whitening transform for the COMBINED (concat) embedding used by the whitened p_hat skills-fit,
# built by build_whitened_concat.py (mu, W=Sigma^-1/2, target). Refit on the live corpus for prod.
V4_FULL_CONCAT_WHITENING_PATH: str = _resolve_under_backend(
    _s(
        "V4_FULL_CONCAT_WHITENING_PATH",
        str(_DEFAULT_MODEL_DIR / "concat_whitening_gemini.npz"),
    )
)
# /match_v4 shortlist sizing (v4-only; v3/zqf keep COSINE_CROSS_ENCODER_RETRIEVE_TOP_K/30). Stage-1 now
# ranks in the WHITENED concat space (the RAW concat cosine was near-uninformative, sd ~0.02); the
# whitened p_hat + coverage re-rank is then applied to the final_top_k CE survivors. Widen both so
# achievable jobs can reach the whitened re-rank: retrieve feeds the CE, final is the pool sent to
# whitening (and the max opportunities returned). Whitening itself is cheap (shortlist-only matmuls);
# the cost is CE rerank (~retrieve_top_k) — monitor latency and dial back via env if needed.
MATCH_V4_RETRIEVE_TOP_K: int = _i("MATCH_V4_RETRIEVE_TOP_K", 100)
MATCH_V4_FINAL_TOP_K: int = _i("MATCH_V4_FINAL_TOP_K", 50)
# Top-k occupations returned by /match_v4's full response.
MATCH_V4_TOP_K_OCCUPATIONS: int = _i("MATCH_V4_TOP_K_OCCUPATIONS", 10)
# Demand tilt applied to /match_v4 OCCUPATION final_score ONLY (opportunities are never tilted):
# final *= M ** gamma, with M = expected-demand score in [0,1] (neutral 1.0 when absent/unknown).
# Mirrors the legacy p_hat market factor (PHAT_GAMMA_MARKET=0.3). Set 0.0 to disable.
MATCH_V4_OCC_DEMAND_GAMMA: float = _f("MATCH_V4_OCC_DEMAND_GAMMA", 0.3)
# Committed NPZ of occupation concat-Gemini embeddings (codes + float32 vectors), built offline
# by app.services.cross_encoder.embed_occupations. Missing => occupations skipped (logged).
OCCUPATION_CONCAT_EMBEDDINGS_PATH: str = _s(
    "OCCUPATION_CONCAT_EMBEDDINGS_PATH",
    str(_RESOURCES / "occupations" / "occupation_concat_embeddings.npz"),
)

PREFERENCE_CONFIG: Dict[str, Any] = {
    "base_constant": PREFERENCE_BASE_CONSTANT,
    "attributes": {
        "earnings_per_month": {
            "enabled": _s("PREF_ENABLE_EARNINGS", "true").lower()
            in ("1", "true", "yes"),
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
            "enabled": _s("PREF_ENABLE_TASK_CONTENT", "false").lower()
            in ("1", "true", "yes"),
            "type": "dummy",
            "beta": 0.2,
            "active_level": "task_creative",
        },
        "physical_demand": {
            "enabled": _s("PREF_ENABLE_PHYSICAL_DEMAND", "true").lower()
            in ("1", "true", "yes"),
            "type": "dummy",
            "beta": -0.4,
            "active_level": "phys_heavy",
        },
        "work_flexibility": {
            "enabled": _s("PREF_ENABLE_WORK_FLEXIBILITY", "false").lower()
            in ("1", "true", "yes"),
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
            "enabled": _s("PREF_ENABLE_CAREER_GROWTH", "true").lower()
            in ("1", "true", "yes"),
            "type": "dummy",
            "beta": 0.5,
            "active_level": "growth_high",
        },
        "social_meaning": {
            "enabled": _s("PREF_ENABLE_SOCIAL_MEANING", "false").lower()
            in ("1", "true", "yes"),
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
    _s(
        "EMBEDDING_MODEL_PATH",
        str(_DEFAULT_MODEL_DIR / "skill_embedding_model_gemini.pt"),
    )
)
# /match_v4 per-skill GATE uses the WHITENED skill artifact (de-anisotropised; carries its own
# rescale target in metadata). Kept separate from the shared EMBEDDING_MODEL_PATH so v2/v3 + the
# Node2Vec/skill-gap/hybrid paths are untouched.
V4_FULL_EMBEDDING_MODEL_PATH: str = _resolve_under_backend(
    _s(
        "V4_FULL_EMBEDDING_MODEL_PATH",
        str(_DEFAULT_MODEL_DIR / "skill_embedding_model_gemini_whitened.pt"),
    )
)
SKILL_TO_ROW_PATH: str = _resolve_under_backend(
    _s("SKILL_TO_ROW_PATH", str(_DEFAULT_MODEL_DIR / "skill_to_row.json"))
)
_TAX = _RESOURCES / "skill_taxonomy"
SKILLS_CSV_PATH: str = _s("SKILLS_CSV_PATH", str(_TAX / "skills.csv"))
SKILL_GROUPS_CSV_PATH: str = _s("SKILL_GROUPS_CSV_PATH", str(_TAX / "skill_groups.csv"))
SKILL_HIERARCHY_CSV_PATH: str = _s(
    "SKILL_HIERARCHY_CSV_PATH", str(_TAX / "skill_hierarchy.csv")
)

DEBUG_MODE = True
