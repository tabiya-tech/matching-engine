from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.languages import CANONICAL_LANGUAGE, language_setting

load_dotenv()

_BACKEND_ROOT = Path(__file__).resolve().parent.parent  # .../backend
_RESOURCES = _BACKEND_ROOT / "resources"
_OCC_BASENAME = "combined_occupation_database_with_wa.json"


def _default_occupation_json() -> Path:
    """Canonical-language occupation database, falling back to the flat legacy layout."""
    pack = _RESOURCES / "occupations" / CANONICAL_LANGUAGE / _OCC_BASENAME
    if pack.is_file():
        return pack
    return _RESOURCES / "occupations" / _OCC_BASENAME


_DEFAULT_OCC = _default_occupation_json()
_DEFAULT_MODEL_DIR = _RESOURCES / "models"


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


class Settings(BaseSettings):
    """
    Typed, validated application settings, read from environment variables (and `.env`).

    Field names are the lowercase form of the environment variable they bind to
    (e.g. `scoring_mode` <- `SCORING_MODE`); pydantic-settings matches env var names
    case-insensitively by default.
    """

    model_config = SettingsConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _blank_env_values_are_unset(cls, data: Any) -> Any:
        """A blank env var (`FOO=`) means "unset", same as `_s`/`_f`/`_i`/`_b` in the pre-settings
        config: a deployment that comments out a value but leaves the `KEY=` line should fall back
        to the field's default rather than fail validation on an empty string."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not (isinstance(v, str) and v.strip() == "")}
        return data

    # -----------------------------------------------------------------------
    # Mongo — job source (enriched collection from migration / reranker)
    # -----------------------------------------------------------------------
    # Default: RankedJobsEnriched (see scripts/enrich_ranked_jobs_to_new_collection.py). Set to RankedJobs for legacy.
    mongo_jobs_collection: str = "RankedJobsEnriched"

    # HTTP /match: when set, load at most N jobs with is_active + (remote OR per-user location),
    # as a superset of matching_service._job_matches_user_location. Set to 0 to disable
    # the extra filter and load all active jobs (scripts, back-compat).
    jobs_retrieval_filter: bool = True
    jobs_retrieval_limit: int = 10_000
    # Mongo find() inclusion projection (fields used by build_job_dict_from_ranked). Set 0 to load full documents.
    jobs_find_use_projection: bool = True

    # GET /jobs cursor-paginated browse endpoint: default and hard-cap page sizes.
    jobs_page_default_limit: int = 20
    jobs_page_max_limit: int = 100

    # -----------------------------------------------------------------------
    # Modes
    # -----------------------------------------------------------------------
    scoring_mode: Literal["multiplicative", "additive"] = "multiplicative"

    # How to combine u_hat and p_hat in "multiplicative" pipelines.
    # - product:        final = u_hat * p_hat
    # - geometric_mean: final = sqrt(u_hat * p_hat)
    final_score_combiner: Literal["product", "geometric_mean"] = "product"

    # Additive (legacy) weights. Consumed as an UN-normalized weighted sum
    # (final_score = w1*U + w2*pref + w3*demand, see matching_service.py), so these three must
    # sum to 1.0 or the additive score lands on the wrong scale with no error raised.
    additive_w1_skills: float = Field(default=0.40, ge=0.0, le=1.0)
    additive_w2_preference: float = Field(default=0.40, ge=0.0, le=1.0)
    additive_w3_market: float = Field(default=0.20, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _additive_weights_sum_to_one(self) -> Settings:
        total = self.additive_w1_skills + self.additive_w2_preference + self.additive_w3_market
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "additive_w1_skills + additive_w2_preference + additive_w3_market must sum to "
                f"1.0 (got {total}); the additive scoring mode uses them as an un-normalized "
                "weighted sum, so a mismatched total silently rescales every final_score."
            )
        return self

    # -----------------------------------------------------------------------
    # Match output sizes
    # -----------------------------------------------------------------------
    match_top_k_opportunities: int = 5
    match_top_k_occupations: int = 5
    match_top_k_skill_gaps: int = 5

    # Cosine batch runner cross-encoder rerank (see cross_encoder.reranker, run_cosine_matching).
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_batch_size: int = 16
    cosine_cross_encoder_retrieve_top_k: int = 50

    # POST /match_v2 — BM25 × cosine hybrid pool fusion (see hybrid_scoring.run_bm25_cosine_hybrid).
    match_v2_hybrid_top_k: int = 20
    match_v2_max_users_per_request: int = 32

    # When false, skip _job_matches_user_location in _match_items (opportunities and occupations).
    # Mongo prefilter is separate: jobs_retrieval_filter. Default true keeps current behaviour.
    match_apply_location_filter: bool = True

    # Per-deployment kill-switch for occupation recommendations on the v4 engine (POST /match, via
    # run_match_v4_full): every user gets an empty occupation_recommendations list AND the whole
    # occupation pipeline is skipped (corpus load, embedding attach, stage-1 retrieval, CE rerank,
    # Phase-2 overrides). Opportunities and skill gaps are unaffected. Default false keeps current
    # behaviour.
    match_v4_disable_occupations: bool = False

    # --- /match_v4 tiered urban-pull location matching (see services/location_tiers.py) ---
    # Kenyan job supply is heavily Nairobi-concentrated, so strict same-county filtering leaves non-hub
    # users (e.g. Kitui ~2 jobs, Kilifi ~10) with near-empty lists. When enabled, a user's pool is widened
    # to their hub chain (local -> regional hub -> national hub) AND v4 opportunities are softly re-ranked
    # by a per-job location tier: local=1.0, regional=W_REGIONAL, national=W_NATIONAL, off-chain=0.0. So
    # local jobs are preferred but a clearly-better hub job can still surface and scarce-local lists fill
    # from the hub. Hub counties (national/regional) do not pull outward. Set false to fully restore the
    # old strict pool + scoring (instant rollback). v4/v5 opportunities only; v1/v3/occupations unchanged.
    location_tier_enabled: bool = True
    # Soft tier weights (cosine + final_score multipliers). Calibrated on live data; tunable. 0.70/0.50
    # encodes a strong local preference: a regional-hub job must be ~1.4x better fit (1/0.70), a national
    # job ~2x better (1/0.50), to outrank a local job. Lower = stronger local preference.
    location_tier_w_regional: float = Field(default=0.70, ge=0.0, le=1.0)
    location_tier_w_national: float = Field(default=0.50, ge=0.0, le=1.0)
    # County -> hub-chain exceptions (small JSON; every other county defaults to [self, national_hub]).
    location_hub_chains_path: str = str(_RESOURCES / "location" / "location_hub_chains.json")

    # -----------------------------------------------------------------------
    # Success propensity  (p_hat = G * E^alpha * R^beta * M^gamma)
    # -----------------------------------------------------------------------
    gate_similarity_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    # Response filtering threshold for scored skill outputs:
    # - essential_skill_matches.similarity in opportunities/occupations
    # - skill_gap_recommendations.proximity_score
    # Defaults to gate_similarity_threshold when unset.
    match_response_skill_min_score: float | None = Field(default=None, ge=0.0, le=1.0)

    # Exponents in p_hat = G * E^alpha * R^beta * M^gamma (success_propensity.py). A negative
    # exponent on a base near 0 blows up toward infinity, so these must stay non-negative.
    phat_alpha_essential: float = Field(default=0.5, ge=0.0)
    phat_beta_readiness: float = Field(default=0.2, ge=0.0)
    phat_gamma_market: float = Field(default=0.3, ge=0.0)

    # -----------------------------------------------------------------------
    # Skill utility (U) / feasibility — aligned with skills_match
    #
    # w_loc/ess/opt/grp feed a self-normalizing weighted average (divided by their own sum in
    # skills_match.py), so they need not sum to 1 — only non-negative, since a negative weight
    # would invert that term's contribution. gap_penalty multiplies a [0,1] gap_share and is
    # subtracted from core_score, so it's non-negative too. tau_elig and geo_floor are compared
    # against / floor similarity values that live in [0,1].
    # -----------------------------------------------------------------------
    skill_u_w_loc: float = Field(default=0.20, ge=0.0)
    skill_u_w_ess: float = Field(default=0.50, ge=0.0)
    skill_u_w_opt: float = Field(default=0.20, ge=0.0)
    skill_u_w_grp: float = Field(default=0.10, ge=0.0)
    skill_u_gap_penalty: float = Field(default=0.25, ge=0.0)
    skill_u_tau_elig: float = Field(default=0.35, ge=0.0, le=1.0)
    skill_min_essential_match_share: float = Field(default=1.0, ge=0.0, le=1.0)
    skill_essential_geo_floor: float = Field(default=1e-6, ge=0.0, le=1.0)
    # Score-weighted geometric mean for essential_fit: weight each row-max by (rowmax ** alpha).
    # alpha=0 -> uniform weights -> recovers the naive GM (default, current behaviour).
    # alpha>0 -> low scores contribute proportionally less; near-zero scores self-abstain via x^a*ln(x) -> 0.
    skill_essential_damping_alpha: float = Field(default=0.0, ge=0.0)

    # -----------------------------------------------------------------------
    # Per-rowmax rescaling target for whitened cosines.
    # -----------------------------------------------------------------------
    # Whitened cosines compress the discriminative band: identity = 1.0 (tautological),
    # strong non-identity sits at ~0.2-0.3, random ~0.0. The score-weighted GM on raw
    # rowmaxes inherits a bimodality (identity vs everything else) that compresses the
    # downstream final_score range. Per-rowmax rescaling — divide each rowmax by SKILL_
    # RESCALE_TARGET, clip at 1.0 — stretches the [0, target] band into [0, 1] so identity
    # and strong non-identity both contribute at the top of the GM input distribution.
    #
    # The natural anchor for skill_rescale_target is the upper edge of the non-identity
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
    skill_rescale_target: float = Field(default=0.0, ge=0.0)

    # -----------------------------------------------------------------------
    # Preference model (enabling attributes, betas) — use config file, not 30 env vars
    # -----------------------------------------------------------------------
    preference_base_constant: float = 0.5
    preference_legacy_score_scale: float = 0.2
    preference_sigmoid_numerator: float = 4.0

    # Unified DCE+BWS additive-RUM scorer (default) vs the old beta-config scorer (A/B escape hatch).
    # "hybrid_v1" is the former name for the v1 scorer that the unified scorer replaces; accepted
    # as a deprecated alias so a stale env value doesn't crash startup.
    preference_scorer_mode: Literal["unified", "legacy"] = "unified"

    hybrid_pref_sigmoid_factor: float = 2.646
    hybrid_pref_vignettes_for_full_confidence: int = 10
    hybrid_pref_schema_path: str = ""

    # BWS work-activity preference integration into u_hat (additive-RUM).
    # u_hat = logistic(gamma * [alpha * V_task_hat + (1-alpha) * V_dce_hat]), each component in [-1,1].
    # alpha = task-vs-DCE weight (0.5 = equal after scale-harmonisation); gamma = logistic gain.
    bws_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    bws_gain_gamma: float = 4.0
    # "additive_rum" (new, default) | "legacy" (old importance x level x bws sum/mean) — escape hatch
    # for A/B comparison and the alpha sensitivity sweep.
    bws_integration_mode: Literal["additive_rum", "legacy"] = "additive_rum"

    # DCE-attribute utility (unified scorer). The per-user value v_k = sigmoid(beta_k) is
    # inverted via beta_hat = logit(clamp(v_k, eps, 1-eps)); eps bounds extreme values. Must
    # stay strictly between 0 and 0.5, or the clamp's own [eps, 1-eps] range inverts.
    dce_logit_eps: float = Field(default=0.01, gt=0.0, lt=0.5)
    # Optional per-attribute multiplicative scale on beta_hat (JSON object string), e.g.
    # '{"earnings_per_month": 5.0}' to compensate the near-neutral continuous earnings term.
    # Default: no scaling (every attribute = 1.0).
    dce_attr_scale: dict[str, float] = Field(default_factory=dict)

    # Per-attribute toggles (true/false, 1/0, yes/no) for PREFERENCE_CONFIG.
    pref_enable_earnings: bool = True
    pref_enable_task_content: bool = False
    pref_enable_physical_demand: bool = True
    pref_enable_work_flexibility: bool = False
    pref_enable_social: bool = True
    pref_enable_career_growth: bool = True
    pref_enable_social_meaning: bool = False

    # --- /match_v4 full-response (MatchResponse via the Gemini-embeddings engine) ---
    # Per-skill threshold for MatchedSkill.meets_threshold / essential-coverage, in the WHITENED+rescaled
    # space (see v4_full_embedding_model_path). Calibrate properly (2026-06-11 skill-eligibility notes);
    # ~0.45 separates same-field (~0.6) from unrelated (~0.27) on current data. NOTE: this is the
    # rescaled-whitened scale, NOT the old raw-Gemini cosine — do not reuse the historical 0.6.
    v4_full_sim_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    # Min essential-coverage (share of a job's essential skills the user meets) for is_eligible=True.
    # Recalibrated 2026-06-18: 0.5 -> 0.38. The old 0.5 bar flagged genuine in-field matches ineligible
    # (electrician cov 0.39 for an "Electrician" posting; masseuse 0.44 for a "Massage Specialist") — and
    # the mean coverage of parsed jobs is only ~0.39, i.e. the TYPICAL real match sat below the bar. 0.38
    # admits both anchor cases; a live 38-user sweep moved aggregate eligible share 32% -> 43% (the
    # per-skill SIM bar stays at 0.45 to avoid loosening what counts as a match). Env-tunable for rollback.
    v4_full_min_ess_share: float = Field(default=0.38, ge=0.0, le=1.0)
    # Graded skill-match badge bands on essential-coverage in [0,1] (strong / partial / weak).
    v4_full_badge_strong: float = Field(default=0.7, ge=0.0, le=1.0)
    v4_full_badge_partial: float = Field(default=0.4, ge=0.0, le=1.0)
    # Kill-switch for /match_v4's per-skill GATE. True (default) = whitened matcher + one-to-one
    # assignment + exact-id (the new meaningful gate). False = revert the per-skill detail to the legacy
    # raw matcher (score_pair, max-over-user, no rescale) — the old saturated behaviour — for fast rollback.
    v4_full_whitened_gate: bool = True
    # --- /match_v4 Phase 2: ranking demotion (whitened-concat p_hat + essential-coverage factor) ---
    # Master toggle. True (default) = p_hat's skills-fit is the WHITENED+rescaled concat cosine AND
    # final_score is demoted by essential-coverage**gamma (achievability); final stays u_hat x p_hat.
    # False = Phase-1 behaviour: p_hat is the (unrescaled) stage-1 concat cosine and the gate is
    # annotation-only (ranking unchanged) — INSTANT ROLLBACK via env V4_FULL_RANK_DEMOTE=false, no redeploy.
    # NOTE: stage-1 retrieval now ranks in the WHITENED concat space whenever the artifact is present
    # (job_embedding is whitened on the DB side, or whitened in-process for raw corpora), independent of
    # this toggle — see match_concat_gemini_ce_service. So Phase-1's p_hat is the whitened stage-1 cosine.
    v4_full_rank_demote: bool = True
    # Demotion strength: p_hat *= essential_coverage ** gamma (0 -> no demotion; 1 -> linear). 1.0 = full
    # achievability ordering (gamma sweep: corr(rank,cov) -0.42, cov@1 0.88, weak items at top ~0). Env-tunable.
    v4_full_coverage_gamma: float = Field(default=1.0, ge=0.0)
    # Floor under the coverage demotion: factor = floor + (1 - floor) * coverage ** gamma. Un-floored,
    # a coverage of exactly 0 ANNIHILATES the score (u_hat x p_hat x 0 = 0.0), which is worse than a
    # strong demotion: every such candidate ties at final_score 0.0, so the u_hat x p_hat ordering is
    # lost (the response falls back to the p_hat tie-break) and the payload reads as broken — a non-zero
    # breakdown next to final_score 0.0. Zero coverage is not rare: a user whose skills all sit outside
    # a posting's essentials scores 0, and a whole shortlist scores 0 whenever skill labels cannot be
    # resolved at all (a taxonomy pack pinned off the embedding id space — see skill_label_packs). The
    # floor keeps the demotion monotone in coverage while leaving the score interpretable.
    # 0.0 restores the un-floored behaviour exactly.
    v4_full_coverage_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    # Ranking coverage for postings with NO parsed essential skills (unparsed). essential_coverage()
    # returns 1.0 for these, which previously gave them a demotion-free ride to the top of the v4 ranking
    # even with zero genuine skill overlap (2026-06 nail-tech failure: a masseuse's top-10 was entirely
    # unparsed white-collar jobs). Default (-1.0) = treat an unparsed job as a TYPICAL job: in the RANKING
    # path, substitute the live mean of the parsed-job coverages in this user's shortlist (per pool). A
    # value in [0,1] overrides with a fixed constant (1.0 reproduces the old free-passage behaviour ->
    # instant rollback). DISPLAY essential_coverage / is_eligible are unchanged. Pilot mean parsed
    # coverage ~0.39 (opportunities) / ~0.43 (occupations).
    v4_full_unparsed_coverage: float = -1.0
    # Whitening transform for the COMBINED (concat) embedding used by the whitened p_hat skills-fit,
    # built by build_whitened_concat.py (mu, W=Sigma^-1/2, target). Refit on the live corpus for prod.
    v4_full_concat_whitening_path: str = str(_DEFAULT_MODEL_DIR / "concat_whitening_gemini.npz")
    # /match_v4 shortlist sizing (v4-only; v3/zqf keep cosine_cross_encoder_retrieve_top_k/30). Stage-1 now
    # ranks in the WHITENED concat space (the RAW concat cosine was near-uninformative, sd ~0.02); the
    # whitened p_hat + coverage re-rank is then applied to the final_top_k CE survivors. Widen both so
    # achievable jobs can reach the whitened re-rank: retrieve feeds the CE, final is the pool sent to
    # whitening (and the max opportunities returned). Whitening itself is cheap (shortlist-only matmuls);
    # the cost is CE rerank (~retrieve_top_k) — monitor latency and dial back via env if needed.
    match_v4_retrieve_top_k: int = 100
    match_v4_final_top_k: int = 50
    # Top-k occupations returned by /match_v4's full response.
    match_v4_top_k_occupations: int = 10
    # Demand tilt applied to /match_v4 OCCUPATION final_score ONLY (opportunities are never tilted):
    # final *= M ** gamma, with M = expected-demand score in [0,1] (neutral 1.0 when absent/unknown).
    # Mirrors the legacy p_hat market factor (phat_gamma_market=0.3). Set 0.0 to disable.
    match_v4_occ_demand_gamma: float = Field(default=0.3, ge=0.0)
    # Committed NPZ of occupation concat-Gemini embeddings (codes + float32 vectors), built offline
    # by app.services.cross_encoder.embed_occupations. Missing => occupations skipped (logged).
    occupation_concat_embeddings_path: str = str(_RESOURCES / "occupations" / "occupation_concat_embeddings.npz")

    # -----------------------------------------------------------------------
    # Data files (server-side)
    # -----------------------------------------------------------------------
    # Canonical-language occupation database. Use `occupation_json_path(language)` when the
    # language matters (labels); this constant stays for the many language-neutral call sites.
    occupation_json_path: str = str(_DEFAULT_OCC)
    embedding_model_path: str = str(_DEFAULT_MODEL_DIR / "skill_embedding_model_gemini.pt")
    # /match_v4 per-skill GATE uses the WHITENED skill artifact (de-anisotropised; carries its own
    # rescale target in metadata). Kept separate from the shared embedding_model_path so v2/v3 + the
    # Node2Vec/skill-gap/hybrid paths are untouched.
    v4_full_embedding_model_path: str = str(_DEFAULT_MODEL_DIR / "skill_embedding_model_gemini_whitened.pt")
    skill_to_row_path: str = str(_DEFAULT_MODEL_DIR / "skill_to_row.json")

    # SKILLS_CSV_PATH / SKILL_GROUPS_CSV_PATH / SKILL_HIERARCHY_CSV_PATH are intentionally NOT
    # fields here: `taxonomy_pack_paths()` below reads them live from the environment on every
    # call (scripts override them per-invocation, after this module — and this singleton — has
    # already been imported), so caching them on `settings` would make that override silently
    # not work. See `taxonomy_pack_paths`.

    debug_mode: bool = True

    @field_validator("preference_scorer_mode", mode="before")
    @classmethod
    def _preference_scorer_mode_alias(cls, v: Any) -> Any:
        """`hybrid_v1` is the former name for the v1 scorer that `unified` replaces."""
        if isinstance(v, str) and v.strip().lower() == "hybrid_v1":
            return "unified"
        return v

    @field_validator(
        "scoring_mode",
        "final_score_combiner",
        "preference_scorer_mode",
        "bws_integration_mode",
        mode="before",
    )
    @classmethod
    def _lowercase_and_strip(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator(
        "location_hub_chains_path",
        "v4_full_concat_whitening_path",
        "embedding_model_path",
        "v4_full_embedding_model_path",
        "skill_to_row_path",
    )
    @classmethod
    def _resolve_path_under_backend(cls, v: str) -> str:
        return _resolve_under_backend(v)

    @field_validator("dce_attr_scale", mode="before")
    @classmethod
    def _parse_dce_attr_scale(cls, v: Any) -> Any:
        if not isinstance(v, str):
            return v
        if not v.strip():
            return {}
        try:
            return {str(k): float(val) for k, val in json.loads(v).items()}
        except (ValueError, TypeError, AttributeError):
            return {}

    @model_validator(mode="after")
    def _default_match_response_skill_min_score(self) -> Settings:
        if self.match_response_skill_min_score is None:
            self.match_response_skill_min_score = self.gate_similarity_threshold
        return self


settings = Settings()

# ---------------------------------------------------------------------------
# Backward-compatible module-level constants.
# Every call site in the codebase imports these UPPERCASE names directly
# (e.g. `from app.config import MATCH_TOP_K_SKILL_GAPS`); keep them as plain
# aliases onto the validated `settings` singleton above so none of those call
# sites need to change.
# ---------------------------------------------------------------------------
MONGO_JOBS_COLLECTION = settings.mongo_jobs_collection

JOBS_RETRIEVAL_FILTER = settings.jobs_retrieval_filter
JOBS_RETRIEVAL_LIMIT = settings.jobs_retrieval_limit
JOBS_FIND_USE_PROJECTION = settings.jobs_find_use_projection

JOBS_PAGE_DEFAULT_LIMIT = settings.jobs_page_default_limit
JOBS_PAGE_MAX_LIMIT = settings.jobs_page_max_limit

SCORING_MODE = settings.scoring_mode
FINAL_SCORE_COMBINER = settings.final_score_combiner

GLOBAL_WEIGHTS: dict[str, float] = {
    "w1_skills": settings.additive_w1_skills,
    "w2_preference": settings.additive_w2_preference,
    "w3_market": settings.additive_w3_market,
}

MATCH_TOP_K_OPPORTUNITIES = settings.match_top_k_opportunities
MATCH_TOP_K_OCCUPATIONS = settings.match_top_k_occupations
MATCH_TOP_K_SKILL_GAPS = settings.match_top_k_skill_gaps

CROSS_ENCODER_MODEL_NAME = settings.cross_encoder_model_name
CROSS_ENCODER_BATCH_SIZE = settings.cross_encoder_batch_size
COSINE_CROSS_ENCODER_RETRIEVE_TOP_K = settings.cosine_cross_encoder_retrieve_top_k

MATCH_V2_HYBRID_TOP_K = settings.match_v2_hybrid_top_k
MATCH_V2_MAX_USERS_PER_REQUEST = settings.match_v2_max_users_per_request

MATCH_APPLY_LOCATION_FILTER = settings.match_apply_location_filter
MATCH_V4_DISABLE_OCCUPATIONS = settings.match_v4_disable_occupations

LOCATION_TIER_ENABLED = settings.location_tier_enabled
LOCATION_TIER_W_REGIONAL = settings.location_tier_w_regional
LOCATION_TIER_W_NATIONAL = settings.location_tier_w_national
LOCATION_HUB_CHAINS_PATH = settings.location_hub_chains_path

GATE_SIMILARITY_THRESHOLD = settings.gate_similarity_threshold
MATCH_RESPONSE_SKILL_MIN_SCORE = settings.match_response_skill_min_score

SUCCESS_PROPENSITY_CONFIG: dict[str, Any] = {
    "alpha_essential": settings.phat_alpha_essential,
    "beta_readiness": settings.phat_beta_readiness,
    "gamma_market": settings.phat_gamma_market,
    "gate_threshold": GATE_SIMILARITY_THRESHOLD,
}

SKILL_U_W_LOC = settings.skill_u_w_loc
SKILL_U_W_ESS = settings.skill_u_w_ess
SKILL_U_W_OPT = settings.skill_u_w_opt
SKILL_U_W_GRP = settings.skill_u_w_grp
SKILL_U_GAP_PENALTY = settings.skill_u_gap_penalty
SKILL_U_TAU_ELIG = settings.skill_u_tau_elig
SKILL_MIN_ESSENTIAL_MATCH_SHARE = settings.skill_min_essential_match_share
SKILL_ESSENTIAL_GEO_FLOOR = settings.skill_essential_geo_floor
SKILL_ESSENTIAL_DAMPING_ALPHA = settings.skill_essential_damping_alpha

SKILL_RESCALE_TARGET = settings.skill_rescale_target

DEMAND_SCORE_MAPPING: dict[str, float] = {
    "Very Low Expected Demand": 0.10,
    "Low Expected Demand": 0.25,
    "Below Average Expected Demand": 0.40,
    "Moderate Expected Demand": 0.5,
    "Above Average Expected Demand": 0.60,
    "High Expected Demand": 0.75,
    "Very High Expected Demand": 1.0,
    "Extremely High Expected Demand": 1.0,
}

PREFERENCE_BASE_CONSTANT = settings.preference_base_constant
PREFERENCE_LEGACY_SCORE_SCALE = settings.preference_legacy_score_scale
PREFERENCE_SIGMOID_NUMERATOR = settings.preference_sigmoid_numerator

PREFERENCE_SCORER_MODE = settings.preference_scorer_mode

HYBRID_PREF_SIGMOID_FACTOR = settings.hybrid_pref_sigmoid_factor
HYBRID_PREF_VIGNETTES_FOR_FULL_CONFIDENCE = settings.hybrid_pref_vignettes_for_full_confidence
HYBRID_PREF_SCHEMA_PATH = settings.hybrid_pref_schema_path

BWS_ALPHA = settings.bws_alpha
BWS_GAIN_GAMMA = settings.bws_gain_gamma
BWS_INTEGRATION_MODE = settings.bws_integration_mode

DCE_LOGIT_EPS = settings.dce_logit_eps
DCE_ATTR_SCALE: dict[str, float] = settings.dce_attr_scale

V4_FULL_SIM_THRESHOLD = settings.v4_full_sim_threshold
V4_FULL_MIN_ESS_SHARE = settings.v4_full_min_ess_share
V4_FULL_BADGE_STRONG = settings.v4_full_badge_strong
V4_FULL_BADGE_PARTIAL = settings.v4_full_badge_partial
V4_FULL_WHITENED_GATE = settings.v4_full_whitened_gate
V4_FULL_RANK_DEMOTE = settings.v4_full_rank_demote
V4_FULL_COVERAGE_GAMMA = settings.v4_full_coverage_gamma
V4_FULL_COVERAGE_FLOOR = settings.v4_full_coverage_floor
V4_FULL_UNPARSED_COVERAGE = settings.v4_full_unparsed_coverage
V4_FULL_CONCAT_WHITENING_PATH = settings.v4_full_concat_whitening_path

MATCH_V4_RETRIEVE_TOP_K = settings.match_v4_retrieve_top_k
MATCH_V4_FINAL_TOP_K = settings.match_v4_final_top_k
MATCH_V4_TOP_K_OCCUPATIONS = settings.match_v4_top_k_occupations
MATCH_V4_OCC_DEMAND_GAMMA = settings.match_v4_occ_demand_gamma
OCCUPATION_CONCAT_EMBEDDINGS_PATH = settings.occupation_concat_embeddings_path

PREFERENCE_CONFIG: dict[str, Any] = {
    "base_constant": PREFERENCE_BASE_CONSTANT,
    "attributes": {
        "earnings_per_month": {
            "enabled": settings.pref_enable_earnings,
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
            "enabled": settings.pref_enable_task_content,
            "type": "dummy",
            "beta": 0.2,
            "active_level": "task_creative",
        },
        "physical_demand": {
            "enabled": settings.pref_enable_physical_demand,
            "type": "dummy",
            "beta": -0.4,
            "active_level": "phys_heavy",
        },
        "work_flexibility": {
            "enabled": settings.pref_enable_work_flexibility,
            "type": "dummy",
            "beta": 0.4,
            "active_level": "flex_high",
        },
        "social_interaction": {
            "enabled": settings.pref_enable_social,
            "type": "dummy",
            "beta": 0.1,
            "active_level": "soc_people",
        },
        "career_growth": {
            "enabled": settings.pref_enable_career_growth,
            "type": "dummy",
            "beta": 0.5,
            "active_level": "growth_high",
        },
        "social_meaning": {
            "enabled": settings.pref_enable_social_meaning,
            "type": "dummy",
            "beta": 0.3,
            "active_level": "mean_high",
        },
    },
}

OCCUPATION_JSON_PATH = settings.occupation_json_path
EMBEDDING_MODEL_PATH = settings.embedding_model_path
V4_FULL_EMBEDDING_MODEL_PATH = settings.v4_full_embedding_model_path
SKILL_TO_ROW_PATH = settings.skill_to_row_path

DEBUG_MODE = settings.debug_mode

_TAX = _RESOURCES / "skill_taxonomy"


def _tax_dir(language: str) -> Path:
    """``resources/skill_taxonomy/<language>/``, falling back to the flat legacy layout.

    The flat fallback keeps a checkout that predates language packs (or a deployment with
    a mounted taxonomy directory) working unchanged.
    """
    subdir = language_setting(language, "resources_subdir", language) or language
    candidate = _TAX / subdir
    if candidate.is_dir():
        return candidate
    return _TAX


def taxonomy_pack_paths(language: str, *, ignore_pins: bool = False) -> dict[str, str]:
    """The three taxonomy CSVs for one language.

    ``SKILLS_CSV_PATH`` / ``SKILL_GROUPS_CSV_PATH`` / ``SKILL_HIERARCHY_CSV_PATH`` pin all
    languages to one file when set — used by scripts that deliberately want a single pack.
    That pin is a footgun in a deployment: pointing it at a non-canonical pack silently takes
    the canonical id space with it, so ``ignore_pins=True`` returns the per-language paths the
    layout implies. ``skill_label_packs`` uses it to recover from exactly that misconfiguration.

    Read live from the environment (not the ``settings`` singleton): scripts intentionally set
    these per-invocation, after ``app.config`` has already been imported.
    """
    directory = _tax_dir(language)

    def pick(key: str, filename: str) -> str:
        default = str(directory / filename)
        pinned = (os.getenv(key) or "").strip()
        return default if (ignore_pins or not pinned) else pinned

    return {
        "skills": pick("SKILLS_CSV_PATH", "skills.csv"),
        "skill_groups": pick("SKILL_GROUPS_CSV_PATH", "skill_groups.csv"),
        "skill_hierarchy": pick("SKILL_HIERARCHY_CSV_PATH", "skill_hierarchy.csv"),
    }


def occupation_json_path(language: str = CANONICAL_LANGUAGE) -> str:
    """Occupation database for a language, falling back to the canonical one.

    Its work-activity and ISCO fields are language-neutral; only the labels differ, so a
    language without its own copy is correctly served by the canonical file.

    Reads ``OCCUPATION_JSON_PATH`` live from the environment, independent of the
    ``settings`` singleton, so a script can override it per-invocation after import.
    """
    raw = (os.getenv("OCCUPATION_JSON_PATH") or "").strip()
    if raw:
        return raw
    subdir = language_setting(language, "resources_subdir", language) or language
    candidate = _RESOURCES / "occupations" / subdir / "combined_occupation_database_with_wa.json"
    if candidate.is_file():
        return str(candidate)
    return str(_DEFAULT_OCC)


def cross_encoder_model_name(language: str = CANONICAL_LANGUAGE) -> str:
    """Cross-encoder checkpoint for a language (``CROSS_ENCODER_MODEL_NAME_<LANG>``).

    The reranker scores skill-label text, so an English-only checkpoint on Spanish labels
    scores poorly — hence one per language rather than one globally.
    """
    return str(language_setting(language, "cross_encoder_model", CROSS_ENCODER_MODEL_NAME))


def stopwords(language: str = CANONICAL_LANGUAGE) -> frozenset[str]:
    """BM25 / hybrid tokenisation stopwords for a language."""
    return frozenset(language_setting(language, "stopwords", ()) or ())


# Canonical-language paths, kept as module constants because most call sites are
# language-neutral (the embedding id space is the canonical language's).
_CANONICAL_TAX = taxonomy_pack_paths(CANONICAL_LANGUAGE)
SKILLS_CSV_PATH: str = _CANONICAL_TAX["skills"]
SKILL_GROUPS_CSV_PATH: str = _CANONICAL_TAX["skill_groups"]
SKILL_HIERARCHY_CSV_PATH: str = _CANONICAL_TAX["skill_hierarchy"]
