# Settings reference

`app/config.py` defines `Settings`, a [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
model that reads and validates every environment variable the matching service's application code
depends on at import time. A field's env var name is its uppercase name (e.g. `scoring_mode` binds
to `SCORING_MODE`); matching is case-insensitive.

`app.config` keeps the historical UPPERCASE module constants (e.g. `MATCH_TOP_K_SKILL_GAPS`) as
aliases onto a module-level `settings` instance, so existing call sites (`from app.config import
MATCH_TOP_K_SKILL_GAPS`) are unaffected. New code may import either form; prefer `from app.config
import settings` and `settings.match_top_k_skill_gaps` going forward.

Four path-shaped variables are deliberately **not** `Settings` fields: `SKILLS_CSV_PATH`,
`SKILL_GROUPS_CSV_PATH`, `SKILL_HIERARCHY_CSV_PATH` and `OCCUPATION_JSON_PATH`. They're read live
from the environment by `taxonomy_pack_paths()` / `occupation_json_path()` on every call instead,
because scripts intentionally override them per-invocation after `app.config` (and its `settings`
singleton) has already been imported — caching them on `settings` would make that override silently
not work. See their docstrings in `app/config.py`.

See `backend/.env.example` for a copy-pasteable `.env` with the same variables and more
operational commentary; this document is the authoritative list of defaults and types.

## Mongo / job retrieval

| Env var | Field | Type | Default |
|---|---|---|---|
| `MONGO_JOBS_COLLECTION` | `mongo_jobs_collection` | str | `RankedJobsEnriched` |
| `JOBS_RETRIEVAL_FILTER` | `jobs_retrieval_filter` | bool | `true` |
| `JOBS_RETRIEVAL_LIMIT` | `jobs_retrieval_limit` | int | `10000` |
| `JOBS_FIND_USE_PROJECTION` | `jobs_find_use_projection` | bool | `true` |
| `JOBS_PAGE_DEFAULT_LIMIT` | `jobs_page_default_limit` | int | `20` |
| `JOBS_PAGE_MAX_LIMIT` | `jobs_page_max_limit` | int | `100` |

Mongo connection settings themselves (`MONGO_URL`, `MONGO_DB_NAME`, `MONGO_*_POOL_SIZE`, TLS
options) are read in `app/database.py`, not `app/config.py` — out of scope for this model.

## Modes

| Env var | Field | Type | Default |
|---|---|---|---|
| `SCORING_MODE` | `scoring_mode` | `"multiplicative"` \| `"additive"` | `multiplicative` |
| `FINAL_SCORE_COMBINER` | `final_score_combiner` | `"product"` \| `"geometric_mean"` | `product` |
| `ADDITIVE_W1_SKILLS` | `additive_w1_skills` | float, range `[0, 1]` | `0.40` |
| `ADDITIVE_W2_PREFERENCE` | `additive_w2_preference` | float, range `[0, 1]` | `0.40` |
| `ADDITIVE_W3_MARKET` | `additive_w3_market` | float, range `[0, 1]` | `0.20` |

An invalid `SCORING_MODE` or `FINAL_SCORE_COMBINER` fails import with a `pydantic.ValidationError`
(a `ValueError` subclass) — the service refuses to start rather than silently mis-score.

The three additive weights must also sum to `1.0` (checked by a cross-field validator) — the
additive scoring mode consumes them as an un-normalized weighted sum
(`final_score = w1*U + w2*pref + w3*demand`), so a mismatched total would otherwise silently
rescale every score instead of failing.

## Match output sizes & reranking

| Env var | Field | Type | Default |
|---|---|---|---|
| `MATCH_TOP_K_OPPORTUNITIES` | `match_top_k_opportunities` | int | `5` |
| `MATCH_TOP_K_OCCUPATIONS` | `match_top_k_occupations` | int | `5` |
| `MATCH_TOP_K_SKILL_GAPS` | `match_top_k_skill_gaps` | int | `5` |
| `CROSS_ENCODER_MODEL_NAME` | `cross_encoder_model_name` | str | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `CROSS_ENCODER_BATCH_SIZE` | `cross_encoder_batch_size` | int | `16` |
| `COSINE_CROSS_ENCODER_RETRIEVE_TOP_K` | `cosine_cross_encoder_retrieve_top_k` | int | `50` |
| `MATCH_V2_HYBRID_TOP_K` | `match_v2_hybrid_top_k` | int | `20` |
| `MATCH_V2_MAX_USERS_PER_REQUEST` | `match_v2_max_users_per_request` | int | `32` |
| `MATCH_APPLY_LOCATION_FILTER` | `match_apply_location_filter` | bool | `true` |
| `MATCH_V4_DISABLE_OCCUPATIONS` | `match_v4_disable_occupations` | bool | `false` |

Per-language cross-encoder checkpoints use `CROSS_ENCODER_MODEL_NAME_<LANG>` (e.g.
`CROSS_ENCODER_MODEL_NAME_ES`), read by `cross_encoder_model_name(language)` via the language
registry (`app/languages/`), not a `Settings` field.

## Location tiers (`/match_v4`)

| Env var | Field | Type | Default |
|---|---|---|---|
| `LOCATION_TIER_ENABLED` | `location_tier_enabled` | bool | `true` |
| `LOCATION_TIER_W_REGIONAL` | `location_tier_w_regional` | float, range `[0, 1]` | `0.70` |
| `LOCATION_TIER_W_NATIONAL` | `location_tier_w_national` | float, range `[0, 1]` | `0.50` |
| `LOCATION_HUB_CHAINS_PATH` | `location_hub_chains_path` | path | `resources/location/location_hub_chains.json` |

## Success propensity / gates

| Env var | Field | Type | Default |
|---|---|---|---|
| `GATE_SIMILARITY_THRESHOLD` | `gate_similarity_threshold` | float, range `[0, 1]` | `0.10` |
| `MATCH_RESPONSE_SKILL_MIN_SCORE` | `match_response_skill_min_score` | float \| none, range `[0, 1]` | `gate_similarity_threshold` |
| `PHAT_ALPHA_ESSENTIAL` | `phat_alpha_essential` | float, `>= 0` | `0.5` |
| `PHAT_BETA_READINESS` | `phat_beta_readiness` | float, `>= 0` | `0.2` |
| `PHAT_GAMMA_MARKET` | `phat_gamma_market` | float, `>= 0` | `0.3` |

`match_response_skill_min_score` defaults to whatever `gate_similarity_threshold` resolves to if
left unset — set both explicitly if you want them to diverge.

## Skill utility / feasibility (U)

`skill_u_w_loc` / `_ess` / `_opt` / `_grp` feed a self-normalizing weighted average (divided by
their own sum in `skills_match.py`), so unlike the additive-mode weights above they do **not**
need to sum to 1 — only non-negative, since a negative weight would invert that term's
contribution.

| Env var | Field | Type | Default |
|---|---|---|---|
| `SKILL_U_W_LOC` | `skill_u_w_loc` | float, `>= 0` | `0.20` |
| `SKILL_U_W_ESS` | `skill_u_w_ess` | float, `>= 0` | `0.50` |
| `SKILL_U_W_OPT` | `skill_u_w_opt` | float, `>= 0` | `0.20` |
| `SKILL_U_W_GRP` | `skill_u_w_grp` | float, `>= 0` | `0.10` |
| `SKILL_U_GAP_PENALTY` | `skill_u_gap_penalty` | float, `>= 0` | `0.25` |
| `SKILL_U_TAU_ELIG` | `skill_u_tau_elig` | float, range `[0, 1]` | `0.35` |
| `SKILL_MIN_ESSENTIAL_MATCH_SHARE` | `skill_min_essential_match_share` | float, range `[0, 1]` | `1.0` |
| `SKILL_ESSENTIAL_GEO_FLOOR` | `skill_essential_geo_floor` | float, range `[0, 1]` | `0.000001` |
| `SKILL_ESSENTIAL_DAMPING_ALPHA` | `skill_essential_damping_alpha` | float, `>= 0` | `0.0` |
| `SKILL_RESCALE_TARGET` | `skill_rescale_target` | float, `>= 0` | `0.0` (disabled sentinel) |

## Preference model

| Env var | Field | Type | Default |
|---|---|---|---|
| `PREFERENCE_BASE_CONSTANT` | `preference_base_constant` | float | `0.5` |
| `PREFERENCE_LEGACY_SCORE_SCALE` | `preference_legacy_score_scale` | float | `0.2` |
| `PREFERENCE_SIGMOID_NUMERATOR` | `preference_sigmoid_numerator` | float | `4.0` |
| `PREFERENCE_SCORER_MODE` | `preference_scorer_mode` | `"unified"` \| `"legacy"` | `unified` |
| `HYBRID_PREF_SIGMOID_FACTOR` | `hybrid_pref_sigmoid_factor` | float | `2.646` |
| `HYBRID_PREF_VIGNETTES_FOR_FULL_CONFIDENCE` | `hybrid_pref_vignettes_for_full_confidence` | int | `10` |
| `HYBRID_PREF_SCHEMA_PATH` | `hybrid_pref_schema_path` | str | `""` (defaults to the built-in schema) |
| `BWS_ALPHA` | `bws_alpha` | float, range `[0, 1]` | `0.5` |
| `BWS_GAIN_GAMMA` | `bws_gain_gamma` | float | `4.0` |
| `BWS_INTEGRATION_MODE` | `bws_integration_mode` | `"additive_rum"` \| `"legacy"` | `additive_rum` |
| `DCE_LOGIT_EPS` | `dce_logit_eps` | float, open range `(0, 0.5)` | `0.01` |
| `DCE_ATTR_SCALE` | `dce_attr_scale` | JSON object string → `dict[str, float]` | `{}` |
| `PREF_ENABLE_EARNINGS` | `pref_enable_earnings` | bool | `true` |
| `PREF_ENABLE_TASK_CONTENT` | `pref_enable_task_content` | bool | `false` |
| `PREF_ENABLE_PHYSICAL_DEMAND` | `pref_enable_physical_demand` | bool | `true` |
| `PREF_ENABLE_WORK_FLEXIBILITY` | `pref_enable_work_flexibility` | bool | `false` |
| `PREF_ENABLE_SOCIAL` | `pref_enable_social` | bool | `true` |
| `PREF_ENABLE_CAREER_GROWTH` | `pref_enable_career_growth` | bool | `true` |
| `PREF_ENABLE_SOCIAL_MEANING` | `pref_enable_social_meaning` | bool | `false` |

`PREFERENCE_SCORER_MODE=hybrid_v1` is accepted as a deprecated alias for `unified` so a stale env
value doesn't crash startup.

## `/match_v4` full response

| Env var | Field | Type | Default |
|---|---|---|---|
| `V4_FULL_SIM_THRESHOLD` | `v4_full_sim_threshold` | float, range `[0, 1]` | `0.45` |
| `V4_FULL_MIN_ESS_SHARE` | `v4_full_min_ess_share` | float, range `[0, 1]` | `0.38` |
| `V4_FULL_BADGE_STRONG` | `v4_full_badge_strong` | float, range `[0, 1]` | `0.7` |
| `V4_FULL_BADGE_PARTIAL` | `v4_full_badge_partial` | float, range `[0, 1]` | `0.4` |
| `V4_FULL_WHITENED_GATE` | `v4_full_whitened_gate` | bool | `true` |
| `V4_FULL_RANK_DEMOTE` | `v4_full_rank_demote` | bool | `true` |
| `V4_FULL_COVERAGE_GAMMA` | `v4_full_coverage_gamma` | float, `>= 0` | `1.0` |
| `V4_FULL_COVERAGE_FLOOR` | `v4_full_coverage_floor` | float, range `[0, 1]` | `0.05` |
| `V4_FULL_UNPARSED_COVERAGE` | `v4_full_unparsed_coverage` | float (unbounded — `-1.0` is a sentinel) | `-1.0` (live mean of parsed coverages) |
| `V4_FULL_CONCAT_WHITENING_PATH` | `v4_full_concat_whitening_path` | path | `resources/models/concat_whitening_gemini.npz` |
| `MATCH_V4_RETRIEVE_TOP_K` | `match_v4_retrieve_top_k` | int | `100` |
| `MATCH_V4_FINAL_TOP_K` | `match_v4_final_top_k` | int | `50` |
| `MATCH_V4_TOP_K_OCCUPATIONS` | `match_v4_top_k_occupations` | int | `10` |
| `MATCH_V4_OCC_DEMAND_GAMMA` | `match_v4_occ_demand_gamma` | float, `>= 0` | `0.3` |
| `OCCUPATION_CONCAT_EMBEDDINGS_PATH` | `occupation_concat_embeddings_path` | path | `resources/occupations/occupation_concat_embeddings.npz` |

## Data files

| Env var | `Settings` field | Type | Default |
|---|---|---|---|
| `OCCUPATION_JSON_PATH` | *(none — read live, see above)* | path | `resources/occupations/en/combined_occupation_database_with_wa.json` |
| `EMBEDDING_MODEL_PATH` | `embedding_model_path` | path | `resources/models/skill_embedding_model_gemini.pt` |
| `V4_FULL_EMBEDDING_MODEL_PATH` | `v4_full_embedding_model_path` | path | `resources/models/skill_embedding_model_gemini_whitened.pt` |
| `SKILL_TO_ROW_PATH` | `skill_to_row_path` | path | `resources/models/skill_to_row.json` |
| `SKILLS_CSV_PATH` | *(none — read live, see above)* | path | `""` (per-language layout; see below) |
| `SKILL_GROUPS_CSV_PATH` | *(none — read live, see above)* | path | `""` |
| `SKILL_HIERARCHY_CSV_PATH` | *(none — read live, see above)* | path | `""` |

Setting any of the three taxonomy CSV paths pins **every** language to that one file — a deployment
footgun (it silently takes the canonical embedding id space with it). Leave them empty in any real
deployment; they exist for scripts that deliberately want a single pack.

A relative path in any `*_path` field resolves against `backend/`, with a leading `backend/`
segment stripped first (so a repo-root-relative path still works when the process's cwd is
`backend/`).

## Misc

| Env var | Field | Type | Default |
|---|---|---|---|
| `DEBUG_MODE` | `debug_mode` | bool | `true` |

## Per-stack overrides

Deployed env vars come from two layers of GitHub Actions configuration on the `.github/workflows/deploy.yml`
workflow, passed through to `pulumi up` and from there into the Cloud Run service's env:

1. **Repository variables** (`vars.*` / `secrets.*` at the repo level) apply to every stack unless
   overridden below.
2. **Environment variables/secrets** (configured per GitHub Environment: `dev`, `shp`, `zmb`, `sa`,
   `eth`, `argentina`) override the repo-level value of the same name for that stack only.

Every secret (`GEMINI_API_KEY`, `MONGO_URL`, `MONGO_DB_NAME`, `MONGO_JOBS_COLLECTION`, and for
`argentina` also the Langfuse keys) is set per-environment, since each stack points at its own
Mongo cluster and job collection.

Repo-level variables currently set (applied to every stack except where an environment overrides
them):

| Variable | Value | Differs from code default? |
|---|---|---|
| `ADDITIVE_W1_SKILLS` | `0.40` | no |
| `ADDITIVE_W3_MARKET` | `0.20` | no |
| `EMBEDDING_MODEL_PATH` | `resources/models/skill_embedding_model.pt` | **yes** — the legacy Node2Vec artefact, not the Gemini default |
| `GATE_SIMILARITY_THRESHOLD` | `0.35` | **yes** — code default is `0.10` |
| `JOBS_RETRIEVAL_FILTER` | `true` | no |
| `MATCH_TOP_K_OCCUPATIONS` | `5` | no |
| `MATCH_TOP_K_OPPORTUNITIES` | `5` | no |
| `MATCH_TOP_K_SKILL_GAPS` | `5` | no |
| `OCCUPATION_JSON_PATH` | `resources/occupations/en/combined_occupation_database_with_wa.json` | no (same path the default resolves to) |
| `PHAT_ALPHA_ESSENTIAL` | `0.5` | no |

Environment-specific overrides, applied on top of the repo-level variables above:

| Variable | `dev` | `shp` | `zmb` | `sa` | `eth` | `argentina` |
|---|---|---|---|---|---|---|
| `JOBS_RETRIEVAL_FILTER` | — | — | — | — | — | `false` |
| `LOCATION_TIER_ENABLED` | — | — | — | — | — | `false` |
| `MATCH_V4_DISABLE_OCCUPATIONS` | — | — | — | — | — | `true` |
| `OCCUPATION_JSON_PATH` | — | — | — | — | — | `resources/occupations/en/combined_occupation_database_with_wa.json` |
| `SKILLS_CSV_PATH` | — | — | — | — | — | `resources/skill_taxonomy/es/skills.csv` |
| `SKILL_GROUPS_CSV_PATH` | — | — | — | — | — | `resources/skill_taxonomy/es/skill_groups.csv` |
| `SKILL_HIERARCHY_CSV_PATH` | — | — | — | — | — | `resources/skill_taxonomy/es/skill_hierarchy.csv` |
| `MATCHING_ENABLE_TRACING` | — | — | — | — | — | `1` |
| `MATCHING_LANGFUSE_HOST` | — | — | — | — | — | `https://cloud.langfuse.com` |
| `MATCHING_TRACING_CONFIG` | — | — | — | — | — | `{}` |

(`—` = no override; the stack runs on the repo-level variable or the code default.) `dev`, `shp`,
`zmb`, `sa` and `eth` currently run entirely on repo-level variables and code defaults — only
`argentina` carries its own overrides, matching its Spanish-language deployment and its own
Langfuse tracing project.

To check the current live values yourself (requires repo admin access):

```bash
gh api repos/tabiya-tech/matching-engine/actions/variables --jq '.variables[] | "\(.name)=\(.value)"'
gh api repos/tabiya-tech/matching-engine/environments/<stack>/variables --jq '.variables[] | "\(.name)=\(.value)"'
```

This table is a snapshot; re-run the commands above before relying on it for an incident or a
deploy decision.
