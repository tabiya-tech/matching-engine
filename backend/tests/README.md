# Testing Protocol — SHP Matching Algorithm

> **All checks listed below must pass before any code is pushed or a PR is merged.**

## Overview

The test suite validates three layers of the matching service:

| Layer | Directory | What it verifies | Needs server? |
|---|---|---|---|
| **Data Validation** | `tests/data_validation/` | Pydantic model shapes — required fields, defaults, validators | No |
| **Data Schema** | `tests/data_schema/` | API wiring — endpoint registration, auth boundaries, config rejection | No (uses mocked `TestClient`) |
| **Smoke** | `tests/smoke/` | Runtime behavior — health endpoint, payload guards, response contracts | No (uses mocked `TestClient`) |

In addition, static analysis checks enforce code quality:

| Check | What it verifies |
|---|---|
| **Lint** | No unused imports, variables, or code issues (`ruff check`) |
| **Formatter** | Consistent code style across all files (`ruff format --check`) |
| **Startup** | FastAPI boots and serves `/docs` + `/openapi.json` |

---

## Pre-Push / Pre-Merge Checklist

Run all commands from `backend/`:

```bash
cd shp-matching-algorithm/backend
```

### 1. Static Analysis

```bash
python tests/sanity_checks/lint_check.py
python tests/sanity_checks/formatter_check.py
```

- **Lint check**: reports unused imports, unused variables, and code issues. Must show `PASS`.
- **Formatter check**: verifies all files match `ruff format` style. Must show `PASS`.
- Both support `--fix` to auto-correct (use only locally, never in CI):
  ```bash
  python tests/sanity_checks/lint_check.py --fix
  python tests/sanity_checks/formatter_check.py --fix
  ```

### 2. Startup Check

```bash
python tests/sanity_checks/startup_check.py
```

Boots the FastAPI app via `TestClient` and verifies `/docs` and `/openapi.json` return HTTP 200. Requires `MONGO_URL` and `MONGO_DB_NAME` in `.env` or environment.

### 3. Data Validation Tests

```bash
python tests/sanity_checks/data_validation_check.py
```

Runs `pytest tests/data_validation/` which includes:

- **`test_request_validation.py`** — Input model validation
  - County suffix stripping (`"Nairobi County"` → `"Nairobi"`)
  - `MatchRequestV5` correctly inherits all `MatchRequest` fields
  - `zqf_level` defaults to `None`

- **`test_response_contracts.py`** — Output model contracts (guards the Swagger schema)
  - `MatchResponse.user_id` is required (not optional)
  - `OpportunityRecommendation` — all 8 required fields, all 16 optional fields default `None`
  - `OccupationRecommendation` — all 8 required fields, list defaults (`typical_tasks`, `career_path_next_steps`)
  - `SkillGapRecommendation` — all 6 required fields
  - `ScoreBreakdown` — all 11 fields exist and default `None`
  - `MatchedSkills` — sub-lists default to `[]`
  - `MatchedSkill` — required fields (`job_skill_id`, `similarity`, `meets_threshold`)
  - `MatchedPreference` — required fields (`attribute`, `user_weight`, `beta`, `encoded_value`, `contribution`, `matched`)
  - `MatchResponseV5` — mirrors V1 structure + `zqf_eligible`/`zqf_gap` on opportunities

### 4. Data Schema Tests

```bash
python tests/sanity_checks/data_schema_check.py
```

Runs `pytest tests/data_schema/` which includes:

- **`test_openapi_schema.py`** — API wiring
  - All 6 endpoints registered (`/health`, `/match`, `/experiments/v2/match`, `/experiments/v3/match`, `/match_v4`, `/experiments/v5/match`)
  - Correct HTTP methods (GET for health, POST for all match endpoints)
  - Auth boundaries: `/health` and `/match` require `x-api-key`; experiment endpoints are public

- **`test_config_validation.py`** — Configuration safety
  - Invalid `FINAL_SCORE_COMBINER` values are rejected at import time
  - Invalid `SCORING_MODE` values are rejected at import time

### 5. Smoke Tests

```bash
python tests/sanity_checks/smoke_check.py
```

Runs `pytest tests/smoke/` which includes:

- **`test_startup_smoke.py`** — Health endpoint
  - `GET /health` with `x-api-key` returns `200 {"status": "ok"}`
  - `GET /health` without `x-api-key` returns `401`

- **`test_endpoint_smoke.py`** — Endpoint behavior
  - Empty payload `[]` returns `400`
  - Invalid `final_score_combiner` query param returns `400`
  - `_zqf_annotation` logic: eligible, ineligible, missing user ZQF, missing job ZQF
  - Unified response contract: all match endpoints (`/match`, `/experiments/v2/match`, `/experiments/v3/match`, `/match_v4`, `/experiments/v5/match`) return `user_id` + three recommendation lists

### Run Everything at Once

```bash
python tests/sanity_checks/run_all_checks.py
```

Runs all 6 checks (lint, format, data validation, data schema, smoke, job dict mapping) and prints a one-line PASS/FAIL per check with a final summary. Paste this output into PR descriptions.

### 6. Job Dict Mapping Tests

```bash
python tests/sanity_checks/job_dict_mapping_check.py
```

Runs `pytest tests/unit/` which validates `build_job_dict_from_ranked()` — the Mongo ranked-job → flat job dict mapper used by every match endpoint.

**Coverage strategy** (not every field gets its own test):

- **Mapping logic** — full coverage: ZQF naming conventions (`min_zqf_level` vs `zqf_min`), province/county fallback, `originUuid` precedence, posted-date chain, embedding dim gate, skill ID filtering, etc.
- **Simple passthrough** — one happy-path test asserts core `classifier_metadata` fields (`title`, `employer`, `salary`, ISCO, URL, …) map correctly together.

To run pytest directly with verbose output:

```bash
python -m pytest tests/data_validation/ tests/data_schema/ tests/smoke/ tests/unit/ -v
```
---

## File Structure

```
tests/
├── conftest.py                          # Shared fixtures, mocking (TestClient, env setup)
├── README.md                            # This file
├── data_validation/
│   ├── test_request_validation.py       # Input model tests (6 tests)
│   └── test_response_contracts.py       # Output model tests (22 tests)
├── data_schema/
│   ├── test_openapi_schema.py           # Endpoint + auth tests (2 tests)
│   └── test_config_validation.py        # Config rejection tests (2 tests)
├── smoke/
│   ├── test_startup_smoke.py            # Health endpoint tests (2 tests)
│   └── test_endpoint_smoke.py           # Endpoint behavior tests (7 tests)
├── unit/
│   └── test_build_job_dict_from_ranked.py  # Mongo job doc → flat dict mapping
└── sanity_checks/
    ├── data_validation_check.py         # Runner: pytest tests/data_validation/
    ├── data_schema_check.py             # Runner: pytest tests/data_schema/
    ├── smoke_check.py                   # Runner: pytest tests/smoke/
    ├── job_dict_mapping_check.py        # Runner: pytest tests/unit/
    ├── lint_check.py                    # Runner: ruff check
    ├── formatter_check.py               # Runner: ruff format --check
    └── startup_check.py                 # Runner: FastAPI boot check
```

---

## Adding New Tests

When adding new endpoints, models, or features:

1. **New Pydantic model or field** → Add contract tests in `tests/data_validation/test_response_contracts.py` or `test_request_validation.py`
2. **New endpoint** → Add it to `EXPECTED_ENDPOINTS` in `tests/data_schema/test_openapi_schema.py` and to `AUTH_REQUIRED_PATHS` if it requires `x-api-key`
3. **New config enum** → Add a parameterized case in `tests/data_schema/test_config_validation.py`
4. **New endpoint behavior** → Add smoke tests in `tests/smoke/test_endpoint_smoke.py`
5. **New Mongo field mapping or fallback in `build_job_dict_from_ranked`** → Add cases in `tests/unit/test_build_job_dict_from_ranked.py` (mapping logic) or extend the happy-path test (simple passthrough)

Always run the full suite after changes to confirm nothing regresses.
