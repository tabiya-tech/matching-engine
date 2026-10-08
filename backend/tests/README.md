# Testing Protocol — SHP Matching Algorithm

> **All checks listed below must pass before any code is pushed or a PR is merged.**

## Overview

The test suite validates three layers of the matching service:

| Layer | Directory | What it verifies | Needs server? |
|---|---|---|---|
| **Data Validation** | `tests/data_validation/` | Pydantic model shapes — required fields, defaults, validators | No |
| **Data Schema** | `tests/data_schema/` | API wiring — endpoint registration, auth boundaries, config rejection, docs served | No (uses mocked `TestClient`) |
| **Smoke** | `tests/smoke/` | Runtime behavior — health endpoint, payload guards, response contracts | No (uses mocked `TestClient`) |
| **ML logic** | `tests/ml_logic/`, `tests/components/`, `tests/integration/` | Matching invariants, metamorphic rules, mocked-embedding ranking | No |

In addition, these checks inspect the code itself (no running server or tests needed):

| Check | What it verifies |
|---|---|
| **Lint** | No unused imports, variables, or code issues; `Any` banned in function parameters/return types (`ruff check`, rule `ANN401`) |
| **Formatter** | Consistent code style across all files (`ruff format --check`) |
| **Type Check** | Strict static typing (`pyright`, `typeCheckingMode: strict`) |

These checks run automatically on every push via `.github/workflows/main.yml` (which calls `backend-ci.yml`).

---

## Pre-Push / Pre-Merge Checklist

Run all commands from `backend/`. One-time setup, if you haven't already:

```bash
python -m venv venv            # use Python 3.11 specifically — see note below
source venv/bin/activate
pip install -r requirements-dev.txt
```

> **Note:** this project targets Python 3.11 (see `backend/Dockerfile` and `backend/pyproject.toml`). If your system's default `python3`/`python` is a different version, create the venv with the 3.11 binary explicitly, e.g. `python3.11 -m venv venv`, otherwise installing dependencies may fail trying to compile packages that only ship pre-built wheels for 3.11.

### 1. Lint, Format, and Type Checks

These checks read the code without running it — no server, no database, no test data needed. All three default to checking the current directory when run with no path argument (run these from `backend/`).

**Check only — none of these commands change any files:**

```bash
ruff check                                        # lint: reports issues only
ruff format --check                               # format: reports which files are misformatted only
pyright                                            # strict type check: reports type errors only
```

- **Lint** (`ruff check`): reports unused imports, unused variables, code issues, and `Any` used as a function parameter or return type.
- **Formatter** (`ruff format --check`): reports which files don't match `ruff format` style. The `--check` flag is what makes it report-only — without it, `ruff format` would rewrite the files instead.
- **Type check** (`pyright`): runs `pyright` in strict mode (config in `backend/pyproject.toml` under `[tool.pyright]`). There is no auto-fix for type errors — a human has to resolve them.

**Auto-fix — these commands do change files. Use only locally, never in CI:**

```bash
ruff check --fix   # rewrites files: auto-fixes safe lint issues (e.g. removes unused imports)
ruff format         # rewrites files: reformats everything to match the style
```

#### Strictness Policy: `Any` usage

`Any` is banned in function parameter and return type annotations via ruff's `ANN401` rule (enabled in `backend/pyproject.toml` under `[tool.ruff.lint]`). An existing, unavoidable use is suppressed inline:

```python
def legacy_adapter(payload: Any) -> None:  # noqa: ANN401 - third-party callback signature we don't control
    ...
```

This keeps the exception visible in the diff and reviewable, rather than silencing the rule project-wide or tracking it in a separate file.

### 2. Data Validation Tests

```bash
python -m pytest tests/data_validation/ -v
```

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

### 3. Data Schema Tests

```bash
python -m pytest tests/data_schema/ -v
```

- **`test_openapi_schema.py`** — API wiring
  - Exactly the served endpoints are registered (`/health`, `/jobs`, `/jobs/stats`, `/match`); retired v2/v3/v5/`/match_v4` routes are absent
  - Correct HTTP methods (GET for health, POST for all match endpoints)
  - Auth boundaries: every endpoint requires `x-api-key`
  - FastAPI boots and serves `/docs` (Swagger UI) with HTTP 200

- **`test_config_validation.py`** — Configuration safety
  - Invalid `FINAL_SCORE_COMBINER` values are rejected at import time
  - Invalid `SCORING_MODE` values are rejected at import time

### 4. Smoke Tests

```bash
python -m pytest tests/smoke/ -v
```

- **`test_startup_smoke.py`** — Health endpoint
  - `GET /health` with `x-api-key` returns `200 {"status": "ok"}`
  - `GET /health` without `x-api-key` returns `401` or `403` (FastAPI version)

- **`test_endpoint_smoke.py`** — Endpoint behavior
  - Empty payload `[]` returns `400`
  - Invalid `final_score_combiner` query param returns `400`
  - `POST /match` without `x-api-key` returns 403
  - `_zqf_annotation` logic (retired v5 code, still unit-tested): eligible, ineligible, missing user ZQF, missing job ZQF
  - Response contract: `POST /match` returns `user_id` + three recommendation lists

### 5. ML Logic Tests

```bash
python -m pytest tests/ml_logic/ tests/components/ tests/integration/ -v
```

- **`ml_logic/`** — education gate, remote/location, skill-gap invariants, ZQF, adversarial inputs, cross-endpoint rules
- **`components/`** — `CosineSkillMatcher` + **metamorphic** tests (reorder/duplicate skills, monotonicity)
- **`integration/`** — v3 concat pipeline with **mocked embeddings** (deterministic rank #1, education gate in stage-1)

See `tests/AI_MATCHING_TEST_PLAN.md` for full scope.

### 6. Job Dict Mapping Tests

```bash
python -m pytest tests/unit/ -v
```

Validates `build_job_dict_from_ranked()` — the Mongo ranked-job → flat job dict mapper used by every match endpoint.

**Coverage strategy** (not every field gets its own test):

- **Mapping logic** — full coverage: ZQF naming conventions (`min_zqf_level` vs `zqf_min`), province/county fallback, `originUuid` precedence, posted-date chain, embedding dim gate, skill ID filtering, etc.
- **Simple passthrough** — one happy-path test asserts core `classifier_metadata` fields (`title`, `employer`, `salary`, ISCO, URL, …) map correctly together.

### All Checks and All Tests, in One Go

The commands below are the same ones from sections 1–6 above, just copy-pasted together: the three checks from [section 1](#1-lint-format-and-type-checks), then one `pytest` run covering every test directory from sections 2–6.

```bash
ruff check
ruff format --check
pyright
python -m pytest tests/data_validation/ tests/data_schema/ tests/smoke/ tests/unit/ tests/ml_logic/ tests/components/ tests/integration/ -v
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
│   ├── test_openapi_schema.py           # Endpoint + auth + docs tests
│   └── test_config_validation.py        # Config rejection tests (2 tests)
├── smoke/
│   ├── test_startup_smoke.py            # Health endpoint tests (2 tests)
│   └── test_endpoint_smoke.py           # Endpoint behavior tests (7 tests)
├── unit/
│   └── test_build_job_dict_from_ranked.py  # Mongo job doc → flat dict mapping
├── ml_logic/                            # Matching invariants (education, location, skill gaps, ZQF)
├── components/                          # Skill scorer + metamorphic tests
├── integration/                         # Mocked-embedding v3 pipeline tests
└── AI_MATCHING_TEST_PLAN.md             # ML logic test plan and scope
```

Lint, format, and type-check configuration lives in `backend/pyproject.toml` (`[tool.ruff]`, `[tool.pyright]`) and is run directly via `ruff check`, `ruff format --check`, and `pyright`.

---

## Adding New Tests

When adding new endpoints, models, or features:

1. **New Pydantic model or field** → Add contract tests in `tests/data_validation/test_response_contracts.py` or `test_request_validation.py`
2. **New endpoint** → Add it to `EXPECTED_ENDPOINTS` in `tests/data_schema/test_openapi_schema.py` and to `AUTH_REQUIRED_PATHS` if it requires `x-api-key`
3. **New config enum** → Add a parameterized case in `tests/data_schema/test_config_validation.py`
4. **New endpoint behavior** → Add smoke tests in `tests/smoke/test_endpoint_smoke.py`
5. **New Mongo field mapping or fallback in `build_job_dict_from_ranked`** → Add cases in `tests/unit/test_build_job_dict_from_ranked.py` (mapping logic) or extend the happy-path test (simple passthrough)
6. **New matching rule or scorer invariant** → Add tests in `tests/ml_logic/`, `tests/components/`, or `tests/integration/` (see `AI_MATCHING_TEST_PLAN.md`)

Always run the full suite after changes to confirm nothing regresses.
