# Tabiya Matching Engine

Tabiya Matching Engine is a matching service that recommends occupations and job opportunities for users based on skills, preferences, and market signals.

## Overview

The repository contains:

- `backend`: FastAPI service for scoring and recommendation APIs.
- `frontend`: React application for interacting with matching outputs.
- shared resources and scripts for benchmarking, diagnostics, and operational maintenance.

The backend supports multi-user requests, Mongo-backed job retrieval, and configurable scoring behavior for both quality and latency tuning.

**API reference:** [`backend/API.md`](backend/API.md): request and response fields, query parameters, errors and examples.

## Core Capabilities

- **User-to-opportunity matching** with ranked recommendations.
- **User-to-occupation matching** for broader career pathways.
- **Skill gap recommendations** to improve future match potential.
- **Configurable scoring and response thresholds** via environment variables.

## Scoring Model

Each user's skills are embedded and compared with jobs and occupations. The closest candidates are re-scored by a cross-encoder and then ranked by:

`final_score = u_hat × p_hat × coverage_factor × location_tier_factor`

Where:

- `u_hat` is preference utility: how well the item matches the user's stated preferences (job attributes and work activities).
- `p_hat` is success propensity: how likely the user is to succeed in the item, driven by skill fit.
- `coverage_factor` lowers the score when the user meets few of the item's essential skills.
- `location_tier_factor` (opportunities only) is 1.0 for jobs in the user's own area and lower for jobs in a regional or national hub.

`FINAL_SCORE_COMBINER=geometric_mean` combines `u_hat` and `p_hat` as `√(u_hat × p_hat)` instead of their product. Every factor is returned in each recommendation's `score_breakdown`.

## API

`POST /match` takes a JSON array of users (with an `x-api-key` header) and returns, for each user:

- `opportunity_recommendations`
- `occupation_recommendations`
- `skill_gap_recommendations`

Other routes: `GET /jobs`, `GET /jobs/stats`, `GET /health`.

See [`backend/API.md`](backend/API.md) for the full reference.

The language a deployment matches in is configured with `TARGET_LANGUAGE` (see
[Languages](#languages)), not per request. Skill matching itself is language-neutral, so a
Spanish posting matches a Spanish profile either way.

Interactive API docs are available at `http://127.0.0.1:8000/docs` when the backend is running.

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./setup.sh
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
VITE_MATCHING_API_KEY=<your-api-key> npm run dev
```

The frontend calls the backend at `http://127.0.0.1:8000/match` and sends `VITE_MATCHING_API_KEY` as the `x-api-key` header.

## Languages

Each deployment is configured for one language with `TARGET_LANGUAGE` (`en` | `es`, or a
locale spelling like `AR-es` / `es_AR` / `spanish`). Requests carry no language. The
important thing to understand is which half of the pipeline is language-neutral and which
is not.

**Skill matching is language-neutral.** Both sides resolve skills by *label* into the
internal id space of the embedding artefact. Every enabled language's taxonomy label pack
is loaded into that one resolver and mapped onto the same English skill ids, so a Spanish job
posting matched against a Spanish user profile scores through exactly the same vectors as
the English equivalent — **with nothing on the request, and with no Spanish retrain.**

That works because skill `ID`s are per-taxonomy-locale but `UUIDHISTORY`'s oldest entry is
not: it is identical across locales for all 13,896 skills. The packs are joined on it at
load time (`app/services/skill_label_packs.py`).

**Text scoring and display are not.** These follow `TARGET_LANGUAGE`:

| What | Where |
|---|---|
| Cross-encoder checkpoint (stage-2 rerank on `/match`) | `cross_encoder_model` per language; `CROSS_ENCODER_MODEL_NAME_<LANG>` overrides |
| BM25 / hybrid stopwords | `stopwords` per language |
| Labels echoed back in the response | `SkillScorer.display_labels(language)` |
| Occupation database labels | `resources/occupations/<lang>/`, falling back to `en` |

An unset `TARGET_LANGUAGE` means `en`; an unregistered value falls back to `en` with a
warning at startup rather than failing the deployment.

```bash
# An Argentina deployment: Spanish postings + Spanish profiles, Spanish-capable reranker
TARGET_LANGUAGE=es uvicorn app.main:app
```

On Cloud Run it is one variable per stack: `TARGET_LANGUAGE` in the stack's GitHub
environment (`vars.TARGET_LANGUAGE`), passed through `iac/backend/env_vars.py`. Leave the
`SKILLS_CSV_PATH` / `SKILL_GROUPS_CSV_PATH` / `SKILL_HIERARCHY_CSV_PATH` /
`OCCUPATION_JSON_PATH` vars **empty** — each one pins every language to a single file (see
`iac/backend/.env.example`).

Registered languages live in `backend/app/languages/` (`en_config.py`, `es_config.py`);
`LANGUAGE_REGISTRY` in `__init__.py` is the only list to edit.

### Adding a language

1. Add the code to `LANGUAGE_REGISTRY` in `backend/app/languages/__init__.py`.
2. Copy `es_config.py` to `<code>_config.py`; set its locales, cross-encoder checkpoint and
   stopwords.
3. Build its taxonomy label pack from a taxonomy CSV export:

```bash
cd backend
python -m scripts.build_language_taxonomy --taxonomy-dir <export-dir> --language <code>
```

   The script validates the columns the resolver reads by name and — the part that matters
   — reports how much of the pack joins onto the English id space. Anything that does not
   join has no embedding row, so labels resolving to it would be silently dropped at match
   time; that almost always means the two packs came from different taxonomy releases.

`tests/unit/test_language_support.py` guards the invariant: every pack must join onto the
English id space, and a Spanish label must resolve to the same id as its English
counterpart.

`ENABLED_LANGUAGES` limits which packs are loaded (default: all — it is a CSV parse, not a
model load). English (`en`) is always included; it defines the id space.

## Configuration

Backend runtime settings are managed through `backend/.env` (see `backend/.env.example`).

Key settings include:

- data source (`MONGO_URL`, `MONGO_DB_NAME`, `MONGO_JOBS_COLLECTION`) and retrieval controls (filters, projection, warmup)
- `GEMINI_API_KEY`, used to embed users' skills
- language defaults (`TARGET_LANGUAGE`, `ENABLED_LANGUAGES`, `CROSS_ENCODER_MODEL_NAME_<LANG>`)
- response sizes: `MATCH_V4_RETRIEVE_TOP_K` (shortlist, default 100), `MATCH_V4_FINAL_TOP_K` (opportunities returned, default 50), `MATCH_V4_TOP_K_OCCUPATIONS` (default 10), `MATCH_TOP_K_SKILL_GAPS` (default 5)
- scoring: `FINAL_SCORE_COMBINER` (`product` or `geometric_mean`), `PREFERENCE_SCORER_MODE`, `V4_FULL_COVERAGE_GAMMA` / `V4_FULL_COVERAGE_FLOOR`, and location tiers (`LOCATION_TIER_ENABLED`, `LOCATION_TIER_W_REGIONAL`, `LOCATION_TIER_W_NATIONAL`)
- skill-gap threshold: `MATCH_RESPONSE_SKILL_MIN_SCORE`. Skill gaps whose proximity is below this value are dropped. If it isn't set, it falls back to `GATE_SIMILARITY_THRESHOLD`.

## Request tracing (Langfuse)

Every `POST /match` request can be traced to [Langfuse](https://langfuse.com),
with the same layer the llm-reranker and compass-connect use (`backend/app/observability/`). It is
**off by default**; a deployment with no Langfuse keys behaves exactly as before.

What one trace holds:

| Observation | What it shows |
|---|---|
| root, named after the route | request id, pseudonymous `user_id` (single-user requests; batches list `user_ids`), query params, HTTP status, embedding totals |
| `retrieval` | Mongo find / build and occupation-cache timings, job and occupation counts |
| `embedding` → `embed_content` | one Langfuse **embedding** per Gemini call: model, dimensionality, token usage, attempts, retries, per-attempt latency, failure |
| `shortlist`, `rerank` | stage-1 cosine and the cross-encoder, per `corpus` (`jobs` / `occupations`) |
| `preference_scoring`, `formatting`, `skill_gaps` | u_hat × p_hat scoring, row building, skill-gap analysis |

A Langfuse dashboard of observation latency (p50 / p95 / p99) grouped by name gives per-stage
percentiles. Traces carry the `route:/match` tag, and you can filter by **environment** for a
deployment (IaC sets it to the Pulumi stack).

- **Find a request:** traced responses carry `X-Request-ID` (a client-sent one is echoed) and
  `X-Trace-ID` — paste the trace id into Langfuse. Or search by the user's id.
- **Failures and retries:** traces are tagged `error` (5xx / exception), `client_error` (4xx),
  `embedding_retried` and `embedding_failed`.
- **Embedding spend:** Langfuse prices each `embed_content` from its token usage and the model's price.
  Gemini's `embed_content` reports no tokens, so a traced call runs `count_tokens` in parallel with it
  and waits at most `TOKEN_COUNT_WAIT_S` (0.2 s) after the embedding returns; if the count is not in
  by then, that call records no usage rather than a guess. If the Langfuse project has no price for `gemini-embedding-001`, add it once under
  *Settings → Models* (match `gemini-embedding-001`, input price per token); cost then shows per trace
  and aggregates per environment.
- **Privacy:** no request or response body is recorded — only counts, timings, error class names,
  the request id and the pseudonymous user id. The text sent to Gemini (the jobseeker's skills) is
  exported only with `"recordEmbeddingInput": true`, masked first. Vectors are never exported.

```bash
MATCHING_ENABLE_TRACING=1
MATCHING_LANGFUSE_HOST=https://cloud.langfuse.com
MATCHING_LANGFUSE_PUBLIC_KEY=pk-lf-...
MATCHING_LANGFUSE_SECRET_KEY=sk-lf-...
MATCHING_TRACING_ENVIRONMENT=dev          # IaC: defaults to the stack name
MATCHING_TRACING_CONFIG={"sampleRate": 1.0, "recordEmbeddingInput": false}
```

In GitHub, set `MATCHING_ENABLE_TRACING` (and optionally `MATCHING_LANGFUSE_HOST`,
`MATCHING_TRACING_CONFIG`) as environment variables and `MATCHING_LANGFUSE_PUBLIC_KEY` /
`MATCHING_LANGFUSE_SECRET_KEY` as environment secrets. The stderr timing blocks
(`app/match_timing_log.py`) are unchanged.

## Deployment

Cloud Run deployment is supported through:

- `backend/build-and-deploy.sh`

Example:

```bash
cd backend
./build-and-deploy.sh <project-id> <env-vars-yaml>
```

