# Tabiya Matching Engine

Tabiya Matching Engine is a matching service that recommends occupations and job opportunities for users based on skills, preferences, and market signals.

## Overview

The repository contains:

- `backend`: FastAPI service for scoring and recommendation APIs.
- `frontend`: React application for interacting with matching outputs.
- shared resources and scripts for benchmarking, diagnostics, and operational maintenance.

The backend supports multi-user requests, Mongo-backed job retrieval, and configurable scoring behavior for both quality and latency tuning.

## Core Capabilities

- **User-to-opportunity matching** with ranked recommendations.
- **User-to-occupation matching** for broader career pathways.
- **Skill gap recommendations** to improve future match potential.
- **Configurable scoring and response thresholds** via environment variables.

## Scoring Model

Default scoring mode is **multiplicative** (`SCORING_MODE=multiplicative`):

`S_total = U_hat × P_hat`

Where:

- `U_hat` captures utility from skills and preferences.
- `P_hat` captures success propensity (gate, essential fit, readiness, market opportunity).

Legacy additive mode is also available (`SCORING_MODE=additive`) for controlled comparisons.

## API

Primary endpoint:

- `POST /match` — accepts one or more users and returns:
  - `opportunity_recommendations`
  - `occupation_recommendations`
  - `skill_gap_recommendations`

Hybrid diagnostic / alternate ranking:

- `POST /match_v2` — same `MatchRequest` body shape as `POST /match` (JSON array); loads **all active jobs** from Mongo **without** the per-user location prefilter used by `POST /match` (`JOBS_RETRIEVAL_FILTER` is effectively bypassed here so hybrid indexes match unrestricted batch runs, e.g. CLI `--mongo-all-active`). Returns **`hybrid_recommendations`** ranked by BM25 × embedding‑cosine **pool fused** scores (optional query: `fusion_top_k`, `alpha_on_cosine`). Does not compute occupations or the full SkillScorer / `p_hat` stack. **`x-api-key` is not required** on this route for now (unlike `/match`).

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
npm run dev
```

## Languages

Each deployment is configured for one language with `TARGET_LANGUAGE` (`en` | `es`, or a
locale spelling like `AR-es` / `es_AR` / `spanish`). Requests carry no language. The
important thing to understand is which half of the pipeline is language-neutral and which
is not.

**Skill matching is language-neutral.** Both sides resolve skills by *label* into the
internal id space of the embedding artefact. Every enabled language's taxonomy label pack
is loaded into that one resolver and mapped onto the same canonical ids, so a Spanish job
posting matched against a Spanish user profile scores through exactly the same vectors as
the English equivalent — **with nothing on the request, and with no Spanish retrain.**

That works because skill `ID`s are per-taxonomy-locale but `UUIDHISTORY`'s oldest entry is
not: it is identical across locales for all 13,896 skills. The packs are joined on it at
load time (`app/services/skill_label_packs.py`).

**Text scoring and display are not.** These follow `TARGET_LANGUAGE`:

| What | Where |
|---|---|
| Cross-encoder checkpoint (stage-2 rerank on `/match_v3`, `/match_v4`) | `cross_encoder_model` per language; `CROSS_ENCODER_MODEL_NAME_<LANG>` overrides |
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
   — reports how much of the pack joins onto the canonical id space. Anything that does not
   join has no embedding row, so labels resolving to it would be silently dropped at match
   time; that almost always means the two packs came from different taxonomy releases.

`tests/unit/test_language_support.py` guards the invariant: every pack must join onto the
canonical id space, and a Spanish label must resolve to the same id as its English
counterpart.

`ENABLED_LANGUAGES` limits which packs are loaded (default: all — it is a CSV parse, not a
model load). The canonical language is always included; it defines the id space.

## Configuration

Backend runtime settings are managed through `backend/.env` (see `backend/.env.example`).

Key settings include:

- data source and retrieval controls (Mongo collection, retrieval filters, projection, warmup)
- language defaults (`TARGET_LANGUAGE`, `ENABLED_LANGUAGES`, `CROSS_ENCODER_MODEL_NAME_<LANG>`)
- scoring mode and weights
- top-k response sizes
- response skill thresholding (`MATCH_RESPONSE_SKILL_MIN_SCORE`)

If `MATCH_RESPONSE_SKILL_MIN_SCORE` is not set, it falls back to `GATE_SIMILARITY_THRESHOLD`.

## Deployment

Cloud Run deployment is supported through:

- `backend/build-and-deploy.sh`

Example:

```bash
cd backend
./build-and-deploy.sh <project-id> <env-vars-yaml>
```

