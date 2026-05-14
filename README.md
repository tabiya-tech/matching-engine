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

Interactive API docs are available at `http://127.0.0.1:8000/docs` when the backend is running.

## Quick Start

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Configuration

Backend runtime settings are managed through `backend/.env` (see `backend/.env.example`).

Key settings include:

- data source and retrieval controls (Mongo collection, retrieval filters, projection, warmup)
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

