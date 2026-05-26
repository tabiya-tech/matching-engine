# Gemini + cross-encoder skills with u_hat × p_hat preference integration

Batch experiment aligned with **`POST /match_v3`** for skill retrieval, then preferences:

```
final_score = u_hat × p_hat
```

| Signal | Source |
|--------|--------|
| **Stage 1–2 (column 1)** | `run_match_concat_gemini_ce` — Gemini concat user embed × Mongo `job_embedding` / `concat_skill_embedding_gemini` → **`concat_cosine_similarity`**, then CE rerank |
| **p_hat** | Raw **`concat_cosine_similarity`** (not `cross_encoder_score`, which is per-user min–max and often 1.0 on rank #1) |
| **u_hat** | `get_preference_scorer()` — use **`hybrid_v1`** (`preference_score_v1`) or **`legacy`** (`PreferenceScorer`) |

## Pipeline order

1. **Column 1:** concat cosine → **p_hat**; CE reranks order (logit only).
2. **Column 2:** same job pool → **u_hat** → **final = u_hat × p_hat** → re-sort.

If the same job appears in both columns, **u_hat is the same** (it depends only on user + job). Columns look different because **ranking** changes: e.g. Baker has p_hat 0.90 but u_hat 0.36, so it stays high on the left and drops on the right; HR roles have u_hat 0.86 and move to the top on the right.



## Prerequisites

- `GEMINI_API_KEY` in `backend/.env`
- Mongo jobs with `job_embedding` (3072-d) or `concat_skill_embedding_gemini.vector_bin`
- `MONGO_JOBS_COLLECTION=ranked_jobs_cleansed_v2` (case-sensitive)
- **`PREFERENCE_SCORER_MODE=hybrid_v1`** for the new preference model (see `preference_score_v1/`)

## Commands

From `backend/`:

```bash
# One-shot (matching + HTML)
chmod +x run_gemini_ce_hybrid_dashboard.sh
./run_gemini_ce_hybrid_dashboard.sh ../data/njila/njila_match_input.jsonl
```

Or step by step:

```bash
python3 -m app.services.gemini_ce_preference_matching.run_matching \
  --users ../data/njila/njila_match_input.jsonl \
  --from-mongo \
  --retrieve-top-k 50 \
  --final-top-k 10 \
  --output output/results_gemini_ce_hybrid_v1.json

python3 -m app.services.gemini_ce_preference_matching.build_dashboard \
  --input output/results_gemini_ce_hybrid_v1.json \
  --output output/dashboards/results_gemini_ce_hybrid_v1_dual.html \
  --top-k 50
```

Open the HTML in a browser: `output/dashboards/results_gemini_ce_hybrid_v1_dual.html`

| Dashboard column | Content |
|------------------|---------|
| **Left** | Skills only: CE order; headline = **p_hat** (concat cosine). **No preferences.** |
| **Right** | **hybrid_v1** → **u_hat**, then **final** = u_hat × p_hat; re-sorted |

### Work activities (BWS) comparison — 3 columns

`run_matching` now writes **`recommendations_attrs_only`** (Part A DCE only) and **`recommendations`** (Part A + Part B BWS).

```bash
python3 -m app.services.gemini_ce_preference_matching.build_dashboard_with_wa \
  --input output/results_gemini_ce_hybrid_v1.json \
  --output output/dashboards/results_gemini_ce_hybrid_v1_with_wa.html \
  --top-k 50
```

| Column | Content |
|--------|---------|
| **1** | Skills only (same as dual left) |
| **2** | hybrid_v1 **attributes only** (`include_work_activities=False`) |
| **3** | hybrid_v1 **attributes + BWS** (`user.bws_scores` × job `onet_work_activities`) |

**Data needed for column 3 to differ from column 2:** users with `bws_scores` keyed by work-activity IDs; Mongo jobs with `onet_work_activities`. Njila sample users often lack BWS — columns 2–3 may match until Compass BWS is on user records.

## Related

- HTTP API: `POST /match_v3` → `app.services.match_concat_gemini_ce_service`
- HTTP API: `POST /match_v4` → `app.services.match_concat_gemini_ce_preference_service` (v3 + hybrid preference final)
- Per-skill cosine only (old batch path): `app.services.cross_encoder.run_cosine_then_cross_encoder`
