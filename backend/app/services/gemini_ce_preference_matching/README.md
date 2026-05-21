# Gemini + cross-encoder skills with u_hat × p_hat preference integration

Batch experiment aligned with **`POST /match_v3`** for skill retrieval, then preferences:

```
final_score = u_hat × p_hat
```

| Signal | Source |
|--------|--------|
| **Stage 1–2 (column 1)** | `run_match_concat_gemini_ce` — Gemini concat user embed × Mongo `job_embedding` / `concat_skill_embedding_gemini` → **`concat_cosine_similarity`**, then CE rerank |
| **p_hat** | Raw **`concat_cosine_similarity`** (not `cross_encoder_score`, which is per-user min–max and often 1.0 on rank #1) |
| **u_hat** | `PreferenceScorer` (step 3 only — not used to sort column 1) |

## Pipeline order

1. **Column 1:** concat cosine → **p_hat**; CE reranks order (logit only).
2. **Column 2:** same job pool → **u_hat** → **final = u_hat × p_hat** → re-sort.

If the same job appears in both columns, **u_hat is the same** (it depends only on user + job). Columns look different because **ranking** changes: e.g. Baker has p_hat 0.90 but u_hat 0.36, so it stays high on the left and drops on the right; HR roles have u_hat 0.86 and move to the top on the right.



## Prerequisites

- `GEMINI_API_KEY` in `backend/.env`
- Mongo jobs with `job_embedding` (3072-d) or `concat_skill_embedding_gemini.vector_bin`
- `MONGO_JOBS_COLLECTION=ranked_jobs_cleansed_v2` (case-sensitive)

## Commands

From `backend/`:

```bash
python3 -m app.services.gemini_ce_preference_matching.run_matching \
  --users ../data/njila/njila_match_input.jsonl \
  --from-mongo \
  --retrieve-top-k 50 \
  --final-top-k 10 \
  --output output/results_gemini_ce_preference.json

python3 -m app.services.gemini_ce_preference_matching.build_dashboard \
  --input output/results_gemini_ce_preference.json \
  --output output/dashboards/results_gemini_ce_preference_dual.html \
  --top-k 50
```

| Dashboard column | Content |
|------------------|---------|
| **Left** | CE order; headline = **p_hat** (concat cosine). No u_hat. |
| **Right** | **u_hat** + **final** = u_hat × p_hat; p_hat chip shown |

## Related

- HTTP API: `POST /match_v3` → `app.services.match_concat_gemini_ce_service`
- Per-skill cosine only (old batch path): `app.services.cross_encoder.run_cosine_then_cross_encoder`
