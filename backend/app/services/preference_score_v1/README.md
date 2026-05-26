# Hybrid preference scoring (v1)

Implements **Hybrid Preference Scoring (1).pdf** in this folder.

- **Part A (new):** DCE attributes — weighted match on demand **level tags** (bucket ids in `job.attributes`).
- **Part B (existing logic):** O*NET work activities — same BWS × (I/5) × (L/7) as `preference_score.py`, aggregated with **mean** (1/N) not sum.

## Formulas

**Part A**

```text
V     = Σᵢ (wᵢ × Vⱼ) / Σᵢ wᵢ
S_attrs = clamp(V × f, 0, 1)
```

- **wᵢ** — `user.preference_vector[i]` or `user.attributes[i].importance`
- **Vⱼ** — from job level tag + **orientation** (client sign convention):

| Orientation | Attributes | Formula |
|-------------|------------|---------|
| **Gain (+)** | earnings, career_growth, social_interaction, task_content, work_flexibility, social_meaning | V′ = ladder position (low→0, high→1) |
| **Cost (−)** | physical_demand | V′ = **1 −** ladder position (`phys_light`→1, `phys_heavy`→0) |

Ladder position = index / (n−1) on `job_attributes_schema (1).json` buckets.
- **f** — `preference_confidence` or `vignette_count / HYBRID_PREF_VIGNETTES_FOR_FULL_CONFIDENCE`

**Part B**

```text
S_wa = (1/N) Σ_c [ BWS(c) × (I_c/5) × (L_c/7) ]
```

**Final**

```text
raw   = S_attrs + S_wa
u_hat = 1 / (1 + exp(-(raw × 2.646)))
```

Default sigmoid factor **2.646** (anchor: raw=2 → ~99.5%).

## Demand side

Uses existing **`job.attributes`** level id strings (LLM / enrichment “tags” / buckets). No new Mongo fields.

## Enable

```bash
# backend/.env
PREFERENCE_SCORER_MODE=hybrid_v1
```

Default remains `legacy` → `PreferenceScorer`.

`calculate_score(..., include_work_activities=False)` skips Part B (`S_wa = 0`) for attrs-only comparison dashboards.

## Code

| File | Role |
|------|------|
| `levels.py` | Level id → Vⱼ |
| `work_activities.py` | Part B (mean BWS) |
| `scorer.py` | `HybridPreferenceScorer` |
| `__init__.py` | `get_preference_scorer()` |
