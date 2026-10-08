# Unified preference scoring (DCE + BWS, additive-RUM)

One `u_hat` from two individual-level random-utility blocks, combined on a common harmonised scale.

- **Part A — DCE attributes:** reconstruct the DCE's own utility from the per-user betas on
  graded job levels.
- **Part B — BWS work activities:** importance-weighted work-activity part-worths.

## What the per-user values are

The elicitation engine estimates per-user MNL/DCE part-worths `β` and sends this repo
`v_k = sigmoid(β_k) ∈ [0,1]` (in `preference_vector`). `0.5` = neutral; `>0.5` prefers the
attribute's **target** level; `<0.5` prefers the **reference** level. BWS `bws_scores` are HB
posterior part-worths (~[-2,2], 0 = neutral), keyed by O*NET WA codes.

## Formulas

**Part A — DCE attributes** (`work_activities.compute_dce_utility`)

```text
β̂_k = logit(clamp(v_k, eps, 1-eps)) · scale_k      # exact inverse of the sigmoid; v=0.5 → 0
ṽ_k = ladder_position(reference→target) ∈ [0,1]     # reference level → 0 (DCE dummy-coding)
V_dce = Σ_k β̂_k · ṽ_k ;  Ṽ_dce = f · clamp(V_dce / Σ_k|β̂_k|, -1, 1)
```
Direction comes from the **sign of β̂_k**, so there is no gain/cost orientation; the schema's
per-attribute level **ordering** fixes the reference (ladder 0) and target (ladder 1). `f` is the
confidence factor (`preference_confidence` / `n_vignettes_completed`; absent ⇒ 1.0).

**Part B — BWS work activities** (`work_activities.compute_task_utility`)

```text
V_task = Σ_c ŵ_c · β_c ,  ŵ_c ∝ WA_Importance (Σŵ = 1) ;  Ṽ_task = clamp(V_task/2, -1, 1)
```

**Combination** (`work_activities.combine_utilities`)

```text
u_hat = logistic( γ · [ α·Ṽ_task + (1-α)·Ṽ_dce ] )   # α=BWS_ALPHA (0.5), γ=BWS_GAIN_GAMMA (4.0)
```

## Schema (`job_attributes_schema.json`)

Each attribute's `levels` are ordered **reference (0) → target (1)**, aligned to the level whose
part-worth the per-user value encodes. Level ids are the canonical job-side `selected_level_id`s
(resolved directly by `levels.resolve_schema_level_id`). `HYBRID_PREF_SCHEMA_PATH` overrides the
committed default.

## Config

```bash
PREFERENCE_SCORER_MODE=unified   # default; 'legacy' = old PreferenceScorer (A/B escape hatch)
BWS_ALPHA=0.5                    # task vs DCE weight (sensitivity sweep)
BWS_GAIN_GAMMA=4.0               # logistic gain
DCE_LOGIT_EPS=0.01               # clamps extreme v_k before logit
DCE_ATTR_SCALE={}                # JSON, optional per-attribute β̂ scale (e.g. earnings)
```

`calculate_score(..., include_work_activities=False)` zeros Part B (`V_task=0`) for attrs-only dashboards.

## Caveats

- **Earnings** is a continuous DCE term with a tiny coefficient, so `sigmoid(β)≈0.5` ⇒ `β̂≈0`: it
  barely moves `V_dce` under the current contract. Use `DCE_ATTR_SCALE` to compensate, or fix upstream.
- The DCE batch covers 4 attributes (earnings, physical_demand, social_interaction, career_growth);
  others sit at 0.5 ⇒ 0 contribution.
- Per-attribute **target-level** semantics (esp. `physical_demand`, `social_interaction`) are gated by
  the directional tests in `backend/tests/test_dce_utility.py` / `test_preference_scorers.py`.

## Code

| File | Role |
|------|------|
| `levels.py` | level id → ladder position (reference→target) |
| `work_activities.py` | `compute_dce_utility` (Part A), `compute_task_utility` (Part B), `combine_utilities` |
| `scorer.py` | `UnifiedPreferenceScorer` |
| `__init__.py` | `get_preference_scorer()` |
| `job_attributes_schema.json` | committed attribute/level schema |
