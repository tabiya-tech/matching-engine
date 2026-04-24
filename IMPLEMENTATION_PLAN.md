# Implementation Plan: Aligning the Matching Algorithm with Bied/Perennes

This plan restructures the Horizon matching algorithm to align with the theoretical framework from the Bied and Perennes doctoral theses. The core change is moving from an **additive three-component score** (`w1*S_skills + w2*S_preference + w3*S_demand`) to a **multiplicative two-factor architecture** (`U_hat * p_hat`) with market-level adjustments applied separately.

---

## Current Architecture (What Exists)

```
final_score = 0.40 * S_skills + 0.40 * S_preference + 0.20 * S_demand
```

| Component | File | Role |
|---|---|---|
| `S_skills` (U_final) | `skill_score.py` + `skills_match.py` | Weighted average of location, essential skill similarity, optional skill similarity, skill group recall, minus gap penalty |
| `S_preference` | `preference_score.py` | `0.5 + 0.2 * sum(beta * user_weight * encoded_value)` over 7 attribute dimensions |
| `S_demand` | `demand_score.py` | Lookup from `expected_demand` label to a float in [0.1, 1.0] |
| Orchestration | `matching_service.py` | Location filter, score all items, sort, format top-K |
| Config | `config.py` | Weights, demand mapping, preference betas |

### Problems identified (per the adaptations document)
1. **Conceptual mismatch**: `S_skills` is a feasibility/capability score, `S_preference` is closest to utility, `S_demand` is a market heuristic. These are not the same objects as U and P in the papers, so adding them is off-model.
2. **Additive combination**: The papers show the ideal ranking is multiplicative (U x P), not additive.
3. **Demand mixed into individual score**: Congestion/market-balancing should be a separate layer, not 20% of the pairwise score.
4. **No hiring-probability proxy**: The system has no distinct P_hat signal.
5. **Hardcoded preference betas**: The utility proxy uses authored coefficients, not estimated ones.
6. **No hard gating**: Missing a mandatory certification is averaged away rather than crushing the score.

---

## Target Architecture

```
Score(i,j) = U_pref(i,j) * p_tilde(i,j)
```

With four clean layers:

| Layer | Purpose | Analogue in papers |
|---|---|---|
| **Filter** | Location filter + hard feasibility gate | Pre-filter |
| **U_hat** | Seeker-side utility proxy | U(i,j) from Bied/Perennes |
| **p_hat** | Cold-start success propensity proxy | P(i,j) from Bied/Perennes |
| **Market rerank** | Demand/congestion adjustment applied after scoring | CAROT / post-processing |

---

## Detailed Change Plan

### Phase 1: Restructure the scoring pipeline

#### 1.1 Create `success_propensity.py` (NEW FILE)

**Location**: `backend/app/services/success_propensity.py`

This is the new `p_hat` — a cold-start hiring-probability proxy built from recruiter-side feasibility signals, not seeker preferences.

```python
class SuccessPropensityScorer:
    def calculate_score(self, user_profile, job_posting) -> dict:
        # Returns: {
        #   "p_hat": float,        # the success propensity [0, 1]
        #   "components": {
        #       "gate": float,     # G_ij: hard feasibility gate (0 or 1)
        #       "essential_fit": float,  # E_ij: essential skill coverage
        #       "recruiter_readiness": float,  # R_ij: optional skills + occupation proximity
        #       "market_opportunity": float,    # M_ij: tightness, freshness, capacity
        #   },
        #   "confidence": float,   # c_ij: data completeness signal
        # }
```

**Components in detail:**

**G_ij — Hard feasibility gate (binary/near-binary)**
- Checks: mandatory certifications, legal eligibility flags, impossible geography, impossible schedule conflicts
- If any hard gate fails, `G_ij = 0` and the entire `p_hat` collapses to 0
- Source data: reuse existing eligibility logic from `skills_match.py` but make it a hard gate, not a soft penalty
- Implementation: extract from `compute_U_complete`'s existing `is_eligible` logic, but apply it as a multiplier rather than a flag

**E_ij — Essential skill fit (geometric mean over Node2Vec similarities)**
- The underlying per-skill similarities remain **Node2Vec graph-based cosine distances** — this is unchanged. Each `sim(s, s')` is still computed as `op_mat @ js_mat.T` using the pretrained Node2Vec embedding matrix (`skill_embedding_model.pt`). The graph structure, embedding model, and `SimilarityEngine` class are fully preserved.
- What changes is only the **aggregation**: switch from arithmetic mean to **geometric mean** of the per-skill best-match similarities
- Formula: `E_ij = (prod over essential skills of max_sim(s, user_skills)^w_s) ^ (1/sum(w_s))`
- This penalizes missing one essential skill much more strongly than arithmetic mean
- Source: refactor the essential skill matching logic currently in `skills_match.py::compute_U_complete`
- Add a new `best_geometric_mean_cos` method to `SimilarityEngine` alongside the existing `best_mean_cos` (which is kept for U-side diagnostics and backward compatibility)

**R_ij — Recruiter-side readiness**
- Optional skill alignment (reuse existing `opt_sim` from `skills_match.py`)
- Occupation proximity (if ESCO/ISCO codes are available in the data)
- Skill group recall (reuse existing `grp_sim`)
- Formula: weighted combination of optional skill fit and occupation proximity

**M_ij — Market opportunity**
- Absorb the current `demand_score.py` logic here
- Include: expected demand label (mapped to float), vacancy freshness if available
- Formula: `sigma(a*log(1+tightness) + b*log(1+absorptive_capacity) + d*freshness)`
- Initially use the existing `DEMAND_SCORE_MAPPING` as the primary input, since richer market data isn't yet available

**Final p_hat formula:**
```
p_hat = G_ij * E_ij^alpha * R_ij^beta * M_ij^gamma
```
Starting exponents: `alpha=0.5, beta=0.2, gamma=0.3`

**Files to modify:**
- Create: `backend/app/services/success_propensity.py`
- Modify: `backend/app/services/skills_match.py` — extract essential skill geometric mean and gate logic into reusable functions
- Modify: `backend/app/config.py` — add `SUCCESS_PROPENSITY_CONFIG` with exponents and gate thresholds

---

#### 1.2 Refactor `skill_score.py` and `skills_match.py` to separate U from P signals

**Current state**: `compute_U_complete` computes a single blended score mixing feasibility (essential skills, eligibility) with utility-adjacent signals (location, optional skills, skill groups).

**Target**: Split into two clean outputs:
1. **Feasibility signals** → feed into `success_propensity.py` (P-side)
2. **Utility signals** → feed into `preference_score.py` (U-side)

**Concrete changes to `skills_match.py`:**

- Add a new function `compute_feasibility_signals(js, op, engine)` that returns:
  - `essential_skill_coverage_geometric`: geometric mean of best-match sims (for E_ij)
  - `essential_skill_coverage_arithmetic`: arithmetic mean (for backward compat / U-side)
  - `gate_passed`: boolean for hard gate
  - `gap_share`: fraction of essentials below threshold
  - `optional_skill_similarity`: centroid cosine (for R_ij)
  - `skill_group_recall`: overlap fraction (for R_ij)
  - `match_details`: the existing detailed match info

- Keep `compute_U_complete` but rename it to `compute_skill_utility(js, op, engine)` and have it return only the utility-relevant components (location score, a softer skill alignment signal). This becomes one input to U_hat alongside preferences.

- Alternatively (simpler): keep `compute_U_complete` largely as-is for now but add `compute_feasibility_signals` as a new function that the success propensity scorer calls. The existing `compute_U_complete` output is NOT used in U_hat directly — only `preference_score.py` contributes to U_hat initially.

**Decision**: Take the simpler path. Add `compute_feasibility_signals` alongside the existing function. The existing `compute_U_complete` can be kept for backward-compatible diagnostics but won't feed into the new final score.

**Important**: All skill similarity computations in both the old and new functions continue to use the Node2Vec graph embedding via `SimilarityEngine`. The embedding matrix, the `skill_to_row` mapping, and the cosine-similarity-over-graph-embeddings approach are architectural invariants — they are not changed by this refactor.

**Files to modify:**
- `backend/app/services/skills_match.py` — add `compute_feasibility_signals()`
- `backend/app/services/skill_score.py` — add method to call the new feasibility function

---

#### 1.3 Redefine `preference_score.py` as the sole U_hat source

**Current state**: `S_preference = 0.5 + 0.2 * sum(beta * weight * encoded_value)`, clamped to [0, 1].

**Target**: `U_hat(i,j)` = seeker-side utility proxy from preference fit + task-content alignment.

**Changes:**
- Remove the `base_constant = 0.5` — this artificially inflates U for all jobs and mutes the signal. Replace with a utility score that genuinely varies between 0 and 1 based on fit.
- Keep the seven preference dimensions as features (earnings, task content, physical demand, flexibility, social interaction, career growth, social meaning).
- Add BWS/work-activity alignment as a task-content alignment signal (already partially implemented).
- Move non-negotiables out: if a user has a hard salary floor or hard location constraint, these should be handled as **filters** (pre-matching) or **hard gates** (in G_ij), not averaged into the utility score.
- Output a clean `u_hat` in [0, 1].

**Concrete changes:**
- Rename the class method output key from `"score"` to `"u_hat"` 
- Remove `base_constant` (or set to 0) and rescale so the score uses the full [0, 1] range
- Add a `normalize_utility()` helper that maps the raw sum to [0, 1] via sigmoid or min-max scaling
- Keep `details` for explainability

**Files to modify:**
- `backend/app/services/preference_score.py`
- `backend/app/config.py` — update `PREFERENCE_CONFIG`

---

#### 1.4 Rewrite `matching_service.py` to use multiplicative scoring

**Current state (line 192-196):**
```python
final_score = (
    GLOBAL_WEIGHTS["w1_skills"] * skill.get("U_final", 0.0)
    + GLOBAL_WEIGHTS["w2_preference"] * pref.get("score", 0.0)
    + GLOBAL_WEIGHTS["w3_market"] * demand.get("score", 0.5)
)
```

**Target:**
```python
u_hat = pref.get("u_hat", 0.0)
p_hat = success.get("p_hat", 0.0)
final_score = u_hat * p_hat
```

**Concrete changes to `_match_items()`:**

1. Replace the three-scorer pattern with two scorers:
   - `scorer_pref.calculate_score(user, item)` → `u_hat`
   - `scorer_success.calculate_score(user, item)` → `p_hat`

2. Compute `final_score = u_hat * p_hat`

3. Keep `scorer_skill` for **diagnostics and explainability** (skill match details are valuable for the user), but its output no longer feeds into `final_score`.

4. Apply demand/market as a **post-processing rerank** (see Phase 2), not as a score component.

5. Update `_create_score_breakdown()` to reflect the new structure:
   ```python
   {
       "u_hat": float,           # seeker-side utility proxy
       "p_hat": float,           # success propensity proxy  
       "p_hat_components": {
           "gate": float,
           "essential_fit": float,
           "recruiter_readiness": float,
           "market_opportunity": float,
       },
       "final_score": float,     # u_hat * p_hat
       "skill_diagnostics": {...}, # legacy skill details for explainability
       "preference_details": [...],
   }
   ```

**Files to modify:**
- `backend/app/services/matching_service.py` — core scoring logic
- `backend/app/config.py` — replace `GLOBAL_WEIGHTS` with new config structure

---

#### 1.5 Update `config.py`

Remove or deprecate:
```python
GLOBAL_WEIGHTS = {
    "w1_skills": 0.40,
    "w2_preference": 0.40,
    "w3_market": 0.20
}
```

Add:
```python
# Success propensity exponents (p_hat = G * E^alpha * R^beta * M^gamma)
SUCCESS_PROPENSITY_CONFIG = {
    "alpha_essential": 0.5,    # essential skill fit exponent
    "beta_readiness": 0.2,     # recruiter readiness exponent
    "gamma_market": 0.3,       # market opportunity exponent
    "essential_threshold": 0.35,  # similarity threshold for "matching"
    "gate_threshold": 0.35,       # threshold for hard gate
}

# Scoring mode
SCORING_MODE = "multiplicative"  # "multiplicative" (U*P) or "additive" (legacy)
```

Keep `DEMAND_SCORE_MAPPING` as it feeds into M_ij.
Keep `PREFERENCE_CONFIG` but update `base_constant` to 0.

---

### Phase 2: Move demand out of the individual score

#### 2.1 Implement market-level post-processing

**Current state**: `S_demand` is 20% of the individual additive score.

**Target**: Demand/market signal is used in one of three ways (per the adaptations document):
1. **Inside p_hat's M_ij** — market opportunity affects the success propensity (already handled in 1.1)
2. **As a tie-breaker** — after sorting by `u_hat * p_hat`, break ties using demand
3. **As a congestion penalty** — if too many seekers are funneled to the same occupation, penalize it

**Implementation for Phase 2 (tie-breaker approach):**

In `matching_service.py`, change the sort key:
```python
recommendations.sort(
    key=lambda x: (x["final_score"], x["demand_score"]),  
    reverse=True
)
```

This keeps demand as a secondary signal without polluting the welfare-aligned primary score.

**Files to modify:**
- `backend/app/services/matching_service.py`

---

### Phase 3: Update schemas and response format

#### 3.1 Update `schemas.py`

**Changes:**
- Update `ScoreBreakdown` to reflect the new U_hat / p_hat structure:
  ```python
  class ScoreBreakdown(BaseModel):
      u_hat: float                    # seeker utility proxy
      p_hat: float                    # success propensity proxy
      p_hat_components: PHatComponents
      final_score: float              # u_hat * p_hat
      # Legacy fields kept for backward compat
      skill_diagnostics: Optional[SkillComponents] = None
      preference_details_score: float
      demand_label: str
  ```

- Add `PHatComponents`:
  ```python
  class PHatComponents(BaseModel):
      gate: float
      essential_fit: float
      recruiter_readiness: float
      market_opportunity: float
      confidence: float
  ```

**Files to modify:**
- `backend/app/schemas.py`

---

### Phase 4: Update justification and explainability

#### 4.1 Update `_build_justification()`

The justification text should now explain the match in terms of:
- Why this job is **good for the seeker** (U_hat reasons: preference alignment)
- Why the seeker **has a good chance** (p_hat reasons: skill coverage, market demand)

Update the justification builder to use the new score structure.

**Files to modify:**
- `backend/app/services/matching_service.py` — `_build_justification()`

---

### Phase 5: Validation

#### 5.1 Monotonicity tests (no new files needed, but add test cases)

Verify these invariants:
- Removing a required credential never increases `p_hat`
- Increasing essential skill coverage never decreases `p_hat`
- Holding suitability fixed, moving from low to high demand increases `p_hat`
- Holding everything fixed, stale jobs don't outrank fresh jobs on the success component
- U and P should **disagree** in sensible cases (high-utility/low-success vs low-utility/high-success)

#### 5.2 Expert face-validity checks

Create test scenarios for:
- **High U, Low P**: Dream job with poor skill fit → low final score
- **Low U, High P**: Easy-to-get job the seeker doesn't want → low final score  
- **High U, High P**: Great fit all around → highest final score
- **Moderate U, Moderate P**: Decent match → middle score

---

## File Change Summary

| File | Action | Description |
|---|---|---|
| `services/success_propensity.py` | **CREATE** | New p_hat scorer with G, E, R, M components |
| `services/skills_match.py` | **MODIFY** | Add `compute_feasibility_signals()`, add geometric mean |
| `services/skill_score.py` | **MODIFY** | Add method to expose feasibility signals for p_hat |
| `services/preference_score.py` | **MODIFY** | Remove base_constant, rescale to true [0,1], rename output to u_hat |
| `services/demand_score.py` | **MODIFY** | Output feeds into M_ij inside success_propensity, no longer a top-level score |
| `services/matching_service.py` | **MODIFY** | Replace additive formula with `u_hat * p_hat`, demand as tie-breaker |
| `config.py` | **MODIFY** | Replace GLOBAL_WEIGHTS, add SUCCESS_PROPENSITY_CONFIG |
| `schemas.py` | **MODIFY** | Update ScoreBreakdown, add PHatComponents |
| `services/skill_gap_analysis.py` | No change | Skill gap logic is independent of scoring formula |
| `routes.py` | No change | API contract stays the same (MatchRequest/MatchResponse) |
| `database.py` | No change | Data access layer is unaffected |

---

## Migration Strategy

1. **Phase 1 first**: Implement the new scoring pipeline alongside the old one. Add a `SCORING_MODE` config flag so both can coexist during testing.
2. **A/B comparison**: Run both formulas on the same inputs and compare rankings to validate the new formula produces sensible results.
3. **Phase 2-4**: Once the multiplicative formula is validated, update the post-processing, schemas, and justifications.
4. **Phase 5**: Run monotonicity and face-validity tests.
5. **Cutover**: Remove the legacy additive path once confident.

---

## Future Extensions (Stage 2 from the adaptations document)

These are noted for reference but **not part of this implementation plan**:

- **Estimate preference betas from data** instead of hardcoding (requires pilot data)
- **Structural value function**: Replace `U * P` with `Gamma_hat(i,j) = sigma * P * log(1 + e^(Delta/sigma))` where `Delta = alpha*U - beta/P + c` (requires click/application data)
- **Confidence-weighted shrinkage**: `p_hat* = c_ij * p_hat + (1 - c_ij) * p_bar_{o,z}` — shrink low-confidence estimates toward a market prior
- **Congestion post-processing**: Implement CAROT-style optimal transport for population-level reranking
