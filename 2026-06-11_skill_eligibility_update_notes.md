# Developer note + plan: make the v4 skills match meaningful (keep `u_hat × p_hat`, keep combined embeddings for ranking)

## Context / why this change

The local `/match_v4` harness + analysis script ([backend/analyze_match_v4_results.py](backend/analyze_match_v4_results.py)) surfaced that the skill side of the algorithm is effectively a no-op:

- **Eligibility is saturated and meaningless.** Every recommendation comes back `is_eligible=True`; binary `essential_fit` is `1.0` for **every** rec. A woodworker (*sand wood, operate wood sawing equipment, repair furniture frames*) is marked as meeting **23/23** essential skills of a "Fullstack Developer" job — via pairings like `debug software ~ repair furniture frames (0.81)`.
- **Skills barely influence ranking.** `p_hat` (combined-embedding cosine) has sd ≈ 0.02; `u_hat` (preferences) sd ≈ 0.14 → ranking is driven almost entirely by preferences.

### Root cause (verified in code)
- **Anisotropic Gemini skill embeddings**: random/unrelated skill pairs cosine ≈ 0.76 (sd ≈ 0.027) — everything looks "similar." Identity = 1.0; unrelated still ~0.78–0.82.
- **Threshold below the noise floor**: `V4_FULL_SIM_THRESHOLD = 0.6` ([config.py:243](backend/app/config.py#L243)) is below the random-pair floor → `meets_threshold` ~always True.
- **Gate disabled + annotation-only**: `V4_FULL_MIN_ESS_SHARE = 0.0` ([config.py:246](backend/app/config.py#L246)) → `is_eligible` always True; and nothing ranks/filters on it.
- **Max-over-all-user-skills aggregation** in `CosineSkillMatcher.score_pair` inflates coverage: with more user skills, *some* skill is ~0.8 to any job skill.

### Reconciliation with the "combined embeddings are more robust" finding (important)
A colleague moved retrieval from per-skill matching to **combined skill-set embeddings** because per-skill matching was brittle. That is correct **for ranking** — pooling all skills into one vector averages out per-skill noise, giving a lower-variance relevance scalar. **This plan keeps combined embeddings for ranking.**

But the eligibility **gate** asks a different question — *"does the user possess each specific required skill?"* — which a pooled vector **cannot answer** (no per-skill decomposition). So per-skill matching for the gate is not a regression; it is the only representation that can answer it. The brittleness the colleague hit was largely **mechanical** (anisotropy, max-inflation) and is fixable at the root — and the prototype confirms **whitening alone does ~84% of the fix**. The **residual** brittleness — a bi-encoder measures *semantic relatedness, not competence* — would be handled by an **escalation path** (a cross-encoder) gated on an empirical test; **that test was run and the baseline passed, so escalation isn't currently needed** (see Empirical validation).

## Constraints & decisions (locked)

- **`final = u_hat × p_hat` is FIXED (from the paper). Do not change the combination.**
- `p_hat` = P(get job) = product of [0,1] factors: **skills-fit × labour-demand × popularity** (popularity not yet implemented — separate follow-up). `p_hat` must stay in **[0,1]**.
- **Ranking skills-fit = whitened *combined* embedding** (keep pooling; only de-compress it). Does **not** reintroduce per-skill brittleness into ranking.
- **Gate = per-skill coverage**, made robust primarily by **whitening** (the dominant lever — ~84% of the fix, see Empirical validation) plus **one-to-one assignment** (secondary) and **calibration**. Exact-id overlap is a high-confidence *bonus* (rare). **Generic-skill down-weighting is dropped** — whitening already subsumes it (validated). **The taxonomy hierarchy is NOT relied upon** (deemed unreliable).
- **Coverage ∈ [0,1] feeds `p_hat` as an additional factor** so unachievable jobs are demoted — `u_hat × p_hat` stays intact. **Calibrate `min_ess_share` with sparse/niche users in mind** (a high bar gates them out of *everything* — false-ineligible); always **demote, never hard-filter**.
- **Count-dependence is mitigated by the two-factor product — but only with the *whitened* concat (Design A).** Validated on the **real Gemini concat**: `corr(#skills,·)` falls 0.73 → **~0.15** when `skills_fit` is the **whitened** concat (which is count-neutral, corr ≈ 0); with the **raw** concat the product stays **+0.71** (no help). So count-robustness is **coupled to shipping Design A** — the same whitening that de-compresses ranking. **Both** cheap metric tweaks were tested and **rejected** (soft coverage; whitened de-dup). Any residual edges → **demote-not-filter + graded badge**, not a metric tweak.
- **Baseline first, measure, escalate** — and we measured: **the cross-occupation separation test was RUN and the bi-encoder baseline separates (AUC ≈ 0.90–0.97 on occupations and real users), so a cross-encoder is NOT currently indicated.** It stays a contingency if live data degrades. **No LLM-as-judge** — it would massively increase inference time and matching must run fast/live.
- **Scope: `/match_v4` only.** Do not change the shared matcher used by the v2/v3 experiment endpoints.

## On the metric question (cosine vs dot vs L2) — it's a red herring
The problem isn't cosine, it's the **geometry of the space** (anisotropy: everything clustered in a narrow cone). The meaningful "change of metric" is exactly **whitening**: subtract the mean, rescale by the inverse covariance (Σ^(−1/2)), re-normalize — a **Mahalanobis** similarity that accounts for the embedding cloud's covariance, which plain cosine/dot/L2 ignore. The vectors are L2-normalized, so **dot = cosine** and **Euclidean is monotonic in cosine** — swapping the operator changes nothing. **Raw (un-normalized) dot product** would only help if vector norms encoded skill specificity/IDF — Gemini norms don't, so no. **Bottom line: keep cosine; change the space (whiten).** Count imbalance is handled by **assignment**, not by dividing by skill counts (cosine already handles per-vector magnitude).

## Key assets that already exist (reuse, don't rebuild)
- **Whitened skill embedding artifact** `skill_embedding_model_gemini_whitened.pt`, built by [build_whitened_embedding.py](backend/app/services/build_whitened_embedding.py): same `skill_to_row.json` mapping + dim 3072; after whitening random pairs ≈ 0 (sd ≈ 0.035), identity ≈ 1, metadata `target_max_p999 ≈ 0.252`. **The core fix.**
- **Rescale pattern** in [skills_utility/skills_match.py](backend/app/services/skills_utility/skills_match.py) (`SKILL_RESCALE_TARGET` hydrated from `target_max_p999`; `min(1, cos/target)`) — copy.
- **Exact-id overlap** is free (user & job skills resolve to the same internal id via `_resolve_label`/`_resolve_origin_uuid`); but with ~14k fine-grained skills it is **rare**, so it's a bonus, not the backbone.
- **Cross-encoder infra already exists** (used for job reranking) — available for the escalation path, run only on the top-K shortlist.
- **The matcher is a shared singleton** (`_get_matcher`, [match_concat_gemini_ce_service.py:57](backend/app/services/match_concat_gemini_ce_service.py#L57)) used by v2/v3/v4 + retrieval detail. → introduce a **separate v4 matcher**; do not repoint it or global `EMBEDDING_MODEL_PATH`.

---

## Design

### A. Ranking — de-compress the combined skills-fit (whiten the concat embedding)
`p_hat`'s skills-fit factor stays the **combined (concat) skill-set embedding cosine** — computed in a **whitened** space so it varies meaningfully and skills move the order. Honours the colleague's pooling; only fixes compression.
- Fit a whitening transform (`μ`, `Σ^(−1/2)` with Tikhonov shrinkage) on the concat-embedding distribution; **apply in-process at load/request time** to the user concat vector and job/occupation concat vectors (no stored-vector migration).
- Caveat: this also changes **stage-1 retrieval ordering** (retrieval uses the same concat cosine) → recalibrate and A/B via the analysis script; keep behind a toggle.
- **Validated bonus:** the whitened concat `skills_fit` is **count-neutral** (corr ≈ 0 with #user skills, vs +0.28 for the raw concat), so Design A *also* removes the count-bias of the coverage demotion in `p_hat` (product count-corr 0.73 → ~0.15) — it doubles as the count-dependence fix. Rescale the whitened concat to [0,1] (p99 target ≈ 0.09) to use it as a probability factor.

### B. Gate — robust per-skill coverage (the baseline, and the *primary* skills signal for the gate)
A v4-only per-skill matcher (do **not** touch the shared singleton):
1. **Whitened (Mahalanobis) skill embeddings + rescale** — purely data-driven, **no taxonomy graph**.
2. **One-to-one assignment** (Hungarian / greedy) instead of max-over-all-user-skills: each user skill covers ≤1 essential skill ⇒ extra user skills can't inflate coverage.
3. **(Dropped) Generic-skill down-weighting.** The prototype showed whitening already collapses "genericness" (post-whiten ≈ 0.003, flat across skills), so down-weighting contributes ~0%. Leave it out. *If* a future calibration ever shows a residual, the lever would be ESCO "skill reuse level" metadata (not the unreliable hierarchy) — but it is not part of this plan.
4. Per essential skill: `meets = exact_id_overlap OR (whitened_assigned_sim ≥ calibrated τ)`.
5. `coverage = Σ(weight·met) / Σ(weight)` over essentials ∈ [0,1]; **unresolved essentials count as not-met**.
6. `is_eligible = coverage ≥ V4_FULL_MIN_ESS_SHARE` (default 0.0 → **0.5**, tunable). Annotation now accurate.

### C. Coverage feeds `p_hat` (demote unachievable) — formula intact
`p_hat = skills_fit(whitened concat) × coverage^γ × demand × popularity` — every factor in [0,1], so `p_hat ∈ [0,1]` and `final = u_hat × p_hat` is unchanged.
- The two skill-derived factors measure **different** things: *skills_fit* = broad profile fit (smooth, robust); *coverage* = essential-requirement satisfaction (the gate). A job with high overall fit but missing must-haves (woodworker × Fullstack) is demoted by the coverage factor.
- `γ` (`V4_FULL_COVERAGE_GAMMA`) tunes demotion strength to avoid over-crushing (watch for double-penalizing skills; use a small γ or a coverage floor); calibrate.

### D. Robustness ladder + decision rule (measure, don't assert)
> **Status: the decision rule has been RUN (see Empirical validation) — the bi-encoder baseline separates (AUC ≈ 0.90–0.97), so we ship it. The escalation options below are contingencies, not planned work.**
- **Floor (this plan):** whitened combined (ranking) + whitened-assignment coverage (gate).
- **Decision rule:** the **cross-occupation self-match matrix** — synthesize a "graduate of occupation A" from A's essential skills, measure coverage against every occupation B. If the diagonal (A vs A) separates cleanly from off-diagonal (A vs unrelated B), the bi-encoder baseline is sufficient. **If it does not separate, escalate.**
- **Escalation options (only if the test fails):**
  - **Cross-encoder on the shortlist (NOT an LLM judge).** A bi-encoder cosine is the weakest semantic match; a cross-encoder that reads *both* skills jointly is far more reliable at "does skill A satisfy requirement B?", and the infra already exists. **Bounded design (required if we escalate):** let the **bi-encoder propose the single best user skill per essential skill** (the assignment), then have the **CE verify only that one pair**, and **only for the final top-K jobs** — that's ≈ `essential_skills × top_K` ≈ a few hundred inferences, batched ~0.1–0.5 s. Live-feasible, but it *is* the thing to watch — which is exactly why **baseline-first, escalate only if measured necessary** matters for the latency budget. The naive all-pairs version (`essential × all user skills × all items`) is **not** live-safe. **We explicitly avoid an LLM-as-judge** — it would massively increase inference time and matching must be fast/live.
  - **Embedding-derived relatedness instead of the taxonomy graph** — treat two skills as "related" if they're **mutual k-nearest-neighbours** (or co-clustered) in the *whitened* space. A self-supervised substitute for the hierarchy: captures "these skills genuinely sit together" from data, without trusting curated edges. More robust than a global threshold because it's local (top-k mutual), not absolute.
  - **Occupation-anchored eligibility** — match user→occupation, then judge a job via its occupation's curated essential-skill bundle (clean, complete skill sets).

---

## Empirical validation (prototype — read-only, `backend/skill_match_prototype.py`)
We ran the decision-rule experiment on local data with **no production/pipeline edits**: the
cross-occupation self-match matrix (occupation skill bundles) and a real-user split-half + user→job
coverage test on the **kenya** and **njila** datasets. Configs: `raw+max` (current), `whitened+max`,
`whitened+assign`, `+dw` (generic down-weighting). Findings:

1. **Whitening is the dominant fix (~84%).** Spurious coverage of UNRELATED occupations:
   `raw+max 0.82 → whitened+max 0.36 → whitened+assign 0.27 → +dw 0.26`. Contribution shares of that
   drop: **whitening 84% · max→assignment 16% · generic down-weighting ~0%.**
2. **Generic-skill down-weighting is unnecessary after whitening — dropped.** Post-whitening,
   "genericness" collapses to ≈0.003 and is flat across skills (whitening removes the common direction
   that made generic skills match everything), so down-weighting adds ~0%.
3. **The bi-encoder separates → no cross-encoder indicated.** `whitened+assign` opens a real absolute gap:
   - occupations (split-half, same-occ vs unrelated): diag **0.68** vs unrelated **0.27**, **AUC ≈ 0.97**.
   - real users (split-half, self vs other users): self **~0.61** vs cross **~0.30**, **AUC ≈ 0.90 kenya / 0.95 njila**.
   - `raw+max` is saturated (self≈cross≈0.83; absolute gap ~0.04). **→ ship the bi-encoder baseline; the cross-encoder stays a contingency, not planned work.**
4. **De-saturation confirmed on real users.** Per-user coverage spread across 150 jobs (within-user std):
   `raw+max 0.019 → whitened+assign ~0.10` (≈5–6× more spread); mean coverage `0.83 → ~0.41`. Skills now
   discriminate between jobs for a real user (kenya & njila identical pattern).
5. **Assignment is secondary.** It lowers absolute levels (useful for stable thresholding) but does NOT
   drive separation (`whitened+max` AUC ≥ `whitened+assign`), and the user-skill-**count** dependence
   persists (`corr(#skills, spurious coverage) ≈ +0.7` under both max and assignment). Keep it (cheap);
   whitening is the lever.
6. **Label resolution clean on test data; count-dependence is the real calibration risk.** kenya/njila
   resolved **100%** of skills by label (the `originUUID` fallback was never exercised — but this is
   well-aligned test data, so the `originUUID` pre-requisite still stands for live). Concretely: a niche
   user (woodworker, 17 skills) now covers everything weakly (top ~0.49) — correct, but a high
   `min_ess_share` would gate them out of **everything** (false-ineligible); a broad 150-skill user covers
   most jobs ~1.0. **→ validates "demote via `coverage` in `p_hat`, don't hard-filter"; calibrate
   `min_ess_share` with the sparse-user case in mind.**
7. **What fixes count-dependence — two cheap levers tested, only the two-factor product works.**
   - **Calibrated soft coverage (one global Platt fit): no help.** `corr(#skills, coverage)` stays
     ≈ **+0.72 kenya / +0.67 njila** (a monotonic recalibration can't undo a monotone count effect) and it
     *compresses* discrimination (within-user std 0.10 → 0.06). **Not adopted.**
   - **Whitened de-dup of near-duplicate user skills: no help (slightly worse).** Across thresholds that
     collapse 4–40% of skills, coverage's count-corr stays ≈ **0.72** and the **product slightly WORSENS
     (0.35 → 0.38)** — the count effect is *genuine distinct breadth*, not redundancy, and de-dup sharpens
     the centroid, nudging `skills_fit`'s count-bias up. **Not adopted.**
   - **Two-factor product — the only effective reducer, but ONLY with the *whitened* concat.** Validated on
     the **real Gemini concat** (job side = stored `job_embedding`; user side embedded live), not the proxy:
     the **raw** concat `skills_fit` is too compressed (sd 0.025), so `skills_fit × coverage` stays **+0.71**
     (no help); the **whitened** concat `skills_fit` is **count-neutral** (corr ≈ **+0.00**), and the product
     (whitened→[0,1] × coverage) drops the count-corr to **≈ +0.15** (from +0.73 for coverage alone) — even
     better than the proxy's +0.3.
   → **Rely on the two-factor `p_hat`; do NOT add soft coverage or de-dup.** Crucially, **count-robustness is
   coupled to Design A**: the *same* whitening that de-compresses ranking also makes `skills_fit` count-neutral
   and removes the demotion's count-bias — with the **raw** concat it does **not** self-correct. The whitened
   concat must be rescaled to [0,1] (p99 target ≈ 0.09) to act as a `p_hat` factor. Handle any residual edges
   via **demote-not-filter + graded badge**. (Concat-whitening was fit here on ~2.2k vectors with shrinkage;
   a production artifact should use a larger corpus.)
8. **Real-concat count-dependence numbers (`--mode concat`, user concat embedded live via Gemini, job side =
   stored `job_embedding`).** `corr(#user skills, metric)` — closer to 0 = less count-driven:

   | metric | kenya | njila |
   |---|---|---|
   | per-skill coverage (whitened+assign) | +0.726 | +0.666 |
   | RAW concat `skills_fit` (sd ≈ 0.025) | +0.277 | +0.173 |
   | **WHITENED concat `skills_fit`** (count-neutral) | **+0.003** | **+0.014** |
   | product: RAW concat × coverage | +0.715 | +0.650 |
   | **product: WHITENED→[0,1] × coverage** | **+0.159** | **+0.137** |

   Takeaway: whitening the concat makes `skills_fit` count-neutral, so the two-factor product's count-bias
   collapses from ~0.73 to ~0.15 — **but only with the whitened concat (Design A); the raw concat does not
   self-correct.** [0,1] rescale target (p99 of whitened concat sim) ≈ 0.085 (kenya) / 0.092 (njila).

Re-run: `python backend/skill_match_prototype.py --mode users` (real users) / `--mode occ` (occupations) / `--mode concat` (real Gemini concat two-factor check).

---

## Detailed changes (file → function)
- **[config.py](backend/app/config.py):** `V4_FULL_EMBEDDING_MODEL_PATH` (whitened skill artifact; leave global `EMBEDDING_MODEL_PATH` untouched); `V4_FULL_SIM_THRESHOLD` → calibrated *rescaled* τ; `V4_FULL_MIN_ESS_SHARE` 0.0 → 0.5; new `V4_FULL_COVERAGE_GAMMA`; concat-whitening artifact path + on/off toggle. (No generic-skill weighting knob — dropped per Empirical validation. Also drop the previously-considered `V4_FULL_USE_TAXONOMY_TIER` / `V4_FULL_SKILL_DEMOTE_BETA`.)
- **[cosine_similarity/skill_score.py](backend/app/services/cosine_similarity/skill_score.py) — `CosineSkillMatcher`:** accept `model_path`/`rescale_target`; load whitened + rescale; replace `score_pair`'s max-of-all with **assignment**; emit per-essential `exact`, rescaled `similarity` (+ `similarity_raw`); return `coverage`.
- **[match_concat_gemini_ce_service.py](backend/app/services/match_concat_gemini_ce_service.py):** `_get_v4_matcher()` (whitened, v4-only); add the in-process **concat-whitening** transform to the user + job/occupation concat vectors feeding the retrieval/`p_hat` cosine (toggle; v2/v3 unaffected).
- **[match_v4_formatting.py](backend/app/services/match_v4_formatting.py):** `build_matched_skills` → `meets_threshold = exact OR (rescaled_sim ≥ τ)` (no taxonomy term); add `match_tier`/`similarity_raw`; `is_eligible_from_skills` on corrected flags with unresolved-as-not-met; surface `coverage`.
- **[gemini_ce_preference_matching/scoring.py](backend/app/services/gemini_ce_preference_matching/scoring.py):** compose `p_hat = skills_fit × coverage^γ × demand × popularity` (add the coverage factor); **keep `final = combine(u_hat, p_hat)` exactly.** (Removes the earlier `final × skill_factor` idea, which would have changed the formula.)
- **New offline scripts (read-only / build):** concat-embedding whitening builder (analogous to `build_whitened_embedding.py`); calibration harness (see appendix).

---

## Pre-requisite: skill-label resolution & `originUUID`
The per-skill gate resolves each user/job skill to a canonical id **label-first** (`_resolve_label`), with an `originUUID` → `UUIDHISTORY` **fallback** that recovers skills renamed across taxonomy versions. A resolution failure **silently drops** the skill → understated coverage → **false-ineligible** users. So, *before* relying on per-skill coverage:
- **Audit the miss rate with existing telemetry (no new script).** `CosineSkillMatcher.get_resolution_stats()` already returns `total_misses` / `distinct_missed_labels` / `top_misses` and is emitted by `run_cosine_vs_legacy.py` / `run_bm25_cosine_hybrid.py`; surface it from a `run_match_v4_local.py` run (one-line dump of `_get_matcher().get_resolution_stats()`) on the kenya/njila/live datasets. A high miss rate ⇒ taxonomy drift ⇒ fix upstream before trusting the gate.
- **Keep resolution label-first (unchanged); ensure the platform sends a non-empty `originUUID`** per skill so the drift fallback actually fires. NB: `Skill.originUUID` is a *required* field in the schema, but an empty string passes validation and **silently disables** the fallback — confirm the live payload sends the real ESCO/Tabiya UUID, not `""`.
- **Blast radius of a missing/empty `originUUID` is narrow:** it is consumed *only* by the `CosineSkillMatcher` user-side fallback — **not** the combined/ranking embedding (labels only), **not** the job side (resolves via `id`/`tabiya_skill_id`), **not** `SkillScorer` (label-only, no UUID fallback). The separate top-level `skill_groups_origin_uuids` feeds only the v1 `/match` group signal (v4's `grp` is already null).

## Calibration (also the escalate/stop decision)
Full methodology in the appendix below. In short: pick τ from the **negative (random/cross-domain) vs positive (altLabels)** distributions at a target FPR (~1–5%); the **cross-occupation matrix** decision rule has **already been run and the baseline separates** (escalate only if live data degrades); freeze an **adversarial probe battery** as a regression test; then tune `min_ess_share` and `γ` against the analysis-script outputs.

## Rollout / back-compat / risk
- **v4 isolation** behind `V4_FULL_*` config + a dedicated matcher; rollback = config. v2/v3 + hybrid untouched.
- **Concat-whitening changes retrieval ordering** (product-affecting) → A/B before merge; keep behind a toggle.
- **Two skill factors in `p_hat`** risk over-demotion → control with `γ`/floor; validate.
- `popularity` factor of `p_hat` still unimplemented (separate follow-up).
- **Recalibrate τ whenever either whitened artifact is rebuilt** (τ is tied to its `target_max_p999`); assert artifact identity in the test.
- Consumer-facing: `meets_threshold`/`is_eligible` become discriminative; `similarity` rescaled (surface `similarity_raw`); rankings shift (coverage factor + whitened concat). Additive fields, no data migration.

---

## Architecture & runtime — checklist for the dev team
The plan keeps the **live request path nearly unchanged** by pushing cost offline. Things to own/check:

- **Load models at startup, never per request.** The v4 whitened skill matrix (~170 MB) loads **once at warmup**, in addition to the existing raw matrix used by v2/v3 (so ~2 matrices in RAM). Ensure (a) it's loaded in `warmup_on_startup` and gated behind the **readiness probe** so the **first live users don't pay the load**, and (b) container memory is sized for both. It must never be constructed inside a request.
- **Jobs need a pre-computed whitened / Mahalanobis-queryable CONCAT vector — an UPSTREAM change, probably outside this repo.** The ranking factor (`skills_fit` = whitened *combined* embedding cosine) uses each job's stored concat embedding. Whitening the whole corpus *per request* is too slow (~0.3–2 s), so the whitened (or mean-centered) concat vector should be **pre-computed and stored where job embeddings are produced — i.e. the embedding/enrichment pipeline + the jobs DB** (you're right: likely at scrape/enrichment time, not in this matching repo). Caveat: the **whitening transform (μ, Σ⁻¹ᐟ²) is owned here** (an artifact, like the skill whitening) and must be **shared with that upstream pipeline** so the identical transform is applied — recalibrate them together. (Short-term you *can* whiten at load to prototype, accepting the latency, before investing in upstream storage.)
- **The eligibility GATE is self-contained — no upstream/job change needed.** It matches the job's existing essential-skill **IDs** against the whitened **skill** matrix (the startup artifact in this repo); it does **not** use the job's concat vector. So the eligibility fix ships entirely within this repo; only the ranking (`p_hat`) improvement needs the upstream vector.
- **Otherwise per-request cost ≈ 0.** The per-skill pass already runs on the shortlist; we swap raw→whitened skill lookups (startup-loaded), replace max-over-user-skills with a cheap one-to-one **assignment** (~k³ on tiny matrices), and add a couple of multiplications (`coverage^γ`, the small user-vector transform). **Rule: never whiten the job corpus inside a request.**
- **Pre-existing bottleneck (orthogonal to this plan):** loading ~2,000 job embeddings from Mongo per request already cost ~13 s in the live test — that is today's real live-latency issue. Worth a separate look (caching / vector index / projection). Prefer **not** storing a *second* per-job vector (use the Mahalanobis-query/mean-centered form, or store whitened *instead of* raw for v4) so this payload doesn't double.
- **Payload must send a non-empty per-skill `originUUID`.** It's the per-skill matcher's only drift-recovery fallback. `Skill.originUUID` is required by the schema but an empty string silently disables the fallback → renamed/aliased skills drop → users wrongly under-credited. Confirm the live payload carries the real ESCO/Tabiya UUID; audit the label miss rate via `get_resolution_stats()` (see *Pre-requisite* section).

---

## Rollout safety & phasing
**Key fact: correctness does NOT depend on the job-DB whitening.** The matching repo applies the whitening transform `(μ, Σ^-1/2)` *in-process* to **both** the user and job concat vectors, so the DB-side precompute is a **latency optimization, not a prerequisite**. The plan's two parts have very different dependencies, so phase them.

> **Core safety rule — never deploy a half-whitened state.** The whitening transform must hit **both** sides with the **same artifact**. Whitening one side and comparing it to the raw other side yields meaningless cosines → broken ranking **and** a broken stage-1 retrieval shortlist. Gate the concat-whitening behind a **single toggle that flips both sides together**.

### Phase 1 — Eligibility gate (Design B). No DB dependency → ship first.
- Whitened per-skill matcher (startup artifact already in this repo) + one-to-one assignment + coverage + accurate `is_eligible` + graded badge.
- Touches only: a v4-only matcher instance, `match_v4_formatting`, and surfacing `coverage`. **No job-DB change, no concat whitening**, negligible added latency (reuses the per-skill pass that already runs).
- Pre-reqs (from earlier sections): load the ~170 MB whitened skill matrix at **warmup behind the readiness probe**; audit the label-resolution miss rate (`get_resolution_stats()`); confirm the live payload sends non-empty `originUUID`.
- **Independently shippable** — delivers the "meaningful eligibility" win on its own.

### Phase 2 — Ranking de-compression + coverage demotion (Design A + `coverage→p_hat`). In-process whitening, toggled.
- Whiten the **concat** embedding for `p_hat`'s `skills_fit`, computed **in-process** from the raw `job_embedding` loaded from Mongo (no DB change); whiten the user concat at request time; **same transform artifact on both sides**. Rescale the whitened concat to [0,1] (p99 target ≈ 0.09) so it's a valid probability factor.
- Feed `coverage^γ` into `p_hat`; `final = u_hat × p_hat` unchanged.
- **Behind a single both-sides toggle**; A/B the retrieval-ordering shift via `analyze_match_v4_results.py` before enabling (product-affecting).
- **Latency:** per-request corpus whitening ≈ **+0.5–1 s** for ~2k jobs (on top of the existing ~13 s corpus load). **Mitigate with the Mahalanobis-query form** (precompute `u' = (u−μ)Σ^-1` per user; mean-center jobs) → ~zero added cost, still no DB change. (Caveat: that form drops per-vector renormalization → calibrate on it.)
- This phase delivers the **count-robustness** (validated: it requires the whitened concat) and lets skills move ranking. If deferred, Phase 1's gate still works, but the `coverage` demotion carries the count-bias (+0.71 vs ~0.15) and skills barely move the order.

#### Phase 2 — IMPLEMENTED (2026-06-13). Toggle `V4_FULL_RANK_DEMOTE` default **ON**, `γ=1.0`.
**Design deviation from the plan above (important):** whitening is applied **only to the CE shortlist at the scoring step**, NOT to the whole corpus during retrieval.
- `p_hat`'s skills-fit is recomputed per shortlist candidate as `min(1, max(0, cos_whitened)/target)` from the snapshotted `job_embedding` (taken *before* retrieval pops it) and the once-whitened user concat; `final = u_hat × p_hat × coverage^γ` (demand tilt still multiplies for occupations). **`final = u_hat × p_hat` combination preserved.**
- Consequences vs the plan: (a) **retrieval ordering & the shared v3 path are untouched** (only v4's final re-rank changes) — cleaner isolation; (b) **no per-request corpus whitening → no Mahalanobis-query needed**, latency cost ≈ the matmuls over ~`final_top_k` vectors (negligible); (c) the cost is that an achievable job the *saturated* stage-1 (raw concat, sd≈0.02) buried below the shortlist can't be recovered. **Mitigation:** v4-only shortlist widened — `MATCH_V4_RETRIEVE_TOP_K=100` (was 50), `MATCH_V4_FINAL_TOP_K=50` (was 30). (Phase 3's DB-precompute would let retrieval itself use the whitened space; deferred.)
- **Artifact:** `backend/resources/models/concat_whitening_gemini.npz` (μ, W=Σ^-1/2, p99 `target`=0.106), built by `app/services/build_whitened_concat.py`. **Fit on the local snapshot `data/kenya_jobs_for_pipeline.json` (~2017 jobs) — REFIT on the full/live corpus for production** (and ideally include occupation + representative user concat vectors; today it's job-fit applied to all three).
- **Verification (32 users, baseline vs Phase 2; `compare_p2.py`, `sweep_gamma.py`):** invariant 0 violations; `p_hat` sd 0.017→0.322; corr(rank,coverage) +0.03→**−0.42** opp / −0.34 occ. γ sweep (opps): weak-items-in-top-5 1.41→0.66(γ.25)→0.41(γ.5)→**0.16(γ1.0)**; cov@1 0.52→0.88(γ1.0). γ=1.0 chosen for maximal achievability ordering.
- **Deployment review / risks:**
  - **Artifact must ship** — committed to git (no LFS; consistent with the 82 MB `.pt` files). If absent/incompatible (dim≠`EMBEDDING_DIM` or target≤0), the loader disables whitening AND `run_match_v4_full` falls back to **pure Phase 1** (annotation-only, no demotion) with a logged error — it does *not* half-apply the count-biased `raw p_hat × coverage`.
  - **Rollback:** set env `V4_FULL_RANK_DEMOTE=false` (no redeploy).
  - **Memory:** W is 3072² float64 ≈ 75 MB resident (warmed at startup when the toggle is on).
  - **Latency:** CE rerank now runs on ~100 candidates (was 50) × (jobs+occ) — monitor; dial back via the `MATCH_V4_*` env vars.
  - **Consumer-facing:** `/match_v4` now returns up to **50** opportunities (was 30); `p_hat` spans **0–1** (was ~0.8–0.94) and `final_score` can be **0.0** for unrelated/zero-coverage items — any downstream recommender consuming `p_hat`/`final_score` should be checked.

### Phase 3 — Move the whitened job concat vector upstream (jobs DB / embedding pipeline). Pure latency optimization.
- Precompute & store the whitened (or mean-centered, for Mahalanobis-query) job concat vector where job embeddings are produced (scrape/enrichment + jobs DB), so the matching repo reads it directly and skips per-request whitening.
- **HARD REQUIREMENT:** the transform artifact `(μ, Σ^-1/2)` is owned in this repo and must be **shared and version-pinned** with the upstream pipeline (assert by artifact hash). A drift between the transform applied upstream (jobs) and in-repo (users) **silently breaks ranking** — this is the *real* hazard, and it's introduced by doing the DB side carelessly, not by deferring it.
- Fit the transform on a **large corpus** (the prototype used only ~2.2k vectors + shrinkage; production should use the full job corpus or more for a stable covariance).
- Consider storing whitened **instead of** raw for v4 (or projecting at read) to avoid doubling the per-request payload (the ~13 s corpus load is today's real bottleneck).

### What you can safely do *now*
- **Phase 1 alone** → eligibility fixed, zero DB dependency, minimal latency. ✓
- **Phase 1 + Phase 2** (in-process, both-sides toggle) → eligibility + ranking + count-robustness, still **no DB dependency**, at a modest latency cost (≈0 via Mahalanobis-query). ✓
- **Never:** whiten only one side; or precompute DB job vectors with a transform that isn't the exact artifact the matching repo uses for users.

### Dependency summary
| Change | Needs job-DB change? | Correctness risk if DB not done | Latency if DB not done |
|---|---|---|---|
| Phase 1 — gate (Design B) | No | None | None |
| Phase 2 — ranking (Design A, in-process) | No | None *(if both sides whitened)* | +~0.5–1 s/req (≈0 with Mahalanobis-query) |
| Phase 3 — DB precompute | Yes (upstream, version-pinned) | — | removes the Phase-2 latency |

## Verification recipe (from `backend/`)
1. **Baseline:** `python run_match_v4_local.py` + `python analyze_match_v4_results.py <run_dir>` → confirms saturation (`pct_eligible≈100%`, compressed essential cosine, woodworker×Fullstack `is_eligible=True`).
2. Apply changes; calibrated τ, `min_ess_share=0.5`, chosen `γ`.
3. **Re-run + analyze** → `pct_eligible` well below 100%, coverage spread widens; **balance** section shows skills now move ranking (within-user skills-fit sd up; rank↔coverage corr > 0).
4. **Woodworker case:** `--user <woodworker> --item "developer"` → `is_eligible=False`; software essentials NOT met; furniture↔software `match_tier=none`; job **demoted** (final drops via the coverage factor).
5. **Calibration audit:** τ, FPR/recall, **cross-occupation separation** (pass ⇒ baseline OK; fail ⇒ escalate to cross-encoder); adversarial battery green.
6. **Regression guard:** v2/v3 experiment outputs unchanged.

## Open / tunable params
- τ (calibration), `min_ess_share` (0.5? — calibrate so sparse/niche users aren't gated out of everything), `γ` (demotion strength), concat-whitening on/off + retrieval recalibration. (Generic-skill down-weighting dropped — whitening subsumes it.)
- **Escalation decision** (cross-encoder vs embedding-derived k-NN vs occupation-anchored) — determined by the cross-occupation test, not pre-committed. **LLM-as-judge is excluded** (latency).
- `popularity` factor of `p_hat` — future.

---

# Appendix: how to test & find a good skills-match threshold

**What we're calibrating** is mainly one number: `τ` = the **whitened, rescaled cosine threshold** above which a user skill "covers" a job essential skill. (`min_ess_share` and the coverage exponent `γ` are tuned *after* `τ`, via the analysis script.)

**The central difficulty:** there is no ground-truth labelled set of "user skill X covers job skill Y." So we **triangulate** several proxies and choose a `τ` where they agree. Don't trust any single method.

## The two errors we are trading off
- **False-eligible** (the bug): woodworker's *repair furniture frames* counted as covering *debug software*.
- **False-ineligible**: a genuine but differently-labelled skill rejected (hurts recall / shrinks a thin youth result set).

Pick a **target per-skill false-positive rate** (~1–5%) and report the recall paid. Frame the pick as "at τ=X, unrelated pairs are wrongly accepted ~F% of the time, and we retain R% of genuine synonyms."

## Tests (label-light → label-heavy)

**A. Intrinsic distribution separation (free, first pass).** Score a NEGATIVE set (random + cross-domain pairs) and a POSITIVE set in the whitened+rescaled space.
- Positives: a skill vs its own **altLabels** (the trustworthy positive ≈ identity). Shared-skill-group siblings can be used only as a **noisy** secondary proxy (we don't rely on the hierarchy in production, and it's noisy as ground truth too — report metrics separately and weight altLabels).
- Metrics: ROC-AUC, **Youden's J**, separation margin = (negative p99) vs (positive p1). Choose τ at the target FPR (≈ negative p99 for ~1%).

**B. Cross-occupation self-match matrix — the headline exercise (free, end-to-end, the decision rule).** ~436 occupations × essential skills. For occupation **A**, synthesise a "graduate of A" = A's essential skills; measure coverage against every occupation **B** → a coverage matrix across a grid of τ.
- Expect diagonal (A vs A) ≈ 1.0, off-diagonal low except genuinely related occupations (Carpenter↔Joiner high; Carpenter↔Software Developer ≈ 0).
- Pick τ that **maximises (diagonal − off-diagonal)** with diagonal ≥ ~0.9. **If no τ separates them, the bi-encoder can't do per-skill competence → escalate to the cross-encoder.** Needs no human labels, uses real skill bundles, tests the actual coverage gate. *(Already run on local data — it **does** separate: AUC ≈ 0.97 occ / 0.90–0.95 real users — so the baseline is sufficient; see Empirical validation.)*

**C. Adversarial probe battery (cheap; becomes a regression test).** Must-be-negative (~20: `debug software ↔ repair furniture frames`, `SQL ↔ digitise documents`) + must-be-positive (~20: `debug software ↔ troubleshoot software`, `Java ↔ object-oriented programming`, altLabels, identical). τ must separate them; freeze as `pytest` so rebuilds/config edits can't silently re-saturate.

**D. Borderline real-pair audit (targeted labelling).** From a harness run, collect `per_job_skill` pairs, stratify by rescaled cosine, sample ~150–300 in the **borderline 0.05–0.40** band, one annotator labels "covers/doesn't"; precision/recall vs τ.

**E. Job-level human-judgment set (most aligned, most effort).** ~50–100 (user, job) "could realistically do / aspirational" labels (incl. woodworker × Fullstack = no, × Carpenter = yes). Sweep τ **and** `min_ess_share`; measure agreement with `is_eligible`.

## Recommended workflow
1. **A** + **B** → candidate τ (FPR ~1–5%; diagonal ≥ 0.9). Expect ~0.10–0.15 rescaled.
2. **C** → confirm forced pairs separate; if not, escalate (cross-encoder).
3. Plug τ into the harness (`V4_FULL_SIM_THRESHOLD=<τ>`), run `run_match_v4_local.py` → `analyze_match_v4_results.py`; read `pct_eligible`, coverage spread, woodworker case.
4. Sweep τ; plot **τ vs pct_eligible** and **τ vs cross-occupation separation**; pick the **elbow**.
5. Then tune `min_ess_share` and `γ` (analysis-script ranking-spread + woodworker rank).
6. Validate on **D**/**E** and a second dataset (kenya + njila), jobs and occupations.

## Deliverable: offline calibration harness (read-only, no Gemini/Mongo)
`backend/calibrate_skill_threshold.py`: load whitened matrix + `target_max_p999`, `skills.csv`; build negative/positive sets (A) and the cross-occupation matrix (B) from the local occupation corpus; sweep τ (e.g. 0.00–0.50 step 0.01) emitting per-τ FPR, recall, F1, ROC-AUC, diagonal/off-diagonal/separation, implied `pct_eligible`; run the adversarial battery (C) as pass/fail; print a recommended τ + CSV. Reuse the build script's seeded RNG and the same `min(1, cos/target)` rescale so τ transfers 1:1 to `V4_FULL_SIM_THRESHOLD`.

## Guardrails
- **Recalibrate whenever a whitened artifact is rebuilt** — τ is tied to its `target_max_p999`; store τ with the artifact metadata and assert the artifact hash in the test.
- Calibrate on **held-out** skills (don't pick τ on the exact altLabel pairs you then report recall on).
- Keep the adversarial battery (C) as a committed regression test — the cheap early warning that the gate has re-saturated.
