# AI / Matching Logic — Test Plan

> **Status:** Implemented (Phases 1–3; corpus audit deferred).  
> **Location:** All new tests live under `backend/tests/`.

---

## 1. What we already have

| Layer | Folder | What it checks | What it does **not** check |
|-------|--------|----------------|----------------------------|
| Request shapes | `data_validation/test_request_validation.py` | County strip, v5 inheritance | Matching behavior |
| Response shapes | `data_validation/test_response_contracts.py` | All Swagger fields exist | Whether scores are correct |
| API wiring | `data_schema/` | Endpoints registered, auth | Ranking logic |
| Smoke | `smoke/` | 4 response keys on all 5 endpoints (mocked matchers) | Real scoring |
| Mongo mapping | `unit/test_build_job_dict_from_ranked.py` | Job dict fields from Mongo | Whether jobs rank well |

**Summary:** We validate **input/output contracts** and **DB mapping**. The **matcher is mocked** in smoke tests — no real AI logic is exercised yet.

---

## 2. What we need to add (agreed scope)

### A. Behavior invariants (CI — fast, no Mongo, no Gemini)

Rules the matcher must **always** follow.

| Area | Examples |
|------|----------|
| **Location** | Remote jobs match any user; user city matches local jobs; wrong province excluded |
| **Education** | `any_post_secondary_educ=0` drops jobs with `requires_post_secondary`; occupations unaffected |
| **Skill gaps** | Count ≤ `top_k`; no skill user already has; empty user skills → `[]` |
| **Ranking shape** | Ranks 1..n contiguous; `final_score` in valid range; required item fields present |
| **ZQF (v5)** | `zqf_eligible` / `zqf_gap` correct on opportunities |
| **Cross-endpoint** | Same user+jobs: education gate consistent across engines; all respect `skill_gap_top_k` |
| **Adversarial** | No skills / bad embedding dim / empty corpus → no crash, graceful empty results |

### B. Component + metamorphic tests (CI — mocked ML)

| Area | Examples |
|------|----------|
| **Skill scorer** | Same skill → high similarity; unrelated → low; threshold behavior |
| **Preference scorer** | Strong preference + matching job → contribution; see §4c below |
| **Metamorphic** | Add relevant skill → score does not drop; reorder skills → same output; duplicate skill → no change |

### C. Mocked-embedding ranking tests (CI — deterministic “AI”)

Patch Gemini / cross-encoder to return **fixed vectors** so we know the expected winner:

- User vector = Job A vector → Job A ranks #1  
- User vector ⊥ Job B → Job B ranks low  

Tests the **pipeline wiring**, not Gemini quality.

### D. Corpus data-quality audit (manual / staging — real Mongo)

Script that reports what % of active jobs have:

- essential skills  
- valid embeddings (`vector_bin` or `job_embedding`)  
- education field  
- ZQF fields  
- location (city / province / remote)  

**Not a CI gate** — run before client handoff or after a new scrape.

### E. Offline ranking eval (optional — real Gemini + Mongo)

Small labeled set (user → relevant job UUIDs). Run `run_local.py`, compute MRR / Precision@k.  
**Only worth it if you have labeled relevance data.** See §4d.

---

## 3. Proposed folder layout (under `backend/tests/`)

```
backend/tests/
├── ml_logic/                          # matching invariants (pure functions + small fixtures)
│   ├── conftest.py
│   ├── test_education_gate.py
│   ├── test_location_matching.py
│   ├── test_skill_gap_invariants.py
│   ├── test_zqf_annotation.py
│   ├── test_cross_endpoint_consistency.py
│   └── test_adversarial_inputs.py
├── components/                        # NEW — individual scorers
│   ├── test_skill_scorer.py
│   ├── test_preference_scorer.py
│   └── test_metamorphic_matching.py
├── integration/                       # NEW — mocked embeddings (deterministic AI pipeline)
│   ├── test_match_v4_mocked_embeddings.py
│   └── test_match_legacy_fixtures.py
├── sanity_checks/
│   └── ml_logic_check.py              # runs ml_logic/ + components/ + integration/
└── AI_MATCHING_TEST_PLAN.md           # this file
```

---

## 4. Answers to your questions

### 4a. Why not test with real Gemini?

| Reason | Detail |
|--------|--------|
| **CI speed** | One embed call = seconds; full suite should finish in &lt;10s |
| **Cost** | Every PR run burns API quota |
| **Flakiness** | Network, rate limits, model drift |
| **Determinism** | Hard to assert “Job A ranks #1” when vectors change |

**Recommendation**

- **CI (default):** mock embeddings → deterministic pipeline tests  
- **Optional later:** a separate **manual or nightly** job with real `GEMINI_API_KEY` on 2–3 tiny users — not blocking merge  

We are **not** saying “never use Gemini.” We separate **logic tests** (mocked, every push) from **quality eval** (real Gemini, occasional).

---

### 4b. Metamorphic tests — agreed

Implement in `components/test_metamorphic_matching.py`. No golden rankings; only relationships that must hold after algo changes.

---

### 4c. Skills only, no preferences — what should happen?

**Today (important):**

- Omitting `preference_vector` fills **neutral 0.5** on every field (Pydantic default).
- On **v4**, `final_score = u_hat × p_hat` (or geometric mean).
- Neutral preferences → `u_hat ≈ 0.5`, so **final score is dampened** even when the client sent “skills only.”
- Preferences are **not** literally added, but neutral `u_hat` still **multiplies** `p_hat`.

**Desired behavior (your ask):**

| User sends | Expected |
|------------|----------|
| Skills only (no preference signal) | `final_score` = skill score (`p_hat` / retrieval score); preferences do not change ranking |
| Skills + real preferences | `final_score` = combined (`u_hat × p_hat`) |
| Neutral preferences explicitly (all 0.5) | Same as skills-only OR clearly documented — team should decide |

**Plan**

1. **Add tests first** that describe the desired rule (may fail today).  
2. **Small product change** (if approved): detect “no preference signal” — e.g. empty `preference_vector` in raw JSON or all fields neutral **and** no `bws_scores` → skip preference combine, use `final_score = p_hat`.  
3. Tests in `components/test_preference_scorer.py` + `integration/test_match_v4_mocked_embeddings.py`.

This is both a **test** and a possible **small code change** — flag for go-ahead.

---

### 4d. Offline MRR / ranking quality — do we need it?

| Need it? | When |
|----------|------|
| **Not for CI** | No labeled data in repo today; rankings change when you tune the algo |
| **Yes for client confidence** | When you have 5–20 users with “good job” UUIDs in Zambia/Kenya test data |
| **How** | Script using `run_local.py` + jsonl; report MRR@10; optional floor (e.g. MRR ≥ 0.25) run **manually** before release |

**Recommendation:** Phase 4 (optional). Corpus audit (D) is higher priority for client handoff.

---

### 4e. Snapshot with tolerance — skip for now

Useful later for one frozen synthetic case. Defer until mocked-embedding tests (C) exist.

---

### 4f. Property-based (Hypothesis) — skip for now

Powerful but adds dependency and complexity. Revisit if hand-written invariants miss edge cases.

---

### 4g. Cross-endpoint consistency — agreed

`behavior/test_cross_endpoint_consistency.py`:

- Same fixture user + jobs through `/match`, v2, v4 (mocked heavy deps).  
- Assert: education filter consistent, response shape identical, `skill_gap_top_k` respected.

---

### 4h. Adversarial / negative cases — agreed

`behavior/test_adversarial_inputs.py`:

- Empty skills, empty jobs, wrong embedding dim, missing `user_id` (per-endpoint behavior documented).

---

## 5. Implementation phases

| Phase | Work | Effort | CI? |
|-------|------|--------|-----|
| **1** | Behavior invariants: education, location, skill gaps, ranking shape, ZQF | ~1 day | Yes |
| **2** | Component + metamorphic + skills-only preference rule (4c) | ~1 day | Yes |
| **3** | Mocked-embedding v4 ranking + cross-endpoint + adversarial | ~1 day | Yes |
| **4** | Corpus audit script (Mongo) | ~0.5 day | Manual |
| **5** | Offline MRR eval (if labeled data exists) | ~0.5–1 day | Manual |

**Total:** ~3–4 days for Phases 1–4.

After Phase 1–3, wire into `run_all_checks.py` as a 7th check: **Behavior Check**.

---

## 6. What changes code vs tests only

| Item | Tests only | Code change needed? |
|------|------------|---------------------|
| Education / location / skill gaps | Yes | No |
| Ranking shape / ZQF | Yes | No |
| Metamorphic | Yes | No |
| Mocked embeddings | Yes (patch) | No |
| **Skills-only → final = skill score** | Yes | **Likely yes** — see §4c |
| Corpus audit | New script | No |
| Offline MRR | New script | No |

---

## 7. Example tests (concrete)

```python
# behavior/test_location_matching.py
def test_remote_job_matches_any_user_city():
    user = {"city": "Lusaka", "province": "Lusaka"}
    job = {"city": "Remote", "province": ""}
    assert _job_matches_user_location(job, user) is True

# behavior/test_education_gate.py
def test_user_without_postsec_excludes_require_ps_jobs():
    user = {"any_post_secondary_educ": 0}
    jobs = [{"uuid": "j1", "requires_post_secondary": True}, {"uuid": "j2"}]
    out = filter_jobs_by_education(user, jobs)
    assert [j["uuid"] for j in out] == ["j2"]

# components/test_preference_scorer.py
def test_skills_only_user_final_score_equals_p_hat():
    # after product rule: no preference signal → final_score == p_hat
    ...

# integration/test_match_v4_mocked_embeddings.py
def test_identical_user_job_vector_ranks_first(mock_embed):
    # user_emb == job_a_emb → job_a rank 1
    ...
```

---

## 8. Go-ahead checklist

Before implementation, confirm:

- [ ] Approve Phases 1–3 for CI  
- [ ] Approve Phase 4 corpus audit (Mongo script)  
- [ ] **Skills-only rule (§4c):** should `final_score = p_hat` when no preferences sent? (likely needs small code change)  
- [ ] Defer Phase 5 (MRR) until labeled relevance JSON exists  
- [ ] Add 7th check to `run_all_checks.py` after Phase 1–3  

**Reply “go ahead” (and §4c decision) to start implementation in `backend/tests/`.**
