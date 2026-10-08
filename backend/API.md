# Matching API

The Matching API recommends **occupations** (career pathways), live **job opportunities** and
**skill gaps** for one or more users, based on their skills, preferences and location.

Matching is done with `POST /match`. Interactive docs (Swagger UI) are served at `/docs` and
the machine-readable spec at `/openapi.json`; both include a description and example for every
field.

- [Endpoints](#endpoints)
- [Authentication](#authentication)
- [POST /match](#post-match)
  - [Query parameters](#query-parameters)
  - [Request body](#request-body)
  - [Response body](#response-body)
  - [Errors](#errors)
  - [Example](#example)
- [Changes](#changes)

## Endpoints

| Method | Path           | Purpose                                                     |
|--------|----------------|-------------------------------------------------------------|
| POST   | `/match`       | Match users to occupations, opportunities and skill gaps.   |
| GET    | `/jobs`        | Browse active jobs (cursor pagination, filters).            |
| GET    | `/jobs/stats`  | Counts over the active jobs catalog.                        |
| GET    | `/health`      | Liveness check: `{"status": "ok"}`.                         |
| GET    | `/docs`        | Swagger UI.                                                 |
| GET    | `/openapi.json`| OpenAPI spec.                                               |

In production, all routes are reached through the API Gateway.

## Authentication

Every API route requires an `x-api-key` header.

- **Through the API Gateway** (production), the key is a Google Cloud API key, validated by the
  gateway. Requests without a valid key get `401`/`403` from the gateway, and requests over quota
  get `429`.
- **Directly against the service** (local or Cloud Run), the service only checks that the header
  is present. A missing header returns `403 {"detail": "Not authenticated"}`.

## POST /match

Send a JSON **array** of `MatchRequest` objects, one per user (use an array of length 1 for a
single user). The response is a JSON array of `MatchResponse` objects in the same order.

At most **32 users** per request. Each `user_id` should be
unique within a request, because results are keyed by it.

How matching works, in brief:

1. **Retrieve**: the user's skills are embedded and compared with every eligible job and
   occupation, and the closest `retrieve_top_k` are shortlisted.
2. **Rerank**: a cross-encoder re-scores the shortlist.
3. **Score**: each item gets
   `final_score = u_hat × p_hat × coverage_factor × location_tier_factor`:
   - `u_hat` is preference utility.
   - `p_hat` is success propensity, driven by skill fit.
   - `coverage_factor` lowers the score when the user meets few of the item's essential skills.
   - `location_tier_factor` (opportunities only) prefers jobs in the user's own area over
     regional or national hubs.

   Items are returned in descending `final_score`, and every factor appears in
   `score_breakdown`.

Jobs are filtered by location and, if `any_post_secondary_educ` is `0`, by the education
requirement.

### Query parameters

All query parameters are optional. Defaults are set per deployment; the values below are the
standard ones.

| Name                   | Type    | Range                       | Default   | Meaning |
|------------------------|---------|-----------------------------|-----------|---------|
| `retrieve_top_k`       | integer | 1–500                       | 100       | Stage-1 shortlist size per user, before rerank. |
| `final_top_k`          | integer | 1–200                       | 50        | Max opportunities returned per user (also the rerank pool size). |
| `final_score_combiner` | string  | `product`, `geometric_mean` | `product` | How `final_score` combines `u_hat` and `p_hat`: `u_hat × p_hat` or `√(u_hat × p_hat)`. |
| `skill_gap_top_k`      | integer | 1–50                        | 5         | Number of skill-gap recommendations per user. |

Up to 10 occupations are returned per user. Some deployments don't offer occupation
recommendations; there, `occupation_recommendations` is always an empty list. Fewer than
`skill_gap_top_k` skill gaps may be returned, because suggestions too far from the user's existing
skills are dropped.

Each deployment works in one language (for example English or Spanish). Labels in the response
are in that language. There is no language parameter on the request.

### Request body

`MatchRequest`: every field is optional, so a partial profile still gets a valid response.

| Field                        | Type              | Default | Meaning |
|------------------------------|-------------------|---------|---------|
| `user_id`                    | string            | null    | Your identifier for the user, echoed back as `MatchResponse.user_id` (`""` if omitted). |
| `city`                       | string            | `""`    | User's city or town; used to prefer nearby opportunities. A trailing " County" is stripped. |
| `province`                   | string            | `""`    | User's province or county; scopes occupations and opportunities to the user's region. A trailing " County" is stripped. Empty means a fallback region is used for occupations. |
| `skills_vector.top_skills[]` | array of `Skill`  | `[]`    | The user's skills (see below). |
| `skill_groups_origin_uuids`  | array of string   | `[]`    | ESCO skill-group UUIDs the user belongs to. |
| `preference_vector`          | `PreferenceVector`| neutral | Job preferences (see below). |
| `any_post_secondary_educ`    | integer (0/1)     | null    | Education gate: `0` excludes jobs that require post-secondary education, `1` keeps them, and omitting the field skips the gate. |
| `number_post_secondary_educ` | integer           | null    | Informational; not used for ranking. |
| `total_duration_postsec`     | number            | null    | Informational; not used for ranking. |

`Skill`

| Field            | Type   | Required | Meaning |
|------------------|--------|----------|---------|
| `originUUID`     | string | yes      | ESCO skill origin UUID. |
| `preferredLabel` | string | no       | Human-readable label; also a fallback if the UUID cannot be resolved. |
| `proficiency`    | number | no       | Proficiency in [0, 1]. |

`PreferenceVector`: every value is in [0, 1]. **0.5 is neutral** (no effect), values above 0.5
mean the user values the attribute, and values below 0.5 mean they are averse to it. Omitted
fields default to 0.5, so leaving a field out means "no signal", not "dislike". Avoid sending `0`
unless you mean a strong aversion.

| Field                | Meaning |
|----------------------|---------|
| `earnings_per_month` | Preference for higher earnings. |
| `task_content`       | Preference for the job's task content. |
| `physical_demand`    | Preference for physically demanding work. |
| `work_flexibility`   | Preference for flexible working arrangements. |
| `social_interaction` | Preference for people-facing work. |
| `career_growth`      | Preference for career-growth potential. |
| `social_meaning`     | Preference for socially meaningful work. |
| `bws_scores`         | Optional. Best-Worst Scaling scores keyed by O*NET work-activity code (e.g. `"4.A.4.a.4": 0.9`). Drives the work-activity part of the preference score. |
| `top_10_bws`         | Optional. The user's ten highest-ranked work-activity codes. |

### Response body

`MatchResponse`

| Field                         | Type                              | Meaning |
|-------------------------------|-----------------------------------|---------|
| `user_id`                     | string                            | Echo of the request's `user_id`. |
| `opportunity_recommendations` | array of `OpportunityRecommendation` | Ranked live jobs (best first). |
| `occupation_recommendations`  | array of `OccupationRecommendation`  | Ranked occupations (best first). |
| `skill_gap_recommendations`   | array of `SkillGapRecommendation`    | Skills that would most improve the user's prospects. |

`OpportunityRecommendation`

| Field | Meaning |
|-------|---------|
| `uuid`, `originUuid` | Job identifier, and its identifier in the source system. |
| `rank` | 1-based rank (1 = best). |
| `opportunity_title`, `employer`, `location`, `URL` | Posting basics. |
| `opportunity_isco_occupation_group`, `opportunity_isco_occupation_group_id`, `related_occupation_id` | Occupation classification of the job. |
| `employment_type`, `contract_type`, `salary_text`, `required_education`, `required_experience`, `posted_date`, `closing_date`, `opportunity_description` | Posting details as published (nullable). |
| `final_score` | Ranking score in [0, 1]; higher is better. |
| `is_eligible` | `true` when the user meets enough of the job's essential skills. |
| `justification` | Short plain-language reason for the recommendation. |
| `score_breakdown` | Components behind `final_score` (see below). |
| `matched_skills` | Per-skill match details (see below). |
| `matched_preferences[]` | Per-attribute preference details: `attribute`, `job_value`, `job_value_label`, `user_weight`, `beta`, `encoded_value`, `contribution` (= beta × user_weight × encoded_value), `matched`. |
| `matched_work_activities` | Work-activity (BWS) details; null when no `bws_scores` were sent. |

`OccupationRecommendation` has the same scoring fields (`rank`, `final_score`, `is_eligible`,
`justification`, `score_breakdown`, `matched_skills`, `matched_preferences`,
`matched_work_activities`) plus `occupation_label`, `occupation_description`, `province`,
`salary_range` (typical monthly earnings level), `typical_tasks[]` (up to 8) and
`career_path_next_steps[]` (currently always empty).

`score_breakdown` (all fields nullable)

| Field | Meaning |
|-------|---------|
| `u_hat` / `preference_score` | Preference utility in [0, 1]. |
| `p_hat` | Success propensity in [0, 1], driven by skill fit. |
| `p_hat_source` | Signal that produced `p_hat` (e.g. `concat_cosine_whitened`). |
| `essential_coverage` | Share of the item's essential skills the user meets, in [0, 1]. |
| `skill_match_level` | `strong`, `partial`, `weak`, or `unknown` (the item lists no essential skills). |
| `total_skill_utility`, `skill_components.ess` | Mean similarity over essential skills, in [0, 1]. |
| `skill_components.opt` | Mean similarity over optional skills, in [0, 1]. |
| `coverage_factor`, `coverage_gamma` | Skill-coverage multiplier on `final_score` (0–1), and the exponent used to compute it. |
| `location_tier_factor` | Location multiplier on `final_score` for opportunities: 1.0 for the user's own area, lower for regional or national hubs. |
| `skill_components.loc`, `skill_components.grp` | Not computed (null). |
| `skill_penalty_applied` | Share of essential skills below the similarity threshold (higher = bigger gap). |
| `demand_score`, `demand_label` | Expected labour-market demand, when known. |
| `preference_score_legacy`, `p_hat_components`, `skill_diagnostics` | Diagnostic or reserved; usually null. |

`matched_skills`

| Field | Meaning |
|-------|---------|
| `essential_skill_matches[]` | One row per essential skill: `job_skill_id`, `job_skill_label`, `best_user_skill_id`, `best_user_skill_label`, `similarity` (0–1), `meets_threshold`, `match_tier` (`exact` / `embedding` / `none`). |
| `optional_exact_matches[]` | Optional skills the user meets: `skill_id`, `skill_label`. |
| `skill_group_matches[]` | Currently always empty. |

`SkillGapRecommendation`

| Field | Meaning |
|-------|---------|
| `skill_id`, `skill_label` | The suggested skill. |
| `proximity_score` | Closeness to the user's existing skills, in [0, 1]. |
| `job_unlock_count` | Weighted count of nearby jobs it would help with (essential = 2, optional = 1). |
| `combined_score` | Ranking score: 0.4 × proximity + 0.6 × normalised `job_unlock_count`. |
| `reasoning` | Plain-language reason. |

### Errors

Errors use FastAPI's shape: `{"detail": ...}`.

| Status | When | `detail` |
|--------|------|----------|
| 400 | More than 32 users in the array | `"Too many users in one request (max 32)."` |
| 400 | Empty array | `"Request body must be a non-empty JSON array."` |
| 400 | Bad `final_score_combiner` | `"final_score_combiner must be 'product' or 'geometric_mean'"` |
| 400 | An input value was rejected during matching | A message describing the problem. |
| 401 / 403 | Missing or invalid API key (gateway) | Gateway error body. |
| 403 | Missing `x-api-key` header (direct to service) | `"Not authenticated"` |
| 422 | Body is not an array, or a field has the wrong type, or a query param is out of range | A list of validation errors (`loc`, `msg`, `type`). |
| 429 | Gateway rate limit exceeded | Gateway error body. |
| 500 | Unexpected server error | `"Internal server error: <ExceptionClass>"` |

### Example

```bash
curl -X POST "$BASE_URL/match?final_top_k=20" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '[{
        "user_id": "u1",
        "city": "Nairobi",
        "province": "Nairobi",
        "any_post_secondary_educ": 1,
        "skills_vector": {"top_skills": [
          {"originUUID": "00000000-0000-4000-8000-000000000001",
           "preferredLabel": "customer service", "proficiency": 0.8}
        ]},
        "preference_vector": {"earnings_per_month": 0.7, "career_growth": 0.8}
      }]'
```

Response (abridged; the full example is in `/docs`):

```json
[
  {
    "user_id": "u1",
    "opportunity_recommendations": [
      {
        "uuid": "6a27a1d40c73458a0b453373",
        "rank": 1,
        "opportunity_title": "Customer Service Representative",
        "employer": "Acme Ltd",
        "location": "Nairobi",
        "final_score": 0.4544,
        "is_eligible": true,
        "justification": "Strong match on your customer service skills. It fits your preferences for career growth (High).",
        "score_breakdown": {"u_hat": 0.71, "p_hat": 0.64, "essential_coverage": 0.75, "skill_match_level": "strong"},
        "matched_skills": {"essential_skill_matches": [], "optional_exact_matches": [], "skill_group_matches": []},
        "matched_preferences": [],
        "matched_work_activities": null
      }
    ],
    "occupation_recommendations": [],
    "skill_gap_recommendations": [
      {"skill_id": "…", "skill_label": "use spreadsheets software", "proximity_score": 0.71,
       "job_unlock_count": 6, "combined_score": 0.684,
       "reasoning": "Similar to your 'customer service' skill. Would help unlock or improve 6 jobs."}
    ]
  }
]
```

## Changes

**2026-10-08**

- `POST /match` now returns the recommendations previously served at `POST /match_v4`. The
  request body, response shape and query parameters are unchanged from `/match_v4`, but it now
  requires the `x-api-key` header. Clients of the previous `POST /match` keep the same request
  and response shape, but will see different scores and rankings.
- Removed: `POST /match_v4`, `POST /experiments/v2/match`, `POST /experiments/v3/match` and
  `POST /experiments/v5/match`. These paths now return `404`; use `POST /match`.
- Qualification-level (ZQF) annotations from `/experiments/v5/match` are no longer available:
  `zqf_level` in the request is ignored, and opportunities don't include `zqf_eligible`,
  `zqf_gap`, `zqf_min_label` or `zqf_max_label`.
