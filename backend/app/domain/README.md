# Domain models and data dictionary

The matching service works on four kinds of values, each a strict pydantic model in `app/domain/`:

| Model | Module | Built from | Built by |
|---|---|---|---|
| `Job` | `job.py` | ranked-job documents in MongoDB (`MONGO_JOBS_COLLECTION`) | `app.jobs.job_document.build_job_from_ranked` |
| `Occupation` | `occupation.py` | the occupation file (`OCCUPATION_JSON_PATH`) and the occupation-embeddings NPZ | `app.occupations.flatten.flatten_occupations`, `app.occupations.loader.attach_occupation_embeddings` |
| `UserProfile` | `user.py` | one item of the `POST /match` body (`MatchRequest`) | `UserProfile.from_request` |
| `ScoreBreakdown` | `scores.py` | the ranking stages' output | `app.matching.formatting` |

Shared parts: `SkillReference` (`skills.py`), `JobAttributes` (`attributes.py`), `WorkActivity`
(`work_activities.py`).

## Rules

- **Strict, closed and frozen.** Every domain model derives from `DomainModel`
  (`strict=True, extra="forbid", frozen=True`). No type coercion happens past the boundary, an
  unknown field is an error, and a changed value is a `model_copy(update=...)`.
- **Validate once, at the boundary.** Upstream documents are first validated by a `SourceRecord`
  model (`RankedJobDocument`, `OccupationFileEntry`), which ignores unread fields and normalises
  loose scalars with `app.domain.coercion`: numbers and dates to text, `0/1`/`"yes"` to booleans,
  numeric strings to numbers. They are then mapped to the domain model. A job document that fails
  validation is logged and skipped (`JobsLoadTiming.n_skipped_invalid`). An invalid occupation
  file fails the load.
- **Names follow the API.** `Job` / `Occupation` field names are the response field names they
  feed (`originUuid`, `opportunity_title`, …), and `UserProfile` mirrors `MatchRequest`.
- **Typed end to end.** Past the boundary nothing is a dict: the ranking stages read these models
  and pass typed intermediates to each other (`app/ranking/types.py`, see *Ranking intermediates*
  below), and the formatters build the response models directly. Mongo query filters and the
  ASGI/tracing plumbing are the only `dict[str, Any]` left, and neither crosses a layer.

## Adding a field

1. Declare it on the `SourceRecord` (`RankedJobDocument` / `OccupationFileEntry`), and for jobs add
   it to `RANKED_JOB_FIND_PROJECTION`.
2. Declare it on the domain model with a `Field(description=...)` and map it in
   `job_from_document` / `flatten_occupations`.
3. Add a row below and a test in `tests/unit/test_build_job_from_ranked.py` or
   `tests/unit/test_occupation_flatten.py`.

## Job

Source document: one ranked job, written by the **llm-reranker** (`app/ranked_job_schema.py:
build_ranked_job_document`). `classifier_metadata` holds the listing fields copied from the
**scraper** ad, via the **classifier** (`classifier_metadata_from_context`). Paths are relative to
the document root.

| Field | Type | Source path (first non-empty wins) | Notes |
|---|---|---|---|
| `uuid` | `str` | `job_id` | Scraper-assigned id; `""` when absent. |
| `originUuid` | `str \| None` | `origin_uuid`, `originUuid`, `job_fingerprint`, `job_id` | Stable id across scrapes. The first two are not projected. |
| `job_fingerprint` | `str \| None` | `job_fingerprint` | Content hash (scraper dedup key). |
| `opportunity_title` | `str` | `classifier_metadata.title` | `"Unknown"` when absent. |
| `city` | `str` | `classifier_metadata.city` | Stripped; `""` when absent. |
| `province` | `str` | `classifier_metadata.province`, `classifier_metadata.county` | `county` is the older name (the llm-reranker now writes `province`). |
| `location` | `str` | `city` + `" "` + `province` | Display only. |
| `location_keys` | `tuple[str, ...]` | `location_keys` | Standardised location keys (`app.location.keys`), written by the llm-reranker; derived from `classifier_metadata` city / county / province when the document predates the field. Decides location eligibility (the `/match` filter and location tiers). |
| `employer` | `str \| None` | `classifier_metadata.employer` | |
| `employment_type` | `str \| None` | `classifier_metadata.employment_type` | As stated in the ad. |
| `contract_type` | `str` | `classifier_metadata.employment_type` | `"full_time"` when absent. |
| `salary_text` | `str \| None` | `classifier_metadata.salary` | Free text. |
| `closing_date` | `str` | `classifier_metadata.closing_date` | Stringified (may be a date upstream); `""` when absent. |
| `posted_date` | `str \| None` | `classifier_metadata.posted_date`, `.date_posted`, `.posted_at` | `posted_at` is not projected. |
| `url` | `str \| None` | `classifier_metadata.application_url` | |
| `opportunity_description` | `str` | `classifier_metadata.job_description`, `.description` | `""` when absent. |
| `opportunity_isco_occupation_group` | `str \| None` | `classifier_metadata.isco_occupation_group` | |
| `opportunity_isco_occupation_group_id` | `str \| None` | `classifier_metadata.isco_occupation_group_id` | Stringified (often stored as a number). |
| `related_occupation_id` | `str \| None` | `related_occupation_id`, then the ISCO group id | `related_occupation_id` is not projected. |
| `category` | `str \| None` | `classifier_metadata.category`, `.sector`, ISCO group label | Jobs-board sector. |
| `source_platform` | `str \| None` | `classifier_metadata.source_platform`, `.source`, `.platform` | Scraper platform. |
| `essential_skills` | `list[SkillReference]` | `llm_classified_skills.essential[]` → `{tabiya_skill_id, label}` | LLM pass 1 (skill ranking). Rows without `tabiya_skill_id` are dropped. |
| `optional_skills` | `list[SkillReference]` | `llm_classified_skills.optional[]` → `{tabiya_skill_id, label}` | As above. |
| `skills` | `list[str]` | labels of `essential_skills` then `optional_skills` | De-duplicated; jobs-board listing. |
| `skill_groups_origin_uuids` | `list[str]` | `skill_groups_origin_uuids` | llm-reranker `derive_skill_groups_origin_uuids`; a scalar becomes a one-item list. |
| `attributes` | `JobAttributes` | `llm_job_attributes.attributes` | LLM pass 2 (job attributes). Unknown names are ignored. |
| `onet_work_activities` | `list[WorkActivity]` | `onet_work_activities[]` | llm-reranker `utils/work_activities_enrichment.py`. Unrated rows (`""`, `"code is not available"`) are dropped. |
| `requires_post_secondary` | `bool \| None` | `llm_job_attributes.attributes.requires_post_secondary` | LLM pass 2; `0/1`/`"true"` accepted. Drives the education gate. |
| `zqf_min` | `int \| None` | `classifier_metadata.min_zqf_level`, `.zqf_min`, root `zqf_min` | Zambia only. |
| `zqf_max` | `int \| None` | `classifier_metadata.max_zqf_level`, `.zqf_max`, root `zqf_max` | Zambia only. |
| `zqf_min_label` | `str \| None` | `classifier_metadata.min_zqf_label`, `.zqf_min_label` | Zambia only. |
| `zqf_max_label` | `str \| None` | `classifier_metadata.max_zqf_label`, `.zqf_max_label` | Zambia only. |
| `embedding` | `np.ndarray \| None` | `concat_skill_embedding_gemini.vector_bin`, else `job_embedding` | float32, `EMBEDDING_DIM` long; either source is ignored at another size. `job_embedding`: llm-reranker `utils/job_embedding.py`. `vector_bin`: NPZ sync script. |
| `embedding_whitened` | `bool` | `llm_reranker_meta.embedding.whitening.enabled is True` | The llm-reranker whitened `job_embedding` with the same artifact the service ships. |

## Occupation

Source: `resources/occupations/<language>/combined_occupation_database_with_wa.json`, a list of
entries with `occupation`, `skills`, `counties_data` and `onet_work_activities`, exported from the
taxonomy with per-county LLM attribute estimates and labour-demand labels. Each entry becomes one
row **per county** in `counties_data` (one county-less row when the list is empty).

| Field | Type | Source path | Notes |
|---|---|---|---|
| `uuid` | `str` | `occupation.code` + `"_"` + `counties_data[].county` | Just the code when there is no county. |
| `originUuid` | `str` | `occupation.code` | Key into the embeddings NPZ. |
| `occupation_label` | `str` | `occupation.preferred_label` | `"Unknown"` when absent. |
| `description` | `str` | `occupation.description` | |
| `county` | `str` | `counties_data[].county` | Also exposed as `city`, `province` and `location` properties. |
| `location_keys` | `tuple[str, ...]` | `counties_data[].county` | Location keys of the county (`app.location.keys`); decide location eligibility. |
| `essential_skills` | `list[SkillReference]` | `skills.essential.uuids[i]` + `skills.essential.labels[i]` | Label `""` when the label list is shorter. |
| `optional_skills` | `list[SkillReference]` | `skills.optional.uuids[i]` + `skills.optional.labels[i]` | As above. |
| `attributes` | `JobAttributes` | `counties_data[].job_attributes.attributes` (list of `{attribute_name, selected_level_id}` or a mapping), plus `counties_data[].labor_demand.expected_demand` | Names outside the schema are ignored. Some entries use display labels such as `"Monthly Earnings"`, which never matched the scorer. |
| `onet_work_activities` | `list[WorkActivity]` | `onet_work_activities[]` | Same on every county row; unrated rows (`WA_Importance: ""`) are dropped. |
| `requires_post_secondary` | `bool \| None` | `occupation.requires_post_secondary`, else entry-level `requires_post_secondary` | |
| `included_tasks` | `str` | `occupation.included_tasks` | Free text `(a) …; (b) …`; feeds `typical_tasks`. |
| `embedding` | `np.ndarray \| None` | `occupation_concat_embeddings.npz` (`codes`, `vectors`) by `originUuid` | Built by `python -m tools.embeddings.embed_occupations`; whitened once at load when the concat artifact is present. |
| `embedding_whitened` | `bool` | loader | True when the load-time whitening ran. |

## UserProfile

Source: one item of the `POST /match` body, sent by Compass. Defaults are applied by `MatchRequest`,
and a trailing `" County"` is stripped from `city` / `province`.

| Field | Type | Request field | Notes |
|---|---|---|---|
| `user_id` | `str \| None` | `user_id` | Echoed as `MatchResponse.user_id`; the only user field sent to tracing. |
| `city` | `str` | `city` | `""` = no location preference. |
| `province` | `str` | `province` | `""` = random-county fallback for occupations. |
| `skills_vector.top_skills[]` | `list[UserSkill]` | `skills_vector.top_skills[]` | `originUUID`, `preferredLabel`, `proficiency` (unused for ranking). |
| `skill_groups_origin_uuids` | `list[str]` | `skill_groups_origin_uuids` | |
| `preference_vector` | `UserPreferences` | `preference_vector` | Attribute values in [0, 1], 0.5 neutral. `bws_scores` maps O\*NET work-activity code to a BWS part-worth. |
| `any_post_secondary_educ` | `int \| None` | `any_post_secondary_educ` | `0` turns on the education gate. |
| `number_post_secondary_educ` | `int \| None` | `number_post_secondary_educ` | Informational. |
| `total_duration_postsec` | `float \| None` | `total_duration_postsec` | Informational. |
| `zqf_level` | `int \| None` | `zqf_level` (`MatchRequestV5` only) | Zambia ZQF annotation. |

## Shared parts

**`SkillReference`**: `id` (Tabiya skill id for jobs, the skill UUID listed in the occupation
file for occupations) and `label` (`""` when absent). The engine resolves skills by label first.

**`JobAttributes`**: level ids per preference dimension: `earnings_per_month`, `task_content`,
`physical_demand`, `work_flexibility`, `social_interaction`, `career_growth`, `social_meaning` (the
attributes of the llm-reranker's `resources/<lang>/job_attributes_schema.json`), plus
`expected_demand` (occupations only). Ids are kept as stored. The preference scorer only scores the
attributes in its own schema (`resources/preference/job_attributes_schema.json`, loaded as
`app.artifacts.types.AttributeSchema`), and
maps legacy ids such as `earn_70k` and `soc_peers` onto it.

**`WorkActivity`**: `code` (`WA_code`), `label` (`WA_label`), `importance` (`WA_Importance`, 1–5)
and `level` (`WA_Level`, 1–7), from the O\*NET work-activities data. It serialises to the `WA_*`
names (`model_dump(by_alias=True)`).

## ScoreBreakdown

Computed per recommendation; nothing comes from upstream. Each field's description in `scores.py`
gives its range. Sources by stage:

| Field(s) | Computed by |
|---|---|
| `u_hat`, `preference_score` | preference scorer (`app.ranking.preference`): DCE attributes + BWS work activities |
| `p_hat`, `p_hat_source` | stage-1 concat cosine; whitened and rescaled when the Phase-2 rank demotion is on |
| `coverage_gamma`, `coverage_factor` | Phase-2 achievability demotion (`app.ranking.scoring`) |
| `location_tier_factor` | urban-pull location tiers (opportunities only) |
| `demand_score`, `demand_label` | `DemandScorer` from `attributes.expected_demand` (occupations) |
| `essential_coverage`, `skill_match_level`, `skill_components`, `total_skill_utility`, `skill_penalty_applied` | per-skill cosine gate (`app.matching.formatting`) |
| `preference_score_legacy` | previous preference scorer, kept for comparison |
| `p_hat_components`, `skill_diagnostics` | legacy engine only; null on `/match` |

## Resource files

Committed files are validated into models when the artifacts repository loads them
(`app/artifacts/types.py`):

| Model | File | Notes |
|---|---|---|
| `AttributeSchema` | `resources/preference/job_attributes_schema.json` | Attributes with their levels, ordered reference (ladder 0) to target (1). |
| `HubChainsFile` | `resources/location/location_hub_chains.json` | National hub, regional hubs and self-only hubs; parsed into `app.location.keys.HubChains`. |
| `ConcatWhitening` | `V4_FULL_CONCAT_WHITENING_PATH` (`.npz`: `mu`, `W`, `target`) | None when absent or the dimension does not match. |

## Ranking intermediates

Values passed between the stages of one `POST /match` (`app/ranking/types.py`). They are computed
per request; nothing comes from upstream.

| Model | Produced by | Consumed by | Holds |
|---|---|---|---|
| `Candidate` | stage-1 `Stage1Retriever.shortlist`; updated by `CrossEncoderRerank.apply` | Phase 2, preference scoring | the item, its stage-1 concat cosine and rank, `per_job_skill` detail, and after rerank `rank_cosine` / `cross_encoder_logit` / `cross_encoder_score` |
| `UserShortlist` | `MatchingService.shortlist_and_rerank` | `MatchingService.rank` | one user's reranked candidates and a `RetrievalSummary` |
| `SkillMatchDetail`, `SkillPairScore` | `CosineSkillMatcher.score_pair` / `score_pair_v4` | rerank passages, coverage gate, formatting | per-skill best match and similarity |
| `SkillDetail` | `scoring.skill_detail` | coverage gate, formatting | gate rows plus the item's resolved essential ids |
| `RankOverrideSet` | `RankOverrides.compute` (Phase 2) | `enrich_recommendations_with_preferences` | whitened `p_hat`, ranking coverage and skill detail per uuid |
| `PreferenceScore` | `UnifiedPreferenceScorer` / legacy `PreferenceScorer` | final score, formatting | `u_hat`, per-attribute rows (`AttributePreference`) and the task utility (`WorkActivityUtility`) |
| `ScoredCandidate`, `FinalScoreBreakdown` | `enrich_recommendations_with_preferences` | formatting | final score and each factor that was applied |
| `DemandScore` | `DemandScorer` | demand tilt, formatting | demand score and label |
