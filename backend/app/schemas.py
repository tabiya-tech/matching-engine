from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional


def _strip_county_suffix(s: str) -> str:
    """Normalize Kenyan county names so user locations match occupation/job locations.

    Occupations/jobs store bare county names ("Nairobi"); consumers often send the official
    "Nairobi County". The Mongo job prefilter matches the user string AS A SUBSTRING of the job
    field, so "nairobi county" would not match a job stored as "Nairobi". Trimming a trailing
    " County" (case-insensitive) to "Nairobi" matches both bare and suffixed job values, and is
    harmless for the (bidirectional) occupation location check.
    """
    s = (s or "").strip()
    if s.lower().endswith(" county"):
        s = s[: -len(" county")].strip()
    return s


class Skill(BaseModel):
    preferredLabel: Optional[str] = Field(
        default=None,
        description="Human-readable ESCO skill label. Used for display and as a fallback when the UUID cannot be resolved.",
        examples=["customer service"],
    )
    originUUID: str = Field(
        description="ESCO skill origin UUID identifying the skill.",
        examples=["00000000-0000-4000-8000-000000000001"],
    )
    proficiency: Optional[float] = Field(
        default=None,
        description="Optional self-reported or inferred proficiency in [0, 1].",
        examples=[0.8],
    )


class SkillsVector(BaseModel):
    top_skills: List[Skill] = Field(
        default_factory=list,
        description="The user's skills. Empty list is allowed (skill-based fit will be weak).",
    )


class PreferenceVector(BaseModel):
    # All fields optional: a consumer may send a subset. Missing/0.5 = neutral in the DCE contract
    # (0.5 = sigmoid(0) => recovered beta_hat = 0 => no contribution), so omitting a preference
    # simply means "no signal" rather than a dislike.
    earnings_per_month: float = Field(
        default=0.5,
        description="Preference for higher earnings, in [0, 1]. 0.5 = neutral, >0.5 = values it, <0.5 = averse.",
        examples=[0.7],
    )
    task_content: Optional[float] = Field(
        default=0.5, description="Preference for the job's task content, in [0, 1]. 0.5 = neutral.", examples=[0.5]
    )
    physical_demand: float = Field(
        default=0.5,
        description="Preference for physically demanding work, in [0, 1]. 0.5 = neutral; low values penalise physical jobs.",
        examples=[0.4],
    )
    work_flexibility: Optional[float] = Field(
        default=0.5, description="Preference for flexible working arrangements, in [0, 1]. 0.5 = neutral.", examples=[0.5]
    )
    social_interaction: float = Field(
        default=0.5, description="Preference for social / people-facing work, in [0, 1]. 0.5 = neutral.", examples=[0.6]
    )
    career_growth: float = Field(
        default=0.5, description="Preference for career-growth potential, in [0, 1]. 0.5 = neutral.", examples=[0.8]
    )
    social_meaning: Optional[float] = Field(
        default=0.5, description="Preference for socially meaningful work, in [0, 1]. 0.5 = neutral.", examples=[0.5]
    )
    bws_scores: Optional[dict] = Field(
        default=None,
        description=(
            "Best-Worst Scaling scores keyed by O*NET work-activity code (e.g. ``4.A.4.a.4``). "
            "Drives the work-activity part of the preference score. Omit if not collected."
        ),
        examples=[{"4.A.4.a.4": 0.9, "4.A.3.a.1": -0.4}],
    )
    top_10_bws: Optional[List[str]] = Field(
        default=None,
        description="The user's ten highest-ranked work-activity codes from the BWS exercise.",
        examples=[["4.A.4.a.4", "4.A.2.b.1"]],
    )


class MatchRequest(BaseModel):
    # Every field is optional so future consumers can send a subset and still get a valid response.
    # Omitting location ("") relaxes the job location prefilter (and triggers the occupation
    # random-county fallback); omitting preferences yields a neutral preference vector.
    user_id: Optional[str] = Field(
        default=None,
        description="Caller-supplied identifier echoed back as ``MatchResponse.user_id``.",
        examples=["u1"],
    )
    city: str = Field(
        default="",
        description=(
            "User's city/town. Used to prefer nearby opportunities. A trailing \" County\" is stripped. "
            "Empty = no location preference."
        ),
        examples=["Nairobi"],
    )
    province: str = Field(
        default="",
        description=(
            "User's province/county. Scopes occupations and opportunities to the user's region. "
            "A trailing \" County\" is stripped. Empty = a fallback region is used for occupations."
        ),
        examples=["Nairobi"],
    )
    skills_vector: SkillsVector = Field(
        default_factory=SkillsVector, description="The user's skills."
    )
    skill_groups_origin_uuids: List[str] = Field(
        default_factory=list,
        description="ESCO skill-group origin UUIDs the user belongs to. Optional.",
        examples=[[]],
    )
    preference_vector: PreferenceVector = Field(
        default_factory=PreferenceVector,
        description="Job-attribute and work-activity preferences. Omitted fields are neutral (0.5).",
    )
    any_post_secondary_educ: Optional[int] = Field(
        default=None,
        description=(
            "Post-secondary education gate. ``0`` = user has no post-secondary "
            "(jobs with ``requires_post_secondary`` are excluded), ``1`` = has post-secondary. "
            "Omit to skip the gate."
        ),
        examples=[1],
    )
    number_post_secondary_educ: Optional[int] = Field(
        default=None,
        description="Number of post-secondary qualifications. Informational; not used for ranking.",
        examples=[1],
    )
    total_duration_postsec: Optional[float] = Field(
        default=None,
        description="Total years of post-secondary education. Informational; not used for ranking.",
        examples=[2.0],
    )

    @field_validator("city", "province", mode="before")
    @classmethod
    def _normalize_location(cls, v: Any) -> Any:
        # Strip a trailing " County" so "Nairobi County" matches occupation/job "Nairobi".
        return _strip_county_suffix(v) if isinstance(v, str) else v


class SkillComponents(BaseModel):
    # Optional so the response can carry a partial, interpretable breakdown (ess/opt in [0,1];
    # loc/grp null = "not computed").
    loc: Optional[float] = Field(default=None, description="Location similarity. Not computed by the current engine (null).")
    ess: Optional[float] = Field(
        default=None, description="Mean best cosine similarity over the item's essential skills, in [0, 1].", examples=[0.62]
    )
    opt: Optional[float] = Field(
        default=None, description="Mean best cosine similarity over the item's optional skills, in [0, 1].", examples=[0.48]
    )
    grp: Optional[float] = Field(default=None, description="Skill-group recall. Not computed by the current engine (null).")


class PHatComponents(BaseModel):
    gate: float = Field(default=0.0, description="Eligibility gate component of p_hat.")
    essential_fit: float = Field(default=0.0, description="Essential-skill fit component of p_hat.")
    recruiter_readiness: float = Field(default=0.0, description="Recruiter-readiness component of p_hat.")
    market_opportunity: float = Field(default=0.0, description="Market-opportunity component of p_hat.")


class ScoreBreakdown(BaseModel):
    # --- Multiplicative (paper-aligned) fields ---
    u_hat: Optional[float] = Field(
        default=None,
        description="Preference utility in [0, 1]: how well the item matches the user's preferences (DCE attributes + BWS).",
        examples=[0.71],
    )
    p_hat: Optional[float] = Field(
        default=None,
        description="Success propensity in [0, 1]: how likely the user is to succeed, driven by skill fit.",
        examples=[0.64],
    )
    p_hat_source: Optional[str] = Field(
        default=None,
        description="Which signal produced p_hat (e.g. ``concat_cosine_whitened``).",
        examples=["concat_cosine_whitened"],
    )
    p_hat_components: Optional[PHatComponents] = Field(
        default=None, description="Decomposition of p_hat. Null when the engine does not compute it."
    )
    # --- Legacy additive fields ---
    total_skill_utility: Optional[float] = Field(
        default=None, description="Essential-skill fit aggregate in [0, 1] (same value as ``skill_components.ess``).", examples=[0.62]
    )
    skill_components: Optional[SkillComponents] = Field(
        default=None, description="Per-component skill-similarity breakdown."
    )
    skill_diagnostics: Optional[SkillComponents] = Field(
        default=None, description="Reserved for raw skill diagnostics. Null in the current engine."
    )
    skill_penalty_applied: Optional[float] = Field(
        default=None,
        description="Share of essential skills below the similarity threshold, in [0, 1]. Higher = bigger gap.",
        examples=[0.25],
    )
    preference_score: Optional[float] = Field(
        default=None, description="Preference score; equal to ``u_hat``.", examples=[0.71]
    )
    preference_score_legacy: Optional[float] = Field(
        default=None, description="Preference score from the previous scorer, kept for comparison.", examples=[0.68]
    )
    demand_score: Optional[float] = Field(
        default=None, description="Expected labour-market demand score for the item. Null when unknown.", examples=[0.75]
    )
    demand_label: Optional[str] = Field(
        default=None, description="Human-readable demand level. Null when unknown.", examples=["High Expected Demand"]
    )
    # Per-skill gate: essential-coverage in [0,1] and the graded badge (strong/partial/weak)
    essential_coverage: Optional[float] = Field(
        default=None,
        description="Share of the item's essential skills the user meets, in [0, 1]. 1.0 when the item lists no essential skills.",
        examples=[0.75],
    )
    skill_match_level: Optional[str] = Field(
        default=None,
        description="Badge derived from essential_coverage: ``strong``, ``partial``, ``weak``, or ``unknown`` (no essential skills).",
        examples=["strong"],
    )
    # Multipliers actually applied to final_score on /match, so a low score is auditable:
    # final_score = u_hat x p_hat x coverage_factor x location_tier_factor. coverage_factor is
    # floor + (1-floor) * ranking-coverage ** coverage_gamma — the RANKING coverage, which for an
    # unparsed posting is the shortlist mean rather than the 1.0 reported in essential_coverage.
    coverage_gamma: Optional[float] = Field(
        default=None,
        description="Exponent applied to skill coverage when computing coverage_factor. Null when not applied.",
        examples=[1.0],
    )
    coverage_factor: Optional[float] = Field(
        default=None,
        description=(
            "Skill-coverage multiplier on final_score, in [0, 1]: lower when the user meets fewer of the "
            "item's essential skills. Null when not applied."
        ),
        examples=[0.76],
    )
    location_tier_factor: Optional[float] = Field(
        default=None,
        description=(
            "Location multiplier on final_score for opportunities, in [0, 1]: 1.0 for jobs in the user's "
            "own area, lower for jobs in a regional or national hub. Null when not applied."
        ),
        examples=[1.0],
    )


class MatchedSkill(BaseModel):
    job_skill_id: str = Field(description="ESCO UUID of the item's skill.", examples=["00000000-0000-4000-8000-000000000001"])
    job_skill_label: Optional[str] = Field(default=None, description="Label of the item's skill.", examples=["customer service"])
    best_user_skill_id: Optional[str] = Field(
        default=None, description="UUID of the user's skill most similar to this one.", examples=["00000000-0000-4000-8000-000000000001"]
    )
    best_user_skill_label: Optional[str] = Field(
        default=None, description="Label of the user's most similar skill.", examples=["customer service"]
    )
    similarity: float = Field(description="Cosine similarity between the two skills, in [0, 1].", examples=[0.93])
    meets_threshold: bool = Field(
        description="True when the user is considered to have this skill (exact UUID match or similarity above threshold).",
        examples=[True],
    )
    match_tier: Optional[str] = Field(
        default=None, description="How it matched: ``exact``, ``embedding`` or ``none``.", examples=["exact"]
    )


class OptionalSkillMatch(BaseModel):
    skill_id: str = Field(description="ESCO UUID of the matched optional skill.")
    skill_label: Optional[str] = Field(default=None, description="Label of the matched optional skill.")


class SkillGroupMatch(BaseModel):
    skill_group_id: str = Field(description="ESCO UUID of the matched skill group.")
    skill_group_label: Optional[str] = Field(default=None, description="Label of the matched skill group.")


class MatchedSkills(BaseModel):
    essential_skill_matches: List[MatchedSkill] = Field(
        default_factory=list, description="One row per essential skill of the item, met or not."
    )
    optional_exact_matches: List[OptionalSkillMatch] = Field(
        default_factory=list, description="Optional skills of the item that the user meets."
    )
    skill_group_matches: List[SkillGroupMatch] = Field(
        default_factory=list, description="Matched skill groups. Empty in the current engine."
    )


class MatchedPreference(BaseModel):
    attribute: str = Field(description="Job attribute name (matches a ``PreferenceVector`` field).", examples=["earnings_per_month"])
    job_value: Optional[str] = Field(default=None, description="The item's raw value for the attribute.", examples=["earn_70k"])
    job_value_label: Optional[str] = Field(default=None, description="Human-readable item value.", examples=["~70k"])
    user_weight: float = Field(description="The user's weight for this attribute as used by the scorer.", examples=[0.7])
    beta: float = Field(description="Model coefficient for the attribute.", examples=[0.35])
    encoded_value: float = Field(description="Numerically encoded item value used in scoring.", examples=[1.0])
    contribution: float = Field(description="Contribution to the preference utility (beta × user_weight × encoded_value).", examples=[0.245])
    matched: bool = Field(description="True when the item has the attribute level the user values (encoded_value > 0).", examples=[True])


class MatchedWorkActivity(BaseModel):
    wa_code: str = Field(description="O*NET work-activity code.", examples=["4.A.4.a.4"])
    wa_label: Optional[str] = Field(default=None, description="O*NET work-activity label.")
    user_bws: float = Field(description="The user's BWS score for this activity.")
    wa_importance: float = Field(description="O*NET importance of the activity for the item (1–5).")
    wa_level: float = Field(description="O*NET level of the activity for the item (1–7).")
    norm_importance: float = Field(description="Importance normalised to [0, 1].")
    norm_level: float = Field(description="Level normalised to [0, 1].")
    wa_contribution: float = Field(description="Contribution of this activity to the work-activity utility.")
    # Additive-RUM diagnostics (BWS_INTEGRATION_MODE="additive_rum")
    weight: Optional[float] = Field(default=None, description="Importance weight of the activity (weights sum to 1).")
    beta: Optional[float] = Field(default=None, description="User's BWS part-worth for this activity.")


class WorkActivityBWS(BaseModel):
    wa_score_sum: float = Field(default=0.0, description="Sum of work-activity contributions.")
    details: List[MatchedWorkActivity] = Field(default_factory=list, description="Per-activity breakdown.")
    # Additive-RUM diagnostics
    wa_aggregation: Optional[str] = Field(default=None, description="Aggregation method used over activities.")
    n_work_activities: Optional[int] = Field(default=None, description="Number of activities considered.")
    V_task: Optional[float] = Field(default=None, description="Raw task (work-activity) utility.")
    V_task_hat: Optional[float] = Field(default=None, description="Task utility rescaled to [-1, 1].")


class OpportunityRecommendation(BaseModel):
    uuid: str = Field(description="Job opportunity identifier.", examples=["6a27a1d40c73458a0b453373"])
    originUuid: Optional[str] = Field(default=None, description="Identifier of the job in its source system.", examples=["src-123"])
    URL: Optional[str] = Field(default=None, description="Link to the job posting.", examples=["https://example.com/jobs/123"])
    rank: int = Field(description="1-based rank within this user's opportunities (1 = best).", examples=[1])
    opportunity_title: str = Field(description="Job title.", examples=["Customer Service Representative"])
    opportunity_isco_occupation_group: Optional[str] = Field(default=None, description="ISCO occupation group label.", examples=["Client information workers"])
    opportunity_isco_occupation_group_id: Optional[str] = Field(default=None, description="ISCO occupation group code.", examples=["422"])
    related_occupation_id: Optional[str] = Field(default=None, description="ESCO occupation the job maps to.", examples=["00000000-0000-4000-8000-0000000000aa"])
    location: Optional[str] = Field(default=None, description="Job location.", examples=["Nairobi"])
    employer: Optional[str] = Field(default=None, description="Employer name.", examples=["Acme Ltd"])
    employment_type: Optional[str] = Field(default=None, description="Employment type (e.g. full-time).", examples=["Full-time"])
    salary_text: Optional[str] = Field(default=None, description="Salary as stated in the posting.", examples=["KES 40,000 per month"])
    required_education: Optional[str] = Field(default=None, description="Education requirement as stated in the posting.", examples=["Diploma"])
    required_experience: Optional[str] = Field(default=None, description="Experience requirement as stated in the posting.", examples=["1 year"])
    closing_date: Optional[str] = Field(default=None, description="Application closing date.", examples=["2026-11-01"])
    posted_date: Optional[str] = Field(default=None, description="Date the job was posted.", examples=["2026-10-01"])
    is_eligible: bool = Field(
        description="True when the user meets enough of the job's essential skills (essential_coverage at or above the eligibility share).",
        examples=[True],
    )
    justification: str = Field(
        description="Short plain-language explanation of why this job was recommended.",
        examples=["Strong match on your customer service skills. It fits your preferences for career growth (high)."],
    )
    opportunity_description: Optional[str] = Field(default=None, description="Job description.", examples=["Handle customer enquiries by phone and email."])
    contract_type: Optional[str] = Field(default=None, description="Contract type.", examples=["permanent"])
    final_score: float = Field(
        description="Ranking score in [0, 1]: u_hat combined with p_hat (product by default). Higher is better.", examples=[0.4544]
    )
    score_breakdown: ScoreBreakdown = Field(description="Components behind final_score.")
    matched_skills: MatchedSkills = Field(description="Skill-level match details.")
    matched_preferences: List[MatchedPreference] = Field(
        default_factory=list, description="Per-attribute preference match details."
    )
    matched_work_activities: Optional[WorkActivityBWS] = Field(
        default=None, description="Work-activity (BWS) match details. Null when the user sent no BWS scores."
    )


class OccupationRecommendation(BaseModel):
    uuid: str = Field(description="Occupation identifier.", examples=["occ-0001"])
    originUuid: Optional[str] = Field(default=None, description="ESCO occupation origin UUID.", examples=["00000000-0000-4000-8000-0000000000aa"])
    rank: int = Field(description="1-based rank within this user's occupations (1 = best).", examples=[1])
    occupation_label: str = Field(description="Occupation name.", examples=["call centre agent"])
    province: Optional[str] = Field(default=None, description="Region the occupation record applies to.", examples=["Nairobi"])
    is_eligible: bool = Field(description="True when the user meets enough of the occupation's essential skills.", examples=[True])
    justification: str = Field(description="Short plain-language explanation of why this occupation was recommended.", examples=["Strong match on your customer service skills. This role is in high demand."])
    occupation_description: Optional[str] = Field(default=None, description="Occupation description.", examples=["Call centre agents handle inbound and outbound customer calls."])
    salary_range: Optional[str] = Field(default=None, description="Typical monthly earnings level.", examples=["~70k"])
    typical_tasks: List[str] = Field(default_factory=list, description="Typical tasks of the occupation (up to 8).", examples=[["Answer customer enquiries", "Record customer interactions"]])
    career_path_next_steps: List[str] = Field(
        default_factory=list, description="Suggested next career steps. Currently always empty.", examples=[[]])
    final_score: float = Field(description="Ranking score in [0, 1]. Higher is better.", examples=[0.52])
    score_breakdown: ScoreBreakdown = Field(description="Components behind final_score.")
    matched_skills: MatchedSkills = Field(description="Skill-level match details.")
    matched_preferences: List[MatchedPreference] = Field(
        default_factory=list, description="Per-attribute preference match details."
    )
    matched_work_activities: Optional[WorkActivityBWS] = Field(
        default=None, description="Work-activity (BWS) match details. Null when the user sent no BWS scores."
    )


class SkillGapRecommendation(BaseModel):
    skill_id: str = Field(description="ESCO UUID of the suggested skill to learn.", examples=["00000000-0000-4000-8000-0000000000bb"])
    skill_label: str = Field(description="Label of the suggested skill.", examples=["use spreadsheets software"])
    proximity_score: float = Field(description="Closeness to the user's existing skills, in [0, 1].", examples=[0.71])
    job_unlock_count: int = Field(
        description="Weighted count of nearby jobs this skill would help with (essential = 2, optional = 1).", examples=[6]
    )
    combined_score: float = Field(
        description="Ranking score: 0.4 × proximity + 0.6 × normalised job_unlock_count.", examples=[0.684]
    )
    reasoning: str = Field(
        description="Plain-language reason for the suggestion.",
        examples=["Similar to your 'customer service' skill. Would help unlock or improve 6 jobs."],
    )


class MatchResponse(BaseModel):
    user_id: str = Field(description="Echo of the request's ``user_id``.", examples=["u1"])
    occupation_recommendations: List[OccupationRecommendation] = Field(
        default_factory=list, description="Ranked occupations (career pathways) for the user."
    )
    opportunity_recommendations: List[OpportunityRecommendation] = Field(
        default_factory=list, description="Ranked live job opportunities for the user."
    )
    skill_gap_recommendations: List[SkillGapRecommendation] = Field(
        default_factory=list, description="Skills that would most improve the user's prospects."
    )


class JobListItem(BaseModel):
    """A single browsable job, built from the same source/shape as matched-job
    opportunities (see app.jobs.documents.build_job_dict_from_ranked) minus the per-user
    scoring/matching fields, which only exist after a /match request."""

    uuid: str
    originUuid: Optional[str] = None
    url: Optional[str] = None
    opportunity_title: str
    opportunity_isco_occupation_group: Optional[str] = None
    opportunity_isco_occupation_group_id: Optional[str] = None
    related_occupation_id: Optional[str] = None
    location: Optional[str] = None
    city: Optional[str] = None
    province: Optional[str] = None
    employer: Optional[str] = None
    employment_type: Optional[str] = None
    contract_type: Optional[str] = None
    salary_text: Optional[str] = None
    closing_date: Optional[str] = None
    posted_date: Optional[str] = None
    opportunity_description: Optional[str] = None
    # Consumer-contract fields (Compass jobs board): sector/category, the platform the
    # posting was scraped from, and the flat list of skill labels for this opportunity.
    category: Optional[str] = None
    source_platform: Optional[str] = None
    skills: List[str] = Field(default_factory=list)


class JobsPage(BaseModel):
    """A cursor-paginated page of jobs. ``next_cursor`` is null on the last page;
    otherwise pass it back as the ``cursor`` query param to fetch the next page.

    ``total`` is the count of all jobs matching the current filters (ignoring the
    cursor); it is only populated when the request asks for it (``include_total=true``)."""

    items: List[JobListItem] = Field(default_factory=list)
    next_cursor: Optional[str] = None
    total: Optional[int] = None


class JobsStats(BaseModel):
    """Aggregate counts over the active jobs catalog (the /jobs/stats endpoint)."""

    total: int
    sectors: int
    platforms: int


class MatchV2JobRecommendation(BaseModel):
    """One job from hybrid ``column_fused_weighted_minmax`` (pool min–max fusion)."""

    rank: int
    job_uuid: str
    opportunity_title: str = ""
    employer: Optional[str] = None
    location: Optional[str] = None
    URL: Optional[str] = None
    fusion_score: float
    bm25_norm_within_candidates: Optional[float] = None
    cos_norm_within_candidates: Optional[float] = None
    mean_best_cosine_raw: Optional[float] = None
    bm25_score_raw: Optional[float] = None
    matched_skills: List[str] = Field(default_factory=list)
    matched_skills_cosine: List[str] = Field(default_factory=list)


class MatchV2Response(BaseModel):
    """``POST /match_v2``: hybrid BM25 × embedding-cosine recommendations only."""

    user_id: str
    n_jobs_scored: int
    hybrid_recommendations: List[MatchV2JobRecommendation]
    hybrid_config_summary: Dict[str, Any] = Field(default_factory=dict)


class MatchConcatGeminiCeJobRecommendation(BaseModel):
    """One job after concat-Gemini cosine shortlist and cross-encoder rerank."""

    rank: int
    rank_cosine: Optional[int] = None
    rank_cross_encoder: Optional[int] = None
    job_uuid: str
    opportunity_title: str = ""
    employer: Optional[str] = None
    location: Optional[str] = None
    URL: Optional[str] = None
    concat_cosine_similarity: Optional[float] = None
    cross_encoder_logit: Optional[float] = None
    cross_encoder_score: Optional[float] = None
    # Stage 3 (``POST /match_v4`` only): hybrid preference × p_hat
    u_hat: Optional[float] = None
    p_hat: Optional[float] = None
    final_score: Optional[float] = None
    score_breakdown: Optional[Dict[str, Any]] = None


class MatchConcatGeminiCeResponse(BaseModel):
    """``POST /match_v3`` / ``POST /match_v4`` — Gemini concat × Mongo → CE (+ prefs on v4)."""

    user_id: str
    n_jobs_scored: int
    n_jobs_active_loaded: int
    concat_gemini_ce_recommendations: List[MatchConcatGeminiCeJobRecommendation]
    config_summary: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# v5 experiment
# ---------------------------------------------------------------------------


class MatchRequestV5(MatchRequest):
    """request extended with the user's ZQF qualification level."""

    zqf_level: Optional[int] = Field(
        default=None,
        description=(
            "User's Zambia Qualifications Framework (ZQF) level (integer). "
            "Used with each job's ``zqf_min`` to populate ``zqf_eligible`` and "
            "``zqf_gap`` on opportunity recommendations. Omit to skip ZQF annotation."
        ),
        examples=[4],
    )


class OpportunityRecommendationV5(OpportunityRecommendation):
    """opportunity extended with ZQF eligibility annotation."""

    zqf_eligible: Optional[bool] = None
    zqf_gap: Optional[int] = None
    zqf_min_label: Optional[str] = None
    zqf_max_label: Optional[str] = None


class MatchResponseV5(BaseModel):
    """response with ZQF-annotated opportunities."""

    user_id: str
    occupation_recommendations: List[OccupationRecommendation] = Field(
        default_factory=list
    )
    opportunity_recommendations: List[OpportunityRecommendationV5] = Field(
        default_factory=list
    )
    skill_gap_recommendations: List[SkillGapRecommendation] = Field(
        default_factory=list
    )
