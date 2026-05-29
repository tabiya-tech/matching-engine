from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class Skill(BaseModel):
    preferredLabel: Optional[str] = None
    originUUID: str
    proficiency: Optional[float] = None

class SkillsVector(BaseModel):
    top_skills: List[Skill] = Field(default_factory=list)

class PreferenceVector(BaseModel):
    # All fields optional: a consumer may send a subset. Missing/0.5 = neutral in the DCE contract
    # (0.5 = sigmoid(0) => recovered beta_hat = 0 => no contribution), so omitting a preference
    # simply means "no signal" rather than a dislike.
    earnings_per_month: float = 0.5
    task_content: Optional[float] = 0.5
    physical_demand: float = 0.5
    work_flexibility: Optional[float] = 0.5
    social_interaction: float = 0.5
    career_growth: float = 0.5
    social_meaning: Optional[float] = 0.5
    bws_scores: Optional[dict] = None
    top_10_bws: Optional[List[str]] = None

class MatchRequest(BaseModel):
    # Every field is optional so future consumers can send a subset and still get a valid response.
    # Omitting location ("") relaxes the job location prefilter (and triggers the occupation
    # random-county fallback); omitting preferences yields a neutral preference vector.
    user_id: Optional[str] = None
    city: str = ""
    province: str = ""
    skills_vector: SkillsVector = Field(default_factory=SkillsVector)
    skill_groups_origin_uuids: List[str] = Field(default_factory=list)
    preference_vector: PreferenceVector = Field(default_factory=PreferenceVector)
    any_post_secondary_educ: Optional[int] = None
    number_post_secondary_educ: Optional[int] = None
    total_duration_postsec: Optional[float] = None

class SkillComponents(BaseModel):
    loc: float
    ess: float
    opt: float
    grp: float

class PHatComponents(BaseModel):
    gate: float = 0.0
    essential_fit: float = 0.0
    recruiter_readiness: float = 0.0
    market_opportunity: float = 0.0

class ScoreBreakdown(BaseModel):
    # --- Multiplicative (paper-aligned) fields ---
    u_hat: Optional[float] = None
    p_hat: Optional[float] = None
    p_hat_components: Optional[PHatComponents] = None
    # --- Legacy additive fields ---
    total_skill_utility: Optional[float] = None
    skill_components: Optional[SkillComponents] = None
    skill_diagnostics: Optional[SkillComponents] = None
    skill_penalty_applied: Optional[float] = None
    preference_score: Optional[float] = None
    preference_score_legacy: Optional[float] = None
    demand_score: Optional[float] = None
    demand_label: Optional[str] = None

class MatchedSkill(BaseModel):
    job_skill_id: str
    job_skill_label: Optional[str] = None
    best_user_skill_id: Optional[str] = None
    best_user_skill_label: Optional[str] = None
    similarity: float
    meets_threshold: bool

class OptionalSkillMatch(BaseModel):
    skill_id: str
    skill_label: Optional[str] = None

class SkillGroupMatch(BaseModel):
    skill_group_id: str
    skill_group_label: Optional[str] = None

class MatchedSkills(BaseModel):
    essential_skill_matches: List[MatchedSkill] = Field(default_factory=list)
    optional_exact_matches: List[OptionalSkillMatch] = Field(default_factory=list)
    skill_group_matches: List[SkillGroupMatch] = Field(default_factory=list)

class MatchedPreference(BaseModel):
    attribute: str
    job_value: Optional[str] = None
    job_value_label: Optional[str] = None
    user_weight: float
    beta: float
    encoded_value: float
    contribution: float
    matched: bool

class MatchedWorkActivity(BaseModel):
    wa_code: str
    wa_label: Optional[str] = None
    user_bws: float
    wa_importance: float
    wa_level: float
    norm_importance: float
    norm_level: float
    wa_contribution: float
    # Additive-RUM diagnostics (BWS_INTEGRATION_MODE="additive_rum")
    weight: Optional[float] = None  # ŵ_c = WA_Importance / Σ WA_Importance (Σ = 1)
    beta: Optional[float] = None    # β_c = user BWS part-worth for this activity

class WorkActivityBWS(BaseModel):
    wa_score_sum: float = 0.0
    details: List[MatchedWorkActivity] = Field(default_factory=list)
    # Additive-RUM diagnostics
    wa_aggregation: Optional[str] = None
    n_work_activities: Optional[int] = None
    V_task: Optional[float] = None
    V_task_hat: Optional[float] = None

class OpportunityRecommendation(BaseModel):
    uuid: str
    URL: Optional[str] = None
    rank: int
    opportunity_title: str
    opportunity_isco_occupation_group: Optional[str] = None
    opportunity_isco_occupation_group_id: Optional[str] = None
    location: Optional[str] = None
    employer: Optional[str] = None
    employment_type: Optional[str] = None
    salary_text: Optional[str] = None
    required_education: Optional[str] = None
    required_experience: Optional[str] = None
    closing_date: Optional[str] = None
    is_eligible: bool
    justification: str
    opportunity_description: Optional[str] = None
    contract_type: Optional[str] = None
    final_score: float
    score_breakdown: ScoreBreakdown
    matched_skills: MatchedSkills
    matched_preferences: List[MatchedPreference] = Field(default_factory=list)
    matched_work_activities: Optional[WorkActivityBWS] = None

class OccupationRecommendation(BaseModel):
    uuid: str
    originUuid: Optional[str] = None
    rank: int
    occupation_label: str
    province: Optional[str] = None
    is_eligible: bool
    justification: str
    occupation_description: Optional[str] = None
    final_score: float
    score_breakdown: ScoreBreakdown
    matched_skills: MatchedSkills
    matched_preferences: List[MatchedPreference] = Field(default_factory=list)
    matched_work_activities: Optional[WorkActivityBWS] = None

class SkillGapRecommendation(BaseModel):
    skill_id: str
    skill_label: str
    proximity_score: float
    job_unlock_count: int
    combined_score: float
    reasoning: str

class MatchResponse(BaseModel):
    user_id: str
    occupation_recommendations: List[OccupationRecommendation] = Field(default_factory=list)
    opportunity_recommendations: List[OpportunityRecommendation] = Field(default_factory=list)
    skill_gap_recommendations: List[SkillGapRecommendation] = Field(default_factory=list)


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
