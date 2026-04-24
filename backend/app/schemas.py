from pydantic import BaseModel, Field
from typing import List, Optional

class Skill(BaseModel):
    preferredLabel: Optional[str] = None
    originUUID: str
    proficiency: Optional[float] = None

class SkillsVector(BaseModel):
    top_skills: List[Skill] = Field(default_factory=list)

class PreferenceVector(BaseModel):
    earnings_per_month: float
    task_content: Optional[float] = 0.0
    physical_demand: float
    work_flexibility: Optional[float] = 0.0
    social_interaction: float
    career_growth: float
    social_meaning: Optional[float] = 0.0
    bws_scores: Optional[dict] = None
    top_10_bws: Optional[List[str]] = None

class MatchRequest(BaseModel):
    user_id: Optional[str] = None
    city: str
    province: str
    skills_vector: SkillsVector = Field(default_factory=SkillsVector)
    skill_groups_origin_uuids: List[str] = Field(default_factory=list)
    preference_vector: PreferenceVector

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

class WorkActivityBWS(BaseModel):
    wa_score_sum: float = 0.0
    details: List[MatchedWorkActivity] = Field(default_factory=list)

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
