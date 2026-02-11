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
    task_content: float
    physical_demand: float
    work_flexibility: float
    social_interaction: float
    career_growth: float
    social_meaning: float

class MatchRequest(BaseModel):
    user_id: Optional[str] = None
    city: str
    province: str
    skills_vector: SkillsVector = Field(default_factory=SkillsVector)
    skill_groups_origin_uuids: List[str] = Field(default_factory=list)
    preference_vector: PreferenceVector

class ScoreBreakdown(BaseModel):
    total_skill_utility: float
    skill_components: dict
    skill_penalty_applied: float
    preference_score: float
    demand_score: float

class OpportunityRecommendation(BaseModel):
    uuid: str
    originUuid: Optional[str] = None
    rank: int
    opportunity_title: str
    location: Optional[str] = None
    is_eligible: bool
    justification: str
    contract_type: Optional[str] = None
    final_score: float
    score_breakdown: ScoreBreakdown

class MatchResponse(BaseModel):
    user_id: str
    opportunity_recommendations: List[OpportunityRecommendation]
    skill_gap_recommendations: List[dict]
