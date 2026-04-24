import os
import pulumi_gcp as gcp

from dataclasses import dataclass


def get_env(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise ValueError(">" + key + "< environment variable is not set")

    return value


class EnvKeys:
    MONGO_URL = "MONGO_URL"
    MONGO_DB_NAME = "MONGO_DB_NAME"
    MONGO_JOBS_COLLECTION = "MONGO_JOBS_COLLECTION"
    SCORING_MODE = "SCORING_MODE"
    ADDITIVE_W1_SKILLS = "ADDITIVE_W1_SKILLS"
    ADDITIVE_W2_PREFERENCE = "ADDITIVE_W2_PREFERENCE"
    ADDITIVE_W3_MARKET = "ADDITIVE_W3_MARKET"
    MATCH_TOP_K_OPPORTUNITIES = "MATCH_TOP_K_OPPORTUNITIES"
    MATCH_TOP_K_OCCUPATIONS = "MATCH_TOP_K_OCCUPATIONS"
    MATCH_TOP_K_SKILL_GAPS = "MATCH_TOP_K_SKILL_GAPS"
    GATE_SIMILARITY_THRESHOLD = "GATE_SIMILARITY_THRESHOLD"
    PHAT_ALPHA_ESSENTIAL = "PHAT_ALPHA_ESSENTIAL"
    PHAT_BETA_READINESS = "PHAT_BETA_READINESS"
    PHAT_GAMMA_MARKET = "PHAT_GAMMA_MARKET"
    SKILL_U_W_LOC = "SKILL_U_W_LOC"
    SKILL_U_W_ESS = "SKILL_U_W_ESS"
    SKILL_U_W_OPT = "SKILL_U_W_OPT"
    SKILL_U_W_GRP = "SKILL_U_W_GRP"
    SKILL_U_GAP_PENALTY = "SKILL_U_GAP_PENALTY"
    SKILL_U_TAU_ELIG = "SKILL_U_TAU_ELIG"
    SKILL_MIN_ESSENTIAL_MATCH_SHARE = "SKILL_MIN_ESSENTIAL_MATCH_SHARE"
    SKILL_ESSENTIAL_GEO_FLOOR = "SKILL_ESSENTIAL_GEO_FLOOR"
    PREFERENCE_BASE_CONSTANT = "PREFERENCE_BASE_CONSTANT"
    PREFERENCE_LEGACY_SCORE_SCALE = "PREFERENCE_LEGACY_SCORE_SCALE"
    PREFERENCE_SIGMOID_NUMERATOR = "PREFERENCE_SIGMOID_NUMERATOR"
    PREF_ENABLE_EARNINGS = "PREF_ENABLE_EARNINGS"
    PREF_ENABLE_TASK_CONTENT = "PREF_ENABLE_TASK_CONTENT"
    PREF_ENABLE_PHYSICAL_DEMAND = "PREF_ENABLE_PHYSICAL_DEMAND"
    PREF_ENABLE_WORK_FLEXIBILITY = "PREF_ENABLE_WORK_FLEXIBILITY"
    PREF_ENABLE_SOCIAL = "PREF_ENABLE_SOCIAL"
    PREF_ENABLE_CAREER_GROWTH = "PREF_ENABLE_CAREER_GROWTH"
    PREF_ENABLE_SOCIAL_MEANING = "PREF_ENABLE_SOCIAL_MEANING"
    OCCUPATION_JSON_PATH = "OCCUPATION_JSON_PATH"
    EMBEDDING_MODEL_PATH = "EMBEDDING_MODEL_PATH"
    SKILL_TO_ROW_PATH = "SKILL_TO_ROW_PATH"
    SKILLS_CSV_PATH = "SKILLS_CSV_PATH"
    SKILL_GROUPS_CSV_PATH = "SKILL_GROUPS_CSV_PATH"
    SKILL_HIERARCHY_CSV_PATH = "SKILL_HIERARCHY_CSV_PATH"


@dataclass(frozen=True)
class EnvVars:
    mongodb_uri: str
    mongodb_name: str
    mongo_jobs_collection: str
    scoring_mode: str
    additive_w1_skills: float
    additive_w2_preference: float
    additive_w3_market: float
    match_top_k_opportunities: float
    match_top_k_occupations: float
    match_top_k_skill_gaps: float
    gate_similarity_threshold: float
    phat_alpha_essential: float
    phat_beta_readiness: float
    phat_gamma_market: float
    skill_u_w_loc: float
    skill_u_w_ess: float
    skill_u_w_opt: float
    skill_u_w_grp: float
    skill_u_gap_penalty: float
    skill_u_tau_elig: float
    skill_min_essential_match_share: float
    skill_essential_geo_floor: float
    preference_base_constant: float
    preference_legacy_score_scale: float
    preference_sigmoid_numerator: float
    pref_enable_earnings: bool
    pref_enable_task_content: bool
    pref_enable_physical_demand: bool
    pref_enable_work_flexibility: bool
    pref_enable_social: bool
    pref_enable_career_growth: bool
    pref_enable_social_meaning: bool
    occupation_json_path: str
    embedding_model_path: str
    skill_to_row_path: str
    skills_csv_path: str
    skill_groups_csv_path: str
    skill_hierarchy_csv_path: str

    def get_env_vars(self) -> list[gcp.cloudrunv2.ServiceTemplateContainerEnvArgs]:
        return [
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.MONGO_URL, value=self.mongodb_uri),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.MONGO_DB_NAME, value=self.mongodb_name),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.MONGO_JOBS_COLLECTION,
                                                           value=self.mongo_jobs_collection),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SCORING_MODE, value=self.scoring_mode),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.ADDITIVE_W1_SKILLS,
                                                           value=self.additive_w1_skills),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.ADDITIVE_W2_PREFERENCE,
                                                           value=self.additive_w2_preference),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.ADDITIVE_W3_MARKET,
                                                           value=self.additive_w3_market),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.MATCH_TOP_K_OPPORTUNITIES,
                                                           value=self.match_top_k_opportunities),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.MATCH_TOP_K_OCCUPATIONS,
                                                           value=self.match_top_k_occupations),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.MATCH_TOP_K_SKILL_GAPS,
                                                           value=self.match_top_k_skill_gaps),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.GATE_SIMILARITY_THRESHOLD,
                                                           value=self.gate_similarity_threshold),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PHAT_ALPHA_ESSENTIAL,
                                                           value=self.phat_alpha_essential),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PHAT_BETA_READINESS,
                                                           value=self.phat_beta_readiness),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PHAT_GAMMA_MARKET,
                                                           value=self.phat_gamma_market),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SKILL_U_W_LOC, value=self.skill_u_w_loc),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SKILL_U_W_ESS, value=self.skill_u_w_ess),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SKILL_U_W_OPT, value=self.skill_u_w_opt),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SKILL_U_W_GRP, value=self.skill_u_w_grp),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SKILL_U_GAP_PENALTY,
                                                           value=self.skill_u_gap_penalty),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SKILL_U_TAU_ELIG, value=self.skill_u_tau_elig),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SKILL_MIN_ESSENTIAL_MATCH_SHARE,
                                                           value=self.skill_min_essential_match_share),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SKILL_ESSENTIAL_GEO_FLOOR,
                                                           value=self.skill_essential_geo_floor),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PREFERENCE_BASE_CONSTANT,
                                                           value=self.preference_base_constant),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PREFERENCE_LEGACY_SCORE_SCALE,
                                                           value=self.preference_legacy_score_scale),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PREFERENCE_SIGMOID_NUMERATOR,
                                                           value=self.preference_sigmoid_numerator),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PREF_ENABLE_EARNINGS,
                                                           value=self.pref_enable_earnings),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PREF_ENABLE_TASK_CONTENT,
                                                           value=self.pref_enable_task_content),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PREF_ENABLE_PHYSICAL_DEMAND,
                                                           value=self.pref_enable_physical_demand),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PREF_ENABLE_WORK_FLEXIBILITY,
                                                           value=self.pref_enable_work_flexibility),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PREF_ENABLE_SOCIAL,
                                                           value=self.pref_enable_social),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PREF_ENABLE_CAREER_GROWTH,
                                                           value=self.pref_enable_career_growth),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.PREF_ENABLE_SOCIAL_MEANING,
                                                           value=self.pref_enable_social_meaning),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.OCCUPATION_JSON_PATH,
                                                           value=self.occupation_json_path),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.EMBEDDING_MODEL_PATH,
                                                           value=self.embedding_model_path),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SKILL_TO_ROW_PATH,
                                                           value=self.skill_to_row_path),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SKILLS_CSV_PATH, value=self.skills_csv_path),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SKILL_GROUPS_CSV_PATH,
                                                           value=self.skill_groups_csv_path),
            gcp.cloudrunv2.ServiceTemplateContainerEnvArgs(name=EnvKeys.SKILL_HIERARCHY_CSV_PATH,
                                                           value=self.skill_hierarchy_csv_path),
        ]

    @staticmethod
    def construct_from_env() -> "EnvVars":
        return EnvVars(
            mongodb_uri=get_env(EnvKeys.MONGO_URL),
            mongodb_name=get_env(EnvKeys.MONGO_DB_NAME),
            mongo_jobs_collection=get_env(EnvKeys.MONGO_JOBS_COLLECTION),
            scoring_mode=get_env(EnvKeys.SCORING_MODE),
            additive_w1_skills=get_env(EnvKeys.ADDITIVE_W1_SKILLS),
            additive_w2_preference=get_env(EnvKeys.ADDITIVE_W2_PREFERENCE),
            additive_w3_market=get_env(EnvKeys.ADDITIVE_W3_MARKET),
            match_top_k_opportunities=get_env(EnvKeys.MATCH_TOP_K_OPPORTUNITIES),
            match_top_k_occupations=get_env(EnvKeys.MATCH_TOP_K_OCCUPATIONS),
            match_top_k_skill_gaps=get_env(EnvKeys.MATCH_TOP_K_SKILL_GAPS),
            gate_similarity_threshold=get_env(EnvKeys.GATE_SIMILARITY_THRESHOLD),
            phat_alpha_essential=get_env(EnvKeys.PHAT_ALPHA_ESSENTIAL),
            phat_beta_readiness=get_env(EnvKeys.PHAT_BETA_READINESS),
            phat_gamma_market=get_env(EnvKeys.PHAT_GAMMA_MARKET),
            skill_u_w_loc=get_env(EnvKeys.SKILL_U_W_LOC),
            skill_u_w_ess=get_env(EnvKeys.SKILL_U_W_ESS),
            skill_u_w_opt=get_env(EnvKeys.SKILL_U_W_OPT),
            skill_u_w_grp=get_env(EnvKeys.SKILL_U_W_GRP),
            skill_u_gap_penalty=get_env(EnvKeys.SKILL_U_GAP_PENALTY),
            skill_u_tau_elig=get_env(EnvKeys.SKILL_U_TAU_ELIG),
            skill_min_essential_match_share=get_env(EnvKeys.SKILL_MIN_ESSENTIAL_MATCH_SHARE),
            skill_essential_geo_floor=get_env(EnvKeys.SKILL_ESSENTIAL_GEO_FLOOR),
            preference_base_constant=get_env(EnvKeys.PREFERENCE_BASE_CONSTANT),
            preference_legacy_score_scale=get_env(EnvKeys.PREFERENCE_LEGACY_SCORE_SCALE),
            preference_sigmoid_numerator=get_env(EnvKeys.PREFERENCE_SIGMOID_NUMERATOR),
            pref_enable_earnings=get_env(EnvKeys.PREF_ENABLE_EARNINGS),
            pref_enable_task_content=get_env(EnvKeys.PREF_ENABLE_TASK_CONTENT),
            pref_enable_physical_demand=get_env(EnvKeys.PREF_ENABLE_PHYSICAL_DEMAND),
            pref_enable_work_flexibility=get_env(EnvKeys.PREF_ENABLE_WORK_FLEXIBILITY),
            pref_enable_social=get_env(EnvKeys.PREF_ENABLE_SOCIAL),
            pref_enable_career_growth=get_env(EnvKeys.PREF_ENABLE_CAREER_GROWTH),
            pref_enable_social_meaning=get_env(EnvKeys.PREF_ENABLE_SOCIAL_MEANING),
            occupation_json_path=get_env(EnvKeys.OCCUPATION_JSON_PATH),
            embedding_model_path=get_env(EnvKeys.EMBEDDING_MODEL_PATH),
            skill_to_row_path=get_env(EnvKeys.SKILL_TO_ROW_PATH),
            skills_csv_path=get_env(EnvKeys.SKILLS_CSV_PATH),
            skill_groups_csv_path=get_env(EnvKeys.SKILL_GROUPS_CSV_PATH),
            skill_hierarchy_csv_path=get_env(EnvKeys.SKILL_HIERARCHY_CSV_PATH),
        )
