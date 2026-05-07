"""
Matching tunables for `/match`.

Effective values = ``DEFAULT_TUNABLE_FLAT`` (code) merged with Mongo ``matching_configuration.values``.
Process env vars for these keys are not used during scoring once this module is active per request.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

DEFAULT_TUNABLE_FLAT: Dict[str, str] = {
    "SCORING_MODE": "multiplicative",
    "ADDITIVE_W1_SKILLS": "0.40",
    "ADDITIVE_W2_PREFERENCE": "0.40",
    "ADDITIVE_W3_MARKET": "0.20",
    "MATCH_TOP_K_OPPORTUNITIES": "5",
    "MATCH_TOP_K_OCCUPATIONS": "5",
    "MATCH_TOP_K_SKILL_GAPS": "5",
    "GATE_SIMILARITY_THRESHOLD": "0.35",
    "PHAT_ALPHA_ESSENTIAL": "0.5",
    "PHAT_BETA_READINESS": "0.2",
    "PHAT_GAMMA_MARKET": "0.3",
    "SKILL_U_W_LOC": "0.20",
    "SKILL_U_W_ESS": "0.50",
    "SKILL_U_W_OPT": "0.20",
    "SKILL_U_W_GRP": "0.10",
    "SKILL_U_GAP_PENALTY": "0.50",
    "SKILL_U_TAU_ELIG": "0.35",
    "SKILL_MIN_ESSENTIAL_MATCH_SHARE": "1.0",
    "SKILL_ESSENTIAL_GEO_FLOOR": "0.000001",
    "PREFERENCE_BASE_CONSTANT": "0.5",
    "PREFERENCE_LEGACY_SCORE_SCALE": "0.2",
    "PREFERENCE_SIGMOID_NUMERATOR": "4.0",
    "PREF_ENABLE_EARNINGS": "true",
    "PREF_ENABLE_TASK_CONTENT": "false",
    "PREF_ENABLE_PHYSICAL_DEMAND": "true",
    "PREF_ENABLE_WORK_FLEXIBILITY": "false",
    "PREF_ENABLE_SOCIAL": "true",
    "PREF_ENABLE_CAREER_GROWTH": "true",
    "PREF_ENABLE_SOCIAL_MEANING": "false",
}

TUNABLE_KEYS = frozenset(DEFAULT_TUNABLE_FLAT.keys())

_PREF_ENV_TO_ATTR = {
    "PREF_ENABLE_EARNINGS": "earnings_per_month",
    "PREF_ENABLE_TASK_CONTENT": "task_content",
    "PREF_ENABLE_PHYSICAL_DEMAND": "physical_demand",
    "PREF_ENABLE_WORK_FLEXIBILITY": "work_flexibility",
    "PREF_ENABLE_SOCIAL": "social_interaction",
    "PREF_ENABLE_CAREER_GROWTH": "career_growth",
    "PREF_ENABLE_SOCIAL_MEANING": "social_meaning",
}


def _parse_bool(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _parse_float(v: Any) -> float:
    return float(str(v).strip())


def _parse_int(v: Any) -> int:
    return int(float(str(v).strip()))


@dataclass
class MatchingRuntimeSettings:
    scoring_mode: str
    global_weights: Dict[str, float]
    match_top_k_opportunities: int
    match_top_k_occupations: int
    match_top_k_skill_gaps: int
    match_apply_location_filter: bool
    gate_similarity_threshold: float
    match_response_skill_min_score: float
    success_propensity_config: Dict[str, Any]
    skill_u_w_loc: float
    skill_u_w_ess: float
    skill_u_w_opt: float
    skill_u_w_grp: float
    skill_u_gap_penalty: float
    skill_u_tau_elig: float
    skill_min_essential_match_share: float
    skill_essential_geo_floor: float
    preference_config: Dict[str, Any]
    preference_legacy_score_scale: float
    preference_sigmoid_numerator: float
    sources: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config_module(cls) -> MatchingRuntimeSettings:
        """Bootstrap nested structures from ``app.config`` (import-time env); overwritten by flat overlays."""
        import app.config as c

        gate = float(c.GATE_SIMILARITY_THRESHOLD)
        mr_min = float(c.MATCH_RESPONSE_SKILL_MIN_SCORE)
        sp = dict(c.SUCCESS_PROPENSITY_CONFIG)
        sp["gate_threshold"] = gate
        return cls(
            scoring_mode=str(c.SCORING_MODE),
            global_weights=dict(c.GLOBAL_WEIGHTS),
            match_top_k_opportunities=int(c.MATCH_TOP_K_OPPORTUNITIES),
            match_top_k_occupations=int(c.MATCH_TOP_K_OCCUPATIONS),
            match_top_k_skill_gaps=int(c.MATCH_TOP_K_SKILL_GAPS),
            match_apply_location_filter=bool(c.MATCH_APPLY_LOCATION_FILTER),
            gate_similarity_threshold=gate,
            match_response_skill_min_score=mr_min,
            success_propensity_config=sp,
            skill_u_w_loc=float(c.SKILL_U_W_LOC),
            skill_u_w_ess=float(c.SKILL_U_W_ESS),
            skill_u_w_opt=float(c.SKILL_U_W_OPT),
            skill_u_w_grp=float(c.SKILL_U_W_GRP),
            skill_u_gap_penalty=float(c.SKILL_U_GAP_PENALTY),
            skill_u_tau_elig=float(c.SKILL_U_TAU_ELIG),
            skill_min_essential_match_share=float(c.SKILL_MIN_ESSENTIAL_MATCH_SHARE),
            skill_essential_geo_floor=float(c.SKILL_ESSENTIAL_GEO_FLOOR),
            preference_config=copy.deepcopy(c.PREFERENCE_CONFIG),
            preference_legacy_score_scale=float(c.PREFERENCE_LEGACY_SCORE_SCALE),
            preference_sigmoid_numerator=float(c.PREFERENCE_SIGMOID_NUMERATOR),
            sources={},
        )

    def with_flat_overrides(
        self,
        flat: Dict[str, Any],
        *,
        source_tag: str,
    ) -> MatchingRuntimeSettings:
        s = copy.deepcopy(self)
        sources = dict(s.sources)
        pref = s.preference_config
        sp = s.success_propensity_config
        gw = dict(s.global_weights)

        for key, raw in flat.items():
            if key not in TUNABLE_KEYS:
                continue
            sources[key] = source_tag
            if key == "SCORING_MODE":
                v = str(raw).strip()
                if v not in ("multiplicative", "additive"):
                    raise ValueError(f"SCORING_MODE must be multiplicative or additive, got {v!r}")
                s.scoring_mode = v
            elif key == "ADDITIVE_W1_SKILLS":
                gw["w1_skills"] = _parse_float(raw)
            elif key == "ADDITIVE_W2_PREFERENCE":
                gw["w2_preference"] = _parse_float(raw)
            elif key == "ADDITIVE_W3_MARKET":
                gw["w3_market"] = _parse_float(raw)
            elif key == "MATCH_TOP_K_OPPORTUNITIES":
                s.match_top_k_opportunities = _parse_int(raw)
            elif key == "MATCH_TOP_K_OCCUPATIONS":
                s.match_top_k_occupations = _parse_int(raw)
            elif key == "MATCH_TOP_K_SKILL_GAPS":
                s.match_top_k_skill_gaps = _parse_int(raw)
            elif key == "GATE_SIMILARITY_THRESHOLD":
                g = _parse_float(raw)
                s.gate_similarity_threshold = g
                sp["gate_threshold"] = g
                s.match_response_skill_min_score = g
            elif key == "PHAT_ALPHA_ESSENTIAL":
                sp["alpha_essential"] = _parse_float(raw)
            elif key == "PHAT_BETA_READINESS":
                sp["beta_readiness"] = _parse_float(raw)
            elif key == "PHAT_GAMMA_MARKET":
                sp["gamma_market"] = _parse_float(raw)
            elif key == "SKILL_U_W_LOC":
                s.skill_u_w_loc = _parse_float(raw)
            elif key == "SKILL_U_W_ESS":
                s.skill_u_w_ess = _parse_float(raw)
            elif key == "SKILL_U_W_OPT":
                s.skill_u_w_opt = _parse_float(raw)
            elif key == "SKILL_U_W_GRP":
                s.skill_u_w_grp = _parse_float(raw)
            elif key == "SKILL_U_GAP_PENALTY":
                s.skill_u_gap_penalty = _parse_float(raw)
            elif key == "SKILL_U_TAU_ELIG":
                s.skill_u_tau_elig = _parse_float(raw)
            elif key == "SKILL_MIN_ESSENTIAL_MATCH_SHARE":
                s.skill_min_essential_match_share = _parse_float(raw)
            elif key == "SKILL_ESSENTIAL_GEO_FLOOR":
                s.skill_essential_geo_floor = _parse_float(raw)
            elif key == "PREFERENCE_BASE_CONSTANT":
                pref["base_constant"] = _parse_float(raw)
            elif key == "PREFERENCE_LEGACY_SCORE_SCALE":
                s.preference_legacy_score_scale = _parse_float(raw)
            elif key == "PREFERENCE_SIGMOID_NUMERATOR":
                s.preference_sigmoid_numerator = _parse_float(raw)
            elif key in _PREF_ENV_TO_ATTR:
                attr = _PREF_ENV_TO_ATTR[key]
                if attr in pref.get("attributes", {}):
                    pref["attributes"][attr]["enabled"] = _parse_bool(raw)

        s.global_weights = gw
        s.sources = sources
        return s

    def to_effective_flat(self) -> Dict[str, str]:
        p = self.preference_config
        attrs = p.get("attributes", {})
        return {
            "SCORING_MODE": self.scoring_mode,
            "ADDITIVE_W1_SKILLS": str(self.global_weights["w1_skills"]),
            "ADDITIVE_W2_PREFERENCE": str(self.global_weights["w2_preference"]),
            "ADDITIVE_W3_MARKET": str(self.global_weights["w3_market"]),
            "MATCH_TOP_K_OPPORTUNITIES": str(self.match_top_k_opportunities),
            "MATCH_TOP_K_OCCUPATIONS": str(self.match_top_k_occupations),
            "MATCH_TOP_K_SKILL_GAPS": str(self.match_top_k_skill_gaps),
            "GATE_SIMILARITY_THRESHOLD": str(self.gate_similarity_threshold),
            "PHAT_ALPHA_ESSENTIAL": str(self.success_propensity_config["alpha_essential"]),
            "PHAT_BETA_READINESS": str(self.success_propensity_config["beta_readiness"]),
            "PHAT_GAMMA_MARKET": str(self.success_propensity_config["gamma_market"]),
            "SKILL_U_W_LOC": str(self.skill_u_w_loc),
            "SKILL_U_W_ESS": str(self.skill_u_w_ess),
            "SKILL_U_W_OPT": str(self.skill_u_w_opt),
            "SKILL_U_W_GRP": str(self.skill_u_w_grp),
            "SKILL_U_GAP_PENALTY": str(self.skill_u_gap_penalty),
            "SKILL_U_TAU_ELIG": str(self.skill_u_tau_elig),
            "SKILL_MIN_ESSENTIAL_MATCH_SHARE": str(self.skill_min_essential_match_share),
            "SKILL_ESSENTIAL_GEO_FLOOR": str(self.skill_essential_geo_floor),
            "PREFERENCE_BASE_CONSTANT": str(p.get("base_constant", 0.5)),
            "PREFERENCE_LEGACY_SCORE_SCALE": str(self.preference_legacy_score_scale),
            "PREFERENCE_SIGMOID_NUMERATOR": str(self.preference_sigmoid_numerator),
            "PREF_ENABLE_EARNINGS": str(attrs.get("earnings_per_month", {}).get("enabled", True)).lower(),
            "PREF_ENABLE_TASK_CONTENT": str(attrs.get("task_content", {}).get("enabled", False)).lower(),
            "PREF_ENABLE_PHYSICAL_DEMAND": str(attrs.get("physical_demand", {}).get("enabled", True)).lower(),
            "PREF_ENABLE_WORK_FLEXIBILITY": str(attrs.get("work_flexibility", {}).get("enabled", False)).lower(),
            "PREF_ENABLE_SOCIAL": str(attrs.get("social_interaction", {}).get("enabled", True)).lower(),
            "PREF_ENABLE_CAREER_GROWTH": str(attrs.get("career_growth", {}).get("enabled", True)).lower(),
            "PREF_ENABLE_SOCIAL_MEANING": str(attrs.get("social_meaning", {}).get("enabled", False)).lower(),
        }


def build_effective_settings(mongo_overrides: Optional[Dict[str, Any]] = None) -> MatchingRuntimeSettings:
    """Apply code defaults, then Mongo. Location filter on for scoring unless extended later."""
    mongo_overrides = mongo_overrides or {}
    base = MatchingRuntimeSettings.from_config_module()
    s = base.with_flat_overrides(DEFAULT_TUNABLE_FLAT, source_tag="default")
    if mongo_overrides:
        s = s.with_flat_overrides(mongo_overrides, source_tag="mongodb")
    s.match_apply_location_filter = True
    return s


def build_env_settings() -> MatchingRuntimeSettings:
    """Use env/.env values from ``app.config`` as the effective runtime settings."""
    s = MatchingRuntimeSettings.from_config_module()
    s.sources = {k: "env" for k in TUNABLE_KEYS}
    return s


def apply_settings_to_config_module(settings: MatchingRuntimeSettings) -> None:
    """Write runtime settings back onto ``app.config`` module constants."""
    import app.config as c

    c.SCORING_MODE = settings.scoring_mode
    c.GLOBAL_WEIGHTS = dict(settings.global_weights)
    c.MATCH_TOP_K_OPPORTUNITIES = int(settings.match_top_k_opportunities)
    c.MATCH_TOP_K_OCCUPATIONS = int(settings.match_top_k_occupations)
    c.MATCH_TOP_K_SKILL_GAPS = int(settings.match_top_k_skill_gaps)
    c.MATCH_APPLY_LOCATION_FILTER = bool(settings.match_apply_location_filter)
    c.GATE_SIMILARITY_THRESHOLD = float(settings.gate_similarity_threshold)
    c.MATCH_RESPONSE_SKILL_MIN_SCORE = float(settings.match_response_skill_min_score)
    c.SUCCESS_PROPENSITY_CONFIG = dict(settings.success_propensity_config)
    c.SKILL_U_W_LOC = float(settings.skill_u_w_loc)
    c.SKILL_U_W_ESS = float(settings.skill_u_w_ess)
    c.SKILL_U_W_OPT = float(settings.skill_u_w_opt)
    c.SKILL_U_W_GRP = float(settings.skill_u_w_grp)
    c.SKILL_U_GAP_PENALTY = float(settings.skill_u_gap_penalty)
    c.SKILL_U_TAU_ELIG = float(settings.skill_u_tau_elig)
    c.SKILL_MIN_ESSENTIAL_MATCH_SHARE = float(settings.skill_min_essential_match_share)
    c.SKILL_ESSENTIAL_GEO_FLOOR = float(settings.skill_essential_geo_floor)
    c.PREFERENCE_CONFIG = copy.deepcopy(settings.preference_config)
    c.PREFERENCE_LEGACY_SCORE_SCALE = float(settings.preference_legacy_score_scale)
    c.PREFERENCE_SIGMOID_NUMERATOR = float(settings.preference_sigmoid_numerator)


_DEFAULT_SNAPSHOT: Optional[MatchingRuntimeSettings] = None


def get_default_runtime_settings() -> MatchingRuntimeSettings:
    """Cached settings when Mongo is empty (same as ``build_effective_settings({})``)."""
    global _DEFAULT_SNAPSHOT
    if _DEFAULT_SNAPSHOT is None:
        _DEFAULT_SNAPSHOT = build_effective_settings({})
    return _DEFAULT_SNAPSHOT
