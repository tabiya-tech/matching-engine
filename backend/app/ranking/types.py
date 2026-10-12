"""Typed values passed between the ranking stages.

The ranking pipeline for one corpus (jobs or occupations) is::

    stage-1 shortlist        -> Candidate            (whitened concat cosine, education + location gates)
    stage-2 cross-encoder    -> Candidate            (same item, with the cross-encoder fields set)
    Phase-2 overrides        -> RankOverrideSet      (whitened p_hat, essential coverage, skill detail)
    preference + final score -> ScoredCandidate      (u_hat x p_hat x demotions, re-sorted)
    formatting               -> app.schemas.*Recommendation (the response)

Every model is a strict, frozen ``DomainModel``: a stage returns new values (``model_copy``) rather
than mutating its input.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.domain.base import DomainModel
from app.domain.job import Job
from app.domain.occupation import Occupation

MatchItem = Job | Occupation
"""A ranked item: a job opportunity or an occupation × county row."""

Corpus = Literal["jobs", "occupations"]
"""Which item collection a ranking pass runs over (labels trace spans)."""


# --------------------------------------------------------------------------------------------------
# skills
# --------------------------------------------------------------------------------------------------


class SkillMatchDetail(DomainModel):
    """How one skill of an item is covered by the user's closest skill (``CosineSkillMatcher``)."""

    job_skill_id: str = Field(
        description="Matcher-resolved id of the item's skill (embedding-row id space)."
    )
    job_skill_label: str | None = Field(
        description="Display label of the item's skill in the deployment's language."
    )
    best_user_skill_id: str | None = Field(
        description="Resolved id of the user's most similar skill; None when no user skill is assigned."
    )
    best_user_skill_label: str | None = Field(
        description="Display label of ``best_user_skill_id``; None when no user skill is assigned."
    )
    cosine_similarity: float = Field(
        description=(
            "Similarity in [0, 1]: the clamped cosine (``score_pair``), or the rescaled whitened "
            "cosine (``score_pair_v4``)."
        )
    )
    cosine_similarity_raw: float | None = Field(
        default=None,
        description="Un-rescaled clamped cosine (``score_pair_v4`` only); None from ``score_pair``.",
    )
    exact: bool = Field(
        default=False,
        description="True when the user holds this exact skill id (``score_pair_v4`` only).",
    )


class SkillPairScore(DomainModel):
    """Pooled per-skill cosine between a user and one item (``CosineSkillMatcher.score_pair``)."""

    mean_best_cosine: float = Field(
        description="Mean over the item's embedded skills of the best user-skill cosine, in [0, 1]."
    )
    min_best_cosine: float = Field(
        description="Minimum over the item's embedded skills of the best user-skill cosine, in [0, 1]."
    )
    n_user_skills_embedded: int = Field(
        description="User skills that resolved to an embedding row."
    )
    n_job_skills_embedded: int = Field(
        description="Item skills that resolved to an embedding row."
    )
    per_job_skill: list[SkillMatchDetail] = Field(
        description="One row per embedded item skill, in item order."
    )


class SkillDetail(DomainModel):
    """The per-skill gate detail of one (user, item) pair, reused by the coverage gate and formatting."""

    per_job_skill: list[SkillMatchDetail] = Field(
        description="Per-skill match rows (gate matcher: ``score_pair_v4`` or ``score_pair``)."
    )
    essential_ids: frozenset[str] = Field(
        description="Matcher-resolved ids of the item's essential skills (same id space as ``per_job_skill``)."
    )


# --------------------------------------------------------------------------------------------------
# retrieval + rerank
# --------------------------------------------------------------------------------------------------


class Candidate(DomainModel):
    """One shortlisted item for one user, after stage-1 retrieval and (optionally) the cross-encoder."""

    item: MatchItem = Field(description="The shortlisted job or occupation row.")
    rank: int = Field(
        description="1-based rank: by concat cosine after stage 1, by cross-encoder logit after stage 2."
    )
    concat_cosine_similarity: float = Field(
        description="Stage-1 concat-embedding cosine (whitened space when the artifact is present)."
    )
    per_job_skill: list[SkillMatchDetail] = Field(
        description="Per-skill cosine detail (``score_pair``); the cross-encoder passage is built from its labels."
    )
    rank_cosine: int | None = Field(
        default=None, description="Stage-1 rank; set by the cross-encoder rerank."
    )
    cross_encoder_logit: float | None = Field(
        default=None, description="Raw cross-encoder logit; set by the rerank."
    )
    cross_encoder_score: float | None = Field(
        default=None,
        description="Cross-encoder logit min-max scaled within the user's slate, in [0, 1]; set by the rerank.",
    )

    @property
    def uuid(self) -> str:
        """The item's id."""
        return self.item.uuid


class RetrievalSummary(DomainModel):
    """Configuration and counts of one shortlist + rerank pass (diagnostics for offline tools)."""

    stage1: str = Field(description="Stage-1 method name.")
    stage2: str = Field(description="Stage-2 method name.")
    gemini_user_embed_model: str = Field(description="User embedding model.")
    embedding_dim: int = Field(description="Concat embedding dimension.")
    n_jobs_with_stage1_embedding: int = Field(
        description="Items that carried a stage-1 vector."
    )
    n_jobs_active_loaded: int = Field(description="Items passed in.")
    cross_encoder_model: str | None = Field(
        default=None,
        description="Cross-encoder checkpoint; None when nothing was reranked.",
    )
    language: str | None = Field(
        default=None, description="Deployment language; None when nothing was reranked."
    )
    retrieve_top_k: int | None = Field(
        default=None,
        description="Stage-1 shortlist size; None when nothing was reranked.",
    )
    final_top_k: int | None = Field(
        default=None, description="Stage-2 slate size; None when nothing was reranked."
    )
    mongo_ranked_find_ms: float | None = Field(
        default=None,
        description="Job-load time, when the caller passed the load timing.",
    )
    jobs_retrieval_filter_applied: bool | None = Field(
        default=None,
        description="Whether the job load used the location prefilter, when the caller passed the timing.",
    )


class UserShortlist(DomainModel):
    """The reranked slate of one corpus for one user (``MatchingService.shortlist_and_rerank``)."""

    user_id: str = Field(description="The user's id (empty when the request had none).")
    n_jobs_scored: int = Field(description="Items with a stage-1 vector.")
    n_jobs_active_loaded: int = Field(description="Items passed in.")
    candidates: list[Candidate] = Field(
        description="Cross-encoder-ordered candidates, at most ``final_top_k``."
    )
    config_summary: RetrievalSummary = Field(
        description="Pass configuration and counts."
    )


# --------------------------------------------------------------------------------------------------
# Phase-2 overrides
# --------------------------------------------------------------------------------------------------


class RankOverrideSet(DomainModel):
    """Phase-2 ranking inputs for one user's slate, keyed by item uuid (``RankOverrides.compute``)."""

    p_hat: dict[str, float] = Field(
        default_factory=dict,
        description="Whitened, rescaled concat cosine in [0, 1]; replaces the stage-1 p_hat.",
    )
    coverage: dict[str, float] = Field(
        default_factory=dict,
        description="Ranking essential coverage in [0, 1] (unparsed items get the slate mean).",
    )
    skill_details: dict[str, SkillDetail] = Field(
        default_factory=dict,
        description="Per-skill gate detail, reused by formatting instead of re-scoring.",
    )


# --------------------------------------------------------------------------------------------------
# preference scoring
# --------------------------------------------------------------------------------------------------


class AttributePreference(DomainModel):
    """How one job attribute contributed to the preference utility (a ``MatchedPreference`` row)."""

    attribute: str = Field(description="Attribute name (a ``PreferenceVector`` field).")
    job_value: str | None = Field(
        description="The item's level id (schema-resolved for the unified scorer)."
    )
    job_value_label: str | None = Field(description="Human-readable item level.")
    user_weight: float = Field(description="The user's [0, 1] preference value.")
    beta: float = Field(
        description="Coefficient: recovered ``logit(user_weight)`` (unified) or the configured beta (legacy)."
    )
    encoded_value: float = Field(
        description="Encoded item level: ladder position (unified) or configured mapping (legacy)."
    )
    contribution: float = Field(description="Contribution to the attribute utility.")
    matched: bool = Field(description="True when the attribute contributed.")
    attr_label: str | None = Field(
        default=None, description="Schema label of the attribute (unified scorer)."
    )
    reference_level: str | None = Field(
        default=None, description="Lowest schema level (ladder position 0)."
    )
    target_level: str | None = Field(
        default=None, description="Highest schema level (ladder position 1)."
    )


class WorkActivityContribution(DomainModel):
    """How one O*NET work activity contributed to the task (BWS) utility."""

    wa_code: str = Field(description="O*NET work-activity code.")
    wa_label: str = Field(
        description="O*NET work-activity label (the code when unlabelled)."
    )
    user_bws: float = Field(description="The user's BWS part-worth for the activity.")
    wa_importance: float = Field(description="O*NET importance (1–5).")
    wa_level: float = Field(description="O*NET level (1–7).")
    norm_importance: float = Field(description="Importance / 5.")
    norm_level: float = Field(description="Level / 7.")
    wa_contribution: float = Field(description="Contribution to the task utility.")
    weight: float | None = Field(
        default=None,
        description="Importance weight (sums to 1 over the item's activities); additive-RUM only.",
    )
    beta: float | None = Field(
        default=None, description="Same as ``user_bws``; additive-RUM only."
    )


class WorkActivityUtility(DomainModel):
    """The task (BWS × O*NET work activities) part of the preference utility."""

    wa_details: list[WorkActivityContribution] = Field(
        description="Per-activity contributions."
    )
    wa_score_sum: float = Field(
        description="Task utility (``V_task`` for additive-RUM)."
    )
    wa_aggregation: str | None = Field(
        default=None,
        description="``importance_weighted`` (additive-RUM) or ``mean`` (legacy); None for the legacy sum.",
    )
    n_work_activities: int | None = Field(
        default=None, description="Activities scored."
    )
    V_task: float | None = Field(default=None, description="Raw task utility.")
    V_task_hat: float | None = Field(
        default=None, description="Task utility harmonised to [-1, 1]."
    )


class PreferenceScore(DomainModel):
    """Output of a preference scorer for one (user, item) pair."""

    u_hat: float = Field(description="Preference utility in [0, 1].")
    score: float = Field(
        description="Legacy preference score (attribute-only for the legacy scorer; equals u_hat for the unified one)."
    )
    attributes: list[AttributePreference] = Field(
        description="Per-attribute contributions."
    )
    work_activities: WorkActivityUtility | None = Field(
        default=None,
        description="Task utility; None when the user sent no work-activity BWS or the item has no rated activities.",
    )
    S_attrs: float | None = Field(
        default=None,
        description="Harmonised attribute utility in [-1, 1] (unified scorer).",
    )
    S_wa: float | None = Field(
        default=None, description="Harmonised task utility in [-1, 1] (unified scorer)."
    )
    V: float | None = Field(
        default=None, description="Combined utility before the logistic."
    )
    V_dce: float | None = Field(default=None, description="Raw attribute utility.")
    V_dce_hat: float | None = Field(
        default=None, description="Attribute utility harmonised to [-1, 1]."
    )
    confidence_f: float | None = Field(
        default=None, description="Preference-confidence shrinkage factor in [0, 1]."
    )
    alpha: float | None = Field(default=None, description="Task-utility weight.")
    gamma: float | None = Field(default=None, description="Logistic gain.")
    bws_score_type: str = Field(
        description="``work_activity_id``, ``occupation_id`` or ``unknown``."
    )
    scoring_model: str = Field(description="Scorer identifier.")


# --------------------------------------------------------------------------------------------------
# final score
# --------------------------------------------------------------------------------------------------


class FinalScoreBreakdown(DomainModel):
    """How ``final_score`` was built for one candidate (``enrich_recommendations_with_preferences``)."""

    final_score_combiner: str = Field(description="``product`` or ``geometric_mean``.")
    u_hat: float = Field(description="Preference utility, rounded.")
    p_hat: float = Field(description="Success propensity, rounded.")
    p_hat_source: str = Field(description="Signal p_hat came from.")
    final_score: float = Field(
        description="Final score after every applied factor, rounded."
    )
    preference_score_legacy: float = Field(description="Legacy preference score.")
    demand_score: float | None = Field(
        default=None,
        description="Demand score, when the occupation demand tilt applied.",
    )
    demand_label: str | None = Field(
        default=None, description="Demand label, when the demand tilt applied."
    )
    demand_gamma: float | None = Field(
        default=None, description="Demand exponent, when the demand tilt applied."
    )
    essential_coverage: float | None = Field(
        default=None,
        description="Ranking coverage, when the coverage demotion applied.",
    )
    coverage_gamma: float | None = Field(
        default=None,
        description="Coverage exponent, when the coverage demotion applied.",
    )
    coverage_floor: float | None = Field(
        default=None, description="Coverage floor, when the coverage demotion applied."
    )
    coverage_factor: float | None = Field(
        default=None,
        description="Coverage multiplier, when the coverage demotion applied.",
    )
    location_tier_factor: float | None = Field(
        default=None,
        description="Location-tier multiplier, when location tiers applied.",
    )


class ScoredCandidate(DomainModel):
    """A candidate with its preference score and final score, in final-score order."""

    candidate: Candidate = Field(description="The reranked candidate.")
    rank: int = Field(description="1-based rank by final score.")
    preference: PreferenceScore = Field(description="Preference scorer output.")
    p_hat: float = Field(description="Success propensity in [0, 1], rounded.")
    p_hat_source: str = Field(description="Signal p_hat came from.")
    final_score: float = Field(description="Final score in [0, 1], rounded.")
    breakdown: FinalScoreBreakdown = Field(
        description="Factors behind ``final_score``."
    )

    @property
    def item(self) -> MatchItem:
        """The scored item."""
        return self.candidate.item

    @property
    def uuid(self) -> str:
        """The scored item's id."""
        return self.candidate.item.uuid


class DemandScore(DomainModel):
    """Labour-market demand of an item (``DemandScorer``)."""

    score: float = Field(description="Demand score in [0, 1]; 0.5 when unknown.")
    label: str | None = Field(description="Demand label; None when absent.")
    present: bool = Field(description="True when the label is known to the mapping.")
