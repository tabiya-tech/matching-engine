"""Score breakdown: the components behind a recommendation's ``final_score``.

Domain models that are also part of the ``POST /match`` response (``OpportunityRecommendation`` and
``OccupationRecommendation`` embed ``ScoreBreakdown``); ``app.schemas`` re-exports them.
"""

from typing import Optional

from pydantic import Field

from app.domain.base import DomainModel


class SkillComponents(DomainModel):
    """Per-component skill similarity, each in [0, 1]; null = not computed by the engine."""

    # Optional so the response can carry a partial, interpretable breakdown (ess/opt in [0,1];
    # loc/grp null = "not computed").
    loc: Optional[float] = Field(
        default=None,
        description="Location similarity. Not computed by the current engine (null).",
    )
    ess: Optional[float] = Field(
        default=None,
        description="Mean best cosine similarity over the item's essential skills, in [0, 1].",
        examples=[0.62],
    )
    opt: Optional[float] = Field(
        default=None,
        description="Mean best cosine similarity over the item's optional skills, in [0, 1].",
        examples=[0.48],
    )
    grp: Optional[float] = Field(
        default=None,
        description="Skill-group recall. Not computed by the current engine (null).",
    )


class PHatComponents(DomainModel):
    """Decomposition of the success propensity ``p_hat`` (legacy engine only)."""

    gate: float = Field(default=0.0, description="Eligibility gate component of p_hat.")
    essential_fit: float = Field(
        default=0.0, description="Essential-skill fit component of p_hat."
    )
    recruiter_readiness: float = Field(
        default=0.0, description="Recruiter-readiness component of p_hat."
    )
    market_opportunity: float = Field(
        default=0.0, description="Market-opportunity component of p_hat."
    )


class ScoreBreakdown(DomainModel):
    """Components behind ``final_score``, so a ranking is auditable.

    On ``POST /match``: ``final_score = u_hat × p_hat × coverage_factor × location_tier_factor``
    (``product`` combiner), with an occupation-only demand tilt; each factor that was applied is
    reported here, and a factor that was not applied is null.
    """

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
        default=None,
        description="Decomposition of p_hat. Null when the engine does not compute it.",
    )
    # --- Legacy additive fields ---
    total_skill_utility: Optional[float] = Field(
        default=None,
        description="Essential-skill fit aggregate in [0, 1] (same value as ``skill_components.ess``).",
        examples=[0.62],
    )
    skill_components: Optional[SkillComponents] = Field(
        default=None, description="Per-component skill-similarity breakdown."
    )
    skill_diagnostics: Optional[SkillComponents] = Field(
        default=None,
        description="Reserved for raw skill diagnostics. Null in the current engine.",
    )
    skill_penalty_applied: Optional[float] = Field(
        default=None,
        description="Share of essential skills below the similarity threshold, in [0, 1]. Higher = bigger gap.",
        examples=[0.25],
    )
    preference_score: Optional[float] = Field(
        default=None,
        description="Preference score; equal to ``u_hat``.",
        examples=[0.71],
    )
    preference_score_legacy: Optional[float] = Field(
        default=None,
        description="Preference score from the previous scorer, kept for comparison.",
        examples=[0.68],
    )
    demand_score: Optional[float] = Field(
        default=None,
        description="Expected labour-market demand score for the item. Null when unknown.",
        examples=[0.75],
    )
    demand_label: Optional[str] = Field(
        default=None,
        description="Human-readable demand level. Null when unknown.",
        examples=["High Expected Demand"],
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
