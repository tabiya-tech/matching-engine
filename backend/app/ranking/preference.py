"""Preference scoring (u_hat): DCE attribute utility + BWS work-activity utility, combined additively (see docs/preference-scoring.md), plus the legacy PreferenceScorer and the u_hat x p_hat final-score combiner."""

from __future__ import annotations

import logging
import math
import re
from typing import Literal

from pydantic import Field

from app.artifacts.types import AttributeSchema, AttributeSpec
from app.config import (
    BWS_ALPHA,
    BWS_GAIN_GAMMA,
    BWS_INTEGRATION_MODE,
    DCE_ATTR_SCALE,
    DCE_LOGIT_EPS,
    PREFERENCE_CONFIG,
    PREFERENCE_LEGACY_SCORE_SCALE,
    PREFERENCE_SIGMOID_NUMERATOR,
)
from app.domain.base import DomainModel
from app.domain.user import UserProfile
from app.ranking.types import (
    AttributePreference,
    MatchItem,
    PreferenceScore,
    WorkActivityContribution,
    WorkActivityUtility,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# final_score
#
# Combine u_hat and p_hat into a final ranking score (configurable).
#
# Part of the hybrid v1 preference+skill integration (see README.md).
# --------------------------------------------------------------------------------------------------

FinalScoreCombiner = Literal["product", "geometric_mean"]


def combine_final_score(
    u_hat: float, p_hat: float, *, combiner: FinalScoreCombiner
) -> float:
    """Return a final score in [0, 1] (inputs assumed in [0, 1])."""

    u = float(u_hat)
    p = float(p_hat)
    u = 0.0 if not math.isfinite(u) else max(0.0, min(1.0, u))
    p = 0.0 if not math.isfinite(p) else max(0.0, min(1.0, p))

    if combiner == "product":
        return u * p
    if combiner == "geometric_mean":
        return math.sqrt(u * p)
    raise ValueError(f"Unknown combiner: {combiner!r}")


# --------------------------------------------------------------------------------------------------
# levels
#
# Map demand-side level tags → Vⱼ using ladder position and gain/cost orientation.
# --------------------------------------------------------------------------------------------------

# Production / KenyaJobs_V2 level ids → schema ids in job_attributes_schema (1).json
LEVEL_ID_ALIASES: dict[str, dict[str, str]] = {
    "earnings_per_month": {
        "earn_15k": "earn_10_20k",
        "earn_30k": "earn_20_35k",
        "earn_50k": "earn_50_80k",
        "earn_70k": "earn_50_80k",
    },
    "social_interaction": {
        "soc_customers": "soc_people",
        "soc_peers": "soc_people",
    },
}

_EARN_K_RE = re.compile(r"^earn_(\d+)k$", re.IGNORECASE)

Orientation = Literal["gain", "cost"]

# Client: gain (+) → V' = ladder; cost (−) → V' = 1 − ladder
GAIN_ORIENTED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "earnings_per_month",  # Earnings
        "career_growth",  # Career Growth
        "social_interaction",  # Social Interaction (higher = more people)
        "task_content",  # Task Routine / task ladder (higher = more creative)
        "work_flexibility",  # default gain: higher flexibility = better
        "social_meaning",  # default gain: higher meaning = better
    }
)
COST_ORIENTED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "physical_demand",  # Physical Demand — light = good → invert
    }
)


def attribute_orientation(attr_name: str) -> Orientation:
    """``cost`` for attributes where less is better (physical demand), else ``gain``."""
    if attr_name in COST_ORIENTED_ATTRIBUTES:
        return "cost"
    return "gain"


def _level_index(level_id: str, level_ids: list[str]) -> int | None:
    if not level_id or level_id not in level_ids:
        return None
    return level_ids.index(level_id)


def _schema_spec(attr_name: str, schema: AttributeSchema) -> AttributeSpec | None:
    return schema.spec(attr_name)


def _earnings_level_from_kes(kes: float, level_ids: list[str]) -> str | None:
    """Map a monthly KES amount to the schema earnings bucket id."""
    if kes < 10_000:
        return "earn_lt10k" if "earn_lt10k" in level_ids else None
    brackets = [
        (20_000, "earn_10_20k"),
        (35_000, "earn_20_35k"),
        (50_000, "earn_35_50k"),
        (80_000, "earn_50_80k"),
        (120_000, "earn_80_120k"),
        (180_000, "earn_180_180k"),
        (300_000, "earn_180_300k"),
    ]
    for bound, lid in brackets:
        if kes < bound:
            return lid if lid in level_ids else None
    return "earn_300k_plus" if "earn_300k_plus" in level_ids else None


def resolve_schema_level_id(
    attr_name: str,
    raw_level_id: str | None,
    schema: AttributeSchema,
) -> str | None:
    """
    Normalize job level tags to ids declared in the attribute schema.

    Returns None if the job did not supply a level or it cannot be mapped.
    """
    if raw_level_id is None:
        return None
    raw = str(raw_level_id).strip()
    if not raw or raw in ("—", "-", "â€", 'â€"'):
        return None

    spec = _schema_spec(attr_name, schema)
    if not spec:
        return None
    level_ids = spec.level_ids
    if raw in level_ids:
        return raw

    aliased = (LEVEL_ID_ALIASES.get(attr_name) or {}).get(raw)
    if aliased and aliased in level_ids:
        return aliased

    if attr_name == "earnings_per_month":
        m = _EARN_K_RE.match(raw)
        if m:
            kes = float(m.group(1)) * 1000.0
            mapped = _earnings_level_from_kes(kes, level_ids)
            if mapped:
                return mapped

    return None


def ladder_position(
    attr_name: str, level_id: str | None, schema: AttributeSchema
) -> float:
    """Ladder position ∈ [0, 1]: lowest schema bucket → 0, highest → 1."""
    resolved = resolve_schema_level_id(attr_name, level_id, schema)
    if resolved is None:
        return 0.0

    spec = _schema_spec(attr_name, schema)
    if not spec:
        return 0.0
    level_ids = spec.level_ids
    if not level_ids:
        return 0.0
    if len(level_ids) == 1:
        return 1.0 if resolved == level_ids[0] else 0.0
    idx = _level_index(resolved, level_ids)
    if idx is None:
        return 0.0
    return float(idx) / float(len(level_ids) - 1)


def job_level_to_vj(
    attr_name: str, level_id: str | None, schema: AttributeSchema
) -> float:
    """
    Vⱼ for Part A (client directional mapping).

    Gain (+): V'ⱼ = ladder position
    Cost (−): V'ⱼ = 1.0 − ladder position  (e.g. phys_light → 1, phys_heavy → 0)
    """
    pos = ladder_position(attr_name, level_id, schema)
    if attribute_orientation(attr_name) == "cost":
        return 1.0 - pos
    return pos


def attribute_label(attr_name: str, schema: AttributeSchema) -> str:
    """Display label of an attribute (title-cased name when the schema has none)."""
    spec = schema.spec(attr_name)
    if spec and spec.label:
        return spec.label
    return attr_name.replace("_", " ").title()


def level_label(attr_name: str, level_id: str | None, schema: AttributeSchema) -> str:
    """Human-readable job level for dashboards (uses schema label after resolve)."""
    resolved = resolve_schema_level_id(attr_name, level_id, schema)
    if not resolved:
        return "—"
    spec = _schema_spec(attr_name, schema)
    if not spec:
        return str(level_id or "—")
    for lv in spec.levels:
        if lv.id == resolved:
            return lv.label or resolved
    return resolved


# --------------------------------------------------------------------------------------------------
# preference_score
# --------------------------------------------------------------------------------------------------

# src/preference_scorer.py


def detect_bws_score_type(bws_scores: dict[str, float]) -> str:
    """Whether BWS keys are occupation ids (2 digits) or work-activity ids (e.g. ``4.A.2.a.4``).

    :return: ``occupation_id``, ``work_activity_id`` or ``unknown``
    """
    if not bws_scores:
        return "unknown"
    occupation_id_pattern = re.compile(r"^\d{2}$")
    work_activity_id_pattern = re.compile(r"^\d+(\.[A-Za-z0-9]+)+$")
    keys = list(bws_scores.keys())
    if all(occupation_id_pattern.match(k) for k in keys):
        return "occupation_id"
    if any(work_activity_id_pattern.match(k) for k in keys):
        return "work_activity_id"
    return "unknown"


def _bws_scores(user: UserProfile) -> dict[str, float]:
    return user.preference_vector.bws_scores or {}


def _pref_value(user: UserProfile, attr_name: str) -> float | None:
    """Per-attribute DCE value v_k = sigmoid(beta_k) in [0,1]; None if not supplied."""
    value = getattr(user.preference_vector, attr_name, None)
    return value if isinstance(value, float) else None


def _job_level(item: MatchItem, attr_name: str) -> str | None:
    """The item's level id for an attribute; None when the item does not describe it."""
    value = getattr(item.attributes, attr_name, None)
    return value if isinstance(value, str) else None


class PreferenceScorer:
    """LEGACY attribute scorer: configured betas × user weights × encoded levels, logistic u_hat."""

    @staticmethod
    def detect_bws_score_type(bws_scores: dict[str, float]) -> str:
        """See :func:`detect_bws_score_type`."""
        return detect_bws_score_type(bws_scores)

    def __init__(self):
        self.config = PREFERENCE_CONFIG
        self.base_constant = self.config["base_constant"]

        self._enabled_attrs = {
            k: v for k, v in self.config["attributes"].items() if v.get("enabled", True)
        }

        # Dynamic sigmoid scaling: sigmoid(max_raw * factor) ≈ 0.98
        # so a perfect match on all enabled attributes reaches ~0.98.
        max_positive_sum = sum(abs(s["beta"]) for s in self._enabled_attrs.values())
        self._sigmoid_factor = (
            PREFERENCE_SIGMOID_NUMERATOR / max_positive_sum
            if max_positive_sum > 0
            else 2.0
        )
        # Analytic normaliser for the additive-RUM DCE term: max achievable |Σ β·w·x|
        # (each of w, x ∈ [0,1]) ⇒ Σ|β|. Harmonises V_dce → [-1,1].
        self._dce_normalizer = max_positive_sum

    @staticmethod
    def _humanize_label(value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value)
        prefixes = ("earn_", "task_", "phys_", "flex_", "soc_", "growth_", "mean_")
        for p in prefixes:
            if text.startswith(p):
                text = text[len(p) :]
                break
        text = text.replace("_", " ").strip()
        return text.title() if text else None

    def calculate_score(
        self,
        user: UserProfile,
        item: MatchItem,
        *,
        include_work_activities: bool = True,
    ) -> PreferenceScore:
        """S_pref = base_constant + raw_sum * scaling_factor, plus the BWS work-activity term.

        :param include_work_activities: Accepted for interface parity with
            ``UnifiedPreferenceScorer``; the legacy scorer always scores work activities
        """

        raw_score_sum = 0.0
        rows: list[AttributePreference] = []
        bws_scores = _bws_scores(user)
        bws_score_type = detect_bws_score_type(bws_scores)

        # Standard preference scoring (only enabled attributes)
        for attr_key, settings in self._enabled_attrs.items():
            beta = settings["beta"]
            user_weight = _pref_value(user, attr_key) or 0.0
            job_value_raw = _job_level(item, attr_key)

            encoded_value = 0.0
            if settings["type"] == "dummy":
                if job_value_raw == settings["active_level"]:
                    encoded_value = 1.0
            elif settings["type"] == "ordered_linear":
                mapping = settings.get("mapping", {})
                encoded_value = mapping.get(job_value_raw, 0.0)

            contribution = beta * user_weight * encoded_value
            raw_score_sum += contribution

            rows.append(
                AttributePreference(
                    attribute=attr_key,
                    job_value=job_value_raw,
                    job_value_label=self._humanize_label(job_value_raw),
                    user_weight=round(float(user_weight), 4),
                    beta=round(float(beta), 4),
                    encoded_value=round(float(encoded_value), 4),
                    contribution=round(float(contribution), 4),
                    matched=encoded_value > 0,
                )
            )

        # ``raw_score_sum`` now holds the pure DCE-attribute utility (V_dce, BWS-excluded).
        dce_sum = raw_score_sum
        work_activities: WorkActivityUtility | None = None
        comb: CombinedUtility | None = None

        if BWS_INTEGRATION_MODE == "additive_rum":
            # Additive-RUM: harmonise V_task and V_dce to [-1,1], combine, logistic.
            v_task, work_activities = compute_task_utility(user, item)
            comb = combine_utilities(
                v_task,
                dce_sum,
                self._dce_normalizer,
                alpha=BWS_ALPHA,
                gamma=BWS_GAIN_GAMMA,
            )
            u_hat = comb.u_hat
        else:
            # Legacy: BWS×(Imp/5)×(Lev/7) summed into the same sigmoid as the attributes.
            if bws_score_type == "work_activity_id":
                wa_details: list[WorkActivityContribution] = []
                wa_score_sum = 0.0
                for wa in item.onet_work_activities:
                    user_bws = float(bws_scores.get(wa.code, 0.0))
                    norm_importance = wa.importance / 5.0 if wa.importance else 0.0
                    norm_level = wa.level / 7.0 if wa.level else 0.0
                    wa_contribution = user_bws * norm_importance * norm_level
                    wa_score_sum += wa_contribution
                    wa_details.append(
                        WorkActivityContribution(
                            wa_code=wa.code,
                            wa_label=wa.label or wa.code,
                            user_bws=user_bws,
                            wa_importance=wa.importance,
                            wa_level=wa.level,
                            norm_importance=round(norm_importance, 4),
                            norm_level=round(norm_level, 4),
                            wa_contribution=round(wa_contribution, 4),
                        )
                    )
                raw_score_sum += wa_score_sum
                work_activities = WorkActivityUtility(
                    wa_details=wa_details, wa_score_sum=round(wa_score_sum, 4)
                )

            sigmoid_input = raw_score_sum * self._sigmoid_factor
            u_hat = (
                1.0 / (1.0 + math.exp(-sigmoid_input))
                if abs(sigmoid_input) < 500
                else (1.0 if sigmoid_input > 0 else 0.0)
            )

        # Legacy score kept for backward compatibility (attribute-only; ranking uses u_hat).
        scaled_sum_legacy = dce_sum * PREFERENCE_LEGACY_SCORE_SCALE
        legacy_score = max(0.0, min(1.0, self.base_constant + scaled_sum_legacy))

        return PreferenceScore(
            u_hat=round(u_hat, 4),
            score=legacy_score,
            attributes=rows,
            work_activities=work_activities,
            V=comb.V if comb else None,
            V_dce_hat=comb.v_dce_h if comb else None,
            alpha=comb.alpha if comb else None,
            gamma=comb.gamma if comb else None,
            bws_score_type=bws_score_type,
            scoring_model="legacy_preference_scorer",
        )


# --------------------------------------------------------------------------------------------------
# work_activities
#
# Part B — O*NET work activities (BWS task utility) and the additive-RUM combination.
#
# Shared by the preference scorers (``UnifiedPreferenceScorer`` and legacy ``PreferenceScorer``).
#
# Additive-RUM integration (``BWS_INTEGRATION_MODE="additive_rum"``):
#     V_task = Σ_c ŵ_c · β_c,  ŵ_c ∝ WA_Importance, Σŵ_c = 1 over the job's activities.
#     β_c = bws_scores.get(WA_code, 0.0)  (HB posterior part-worth, ~[-2,2], 0 = neutral).
#     u_hat = logistic( γ · [ α·Ṽ_task + (1-α)·Ṽ_dce ] ), each component harmonised to [-1,1].
#
# ``work_activity_block`` is the legacy (mean of BWS×Imp×Level) path, kept for
# ``BWS_INTEGRATION_MODE="legacy"``.
# --------------------------------------------------------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


class DceUtility(DomainModel):
    """Part A of the unified preference score: the DCE attribute utility."""

    rows: list[AttributePreference] = Field(
        description="One row per schema attribute the user valued and the item describes."
    )
    V_dce: float = Field(description="Σ β̂_k · ṽ_k, rounded.")
    V_dce_hat: float = Field(
        description="confidence · clamp(V_dce / Σ|β̂_k|, -1, 1), rounded (0 when Σ|β̂_k| = 0)."
    )
    confidence_f: float = Field(description="Confidence shrinkage factor applied.")


def compute_dce_utility(
    user: UserProfile,
    item: MatchItem,
    schema: AttributeSchema,
    *,
    logit_eps: float = 0.01,
    attr_scale: dict[str, float] | None = None,
    confidence: float = 1.0,
) -> tuple[float, float, DceUtility]:
    """DCE-attribute utility from per-user betas recovered from the [0,1] contract.

    For each schema attribute with a user value v_k and a job level:
        beta_hat = logit(clamp(v_k, eps, 1-eps)) * scale_k     (v=0.5 -> 0)
        v_tilde  = ladder_position(reference->target) in [0,1]  (reference job -> 0)
        contribution = beta_hat * v_tilde
    V_dce = sum(contribution);  D = sum(|beta_hat|).
    V_dce_hat = confidence * clamp(V_dce / D, -1, 1)   (0 when D == 0).

    Direction comes from the sign of beta_hat, so no gain/cost orientation is applied; the schema
    ordering fixes each attribute's reference (0) and target (1).

    :return: ``(V_dce, V_dce_hat, detail)``
    """
    scale = attr_scale or {}

    v_dce = 0.0
    denom = 0.0
    rows: list[AttributePreference] = []
    for spec in schema.attributes:
        attr = spec.name
        v = _pref_value(user, attr)
        if v is None:
            continue
        raw_level = _job_level(item, attr)
        resolved = resolve_schema_level_id(attr, raw_level, schema)
        if resolved is None:
            continue  # job doesn't describe this attribute

        beta_hat = math.log(
            _clamp(v, logit_eps, 1.0 - logit_eps)
            / (1.0 - _clamp(v, logit_eps, 1.0 - logit_eps))
        )
        beta_hat *= float(scale.get(attr, 1.0))
        v_tilde = ladder_position(attr, raw_level, schema)
        contribution = beta_hat * v_tilde

        v_dce += contribution
        denom += abs(beta_hat)

        levels = spec.level_ids
        rows.append(
            AttributePreference(
                attribute=attr,
                attr_label=attribute_label(attr, schema),
                job_value=resolved,
                job_value_label=level_label(attr, raw_level, schema),  # e.g. "~70k"
                user_weight=round(v, 4),  # per-user [0,1] preference value v_k
                beta=round(beta_hat, 4),  # recovered signed coefficient β̂_k
                encoded_value=round(v_tilde, 4),  # graded ladder position ṽ_k
                contribution=round(contribution, 4),
                matched=contribution != 0.0,
                reference_level=levels[0] if levels else None,
                target_level=levels[-1] if levels else None,
            )
        )

    v_dce_hat = (confidence * _clamp(v_dce / denom, -1.0, 1.0)) if denom > 0 else 0.0
    detail = DceUtility(
        rows=rows,
        V_dce=round(v_dce, 4),
        V_dce_hat=round(v_dce_hat, 4),
        confidence_f=round(float(confidence), 4),
    )
    return v_dce, v_dce_hat, detail


def compute_task_utility(
    user: UserProfile,
    item: MatchItem,
) -> tuple[float, WorkActivityUtility | None]:
    """Importance-weighted BWS task utility.

    ``V_task = Σ_c ŵ_c · β_c`` with ``ŵ_c = importance_c / Σ importance`` (sum to 1).
    Level is intentionally NOT used (it is a skill-demand, not task-preference, signal).

    :return: ``(V_task ∈ [-2, 2], detail)``; ``(0.0, None)`` when the BWS keys are not
        work-activity ids, the item has no activities, or total importance is zero
    """
    bws_scores = _bws_scores(user)
    if detect_bws_score_type(bws_scores) != "work_activity_id":
        return 0.0, None

    rated = [wa for wa in item.onet_work_activities if wa.code and wa.importance > 0]
    total_importance = sum(wa.importance for wa in rated)
    if total_importance <= 0 or not rated:
        return 0.0, None

    v_task = 0.0
    wa_details: list[WorkActivityContribution] = []
    for wa in rated:
        beta = float(bws_scores.get(wa.code, 0.0))
        weight = wa.importance / total_importance  # ŵ_c, Σ = 1
        contribution = weight * beta
        v_task += contribution
        wa_details.append(
            WorkActivityContribution(
                wa_code=wa.code,
                wa_label=wa.label or wa.code,
                user_bws=beta,
                wa_importance=wa.importance,
                wa_level=wa.level,
                norm_importance=round(wa.importance / 5.0, 4),  # display only
                norm_level=round(wa.level / 7.0, 4),  # display only (unused in score)
                weight=round(weight, 6),  # ŵ_c (drives V_task)
                beta=round(beta, 4),  # β_c
                wa_contribution=round(contribution, 6),  # ŵ_c · β_c
            )
        )

    detail = WorkActivityUtility(
        wa_details=wa_details,
        wa_score_sum=round(v_task, 4),  # == V_task
        wa_aggregation="importance_weighted",
        n_work_activities=len(wa_details),
        V_task=round(v_task, 4),
        V_task_hat=round(_clamp(v_task / 2.0, -1.0, 1.0), 4),
    )
    return v_task, detail


class CombinedUtility(DomainModel):
    """The additive-RUM combination of the task and attribute utilities (all rounded)."""

    v_task: float = Field(description="Raw task utility.")
    v_task_h: float = Field(description="Task utility harmonised to [-1, 1].")
    v_dce_h: float = Field(description="Attribute utility harmonised to [-1, 1].")
    alpha: float = Field(description="Task-utility weight.")
    gamma: float = Field(description="Logistic gain.")
    V: float = Field(description="γ·[α·Ṽ_task + (1-α)·Ṽ_dce].")
    u_hat: float = Field(description="logistic(V), in [0, 1].")


def combine_utilities(
    v_task: float,
    v_dce: float,
    dce_normalizer: float,
    *,
    alpha: float,
    gamma: float,
    v_dce_already_harmonized: bool = False,
) -> CombinedUtility:
    """Additive-RUM combination of the task and DCE-attribute utilities into ``u_hat``.

    ``Ṽ_task = clamp(v_task/2, -1, 1)`` (β∈[-2,2], Σŵ=1 ⇒ v_task∈[-2,2]).
    ``Ṽ_dce``  = ``v_dce`` (if already harmonised to [-1,1]) else ``clamp(v_dce/dce_normalizer, -1, 1)``.
    ``V = γ·[α·Ṽ_task + (1-α)·Ṽ_dce]``;  ``u_hat = logistic(V)``.
    """
    v_task_h = _clamp(v_task / 2.0, -1.0, 1.0)
    if v_dce_already_harmonized:
        v_dce_h = _clamp(v_dce, -1.0, 1.0)
    elif dce_normalizer and dce_normalizer > 0:
        v_dce_h = _clamp(v_dce / dce_normalizer, -1.0, 1.0)
    else:
        v_dce_h = 0.0

    v = gamma * (alpha * v_task_h + (1.0 - alpha) * v_dce_h)
    if v >= 500:
        u_hat = 1.0
    elif v <= -500:
        u_hat = 0.0
    else:
        u_hat = 1.0 / (1.0 + math.exp(-v))

    return CombinedUtility(
        v_task=round(v_task, 4),
        v_task_h=round(v_task_h, 4),
        v_dce_h=round(v_dce_h, 4),
        alpha=round(float(alpha), 4),
        gamma=round(float(gamma), 4),
        V=round(v, 4),
        u_hat=round(u_hat, 4),
    )


def work_activity_block(
    user: UserProfile,
    item: MatchItem,
) -> tuple[float, WorkActivityUtility | None]:
    """LEGACY (``BWS_INTEGRATION_MODE="legacy"``): S_wa = (1/N) Σ [ BWS(c) × (I_c/5) × (L_c/7) ].

    :return: ``(S_wa, detail)``; ``(0.0, None)`` when there is nothing to score
    """
    bws_scores = _bws_scores(user)
    if detect_bws_score_type(bws_scores) != "work_activity_id":
        return 0.0, None
    if not item.onet_work_activities:
        return 0.0, None

    wa_details: list[WorkActivityContribution] = []
    contributions: list[float] = []
    for wa in item.onet_work_activities:
        if not wa.code:
            continue
        user_bws = float(bws_scores.get(wa.code, 0.0))
        norm_importance = wa.importance / 5.0 if wa.importance else 0.0
        norm_level = wa.level / 7.0 if wa.level else 0.0
        wa_contribution = user_bws * norm_importance * norm_level
        contributions.append(wa_contribution)
        wa_details.append(
            WorkActivityContribution(
                wa_code=wa.code,
                wa_label=wa.label or wa.code,
                user_bws=user_bws,
                wa_importance=wa.importance,
                wa_level=wa.level,
                norm_importance=round(norm_importance, 4),
                norm_level=round(norm_level, 4),
                wa_contribution=round(wa_contribution, 4),
            )
        )

    n = len(contributions)
    wa_score_sum = sum(contributions) / n if n else 0.0
    return wa_score_sum, WorkActivityUtility(
        wa_details=wa_details,
        wa_score_sum=round(wa_score_sum, 4),
        wa_aggregation="mean",
        n_work_activities=n,
    )


# --------------------------------------------------------------------------------------------------
# scorer
#
# Unified preference scorer: DCE attribute utility + BWS task utility → additive-RUM u_hat.
#
# Part A (DCE attributes): V_dce = Σ_k β̂_k · ṽ_k, with β̂_k recovered from the per-user
# [0,1] contract via logit, and ṽ_k the graded ladder position (reference→target) from the
# committed schema. See compute_dce_utility.
#
# Part B (BWS work activities): importance-weighted V_task (compute_task_utility).
#
# Combination: u_hat = logistic(γ·[α·Ṽ_task + (1-α)·Ṽ_dce]) (combine_utilities).
# --------------------------------------------------------------------------------------------------

# The request carries no preference-confidence signal (vignette count / explicit confidence),
# so the DCE term is never shrunk.
_PREFERENCE_CONFIDENCE = 1.0


class UnifiedPreferenceScorer:
    """DCE-attribute utility + BWS task utility, combined via additive-RUM into u_hat."""

    def __init__(self, schema: AttributeSchema):
        self._schema = schema

    def calculate_score(
        self,
        user: UserProfile,
        item: MatchItem,
        *,
        include_work_activities: bool = True,
    ) -> PreferenceScore:
        """Preference utility of the item for the user.

        :param user: The user
        :param item: The job or occupation
        :param include_work_activities: Score Part B (BWS work activities)
        :return: u_hat and its per-attribute / per-activity contributions
        """
        # Part A — DCE attributes (signed, from logit-recovered per-user betas).
        f = _PREFERENCE_CONFIDENCE
        _, v_dce_hat, dce = compute_dce_utility(
            user,
            item,
            self._schema,
            logit_eps=DCE_LOGIT_EPS,
            attr_scale=DCE_ATTR_SCALE,
            confidence=f,
        )

        # Part B — BWS work activities (importance-weighted task utility).
        work_activities: WorkActivityUtility | None = None
        v_task = 0.0
        if include_work_activities:
            v_task, work_activities = compute_task_utility(user, item)

        # Combine (additive-RUM). v_dce_hat is already harmonised to [-1,1].
        comb = combine_utilities(
            v_task,
            v_dce_hat,
            1.0,
            alpha=BWS_ALPHA,
            gamma=BWS_GAIN_GAMMA,
            v_dce_already_harmonized=True,
        )

        return PreferenceScore(
            u_hat=comb.u_hat,
            score=comb.u_hat,
            attributes=dce.rows,
            work_activities=work_activities,
            S_attrs=round(v_dce_hat, 4),  # harmonised DCE utility ∈ [-1,1]
            S_wa=comb.v_task_h,  # harmonised task utility ∈ [-1,1]
            V=comb.V,
            V_dce=dce.V_dce,
            V_dce_hat=comb.v_dce_h,
            confidence_f=round(f, 4),
            alpha=comb.alpha,
            gamma=comb.gamma,
            bws_score_type=detect_bws_score_type(_bws_scores(user)),
            scoring_model="unified_dce_bws_v1",
        )
