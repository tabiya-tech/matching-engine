"""Types for the matching feature."""

from pydantic import BaseModel, Field


class MatchOptions(BaseModel):
    """Per-request tuning for ``POST /match``, as received (None = server default).

    Defaults and validation are applied by ``MatchingService.match``.
    """

    retrieve_top_k: int | None = Field(
        default=None,
        description="Stage-1 shortlist size per user (default ``MATCH_V4_RETRIEVE_TOP_K``).",
    )
    final_top_k: int | None = Field(
        default=None,
        description=(
            "Cross-encoder pool size and max opportunities per user "
            "(default ``MATCH_V4_FINAL_TOP_K``)."
        ),
    )
    final_score_combiner: str | None = Field(
        default=None,
        description="``product`` or ``geometric_mean`` (default ``FINAL_SCORE_COMBINER``).",
    )
    skill_gap_top_k: int | None = Field(
        default=None,
        description="Skill-gap recommendations per user (default ``MATCH_TOP_K_SKILL_GAPS``).",
    )
