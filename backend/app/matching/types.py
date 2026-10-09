"""Types for the matching feature."""

from pydantic import BaseModel


class MatchOptions(BaseModel):
    """Per-request tuning for ``POST /match``, as received (None = server default).

    Defaults and validation are applied by ``MatchingService.match``.
    """

    retrieve_top_k: int | None = None
    """Stage-1 shortlist size per user (default ``MATCH_V4_RETRIEVE_TOP_K``)."""

    final_top_k: int | None = None
    """Cross-encoder pool size and max opportunities per user (default ``MATCH_V4_FINAL_TOP_K``)."""

    final_score_combiner: str | None = None
    """``product`` or ``geometric_mean`` (default ``FINAL_SCORE_COMBINER``)."""

    skill_gap_top_k: int | None = None
    """Skill-gap recommendations per user (default ``MATCH_TOP_K_SKILL_GAPS``)."""
