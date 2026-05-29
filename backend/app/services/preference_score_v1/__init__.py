"""Unified DCE + BWS preference scoring (additive-RUM). See README.md."""

from app.config import PREFERENCE_SCORER_MODE

from .scorer import UnifiedPreferenceScorer


def get_preference_scorer():
    """Return the configured preference scorer.

    Default ``unified`` → ``UnifiedPreferenceScorer`` (DCE attributes + BWS, additive-RUM).
    ``legacy`` → the old hardcoded-beta ``PreferenceScorer`` (A/B escape hatch only).
    """
    if PREFERENCE_SCORER_MODE == "legacy":
        from app.services.preference_score import PreferenceScorer

        return PreferenceScorer()
    return UnifiedPreferenceScorer()


__all__ = ["UnifiedPreferenceScorer", "get_preference_scorer"]
