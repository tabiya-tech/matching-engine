"""Hybrid preference scoring v1 (see README.md and PDF spec in this folder)."""

from app.config import PREFERENCE_SCORER_MODE

from .scorer import HybridPreferenceScorer


def get_preference_scorer():
    if PREFERENCE_SCORER_MODE == "hybrid_v1":
        return HybridPreferenceScorer()
    from app.services.preference_score import PreferenceScorer

    return PreferenceScorer()


__all__ = ["HybridPreferenceScorer", "get_preference_scorer"]
