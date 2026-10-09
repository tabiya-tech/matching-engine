"""Legacy **embedding** skill matcher (:class:`CosineSkillMatcher`) — production cosine path.

This is separate from ``app.ranking.skills.SkillScorer`` / ``compute_U_complete``.
Use :mod:`run_cosine_matching` for cosine-only batches, or ``app.services.hybrid_scoring`` for BM25 + cosine + hybrid fusion.
"""

from app.ranking.skills import CosineSkillMatcher

__all__ = ["CosineSkillMatcher"]
