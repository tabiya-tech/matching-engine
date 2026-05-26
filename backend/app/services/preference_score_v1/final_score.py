"""Combine u_hat and p_hat into a final ranking score (configurable).

This lives under preference_score_v1 because it is part of the hybrid v1
preference+skill integration experiments (dashboards + batch runner).
"""

from __future__ import annotations

import math
from typing import Literal

FinalScoreCombiner = Literal["product", "geometric_mean"]


def combine_final_score(u_hat: float, p_hat: float, *, combiner: FinalScoreCombiner) -> float:
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

