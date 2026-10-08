"""Concat-embedding whitening ((x - mu) @ W, re-normalised) and its p99 rescale target."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from app.ranking.vectors import l2_normalize_rows


class ConcatWhitener:
    """Whitening for the user/job concat embedding space.

    ``artifact`` is ``{mu, W, target}`` (see ``app.artifacts.repository.load_concat_whitening``), or
    empty/None when the artifact is unavailable — then rows are only L2-normalised and the rescale
    target is 0.0.
    """

    def __init__(self, artifact: Optional[Dict[str, Any]]):
        self._artifact = artifact or None

    def whiten_rows(self, vecs: np.ndarray) -> np.ndarray:
        """L2-normalise rows, apply the concat whitening ((.-mu)@W), re-normalise -> unit whitened rows.
        If the artifact is unavailable, returns the L2-normalised rows unchanged."""
        v = l2_normalize_rows(np.asarray(vecs, dtype=np.float64))
        cw = self._artifact
        if cw is None:
            return v
        return l2_normalize_rows((v - cw["mu"]) @ cw["W"])

    def rescale_target(self) -> float:
        """p99 rescale target for the whitened concat cosine (0.0 if the artifact is unavailable)."""
        cw = self._artifact
        return cw["target"] if cw else 0.0
