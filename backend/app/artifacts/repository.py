"""Repository for the model, taxonomy and resource files the matching engine loads from disk."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from app.ranking.retrieval import HubChains

logger = logging.getLogger(__name__)

# Skill DESCRIPTION cells run past the default field size limit on some exports.
csv.field_size_limit(10_000_000)

# backend/app/artifacts/repository.py -> parents[2] == backend/
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
ATTRIBUTE_SCHEMA_PATH = (
    _BACKEND_ROOT / "resources" / "preference" / "job_attributes_schema.json"
)


class SkillEmbeddingArtifact(BaseModel):
    """A skill-embedding checkpoint as stored (``torch.save`` of ``state_dict`` + metadata)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    weights: np.ndarray
    """``embedding.weight`` in its stored dtype (fp16 or fp32), one row per skill."""

    whitening_target: float | None
    """``whitening.target_max_p999`` for whitened artifacts, else None."""

    model_name: str | None
    """``model_name`` recorded in the checkpoint, if any."""


class IArtifactsRepository(ABC):
    """Interface for the files the matching engine loads from local disk."""

    @abstractmethod
    def load_skill_embedding(self, path: str | Path) -> SkillEmbeddingArtifact:
        """
        Loads a skill-embedding checkpoint.

        :param path: The ``.pt`` checkpoint
        :return: The stored weights and metadata
        :raises Exception: If the checkpoint cannot be read
        """
        raise NotImplementedError()

    @abstractmethod
    def load_skill_to_row(
        self, path: str | Path, *, encoding: str | None = "utf-8"
    ) -> dict[str, int]:
        """
        Loads the skill-id → embedding-row map.

        :raises Exception: If the file cannot be read
        """
        raise NotImplementedError()

    @abstractmethod
    def read_csv_rows(
        self, path: str | Path, *, newline: str | None = None
    ) -> list[dict]:
        """
        Reads a taxonomy CSV into dict rows.

        :raises FileNotFoundError: If the file does not exist
        """
        raise NotImplementedError()

    @abstractmethod
    def load_concat_whitening(self, path: str, *, expected_dim: int) -> dict[str, Any]:
        """
        Loads the concat-whitening artifact.

        :param path: The ``.npz`` artifact
        :param expected_dim: The user/job concat embedding dimension it must match
        :return: ``{mu, W, target}``, or ``{}`` if the file is absent or incompatible
        """
        raise NotImplementedError()

    @abstractmethod
    def file_sha256(self, path: str) -> str:
        """
        SHA-256 of a file's bytes.

        :raises OSError: If the file cannot be read
        """
        raise NotImplementedError()

    @abstractmethod
    def load_hub_chains(self, path: str) -> HubChains | None:
        """
        Loads (and caches) the location hub-chain map.

        :return: The parsed map, or None if the file is missing or malformed
        """
        raise NotImplementedError()

    @abstractmethod
    def load_attribute_schema(self, path: str | None = None) -> dict:
        """
        Loads (and caches) the job-attribute schema used by preference scoring.

        :param path: Schema file, or None for the bundled one
        :raises OSError: If the file cannot be read
        :raises ValueError: If the file is not valid JSON
        """
        raise NotImplementedError()


class ArtifactsRepository(IArtifactsRepository):
    def __init__(self):
        self._hub_chains_cache: dict[str, HubChains | None] = {}
        # (path, schema) of the last schema loaded: a one-entry cache keyed by the path argument.
        self._attribute_schema_cache: tuple[str | None, dict] | None = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def load_skill_embedding(self, path: str | Path) -> SkillEmbeddingArtifact:
        state = torch.load(Path(path), map_location="cpu")
        weights = state["state_dict"]["embedding.weight"].numpy()
        whitening_target = (
            (state.get("whitening") or {}).get("target_max_p999")
            if isinstance(state, dict)
            else None
        )
        return SkillEmbeddingArtifact(
            weights=weights,
            whitening_target=whitening_target,
            model_name=state.get("model_name"),
        )

    def load_skill_to_row(
        self, path: str | Path, *, encoding: str | None = "utf-8"
    ) -> dict[str, int]:
        with open(path, "r", encoding=encoding) as f:
            return json.load(f)

    def read_csv_rows(
        self, path: str | Path, *, newline: str | None = None
    ) -> list[dict]:
        with open(path, "r", encoding="utf-8", newline=newline) as f:
            return list(csv.DictReader(f))

    def load_concat_whitening(self, path: str, *, expected_dim: int) -> dict[str, Any]:
        if path and os.path.exists(path):
            z = np.load(path)
            mu = z["mu"].astype(np.float64)
            W = z["W"].astype(np.float64)
            target = float(z["target"])
            if mu.shape[0] != expected_dim or W.shape[0] != expected_dim or target <= 0:
                # Dim/target mismatch (e.g. embedding model changed without rebuilding the
                # artifact). Disable rather than risk a mid-request matmul error / bad rescale.
                logger.error(
                    "concat whitening artifact %s incompatible (mu_dim=%d W_dim=%d target=%.4f, "
                    "expected dim=%d, target>0); whitened p_hat disabled",
                    path,
                    mu.shape[0],
                    W.shape[0],
                    target,
                    expected_dim,
                )
                return {}
            logger.info(
                "loaded concat whitening artifact %s (target=%.4f)",
                path,
                target,
            )
            return {"mu": mu, "W": W, "target": target}
        logger.warning(
            "concat whitening artifact not found at %s; whitened p_hat disabled",
            path,
        )
        return {}

    def file_sha256(self, path: str) -> str:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()

    def load_hub_chains(self, path: str) -> HubChains | None:
        if path in self._hub_chains_cache:
            return self._hub_chains_cache[path]
        hc: HubChains | None = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("hub-chains JSON must be an object")
            hc = HubChains(
                raw.get("national_hub", ""),
                raw.get("regional_hubs"),
                raw.get("hub_self_only"),
            )
            if not hc.national:
                raise ValueError("hub-chains JSON missing 'national_hub'")
        except (OSError, ValueError, TypeError, AttributeError) as e:
            logger.error(
                "location_tiers: could not load hub chains from %s (%s); urban-pull tiering disabled.",
                path,
                e,
            )
            hc = None
        self._hub_chains_cache[path] = hc
        return hc

    def load_attribute_schema(self, path: str | None = None) -> dict:
        if (
            self._attribute_schema_cache is not None
            and self._attribute_schema_cache[0] == path
        ):
            return self._attribute_schema_cache[1]
        p = Path(path) if path else ATTRIBUTE_SCHEMA_PATH
        with open(p, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self._attribute_schema_cache = (path, schema)
        return schema


_artifacts_repository_singleton: IArtifactsRepository = ArtifactsRepository()


def get_artifacts_repository() -> IArtifactsRepository:
    """Return the process-wide ArtifactsRepository."""
    return _artifacts_repository_singleton
