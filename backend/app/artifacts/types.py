"""Typed contents of the resource files the matching engine loads (see ``ArtifactsRepository``)."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.domain.base import SourceRecord


class AttributeLevel(SourceRecord):
    """One level of a job attribute in the attribute schema."""

    id: str = Field(description="Level id, e.g. ``earn_50_80k``.")
    label: str | None = Field(default=None, description="Display label, e.g. ``~70k``.")


class AttributeSpec(SourceRecord):
    """One job attribute of the attribute schema: its levels, ordered reference (0) to target (1)."""

    name: str = Field(description="Attribute name (a ``PreferenceVector`` field).")
    label: str | None = Field(
        default=None, description="Display label of the attribute."
    )
    levels: list[AttributeLevel] = Field(
        default_factory=list, description="Levels in ladder order."
    )

    @property
    def level_ids(self) -> list[str]:
        """The level ids in ladder order."""
        return [lv.id for lv in self.levels]


class AttributeSchema(SourceRecord):
    """The job-attribute schema (``resources/preference/job_attributes_schema.json``)."""

    attributes: list[AttributeSpec] = Field(
        default_factory=list, description="The attributes, in scoring order."
    )

    def spec(self, name: str) -> AttributeSpec | None:
        """The attribute named ``name``, or None."""
        for a in self.attributes:
            if a.name == name:
                return a
        return None


class HubChainsFile(SourceRecord):
    """The location hub-chain exceptions file (``resources/location/location_hub_chains.json``)."""

    national_hub: str = Field(
        default="", description="The national fallback hub county."
    )
    regional_hubs: dict[str, list[str]] | None = Field(
        default=None, description="Regional hub -> the counties that fall back to it."
    )
    hub_self_only: list[str] | None = Field(
        default=None, description="Hub counties that only surface their own jobs."
    )


class ConcatWhitening(BaseModel):
    """The concat-embedding whitening artifact (``concat_whitening_gemini.npz``)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    mu: np.ndarray = Field(
        description="Mean of the fitted corpus (float64, ``[dim]``)."
    )
    W: np.ndarray = Field(
        description="Whitening matrix Σ^-1/2 (float64, ``[dim, dim]``)."
    )
    target: float = Field(
        description="p99 of whitened cosines; rescales whitened cosines toward [0, 1]."
    )
