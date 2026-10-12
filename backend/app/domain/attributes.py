"""Job attributes: the levels a job or occupation has on each preference dimension."""

from collections.abc import Mapping

from pydantic import Field

from app.domain.base import DomainModel
from app.domain.coercion import to_optional_text


class JobAttributes(DomainModel):
    """Attribute levels of a job or occupation, compared against the user's ``PreferenceVector``.

    Values are level ids from the job-attributes schema (e.g. ``earn_70k``, ``phys_light``). They
    are kept as the producer wrote them: the preference scorer maps legacy ids onto the current
    schema (``app.ranking.preference.resolve_schema_level_id``). None means the source did
    not describe the attribute, which the scorer treats as "no signal".
    """

    earnings_per_month: str | None = Field(
        default=None,
        description="Monthly earnings level id, e.g. ``earn_50_80k`` (legacy: ``earn_70k``).",
    )
    task_content: str | None = Field(
        default=None,
        description="Task content level id: ``task_routine``, ``task_mix`` or ``task_creative``.",
    )
    physical_demand: str | None = Field(
        default=None,
        description="Physical demand level id: ``phys_light`` or ``phys_heavy``.",
    )
    work_flexibility: str | None = Field(
        default=None,
        description="Work flexibility level id: ``flex_fixed``, ``flex_some`` or ``flex_high``.",
    )
    social_interaction: str | None = Field(
        default=None,
        description="Social interaction level id, e.g. ``soc_alone`` or ``soc_people`` (legacy: ``soc_peers``, ``soc_customers``).",
    )
    career_growth: str | None = Field(
        default=None,
        description="Career growth level id: ``growth_low``, ``growth_med`` or ``growth_high``.",
    )
    social_meaning: str | None = Field(
        default=None,
        description="Social meaning level id: ``mean_low`` or ``mean_high``.",
    )
    expected_demand: str | None = Field(
        default=None,
        description=(
            "Expected labour-market demand label, e.g. ``High Expected Demand``. Set on occupations "
            "(from the occupation file's ``labor_demand``); absent on jobs."
        ),
    )

    @classmethod
    def from_levels(cls, levels: Mapping[str, object]) -> "JobAttributes":
        """Build from an ``{attribute_name: level_id}`` mapping, keeping only known attributes.

        Unknown attribute names (and flags such as ``requires_post_secondary`` that the
        llm-reranker stores alongside the levels) are ignored; blank and non-scalar values are
        treated as absent.

        :param levels: Attribute name to level id, as stored upstream
        :return: The attributes
        """
        values: dict[str, str] = {}
        for name in cls.model_fields:
            raw = levels.get(name)
            if isinstance(raw, bool) or isinstance(raw, (dict, list)):
                continue
            text = to_optional_text(raw)
            if text is not None and text.strip():
                values[name] = text
        return cls(**values)
