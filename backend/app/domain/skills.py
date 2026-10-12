"""Skill references shared by jobs and occupations."""

from pydantic import Field

from app.domain.base import DomainModel


class SkillReference(DomainModel):
    """One taxonomy skill required by a job or an occupation.

    The ranking engine resolves skills by ``label`` first (label-primary resolver) and falls
    back to ``id``, so an empty label weakens matching for that skill.
    """

    id: str = Field(
        description=(
            "Taxonomy skill id: ``tabiya_skill_id`` for jobs, the skill UUID listed in the "
            "occupation file for occupations."
        ),
    )
    label: str = Field(
        description="Preferred label of the skill in the deployment's language; empty when the source has none.",
    )
