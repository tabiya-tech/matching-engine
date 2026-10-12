"""User profile: the person being matched, built from one ``POST /match`` request item."""

from pydantic import Field

from app.domain.base import DomainModel
from app.schemas import MatchRequest, PreferenceVector, Skill


class UserSkill(DomainModel):
    """One skill the user has, identified by its taxonomy origin UUID."""

    originUUID: str = Field(description="Taxonomy skill origin UUID.")
    preferredLabel: str | None = Field(
        default=None,
        description="Skill label; the engine embeds and resolves skills by label, falling back to the UUID.",
    )
    proficiency: float | None = Field(
        default=None,
        description="Self-reported or inferred proficiency in [0, 1]; not used for ranking.",
    )

    @classmethod
    def from_request(cls, skill: Skill) -> "UserSkill":
        """Build from a request ``Skill``.

        :param skill: The validated request skill
        :return: The user skill
        """
        return cls(
            originUUID=skill.originUUID,
            preferredLabel=skill.preferredLabel,
            proficiency=skill.proficiency,
        )


class UserSkills(DomainModel):
    """The user's skills (``skills_vector`` in the request)."""

    top_skills: list[UserSkill] = Field(description="The user's skills; may be empty.")


class UserPreferences(DomainModel):
    """The user's job preferences (``preference_vector`` in the request).

    Each attribute value is in [0, 1] with 0.5 neutral: the DCE scorer recovers a signed
    coefficient as ``logit(value)``, so 0.5 contributes nothing.
    """

    earnings_per_month: float = Field(
        description="Preference for higher earnings, in [0, 1]."
    )
    task_content: float | None = Field(
        description="Preference for the job's task content, in [0, 1]."
    )
    physical_demand: float = Field(
        description="Preference for light (low physical demand) work, in [0, 1]."
    )
    work_flexibility: float | None = Field(
        description="Preference for flexible working arrangements, in [0, 1]."
    )
    social_interaction: float = Field(
        description="Preference for social / people-facing work, in [0, 1]."
    )
    career_growth: float = Field(
        description="Preference for career-growth potential, in [0, 1]."
    )
    social_meaning: float | None = Field(
        description="Preference for socially meaningful work, in [0, 1]."
    )
    bws_scores: dict[str, float] | None = Field(
        default=None,
        description="Best-Worst Scaling part-worths keyed by O*NET work-activity code; None when not collected.",
    )
    top_10_bws: list[str] | None = Field(
        default=None, description="The user's ten highest-ranked work-activity codes."
    )

    @classmethod
    def from_request(cls, preferences: PreferenceVector) -> "UserPreferences":
        """Build from a request ``PreferenceVector`` (defaults already applied).

        :param preferences: The validated request preferences
        :return: The user preferences
        """
        return cls(
            earnings_per_month=preferences.earnings_per_month,
            task_content=preferences.task_content,
            physical_demand=preferences.physical_demand,
            work_flexibility=preferences.work_flexibility,
            social_interaction=preferences.social_interaction,
            career_growth=preferences.career_growth,
            social_meaning=preferences.social_meaning,
            bws_scores=dict(preferences.bws_scores)
            if preferences.bws_scores is not None
            else None,
            top_10_bws=list(preferences.top_10_bws)
            if preferences.top_10_bws is not None
            else None,
        )


class UserProfile(DomainModel):
    """The person being matched: location, skills, preferences and education.

    Field names mirror ``MatchRequest``.
    """

    user_id: str | None = Field(
        default=None, description="Caller-supplied id, echoed in the response."
    )
    city: str = Field(
        description="City/town, county suffix stripped; empty = no location preference."
    )
    province: str = Field(
        description="Province/county, county suffix stripped; empty = fallback region."
    )
    skills_vector: UserSkills = Field(description="The user's skills.")
    skill_groups_origin_uuids: list[str] = Field(
        description="Skill-group origin UUIDs of the user."
    )
    preference_vector: UserPreferences = Field(
        description="Attribute and work-activity preferences."
    )
    any_post_secondary_educ: int | None = Field(
        default=None,
        description="``0`` = no post-secondary education (education gate applies), ``1`` = has some, None = unknown.",
    )
    number_post_secondary_educ: int | None = Field(
        default=None,
        description="Number of post-secondary qualifications; not used for ranking.",
    )
    total_duration_postsec: float | None = Field(
        default=None,
        description="Years of post-secondary education; not used for ranking.",
    )
    zqf_level: int | None = Field(
        default=None,
        description="Zambia Qualifications Framework level (``MatchRequestV5`` only); None elsewhere.",
    )

    @classmethod
    def from_request(cls, request: MatchRequest) -> "UserProfile":
        """Build from one validated ``MatchRequest`` (or ``MatchRequestV5``) item.

        :param request: The validated request item
        :return: The user profile
        """
        return cls(
            user_id=request.user_id,
            city=request.city,
            province=request.province,
            skills_vector=UserSkills(
                top_skills=[
                    UserSkill.from_request(s) for s in request.skills_vector.top_skills
                ]
            ),
            skill_groups_origin_uuids=list(request.skill_groups_origin_uuids),
            preference_vector=UserPreferences.from_request(request.preference_vector),
            any_post_secondary_educ=request.any_post_secondary_educ,
            number_post_secondary_educ=request.number_post_secondary_educ,
            total_duration_postsec=request.total_duration_postsec,
            zqf_level=getattr(request, "zqf_level", None),
        )
