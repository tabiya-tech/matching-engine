"""Swagger request-body examples and descriptions for the matching endpoints."""

from typing import Any

# Swagger default request body for POST /match (Kenya, post-secondary user).
MATCH_BODY_EXAMPLE: list[dict[str, Any]] = [
    {
        "user_id": "u1",
        "city": "Nairobi",
        "province": "Nairobi",
        "any_post_secondary_educ": 1,
        "skills_vector": {
            "top_skills": [
                {
                    "originUUID": "00000000-0000-4000-8000-000000000001",
                    "preferredLabel": "customer service",
                    "proficiency": 0.8,
                }
            ]
        },
        "skill_groups_origin_uuids": [],
        "preference_vector": {
            "earnings_per_month": 0.7,
            "physical_demand": 0.4,
            "social_interaction": 0.6,
            "career_growth": 0.8,
        },
    }
]

MATCH_BODY_DESCRIPTION = (
    "JSON **array** of MatchRequest (one object per user). "
    "``any_post_secondary_educ``: ``0`` = no post-secondary (jobs with "
    "``requires_post_secondary`` are filtered out), ``1`` = has post-secondary, "
    "omit to disable the education gate."
)

# Swagger default for /experiments/v5/match (Zambia: ZQF annotation on opportunities).
MATCH_V5_BODY_EXAMPLE: list[dict[str, Any]] = [
    {
        "user_id": "u1",
        "city": "Lusaka",
        "province": "Lusaka",
        "zqf_level": 4,
        "skills_vector": {
            "top_skills": [
                {
                    "originUUID": "00000000-0000-4000-8000-000000000001",
                    "preferredLabel": "prepare bakery products",
                    "proficiency": 0.85,
                },
                {
                    "originUUID": "00000000-0000-4000-8000-000000000002",
                    "preferredLabel": "bake goods",
                    "proficiency": 0.78,
                },
            ]
        },
        "skill_groups_origin_uuids": [],
        "preference_vector": {
            "earnings_per_month": 0.6,
            "physical_demand": 0.5,
            "social_interaction": 0.5,
            "career_growth": 0.6,
        },
    }
]
