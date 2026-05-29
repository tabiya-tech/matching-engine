"""Post-secondary education eligibility gate (shared by all matching endpoints).

A job may declare ``requires_post_secondary`` (boolean) under
``llm_job_attributes.attributes`` in Mongo; it is surfaced as a top-level
``requires_post_secondary`` key on the flat job dict by
:func:`app.database.build_job_dict_from_ranked`.

The user side comes from ``MatchRequest.any_post_secondary_educ`` (0/1, optional).

Gate (applied uniformly across /match, /match_v2, /match_v3, /match_v4):
a user is **ineligible** for a job only when the job requires post-secondary
education *and* the user explicitly reported having none.

Fail-open by design — only positive evidence excludes:
* Job has no / false ``requires_post_secondary``  -> eligible (never hide a job
  that simply lacks the field).
* User did not supply ``any_post_secondary_educ`` (None) -> eligible; preserves
  behaviour for clients that omit the optional field. Only an explicit ``0`` excludes.
"""

from __future__ import annotations

from typing import Any, Dict, List


def job_requires_post_secondary(job: Dict[str, Any]) -> bool:
    """True only when the job explicitly requires post-secondary education."""
    val = job.get("requires_post_secondary")
    if val is None:
        attrs = job.get("attributes")
        if isinstance(attrs, dict):
            val = attrs.get("requires_post_secondary")
    return val is True or val == 1


def user_lacks_post_secondary(user: Dict[str, Any]) -> bool:
    """True only when the user explicitly reported no post-secondary education (``== 0``)."""
    return user.get("any_post_secondary_educ") == 0


def is_education_eligible(user: Dict[str, Any], job: Dict[str, Any]) -> bool:
    """Education gate: ineligible iff the job requires post-secondary and the user has none."""
    return not (job_requires_post_secondary(job) and user_lacks_post_secondary(user))


def filter_jobs_by_education(
    user: Dict[str, Any], jobs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Drop jobs the user is education-ineligible for.

    No-op (returns a shallow copy) when the user has — or did not report lacking —
    post-secondary education, so the common path stays cheap.
    """
    if not user_lacks_post_secondary(user):
        return list(jobs)
    return [j for j in jobs if not job_requires_post_secondary(j)]
