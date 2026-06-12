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

from typing import Any, Dict, List, Optional


def _coerce_flag(val: Any) -> Optional[bool]:
    """Normalise a binary flag that may arrive as bool, int (0/1), float, or string.

    Both the job side (``requires_post_secondary``) and the user side
    (``any_post_secondary_educ``) may be stored as a boolean (``true``/``false``) OR as an
    integer (``1``/``0``), depending on the producer. This collapses both representations:

    * Returns ``True``  for ``True`` / ``1`` / ``"1"`` / ``"true"`` / ``"yes"``.
    * Returns ``False`` for ``False`` / ``0`` / ``"0"`` / ``"false"`` / ``"no"`` / ``""``.
    * Returns ``None`` for missing/unrecognised values (e.g. ``None`` or a stray ``2``),
      so callers can apply their own fail-open default rather than guessing.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if val == 1:
            return True
        if val == 0:
            return False
        return None
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("1", "true", "yes", "y", "t"):
            return True
        if s in ("0", "false", "no", "n", "f", ""):
            return False
    return None


def job_requires_post_secondary(job: Dict[str, Any]) -> bool:
    """True only when the job explicitly requires post-secondary education.

    Accepts the flag as bool or int (``true``/``1`` => requires), at the top level or nested
    under ``attributes``. Anything else (absent / unrecognised) is treated as "not required"
    so a job is never hidden for lacking the field.
    """
    val = job.get("requires_post_secondary")
    if val is None:
        attrs = job.get("attributes")
        if isinstance(attrs, dict):
            val = attrs.get("requires_post_secondary")
    return _coerce_flag(val) is True


def user_lacks_post_secondary(user: Dict[str, Any]) -> bool:
    """True only when the user explicitly reported no post-secondary education.

    Accepts the flag as bool or int (``false``/``0`` => lacks). A stray/unknown value or an
    absent field is NOT treated as lacking here — but note ``MatchRequest`` defaults
    ``any_post_secondary_educ`` to ``0``, so an omitted field becomes "lacks" after validation.
    """
    return _coerce_flag(user.get("any_post_secondary_educ")) is False


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
