"""Location eligibility for a /match request: the users' eligible location keys and the Mongo filter.

Pure (no I/O beyond the cached hub-chain artifact): ``resolve_location_keys`` turns the users into
their eligible keys once per request, ``build_location_filter`` turns those into the job query handed
to ``IJobsRepository.find_jobs_with_timing``. The filter is the whole location decision for the job
pool — an exact ``$in`` on the indexed ``location_keys`` array (see ``app.location.keys``).
"""

from collections.abc import Sequence
from typing import Any

from app.artifacts.repository import IArtifactsRepository
from app.config import (
    LOCATION_HUB_CHAINS_PATH,
    LOCATION_TIER_ENABLED,
    LOCATION_TIER_W_NATIONAL,
    LOCATION_TIER_W_REGIONAL,
)
from app.domain.user import UserProfile
from app.jobs.job_document import LOCATION_KEYS_FIELD, RANKED_JOBS_ACTIVE_FILTER
from app.location.keys import HubChains, UserLocationKeys, user_location_keys


def location_hub_chains(artifacts_repository: IArtifactsRepository) -> HubChains | None:
    """The deployment's hub chains when location tiers are on (``LOCATION_TIER_ENABLED``), else None.

    None also when the hub-chain file is missing or malformed: users are then matched to their own
    city and province only.
    """
    if not LOCATION_TIER_ENABLED:
        return None
    return artifacts_repository.load_hub_chains(LOCATION_HUB_CHAINS_PATH)


def resolve_location_keys(
    users: Sequence[UserProfile], hub_chains: HubChains | None
) -> list[UserLocationKeys]:
    """Each user's eligible location keys and tiers, aligned with ``users``."""
    return [
        user_location_keys(
            u,
            hub_chains,
            w_regional=LOCATION_TIER_W_REGIONAL,
            w_national=LOCATION_TIER_W_NATIONAL,
        )
        for u in users
    ]


def build_location_filter(location_keys: Sequence[UserLocationKeys]) -> dict[str, Any]:
    """Active jobs having at least one location key any of the users is eligible for.

    Served by the ``is_active_location_keys_-_id`` index (``JOBS_INDEX_MODELS``).
    """
    keys = sorted(set().union(*(k.keys for k in location_keys)))
    return {**RANKED_JOBS_ACTIVE_FILTER, LOCATION_KEYS_FIELD: {"$in": keys}}
