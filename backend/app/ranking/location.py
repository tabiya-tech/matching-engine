"""Tiered urban-pull location matching.

Maps a user's county to an ordered fallback **hub chain** (local -> regional hub -> national hub) and
scores each job by which tier it falls in, so non-hub users (whose own county has few jobs) still get a
full list while local jobs stay preferred. Hub counties do not pull outward.

Two consumers:
  * the Mongo prefilter (app.jobs.documents._location_or_clauses_for_one_user) uses ``chain_for`` to widen a
    user's candidate pool to include their hub regions;
  * the v4 opportunity ranker uses ``tier_factor_for_job`` as a per-uuid [0,1] multiplier on final_score
    (local=1.0, regional=W_REGIONAL, national=W_NATIONAL, off-chain=0.0).

Region matching uses ``job_matches_user_location`` — the same lenient casefold-substring + always-remote
rule every matching path uses.

The hub data is a small JSON of exceptions (see resources/location/location_hub_chains.json, loaded by
``app.artifacts.repository.load_hub_chains``); every county not named there defaults to
``[self, national_hub]``. If the file is missing/malformed the loader returns None and callers fall back to
today's strict behaviour (the feature becomes a no-op).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _cf(s: Any) -> str:
    """Casefold + strip, aligned with _norm / app.jobs.documents._norm_loc_value."""
    if s is None:
        return ""
    t = str(s).strip()
    return t.casefold() if t else ""


def _norm(v: Optional[str]) -> str:
    return str(v).strip().casefold() if v is not None else ""


def job_matches_user_location(job: Dict[str, Any], user: Dict[str, Any]) -> bool:
    """Lenient location match.
    - Always matches 'Remote' jobs
    - Matches if city or province match (case-insensitive, substring)
    """
    user_city = _norm(user.get("city"))
    user_province = _norm(user.get("province"))

    job_city = _norm(job.get("city"))
    job_province = _norm(job.get("province"))
    job_loc = _norm(job.get("location"))

    #  Always include Remote jobs
    if "remote" in job_city or "remote" in job_province or "remote" in job_loc:
        return True

    if not user_city or not user_province:
        return False

    # Check City Match (Lenient)
    if job_city and (user_city in job_city or job_city in user_city):
        return True

    # Check Province Match (Lenient)
    if job_province and (
        user_province in job_province or job_province in user_province
    ):
        return True

    # Fallback to location string match
    if job_loc:
        return user_city in job_loc or user_province in job_loc

    return False


def user_matches_any_county(user: Dict[str, Any], counties: List[str]) -> bool:
    """True if the user's location matches at least one of the given occupation counties."""
    for c in counties:
        if job_matches_user_location({"city": c, "province": c, "location": c}, user):
            return True
    return False


def _job_matches(job: Dict[str, Any], region_user: Dict[str, Any]) -> bool:
    return job_matches_user_location(job, region_user)


class HubChains:
    """Parsed county -> hub-chain map. Built from the exceptions JSON; chains derived lazily."""

    def __init__(
        self,
        national_hub: str,
        regional_hubs: Optional[Dict[str, List[str]]],
        hub_self_only: Optional[List[str]],
    ) -> None:
        self.national: str = _cf(national_hub)
        self.regional_of: Dict[str, str] = {}  # county_cf -> regional hub_cf
        for hub, counties in (regional_hubs or {}).items():
            h = _cf(hub)
            if not h:
                continue
            for c in counties or []:
                cc = _cf(c)
                if cc:
                    self.regional_of[cc] = h
        self.self_only: set = {_cf(h) for h in (hub_self_only or []) if _cf(h)}

    def chain_for(self, county_cf: str) -> List[str]:
        """Ordered fallback regions for a user county: [local, regional hub?, national hub?].

        Hubs in ``hub_self_only`` -> ``[self]`` (no outward pull). Unknown/empty counties -> the
        national hub only (safe default). Order encodes tier: index 0 = local.
        """
        c = _cf(county_cf)
        if not c:
            return [self.national] if self.national else []
        if c in self.self_only:
            return [c]
        chain = [c]
        reg = self.regional_of.get(c)
        if reg and reg not in chain:
            chain.append(reg)
        if self.national and self.national not in chain:
            chain.append(self.national)
        return chain

    def tier_factor_for_job(
        self, job: Dict[str, Any], county_cf: str, *, w_regional: float, w_national: float
    ) -> float:
        """Location multiplier for one job given the user's county.

        local (chain[0]) -> 1.0; national hub -> ``w_national``; any other in-chain (regional) hub ->
        ``w_regional``; remote jobs -> 1.0 (always allowed); anything off-chain -> 0.0. Roles are keyed
        by identity (local / national), NOT chain index, so a county whose chain skips the regional tier
        (e.g. Kitui -> [kitui, nairobi]) still scores Nairobi at the national weight.
        """
        chain = self.chain_for(county_cf)
        if not chain:
            return 1.0  # no chain (unknown national hub) -> don't penalise; degrade to neutral
        local = chain[0]
        for region in chain:
            ruser = {"city": region, "province": region, "location": region}
            if _job_matches(job, ruser):
                if region == local:
                    return 1.0
                if region == self.national:
                    return float(w_national)
                return float(w_regional)
        return 0.0
