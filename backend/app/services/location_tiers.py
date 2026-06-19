"""Tiered urban-pull location matching.

Maps a user's county to an ordered fallback **hub chain** (local -> regional hub -> national hub) and
scores each job by which tier it falls in, so non-hub users (whose own county has few jobs) still get a
full list while local jobs stay preferred. Hub counties do not pull outward.

Two consumers:
  * the Mongo prefilter (database._location_or_clauses_for_one_user) uses ``hub_chain_for`` to widen a
    user's candidate pool to include their hub regions;
  * the v4 opportunity ranker (match_v4_full_service) uses ``tier_factor_for_job`` as a per-uuid [0,1]
    multiplier on final_score (local=1.0, regional=W_REGIONAL, national=W_NATIONAL, off-chain=0.0).

Dependency-light (no torch). The job<->region test lazily imports
``matching_service._job_matches_user_location`` so region matching uses the EXACT same casefold-substring
+ always-remote semantics as the rest of the system.

The hub data is a small JSON of exceptions (see resources/location/location_hub_chains.json); every county
not named there defaults to ``[self, national_hub]``. If the file is missing/malformed the loader returns
None and callers fall back to today's strict behaviour (the feature becomes a no-op).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _cf(s: Any) -> str:
    """Casefold + strip, aligned with matching_service._norm / database._norm_loc_value."""
    if s is None:
        return ""
    t = str(s).strip()
    return t.casefold() if t else ""


_JMUL = None  # cached matching_service._job_matches_user_location (lazy; avoids torch at import)


def _job_matches(job: Dict[str, Any], region_user: Dict[str, Any]) -> bool:
    global _JMUL
    if _JMUL is None:
        from app.services.matching_service import _job_matches_user_location

        _JMUL = _job_matches_user_location
    return _JMUL(job, region_user)


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


_CACHE: Dict[str, Optional[HubChains]] = {}


def load_hub_chains(path: str) -> Optional[HubChains]:
    """Load + cache the hub-chain map from ``path``. Returns None (and logs) on any failure, so the
    caller can disable tiering and keep today's strict behaviour."""
    if path in _CACHE:
        return _CACHE[path]
    hc: Optional[HubChains] = None
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
    _CACHE[path] = hc
    return hc
