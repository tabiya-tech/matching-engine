"""Location eligibility as standardised location keys: the one rule every matching path uses.

A **location key** is a canonical region name: one comma-separated part of a city / county /
province string, stripped of surrounding punctuation, whitespace-collapsed and casefolded
(``"Nairobi, Busia"`` -> ``nairobi``, ``busia``). Any place string containing "remote" also yields
the ``remote`` key.

* Jobs carry their keys in the stored ``location_keys`` array (written by the llm-reranker, or by
  ``tools/location/backfill_location_keys.py`` for older documents). The array is indexed, so the database
  decides which jobs are eligible for a request (``app.services.job_retrieval``).
* Occupation rows get the keys of their county when the occupation file is flattened.
* A user's eligible keys are resolved once per request by ``user_location_keys``: their own
  city/province, or — with location tiers on — the regions of their hub chain (local -> regional
  hub -> national hub), each with its tier weight. ``remote`` is always eligible.

Keys are compared exactly. The llm-reranker keeps a copy of ``canonical_location`` /
``location_keys_for`` (it writes the stored keys); ``tests/fixtures/location_key_vectors.json``
holds the cases both copies must agree on.
"""

from __future__ import annotations

import re
from collections.abc import Collection

from pydantic import Field

from app.domain.base import DomainModel
from app.domain.user import UserProfile

REMOTE_KEY = "remote"

# Separators between several regions in one field, e.g. "Nairobi, Busia, Kisumu" or "National/Remote".
_PART_SEPARATORS = re.compile(r"[,;/]")
# Punctuation left around a region name by the scraper, e.g. "CABA)" or "Lusaka.".
_EDGE_PUNCTUATION = " \t\n.()[]\"'"
_WHITESPACE = re.compile(r"\s+")


def canonical_location(value: str | None) -> str:
    """One region name as a key: punctuation-trimmed, whitespace-collapsed, casefolded.

    :return: The key, or ``""`` for a blank value
    """
    if not value:
        return ""
    return _WHITESPACE.sub(" ", value.strip(_EDGE_PUNCTUATION)).casefold()


def location_keys_for(*values: str | None) -> tuple[str, ...]:
    """The sorted, de-duplicated location keys of a place given as one or more strings.

    Each value is split on ``,`` ``;`` and ``/`` and every part becomes a key; ``remote`` is added
    when any value contains "remote".
    """
    keys: set[str] = set()
    for value in values:
        if not value:
            continue
        if REMOTE_KEY in value.casefold():
            keys.add(REMOTE_KEY)
        for part in _PART_SEPARATORS.split(value):
            key = canonical_location(part)
            if key:
                keys.add(key)
    return tuple(sorted(keys))


class HubChains:
    """Parsed county -> hub-chain map. Built from the exceptions JSON; chains derived lazily.

    Maps a user's county to an ordered fallback chain (local -> regional hub -> national hub), so
    non-hub users (whose own county has few jobs) still get a full list while local jobs stay
    preferred. Hub counties do not pull outward. The data is a small JSON of exceptions
    (``LOCATION_HUB_CHAINS_PATH``, loaded by ``ArtifactsRepository.load_hub_chains``); every county
    not named there defaults to ``[self, national_hub]``.
    """

    def __init__(
        self,
        national_hub: str,
        regional_hubs: dict[str, list[str]] | None,
        hub_self_only: list[str] | None,
    ) -> None:
        self.national: str = canonical_location(national_hub)
        self.regional_of: dict[str, str] = {}  # county key -> regional hub key
        for hub, counties in (regional_hubs or {}).items():
            h = canonical_location(hub)
            if not h:
                continue
            for c in counties or []:
                cc = canonical_location(c)
                if cc:
                    self.regional_of[cc] = h
        self.self_only: set[str] = {
            k for k in (canonical_location(h) for h in hub_self_only or []) if k
        }

    def chain_for(self, county: str) -> list[str]:
        """Ordered fallback region keys for a user county: [local, regional hub?, national hub?].

        Hubs in ``hub_self_only`` -> ``[self]`` (no outward pull). Empty county -> the national hub
        only. Order encodes tier: index 0 = local.
        """
        c = canonical_location(county)
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


class UserLocationKeys(DomainModel):
    """The location keys a user may be matched to, each with its tier weight, in preference order."""

    tiers: tuple[tuple[str, float], ...] = Field(
        description="``(key, tier)`` pairs, best first; ``remote`` is always first at 1.0."
    )

    @property
    def keys(self) -> frozenset[str]:
        """The eligible keys (what the database filters on)."""
        return frozenset(k for k, _ in self.tiers)

    def tier_for(self, item_keys: Collection[str]) -> float:
        """Tier of an item with ``item_keys``: that of the user's first key the item has, else 0.0
        (not eligible)."""
        for key, tier in self.tiers:
            if key in item_keys:
                return tier
        return 0.0


def user_home_keys(user: UserProfile) -> tuple[str, ...]:
    """Keys of the user's own city and province; empty unless both are given."""
    if not user.city.strip() or not user.province.strip():
        return ()
    return location_keys_for(user.city, user.province)


def user_location_keys(
    user: UserProfile,
    hub_chains: HubChains | None = None,
    *,
    w_regional: float = 1.0,
    w_national: float = 1.0,
) -> UserLocationKeys:
    """Resolve which job location keys ``user`` is eligible for.

    * No city or no province -> ``remote`` only.
    * With ``hub_chains`` -> ``remote`` plus the regions of the province's hub chain: local 1.0,
      the national hub ``w_national``, any other hub ``w_regional``.
    * Without -> ``remote`` plus the user's own city and province keys, all 1.0.
    """
    tiers: list[tuple[str, float]] = [(REMOTE_KEY, 1.0)]
    home = user_home_keys(user)
    if not home:
        return UserLocationKeys(tiers=tuple(tiers))
    if hub_chains is None:
        tiers.extend((k, 1.0) for k in home if k != REMOTE_KEY)
        return UserLocationKeys(tiers=tuple(tiers))
    chain = hub_chains.chain_for(user.province)
    for region in chain:
        if any(k == region for k, _ in tiers):
            continue
        if region == chain[0]:
            tier = 1.0
        elif region == hub_chains.national:
            tier = float(w_national)
        else:
            tier = float(w_regional)
        tiers.append((region, tier))
    return UserLocationKeys(tiers=tuple(tiers))
