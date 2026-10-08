"""Occupation slate selection: random-county fallback and de-duplication by occupation code."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from typing import Any

from app.ranking.location import user_matches_any_county

logger = logging.getLogger(__name__)


def occupation_counties(occupations: Iterable[dict[str, Any]]) -> list[str]:
    """Sorted distinct counties the occupation corpus is flattened over."""
    return sorted({str(o.get("province")) for o in occupations if o.get("province")})


class OccupationSelector:
    """Occupations are flattened into one identical-embedding row per (code, county). These helpers
    keep one row per code in the user's county."""

    def fallback_location(
        self, user: dict[str, Any], counties: list[str]
    ) -> dict[str, str] | None:
        """A random available county to filter by when the user's province matches none of them
        (location filter only — the user's real preferences still drive u_hat); None otherwise."""
        if counties and not user_matches_any_county(user, counties):
            fallback = random.choice(counties)
            logger.warning(
                "User %r province=%r matches no occupation county %s; using random fallback county %r.",
                str(user.get("user_id") or ""),
                user.get("province"),
                counties,
                fallback,
            )
            return {
                "city": fallback,
                "province": fallback,
                "location": fallback,
            }
        return None

    def unique_by_code(
        self,
        recs: list[dict[str, Any]],
        item_index: dict[str, dict[str, Any]],
        limit: int,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """The first ``limit`` (rec, item) pairs, in rank order, with one row per occupation code."""
        out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        seen_codes: set = set()
        for rec in recs:
            item = item_index.get(str(rec.get("job_uuid") or ""))
            if not item:
                continue
            code = str(item.get("originUuid") or item.get("uuid") or "")
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            out.append((rec, item))
            if len(out) >= limit:
                break
        return out
