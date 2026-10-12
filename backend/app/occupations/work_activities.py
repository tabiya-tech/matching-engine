"""O*NET work-activity importance/level lookup built from the occupation file.

Enriched job documents already carry ``onet_work_activities``, so this is only built at startup when
``WARMUP_WA_LOOKUP`` is set.
"""

import json
import logging
from collections.abc import Iterable, Mapping

from pydantic import Field

from app.config import OCCUPATION_JSON_PATH
from app.domain.base import DomainModel
from app.domain.work_activities import WorkActivity
from app.occupations.occupation_file import OCCUPATION_FILE

logger = logging.getLogger(__name__)


class WorkActivityRating(DomainModel):
    """O*NET importance and level of one work activity."""

    importance: float = Field(description="Importance on the 1–5 scale.")
    level: float = Field(description="Level on the 1–7 scale.")


# Neutral rating for a work-activity code no occupation rates.
_DEFAULT_RATING = WorkActivityRating(importance=3.5, level=3.5)

WaLookup = dict[str, dict[str, WorkActivityRating]]
"""Occupation label (lower-cased) → work-activity code → rating."""

WaAverages = dict[str, WorkActivityRating]
"""Work-activity code → rating averaged over every occupation that rates it."""

_wa_lookup: WaLookup | None = None
_wa_averages: WaAverages | None = None


def load_wa_lookup() -> tuple[WaLookup, WaAverages]:
    """Build (once) the WA importance/level lookup from the occupation file.

    :return: ``(per_occupation_lookup, cross_occupation_averages)``
    """
    global _wa_lookup, _wa_averages
    if _wa_lookup is not None and _wa_averages is not None:
        return _wa_lookup, _wa_averages

    with open(OCCUPATION_JSON_PATH, "r", encoding="utf-8") as f:
        entries = OCCUPATION_FILE.validate_python(json.load(f))

    per_occ: WaLookup = {}
    sums: dict[str, list[float]] = {}  # code -> [importance_sum, level_sum, n]

    for entry in entries:
        label = (entry.occupation.preferred_label or "").lower().strip()
        ratings: dict[str, WorkActivityRating] = {}
        for w in entry.onet_work_activities:
            # Zero importance/level counts as unrated, as in the source export.
            if not w.WA_code or not w.WA_Importance or not w.WA_Level:
                continue
            ratings[w.WA_code] = WorkActivityRating(
                importance=w.WA_Importance, level=w.WA_Level
            )
            acc = sums.setdefault(w.WA_code, [0.0, 0.0, 0.0])
            acc[0] += w.WA_Importance
            acc[1] += w.WA_Level
            acc[2] += 1
        if ratings:
            per_occ[label] = ratings

    averages: WaAverages = {
        code: WorkActivityRating(importance=round(imp / n, 2), level=round(lvl / n, 2))
        for code, (imp, lvl, n) in sums.items()
    }

    _wa_lookup = per_occ
    _wa_averages = averages
    logger.info(
        "Built WA lookup: %d occupations, %d WA codes", len(per_occ), len(averages)
    )
    return per_occ, averages


def enrich_work_activities(
    wa_items: Iterable[Mapping[str, str]],
    classified_occupations: Iterable[Mapping[str, str]],
) -> list[WorkActivity]:
    """Attach importance/level to a job's work-activity items.

    Strategy:
      1. If the job has a classified occupation that matches the taxonomy → use
         that occupation's importance/level per WA code.
      2. Otherwise → use the cross-occupation average for each WA code (neutral 3.5 / 3.5 for a
         code no occupation rates).

    :param wa_items: The job's ``work_activity.items`` (``{id, name}``)
    :param classified_occupations: The job's ``classified_occupations`` (``{label, …}``)
    :return: The rated work activities, in input order
    """
    per_occ, averages = load_wa_lookup()

    # Try to find a matching occupation
    occ_wa: dict[str, WorkActivityRating] | None = None
    for co in classified_occupations:
        label = (co.get("label") or "").lower().strip()
        if label in per_occ:
            occ_wa = per_occ[label]
            break

    enriched: list[WorkActivity] = []
    for item in wa_items:
        code = item.get("id")
        if not code:
            continue
        if occ_wa and code in occ_wa:
            rating = occ_wa[code]
        else:
            rating = averages.get(code, _DEFAULT_RATING)
        enriched.append(
            WorkActivity(
                code=code,
                label=item.get("name", ""),
                importance=rating.importance,
                level=rating.level,
            )
        )

    return enriched
