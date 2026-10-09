"""O*NET work-activity importance/level lookup built from the occupation taxonomy JSON.

Enriched job documents already carry ``onet_work_activities``, so this is only built at startup when
``WARMUP_WA_LOOKUP`` is set.
"""

import json
import logging
from collections import defaultdict

from app.config import OCCUPATION_JSON_PATH

logger = logging.getLogger(__name__)

_wa_lookup = None  # {occupation_label_lower: {WA_code: {importance, level}}}
_wa_averages = None  # {WA_code: {importance, level}} — fallback for unmatched


def load_wa_lookup():
    """Build WA importance/level lookup from the occupation taxonomy JSON.

    Returns (per_occupation_lookup, cross_occupation_averages).
    """
    global _wa_lookup, _wa_averages
    if _wa_lookup is not None:
        return _wa_lookup, _wa_averages

    with open(OCCUPATION_JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    per_occ = {}

    sums = defaultdict(lambda: {"imp": 0.0, "lvl": 0.0, "n": 0})

    for entry in raw:
        label = entry.get("occupation", {}).get("preferred_label", "").lower().strip()
        wa_dict = {}
        for w in entry.get("onet_work_activities", []):
            code = w.get("WA_code")
            imp = w.get("WA_Importance", "")
            lvl = w.get("WA_Level", "")
            if code and imp and lvl and imp != "" and lvl != "":
                imp_f, lvl_f = float(imp), float(lvl)
                wa_dict[code] = {"importance": imp_f, "level": lvl_f}
                sums[code]["imp"] += imp_f
                sums[code]["lvl"] += lvl_f
                sums[code]["n"] += 1
        if wa_dict:
            per_occ[label] = wa_dict

    averages = {}
    for code, s in sums.items():
        averages[code] = {
            "importance": round(s["imp"] / s["n"], 2),
            "level": round(s["lvl"] / s["n"], 2),
        }

    _wa_lookup = per_occ
    _wa_averages = averages
    logger.info(
        "Built WA lookup: %d occupations, %d WA codes", len(per_occ), len(averages)
    )
    return per_occ, averages


def enrich_work_activities(wa_items: list, classified_occupations: list) -> list:
    """Attach importance/level to a job's work activity items.

    Strategy:
      1. If the job has a classified occupation that matches the taxonomy → use
         that occupation's importance/level per WA code.
      2. Otherwise → use the cross-occupation average for each WA code.
    """
    per_occ, averages = load_wa_lookup()

    # Try to find a matching occupation
    occ_wa = None
    for co in classified_occupations:
        label = (co.get("label") or "").lower().strip()
        if label in per_occ:
            occ_wa = per_occ[label]
            break

    enriched = []
    for item in wa_items:
        code = item.get("id")
        if not code:
            continue
        if occ_wa and code in occ_wa:
            vals = occ_wa[code]
        elif code in averages:
            vals = averages[code]
        else:
            vals = {"importance": 3.5, "level": 3.5}

        enriched.append(
            {
                "WA_code": code,
                "WA_label": item.get("name", ""),
                "WA_Importance": vals["importance"],
                "WA_Level": vals["level"],
            }
        )

    return enriched
