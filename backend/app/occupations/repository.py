"""Repository for the occupation corpus and its precomputed concat embeddings (local files)."""

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from app.config import OCCUPATION_CONCAT_EMBEDDINGS_PATH, OCCUPATION_JSON_PATH
from app.ranking.retrieval import ConcatWhitener

logger = logging.getLogger(__name__)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _occ_skill_pairs(uuids: list, labels: list) -> list[dict[str, str]]:
    """Zip occupation skill uuids with their labels (from the occupation JSON; '' if absent)."""
    pairs: list[dict[str, str]] = []
    for i, u in enumerate(uuids):
        lab = labels[i] if i < len(labels) else ""
        pairs.append({"id": str(u), "label": str(lab) if lab else ""})
    return pairs


def _enrich_work_activities(
    wa_items: list, classified_occupations: list, wa_lookup: tuple
) -> list:
    """Attach importance/level to a job's work activity items.

    Strategy:
      1. If the job has a classified occupation that matches the taxonomy → use
         that occupation's importance/level per WA code.
      2. Otherwise → use the cross-occupation average for each WA code.
    """
    per_occ, averages = wa_lookup

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


class IOccupationsRepository(ABC):
    """Interface for the occupation corpus (one row per occupation and county)."""

    @abstractmethod
    async def load_with_timing(self) -> tuple[list[dict], dict[str, Any]]:
        """
        Loads the flattened occupation corpus (cached after the first call).

        :return: ``(rows, timing)``; on a cache hit ``occupation_cache_hit`` is True
        :raises RuntimeError: If the occupation file cannot be loaded
        """
        raise NotImplementedError()

    @abstractmethod
    def attach_embeddings(self, occupations: Sequence[dict]) -> list[dict]:
        """
        Returns occupation rows with ``job_embedding`` / ``job_embedding_whitened`` attached by code.

        Rows with no matching embedding are returned unchanged.

        :param occupations: Rows from ``load_with_timing``
        :return: New list of rows
        """
        raise NotImplementedError()

    @abstractmethod
    def load_wa_lookup(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Builds the work-activity importance/level lookup from the occupation taxonomy.

        :return: ``(per_occupation_lookup, cross_occupation_averages)``
        :raises Exception: If the occupation file cannot be read
        """
        raise NotImplementedError()


class OccupationsRepository(IOccupationsRepository):
    def __init__(self, *, whitener_provider: Callable[[], ConcatWhitener]):
        # Whitens the occupation embeddings once at load (see _load_occupation_embeddings).
        self._whitener_provider = whitener_provider
        self._cached_occupations: list[dict] | None = None
        # {occupation_code: np.ndarray(float32, EMBEDDING_DIM)}
        self._cached_occ_embeddings: dict[str, Any] | None = None
        # True once the cached occ embeddings are whitened (consumed directly, no per-request whitening)
        self._occ_prewhitened = False
        # {occupation_label_lower: {WA_code: {importance, level}}}
        self._wa_lookup: dict[str, Any] | None = None
        # {WA_code: {importance, level}} — fallback for unmatched
        self._wa_averages: dict[str, Any] | None = None
        self._logger = logging.getLogger(self.__class__.__name__)

    async def load_with_timing(self) -> tuple[list[dict], dict[str, Any]]:
        """Load occupations; returns (flat_list, timing_dict).

        On cache hit, occupation_file_read_ms is 0 and occupation_cache_hit is True.
        """
        t_total = time.perf_counter()

        if self._cached_occupations is not None:
            total_ms = _ms(t_total)
            return self._cached_occupations, {
                "occupation_cache_hit": True,
                "occupation_file_read_ms": 0.0,
                "occupation_json_parse_and_flatten_ms": 0.0,
                "n_occupation_rows": len(self._cached_occupations),
                "get_all_occupations_total_ms": total_ms,
            }

        try:
            t0 = time.perf_counter()
            with open(OCCUPATION_JSON_PATH, "r", encoding="utf-8") as f:
                raw_occupations = json.load(f)
            file_read_and_json_ms = _ms(t0)

            t1 = time.perf_counter()
            flattened = []
            for entry in raw_occupations:
                occ = entry.get("occupation", {})
                skills = entry.get("skills", {})
                ess_block = (
                    skills.get("essential", {})
                    if isinstance(skills.get("essential"), dict)
                    else {}
                )
                opt_block = (
                    skills.get("optional", {})
                    if isinstance(skills.get("optional"), dict)
                    else {}
                )
                ess_uuids = ess_block.get("uuids", []) or []
                opt_uuids = opt_block.get("uuids", []) or []
                ess_labels = ess_block.get("labels", []) or []
                opt_labels = opt_block.get("labels", []) or []
                counties = entry.get("counties_data", [])

                code = occ.get("code", "")
                label = occ.get("preferred_label", "Unknown")
                description = occ.get("description", "")

                # Post-secondary education gate (see app.ranking.retrieval):
                # occupation-level flag, applied to all of this occupation's county rows.
                requires_post_secondary = occ.get("requires_post_secondary")
                if requires_post_secondary is None:
                    requires_post_secondary = entry.get("requires_post_secondary")

                raw_wa = entry.get("onet_work_activities", [])
                onet_wa = []
                for w in raw_wa:
                    wc = w.get("WA_code")
                    imp = w.get("WA_Importance", "")
                    lvl = w.get("WA_Level", "")
                    if wc and imp != "" and lvl != "":
                        onet_wa.append(
                            {
                                "WA_code": wc,
                                "WA_label": w.get("WA_label", ""),
                                "WA_Importance": float(imp),
                                "WA_Level": float(lvl),
                            }
                        )

                if not counties:
                    counties = [{"county": "", "job_attributes": {}}]

                for cd in counties:
                    county = cd.get("county", "")
                    job_attrs = cd.get("job_attributes", {})
                    attrs_raw = job_attrs.get("attributes", [])
                    attributes = {}
                    if isinstance(attrs_raw, list):
                        for a in attrs_raw:
                            name = a.get("attribute_name")
                            val = a.get("selected_level_id")
                            if name and val:
                                attributes[name] = val
                    elif isinstance(attrs_raw, dict):
                        attributes = attrs_raw

                    # Demand label so DemandScorer can read attributes["expected_demand"]
                    # (engine-agnostic; powers score_breakdown.demand_* on /match_v4).
                    expected_demand = (cd.get("labor_demand") or {}).get(
                        "expected_demand"
                    )
                    if expected_demand:
                        attributes = {**attributes, "expected_demand": expected_demand}

                    # Wrap skills in the same {id, label} shape used by job dicts. Labels come
                    # from the occupation JSON (skills.*.labels) when present, else empty; gap
                    # analysis reads id directly without going through label resolution.
                    flattened.append(
                        {
                            "uuid": f"{code}_{county}" if county else code,
                            "originUuid": code,
                            "occupation_label": label,
                            "preferredLabel": label,
                            "description": description,
                            "location": county,
                            "city": county,
                            "province": county,
                            "essential_skills": _occ_skill_pairs(ess_uuids, ess_labels),
                            "optional_skills": _occ_skill_pairs(opt_uuids, opt_labels),
                            "skill_groups_origin_uuids": [],
                            "attributes": attributes,
                            "requires_post_secondary": requires_post_secondary,
                            "onet_work_activities": onet_wa,
                            # Occupation-specific tasks (sparse in source); formatter falls back to
                            # O*NET WA labels when absent. See app.matching.formatting._typical_tasks.
                            "included_tasks": occ.get("included_tasks") or "",
                        }
                    )

            flatten_ms = _ms(t1)
            self._cached_occupations = flattened
            total_ms = _ms(t_total)
            logger.info(
                "Loaded %d occupation-county items from %d raw occupations",
                len(flattened),
                len(raw_occupations),
            )
            return self._cached_occupations, {
                "occupation_cache_hit": False,
                "occupation_file_read_ms": file_read_and_json_ms,
                "occupation_json_parse_and_flatten_ms": flatten_ms,
                "n_occupation_rows": len(flattened),
                "n_raw_occupation_entries": len(raw_occupations),
                "get_all_occupations_total_ms": total_ms,
            }
        except Exception as e:
            logger.exception(e)
            raise RuntimeError(f"Failed to load occupations: {e}")

    def _load_occupation_embeddings(self) -> dict[str, Any]:
        """Lazy/cached load of the committed occupation concat-embeddings NPZ (code -> vector).

        Returns {} (with a warning) if the artifact is missing/unreadable, so occupations are
        simply skipped by the /match_v4 retrieval rather than crashing the request.
        """
        if self._cached_occ_embeddings is not None:
            return self._cached_occ_embeddings
        out: dict[str, Any] = {}
        try:
            import numpy as np

            with np.load(OCCUPATION_CONCAT_EMBEDDINGS_PATH, allow_pickle=True) as data:
                codes = [str(c) for c in data["codes"].tolist()]
                vectors = np.asarray(data["vectors"], dtype=np.float32)
            for code, vec in zip(codes, vectors):
                out[code] = np.ascontiguousarray(vec, dtype=np.float32)
            logger.info(
                "Loaded %d occupation concat embeddings from %s",
                len(out),
                OCCUPATION_CONCAT_EMBEDDINGS_PATH,
            )
        except FileNotFoundError:
            logger.warning(
                "Occupation embeddings NPZ not found at %s; /match_v4 occupations will be skipped. "
                "Build it via `python -m tools.embeddings.embed_occupations`.",
                OCCUPATION_CONCAT_EMBEDDINGS_PATH,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "Failed to load occupation embeddings (%s): %s; occupations skipped.",
                OCCUPATION_CONCAT_EMBEDDINGS_PATH,
                e,
            )
        # Whiten the (static) occupation embeddings ONCE here, into the same whitened concat space the
        # matcher uses, so the engine consumes them directly instead of re-whitening ~1700 vectors on every
        # request (the NPZ is raw L2-normalized). Falls back to raw if the concat artifact is unavailable.
        self._occ_prewhitened = False
        if out:
            try:
                import numpy as np

                whitener = self._whitener_provider()
                if whitener.rescale_target() > 0:
                    codes_list = list(out.keys())
                    wmat = whitener.whiten_rows(
                        np.stack([out[c] for c in codes_list], axis=0)
                    ).astype(np.float32)
                    for c, wv in zip(codes_list, wmat):
                        out[c] = np.ascontiguousarray(wv, dtype=np.float32)
                    self._occ_prewhitened = True
                    logger.info(
                        "Whitened %d occupation embeddings once at load (consumed directly thereafter).",
                        len(out),
                    )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "Could not pre-whiten occupation embeddings (%s); whitening in-process per request.",
                    e,
                )
                self._occ_prewhitened = False
        self._cached_occ_embeddings = out
        return out

    def attach_embeddings(self, occupations: Sequence[dict]) -> list[dict]:
        """Return occupation dicts with ``job_embedding`` (shared np.ndarray) attached by code.

        Vector is shared across all county-rows of the same occupation code (skills are identical),
        so memory stays at one array per code. Rows with no matching embedding are returned
        unchanged (the v4 engine skips items without a stage-1 vector).
        """
        emb = self._load_occupation_embeddings()
        if not emb:
            return list(occupations)
        out: list[dict] = []
        for occ in occupations:
            vec = emb.get(str(occ.get("originUuid") or ""))
            if vec is None:
                out.append(occ)
            else:
                o = dict(occ)
                o["job_embedding"] = vec
                # True once the occ cache has been whitened at load -> engine consumes it directly (mirrors
                # DB-whitened jobs); False -> raw, whitened in-process per request.
                o["job_embedding_whitened"] = self._occ_prewhitened
                out.append(o)
        return out

    def load_wa_lookup(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build WA importance/level lookup from the occupation taxonomy JSON.

        Returns (per_occupation_lookup, cross_occupation_averages).
        """
        if self._wa_lookup is not None:
            return self._wa_lookup, self._wa_averages

        with open(OCCUPATION_JSON_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)

        per_occ = {}
        from collections import defaultdict

        sums = defaultdict(lambda: {"imp": 0.0, "lvl": 0.0, "n": 0})

        for entry in raw:
            label = (
                entry.get("occupation", {}).get("preferred_label", "").lower().strip()
            )
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

        self._wa_lookup = per_occ
        self._wa_averages = averages
        logger.info(
            "Built WA lookup: %d occupations, %d WA codes", len(per_occ), len(averages)
        )
        return per_occ, averages
