"""Mongo persistence for matching tunables."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from app.database import client
import app.config as c
from app.matching_runtime import (
    MatchingRuntimeSettings,
    apply_settings_to_config_module,
    build_effective_settings,
    build_env_settings,
)

logger = logging.getLogger(__name__)

def _cfg_collection():
    return client[c.MONGO_MATCHING_CONFIG_DB_NAME][c.MONGO_MATCHING_CONFIG_COLLECTION]


def _routing_collection():
    return client[c.MONGO_ROUTING_CONFIG_DB_NAME][c.MONGO_MATCHING_CONFIG_COLLECTION]


async def fetch_override_flat() -> Dict[str, Any]:
    doc = await _cfg_collection().find_one({"_id": "default"})
    if not doc:
        return {}
    return dict(doc.get("values") or {})


async def save_override_flat(flat: Dict[str, Any]) -> None:
    await _cfg_collection().update_one(
        {"_id": "default"},
        {"$set": {"values": flat}},
        upsert=True,
    )


async def fetch_config_document() -> Optional[Dict[str, Any]]:
    return await _cfg_collection().find_one({"_id": "default"})


async def fetch_mongo_routing_document() -> Optional[Dict[str, Any]]:
    return await _routing_collection().find_one({"_id": "mongo_routing"})


async def save_mongo_routing_config(values: Dict[str, Any]) -> None:
    await _routing_collection().update_one(
        {"_id": "mongo_routing"},
        {"$set": {"values": dict(values)}},
        upsert=True,
    )


async def load_and_apply_mongo_routing_config() -> Dict[str, str]:
    doc = await fetch_mongo_routing_document()
    if not doc:
        return {
            "MONGO_DB_NAME": c.MONGO_DB_NAME,
            "MONGO_JOBS_COLLECTION": c.MONGO_JOBS_COLLECTION,
        }
    vals = doc.get("values") or {}
    db_name = str(vals.get("MONGO_DB_NAME") or c.MONGO_DB_NAME).strip()
    jobs_collection = str(vals.get("MONGO_JOBS_COLLECTION") or c.MONGO_JOBS_COLLECTION).strip()
    if db_name:
        c.MONGO_DB_NAME = db_name
    if jobs_collection:
        c.MONGO_JOBS_COLLECTION = jobs_collection
    return {
        "MONGO_DB_NAME": c.MONGO_DB_NAME,
        "MONGO_JOBS_COLLECTION": c.MONGO_JOBS_COLLECTION,
    }


async def load_effective_matching_settings() -> MatchingRuntimeSettings:
    mode = c.MATCHING_CONFIG_SOURCE

    if mode == "env":
        settings = build_env_settings()
        apply_settings_to_config_module(settings)
        return settings

    try:
        mongo = await fetch_override_flat()
    except Exception:
        if mode == "mongodb":
            logger.warning(
                "Matching config source=mongodb: Mongo fetch failed; using code defaults only",
                exc_info=True,
            )
            settings = build_effective_settings({})
            apply_settings_to_config_module(settings)
            return settings
        logger.warning(
            "Matching config source=auto: Mongo fetch failed; falling back to env",
            exc_info=True,
        )
        settings = build_env_settings()
        apply_settings_to_config_module(settings)
        return settings

    if mode == "mongodb":
        settings = build_effective_settings(mongo or {})
        apply_settings_to_config_module(settings)
        return settings

    # mode == auto
    if mongo:
        settings = build_effective_settings(mongo)
        apply_settings_to_config_module(settings)
        return settings
    settings = build_env_settings()
    apply_settings_to_config_module(settings)
    return settings
