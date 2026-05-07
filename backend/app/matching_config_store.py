"""Mongo persistence for matching tunables."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
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

_COLLECTION = os.getenv("MONGO_MATCHING_CONFIG_COLLECTION", "matching_configuration")
_CFG_DB_NAME = os.getenv("MONGO_TEST_USERS_DB_NAME") or os.getenv("MONGO_DB_NAME")
_CFG_DB = client[_CFG_DB_NAME]


async def fetch_override_flat() -> Dict[str, Any]:
    doc = await _CFG_DB[_COLLECTION].find_one({"_id": "default"})
    if not doc:
        return {}
    return dict(doc.get("values") or {})


async def save_override_flat(flat: Dict[str, Any]) -> datetime:
    now = datetime.now(timezone.utc)
    await _CFG_DB[_COLLECTION].update_one(
        {"_id": "default"},
        {"$set": {"values": flat, "updated_at": now}},
        upsert=True,
    )
    return now


async def fetch_config_document() -> Optional[Dict[str, Any]]:
    return await _CFG_DB[_COLLECTION].find_one({"_id": "default"})


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
