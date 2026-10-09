"""Providers for the heavy, process-lifetime model objects.

Each is loaded on first use and then reused (thread-safe), exactly once per process:
CosineSkillMatcher torch-loads ~14k embedding rows + the taxonomy packs, the cross-encoder pulls
Hugging Face weights. Rebuilding either per request is what produced 80–100s requests in early testing.
The ``build_*`` functions construct a fresh instance from the artifacts on disk.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from app.artifacts.repository import get_artifacts_repository
from app.clients.cross_encoder_client import CrossEncoderClient, ICrossEncoderClient
from app.clients.gemini_embedding_client import (
    EMBEDDING_DIM,
    GeminiEmbeddingClient,
    IGeminiEmbeddingClient,
)
from app.config import (
    CROSS_ENCODER_BATCH_SIZE,
    EMBEDDING_MODEL_PATH,
    HYBRID_PREF_SCHEMA_PATH,
    PREFERENCE_SCORER_MODE,
    SKILL_GROUPS_CSV_PATH,
    SKILL_HIERARCHY_CSV_PATH,
    SKILL_RESCALE_TARGET,
    SKILL_TO_ROW_PATH,
    V4_FULL_CONCAT_WHITENING_PATH,
    V4_FULL_EMBEDDING_MODEL_PATH,
    V4_FULL_RANK_DEMOTE,
    V4_FULL_WHITENED_GATE,
)
from app.languages import default_language
from app.ranking.preference import UnifiedPreferenceScorer
from app.ranking.skills import SkillLabelPacks
from app.ranking.skills import CosineSkillMatcher
from app.ranking.skills import SkillScorer
from app.ranking.retrieval import ConcatWhitener

logger = logging.getLogger(__name__)


def _label_packs(embedding_ids: set) -> SkillLabelPacks:
    repo = get_artifacts_repository()
    return SkillLabelPacks(
        embedding_ids, read_rows=lambda p: repo.read_csv_rows(p, newline="")
    )


def build_skill_matcher(model_path: str | None = None) -> CosineSkillMatcher:
    """A new CosineSkillMatcher over ``model_path`` (default ``EMBEDDING_MODEL_PATH``)."""
    repo = get_artifacts_repository()
    path = Path(model_path or EMBEDDING_MODEL_PATH)
    artifact = repo.load_skill_embedding(path)
    skill_to_row = repo.load_skill_to_row(SKILL_TO_ROW_PATH)
    return CosineSkillMatcher(
        weights=artifact.weights,
        skill_to_row=skill_to_row,
        packs=_label_packs(set(skill_to_row.keys())),
        display_language=default_language(),
        rescale_target=float(artifact.whitening_target or 0.0),
        model_label=artifact.model_name or path.name,
    )


def _optional_csv_rows(path: str) -> list | None:
    try:
        return get_artifacts_repository().read_csv_rows(path)
    except FileNotFoundError:
        return None


def build_skill_scorer() -> SkillScorer:
    """A new SkillScorer over ``EMBEDDING_MODEL_PATH``.

    The per-rowmax rescale target comes from the artefact metadata, unless the operator has
    explicitly set ``SKILL_RESCALE_TARGET`` in env (operator wins). Whitened artefacts carry a
    ``target_max_p999`` value computed at build time; raw / non-whitened artefacts don't, in which
    case the configured default applies.
    """
    repo = get_artifacts_repository()
    path = Path(EMBEDDING_MODEL_PATH)
    artifact = repo.load_skill_embedding(path)

    rescale_target = SKILL_RESCALE_TARGET
    if "SKILL_RESCALE_TARGET" not in os.environ:
        if artifact.whitening_target is not None:
            rescale_target = float(artifact.whitening_target)
            logger.info(
                "SkillScorer: SKILL_RESCALE_TARGET set from artefact metadata = %.4f",
                rescale_target,
            )

    skill_to_row = repo.load_skill_to_row(SKILL_TO_ROW_PATH, encoding=None)
    return SkillScorer(
        weights=artifact.weights,
        skill_to_row=skill_to_row,
        packs=_label_packs(set(skill_to_row.keys())),
        display_language=default_language(),
        rescale_target=rescale_target,
        skill_group_rows=_optional_csv_rows(SKILL_GROUPS_CSV_PATH),
        hierarchy_rows=_optional_csv_rows(SKILL_HIERARCHY_CSV_PATH),
        model_label=artifact.model_name or path.name,
    )


_matcher_lock = threading.Lock()
_matcher_instance: CosineSkillMatcher | None = None


def get_skill_matcher() -> CosineSkillMatcher:
    """The shared CosineSkillMatcher (stage-1 retrieval detail, v2/v3 matched skills)."""
    global _matcher_instance
    if _matcher_instance is None:
        with _matcher_lock:
            if _matcher_instance is None:
                _matcher_instance = build_skill_matcher()
    return _matcher_instance


_v4_matcher_lock = threading.Lock()
_v4_matcher_instance: CosineSkillMatcher | None = None


def get_v4_skill_matcher() -> CosineSkillMatcher:
    """v4-only matcher backed by the WHITENED skill artifact (de-anisotropised + rescaled). Separate
    singleton from ``get_skill_matcher`` so the per-skill GATE in /match is fixed without changing the
    shared matcher used by v2/v3 and the retrieval detail."""
    global _v4_matcher_instance
    if _v4_matcher_instance is None:
        with _v4_matcher_lock:
            if _v4_matcher_instance is None:
                _v4_matcher_instance = build_skill_matcher(V4_FULL_EMBEDDING_MODEL_PATH)
    return _v4_matcher_instance


_skill_scorer_lock = threading.Lock()
_skill_scorer_instance: SkillScorer | None = None


def get_skill_scorer() -> SkillScorer:
    """The shared SkillScorer (legacy /match utility + the skill-gap engine)."""
    global _skill_scorer_instance
    if _skill_scorer_instance is None:
        with _skill_scorer_lock:
            if _skill_scorer_instance is None:
                _skill_scorer_instance = build_skill_scorer()
    return _skill_scorer_instance


_concat_white_lock = threading.Lock()
_concat_whitener: ConcatWhitener | None = None


def get_concat_whitener() -> ConcatWhitener:
    """Lazy-load the concat-whitening artifact (mu, W=Sigma^-1/2, target) for whitened stage-1 retrieval
    and the Phase-2 p_hat. Unavailable artifact -> a whitener that only L2-normalises (target 0.0)."""
    global _concat_whitener
    if _concat_whitener is None:
        with _concat_white_lock:
            if _concat_whitener is None:
                _concat_whitener = ConcatWhitener(
                    get_artifacts_repository().load_concat_whitening(
                        V4_FULL_CONCAT_WHITENING_PATH, expected_dim=EMBEDDING_DIM
                    )
                )
    return _concat_whitener


_cross_encoder_lock = threading.Lock()
# One cross-encoder per language: the checkpoint has to understand the label text it scores.
_cross_encoder_instances: dict[str, ICrossEncoderClient] = {}


def get_cross_encoder_client() -> ICrossEncoderClient:
    """Cross-encoder for the deployment's language, loaded on first use and then reused.

    Keyed by language rather than a single global so that a process whose
    ``TARGET_LANGUAGE`` changes (tests, a script) does not reuse the wrong checkpoint.
    """
    lang = default_language()
    existing = _cross_encoder_instances.get(lang)
    if existing is not None:
        return existing
    with _cross_encoder_lock:
        existing = _cross_encoder_instances.get(lang)
        if existing is not None:
            return existing
        inst = CrossEncoderClient(
            batch_size=CROSS_ENCODER_BATCH_SIZE,
            language=lang,
        )
        inst.warmup()
        _cross_encoder_instances[lang] = inst
        return inst


_gemini_client: IGeminiEmbeddingClient = GeminiEmbeddingClient()


def get_gemini_embedding_client() -> IGeminiEmbeddingClient:
    """The Gemini embedding client (reads ``GEMINI_API_KEY`` on every call)."""
    return _gemini_client


def get_preference_scorer():
    """Return a new preference scorer of the configured kind.

    Default ``unified`` → ``UnifiedPreferenceScorer`` (DCE attributes + BWS, additive-RUM).
    ``legacy`` → the old hardcoded-beta ``PreferenceScorer`` (A/B escape hatch only).
    """
    if PREFERENCE_SCORER_MODE == "legacy":
        from app.ranking.preference import PreferenceScorer

        return PreferenceScorer()
    try:
        schema = get_artifacts_repository().load_attribute_schema(
            HYBRID_PREF_SCHEMA_PATH or None
        )
    except (OSError, ValueError) as e:
        # Never let a missing/invalid schema take down the whole matcher. Degrade to BWS-only
        # and log loudly.
        logger.error(
            "UnifiedPreferenceScorer: could not load attribute schema (%s). DCE attribute "
            "term DISABLED — scoring on BWS only. Commit/deploy job_attributes_schema.json "
            "(or set HYBRID_PREF_SCHEMA_PATH) to restore.",
            e,
        )
        schema = {"attributes": []}
    return UnifiedPreferenceScorer(schema)


def preload_models() -> dict[str, float]:
    """Warm CosineSkillMatcher + CrossEncoder once (call from FastAPI lifespan to avoid per-request cost)."""
    t0 = time.perf_counter()
    get_skill_matcher()
    t1 = time.perf_counter()
    if V4_FULL_WHITENED_GATE:
        get_v4_skill_matcher()  # warm the whitened v4 gate matrix so the first /match doesn't pay the load
    if V4_FULL_RANK_DEMOTE:
        whitener = (
            get_concat_whitener()
        )  # warm the concat-whitening artifact for the Phase-2 whitened p_hat
        # Log the in-process artifact's sha256 + target so ops can confirm they MATCH the DB's recorded
        # whitening.artifact_sha256 / target. If they ever diverge, whitened-user (in-process) vs
        # job_embedding (DB) would be an inconsistent cosine — this is the one hard dependency.
        try:
            _sha = get_artifacts_repository().file_sha256(V4_FULL_CONCAT_WHITENING_PATH)
            logger.info(
                "concat whitening artifact: path=%s sha256=%s target=%.6f (must match DB job whitening)",
                V4_FULL_CONCAT_WHITENING_PATH,
                _sha,
                whitener.rescale_target(),
            )
        except OSError:
            logger.warning(
                "concat whitening artifact not readable at %s; DB-whitened jobs will be degraded.",
                V4_FULL_CONCAT_WHITENING_PATH,
            )
    t1b = time.perf_counter()
    get_cross_encoder_client()
    t2 = time.perf_counter()
    return {
        "cosine_skill_matcher_ms": (t1 - t0) * 1000.0,
        "v4_whitened_matcher_ms": (t1b - t1) * 1000.0,
        "cross_encoder_ms": (t2 - t1b) * 1000.0,
    }
