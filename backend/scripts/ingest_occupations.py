#!/usr/bin/env python3
"""Load a deployment's occupations (and occupation groups) into its database.

Input is an occupation file (JSON list of ``OccupationFileEntry``: the committed
``combined_occupation_database_with_wa.json``, or one built by ``scripts/build_occupation_file.py``).
Output is one document per occupation in ``MONGO_OCCUPATIONS_COLLECTION`` (shape in
``app.occupations.occupation_document``), read by the matching service when
``OCCUPATIONS_SOURCE=mongo``. With ``--taxonomy-dir``, the export's occupation groups are also
written to ``MONGO_OCCUPATION_GROUPS_COLLECTION`` and each occupation records its group codes.

* The whole file is validated before anything is read from or written to the database; a bad
  file is refused with every bad entry listed (exit code 2).
* Idempotent: an unchanged occupation is not written, so a second run reports 0 added/updated.
* Embeddings: an occupation keeps its stored vector while its skill text is unchanged. Otherwise
  the vector comes from ``--embeddings-npz`` (by code; the committed NPZ makes a Kenya load exact
  with no API calls), else from Gemini (``gemini-embedding-001``, ``GEMINI_API_KEY``). The raw and
  the whitened vector (``V4_FULL_CONCAT_WHITENING_PATH``) are both stored, with the model and the
  whitening artifact's SHA-256; a new artifact re-whitens every occupation on the next run.
* Occupations no longer in the file are reported; ``--prune`` deletes them.

Usage (from ``backend/``; connection from ``--env-file`` or ``MONGO_URL`` / ``MONGO_DB_NAME``)::

    # Kenya-style stacks: the committed file and NPZ
    python -m scripts.ingest_occupations \\
        --file resources/occupations/en/combined_occupation_database_with_wa.json \\
        --embeddings-npz resources/occupations/occupation_concat_embeddings.npz \\
        --env-file .env.shp --dry-run

    # Argentina: a built file, with the export's groups
    python -m scripts.ingest_occupations \\
        --file resources/occupations/es/occupations_argentina.json \\
        --taxonomy-dir ../../empujar/empujar-taxonomy --env-file .env.argentina --prune
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _env_file_from_argv(argv: list[str]) -> Path | None:
    """The ``--env-file`` value, found before argparse runs."""
    for i, arg in enumerate(argv):
        if arg == "--env-file" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--env-file="):
            return Path(arg.split("=", 1)[1])
    return None


# app.config reads the environment once, on first import (below), so the env file is loaded
# before any app module; main() refuses an --env-file that was not loaded here.
_EARLY_ENV_FILE = _env_file_from_argv(sys.argv[1:]) if __name__ == "__main__" else None
_EARLY_ENV_FILE_LOADED = False
if _EARLY_ENV_FILE is not None:
    from dotenv import load_dotenv

    _EARLY_ENV_FILE_LOADED = load_dotenv(_EARLY_ENV_FILE, override=True)

import numpy as np  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.clients.gemini_embedding_client import (  # noqa: E402
    EMBEDDING_DIM,
    MODEL_NAME,
)
from app.domain.base import DomainModel  # noqa: E402
from app.occupations.ingest import (  # noqa: E402
    IngestPlan,
    OccupationFileInvalid,
    ParsedOccupation,
    ResolvedEmbedding,
    build_occupation_document,
    parse_occupation_file,
    plan_ingest,
    resolve_embeddings,
)
from app.occupations.occupation_document import OccupationDocument  # noqa: E402
from app.occupations.repository import load_embeddings_npz  # noqa: E402
from app.occupations.taxonomy_export import (  # noqa: E402
    build_group_documents,
    normalise_occupation_code,
    occupation_group_ancestry,
    read_taxonomy_export,
)

_SHOW = 10
logger = logging.getLogger("ingest_occupations")


class DocumentCollection(Protocol):
    """The part of a ``pymongo`` collection the ingest uses (a fake in tests)."""

    def find(self, filter: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]: ...

    def replace_one(
        self, filter: Mapping[str, Any], replacement: Mapping[str, Any], upsert: bool
    ) -> object: ...

    def delete_many(self, filter: Mapping[str, Any]) -> object: ...


class Whitening(DomainModel):
    """The concat-whitening transform applied at ingest."""

    whiten_rows: Callable[[np.ndarray], np.ndarray]
    artifact_sha256: str


class CollectionResult(DomainModel):
    """The plan for one collection and whether it was applied."""

    collection: str
    plan: IngestPlan
    written: bool


class IngestResult(DomainModel):
    """What one ingest did."""

    occupations: CollectionResult
    groups: CollectionResult | None
    embedding_origins: dict[str, int]


def _whitened_vectors(
    resolved: Mapping[str, ResolvedEmbedding], whitening: Whitening | None
) -> dict[str, np.ndarray]:
    """Whiten every resolved vector in one batch (as the runtime does), by code."""
    vectors = {code: r.vector for code, r in resolved.items() if r.vector is not None}
    if whitening is None or not vectors:
        return {}
    codes = list(vectors)
    whitened = whitening.whiten_rows(np.stack(list(vectors.values()))).astype(
        np.float32
    )
    return {
        code: np.ascontiguousarray(vec, dtype=np.float32)
        for code, vec in zip(codes, whitened, strict=True)
    }


def _stored_documents(
    collection: DocumentCollection,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, OccupationDocument]]:
    """Stored documents by ``_id``: as stored, and validated (invalid ones only in the first)."""
    raw = {str(doc["_id"]): doc for doc in collection.find({})}
    parsed: dict[str, OccupationDocument] = {}
    for key, doc in raw.items():
        try:
            parsed[key] = OccupationDocument.model_validate(doc)
        except ValidationError:
            logger.warning("Stored occupation %s is invalid; it will be rewritten", key)
    return raw, parsed


def _apply(
    collection: DocumentCollection,
    desired: Mapping[str, Mapping[str, Any]],
    stored: Mapping[str, Mapping[str, Any]],
    plan: IngestPlan,
    now: datetime,
) -> None:
    for key in [*plan.added, *plan.updated]:
        previous = stored.get(key) or {}
        collection.replace_one(
            {"_id": key},
            {
                **desired[key],
                "ingested_at": previous.get("ingested_at") or now,
                "updated_at": now,
            },
            upsert=True,
        )
    if plan.prune and plan.removed:
        collection.delete_many({"_id": {"$in": plan.removed}})


def run_ingest(
    occupations: list[ParsedOccupation],
    *,
    occupations_collection: DocumentCollection,
    occupations_collection_name: str,
    groups_collection: DocumentCollection | None,
    groups_collection_name: str,
    group_documents: Mapping[str, Mapping[str, Any]] | None,
    ancestry: Mapping[str, list[str]],
    seed: Mapping[str, np.ndarray],
    embed: Callable[[list[str]], np.ndarray] | None,
    whitening: Whitening | None,
    source: Mapping[str, Any],
    dry_run: bool,
    prune: bool,
    now: datetime,
) -> IngestResult:
    """Plan (and unless ``dry_run``, write) the occupations and the occupation groups.

    :raises RuntimeError: If an occupation still has no embedding and this is not a dry run
    """
    stored_raw, stored = _stored_documents(occupations_collection)
    resolved = resolve_embeddings(
        occupations,
        stored=stored,
        seed=seed,
        embed=embed,
        model=MODEL_NAME,
        dim=EMBEDDING_DIM,
    )
    pending = [code for code, r in resolved.items() if r.origin == "pending"]
    if pending and not dry_run:
        raise RuntimeError(
            f"{len(pending)} occupations have no embedding (e.g. {pending[:_SHOW]}); "
            "set GEMINI_API_KEY or pass --embeddings-npz"
        )
    whitened = _whitened_vectors(resolved, whitening)
    desired = {
        occ.code: build_occupation_document(
            occ,
            resolved[occ.code],
            whitened=whitened.get(occ.code),
            whitening_artifact_sha256=whitening.artifact_sha256 if whitening else None,
            model=MODEL_NAME,
            dim=EMBEDDING_DIM,
            group_codes=list(ancestry.get(normalise_occupation_code(occ.code), [])),
            source=source,
        )
        for occ in occupations
    }
    plan = plan_ingest(desired, stored_raw, prune=prune)
    if not dry_run:
        _apply(occupations_collection, desired, stored_raw, plan, now)
    result_groups: CollectionResult | None = None
    if groups_collection is not None and group_documents is not None:
        stored_groups = {str(d["_id"]): d for d in groups_collection.find({})}
        group_plan = plan_ingest(group_documents, stored_groups, prune=prune)
        if not dry_run:
            _apply(groups_collection, group_documents, stored_groups, group_plan, now)
        result_groups = CollectionResult(
            collection=groups_collection_name, plan=group_plan, written=not dry_run
        )
    return IngestResult(
        occupations=CollectionResult(
            collection=occupations_collection_name, plan=plan, written=not dry_run
        ),
        groups=result_groups,
        embedding_origins=dict(Counter(r.origin for r in resolved.values())),
    )


def _print_plan(label: str, result: CollectionResult) -> None:
    plan = result.plan
    verb = "" if result.written else "would be "

    def codes(name: str, values: list[str]) -> str:
        more = f" (+{len(values) - _SHOW} more)" if len(values) > _SHOW else ""
        shown = f": {', '.join(values[:_SHOW])}{more}" if values else ""
        return f"    {name}: {len(values)}{shown}"

    print(f"  {label} -> {result.collection}")
    print(codes(f"{verb}added", plan.added))
    print(codes(f"{verb}updated", plan.updated))
    print(codes("unchanged", plan.unchanged))
    removed = f"{verb}removed" if plan.prune else "not in file (kept; --prune deletes)"
    print(codes(removed, plan.removed))


def _print_problems(e: OccupationFileInvalid) -> None:
    print(f"[ingest_occupations] refused: {e}", file=sys.stderr)
    for problem in e.problems[:50]:
        print(f"  {problem}", file=sys.stderr)
    if len(e.problems) > 50:
        print(f"  … and {len(e.problems) - 50} more", file=sys.stderr)


def _load_whitening() -> Whitening | None:
    from app.artifacts.repository import get_artifacts_repository
    from app.config import V4_FULL_CONCAT_WHITENING_PATH
    from app.server_dependencies.model_dependencies import get_concat_whitener

    whitener = get_concat_whitener()
    if whitener.rescale_target() <= 0:
        return None
    return Whitening(
        whiten_rows=whitener.whiten_rows,
        artifact_sha256=get_artifacts_repository().file_sha256(
            V4_FULL_CONCAT_WHITENING_PATH
        ),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Load occupations into a deployment's database."
    )
    ap.add_argument("--file", required=True, type=Path, help="Occupation file (JSON).")
    ap.add_argument(
        "--taxonomy-dir",
        type=Path,
        help="Taxonomy export: also load its occupation groups and set each occupation's group codes.",
    )
    ap.add_argument(
        "--embeddings-npz",
        type=Path,
        help="Vectors by code (embed_occupations NPZ) for occupations without a current stored one.",
    )
    ap.add_argument(
        "--env-file",
        type=Path,
        help="Env file with MONGO_URL / MONGO_DB_NAME (and GEMINI_API_KEY); overrides the environment.",
    )
    ap.add_argument("--mongo-url", help="Overrides MONGO_URL.")
    ap.add_argument("--db", help="Overrides MONGO_DB_NAME.")
    ap.add_argument(
        "--collection",
        help="Overrides MONGO_OCCUPATIONS_COLLECTION (default occupations).",
    )
    ap.add_argument(
        "--groups-collection",
        help="Overrides MONGO_OCCUPATION_GROUPS_COLLECTION (default occupation_groups).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change; write nothing and call no embedding API.",
    )
    ap.add_argument(
        "--prune", action="store_true", help="Delete stored entries not in the input."
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    # 1. Validate the input before touching the database.
    try:
        raw = json.loads(args.file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"[ingest_occupations] cannot read {args.file}: {e}", file=sys.stderr)
        return 2
    try:
        occupations = parse_occupation_file(raw)
    except OccupationFileInvalid as e:
        _print_problems(e)
        return 2
    print(f"[ingest_occupations] {args.file}: {len(occupations)} valid occupations")

    # 2. Configuration (the env file was loaded at import, before app.config).
    if args.env_file is not None:
        if args.env_file != _EARLY_ENV_FILE:
            print(
                "[ingest_occupations] --env-file is only read when the script is run from the "
                "command line (app.config is already loaded)",
                file=sys.stderr,
            )
            return 2
        if not _EARLY_ENV_FILE_LOADED:
            print(f"[ingest_occupations] cannot read {args.env_file}", file=sys.stderr)
            return 2
    for flag, name in (
        (args.mongo_url, "MONGO_URL"),
        (args.db, "MONGO_DB_NAME"),
        (args.collection, "MONGO_OCCUPATIONS_COLLECTION"),
        (args.groups_collection, "MONGO_OCCUPATION_GROUPS_COLLECTION"),
    ):
        if flag:
            os.environ[name] = flag

    from pymongo import MongoClient

    from app.clients.gemini_embedding_client import embed_text_list
    from app.server_dependencies.database_collections import Collections
    from app.server_dependencies.db_dependencies import (
        MongoDbSettings,
        build_client_kwargs,
    )

    settings = MongoDbSettings.from_env()
    occupations_name = (
        os.getenv("MONGO_OCCUPATIONS_COLLECTION") or Collections.OCCUPATIONS
    )
    groups_name = (
        os.getenv("MONGO_OCCUPATION_GROUPS_COLLECTION") or Collections.OCCUPATION_GROUPS
    )

    group_documents = None
    ancestry: dict[str, list[str]] = {}
    if args.taxonomy_dir is not None:
        export = read_taxonomy_export(args.taxonomy_dir)
        group_documents = build_group_documents(export)
        ancestry = occupation_group_ancestry(export)

    seed = load_embeddings_npz(str(args.embeddings_npz)) if args.embeddings_npz else {}
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    embed: Callable[[list[str]], np.ndarray] | None = None
    if api_key and not args.dry_run:

        def embed_with_gemini(texts: list[str]) -> np.ndarray:
            return embed_text_list(texts, api_key=api_key)

        embed = embed_with_gemini

    meta_file = args.file.with_suffix(".meta.json")
    meta = (
        json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
    )
    source = {
        "file": args.file.name,
        "taxonomy_model": meta.get("taxonomy_model"),
    }

    print(
        f"[ingest_occupations] target: db={settings.database_name} "
        f"collections={occupations_name}"
        + (f", {groups_name}" if group_documents is not None else "")
        + (" (dry run)" if args.dry_run else "")
    )
    client: MongoClient = MongoClient(
        settings.mongo_url, **build_client_kwargs(settings, logger)
    )
    try:
        db = client[settings.database_name]
        occupations_collection = db[occupations_name]
        result = run_ingest(
            occupations,
            occupations_collection=occupations_collection,
            occupations_collection_name=occupations_name,
            groups_collection=db[groups_name] if group_documents is not None else None,
            groups_collection_name=groups_name,
            group_documents=group_documents,
            ancestry=ancestry,
            seed=seed,
            embed=embed,
            whitening=_load_whitening(),
            source=source,
            dry_run=args.dry_run,
            prune=args.prune,
            now=datetime.now(timezone.utc),
        )
        if not args.dry_run:
            occupations_collection.create_index("group_codes", name="group_codes")
    except RuntimeError as e:
        print(f"[ingest_occupations] {e}", file=sys.stderr)
        return 1
    finally:
        client.close()

    origins = ", ".join(f"{k} {v}" for k, v in sorted(result.embedding_origins.items()))
    print(f"[ingest_occupations] embeddings: {origins}")
    _print_plan("occupations", result.occupations)
    if result.groups is not None:
        _print_plan("occupation groups", result.groups)
    print(
        "[ingest_occupations] "
        + ("done" if not args.dry_run else "dry run: nothing written")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
