#!/usr/bin/env python3
"""Parity report: occupations returned by the v4 engine from the file source vs the database source.

Runs ``MatchingService.rank`` (the code behind ``/match``) for a fixed set of users twice: once
with the committed occupation file + NPZ (``OCCUPATIONS_SOURCE=file``, the "before"), once with the
occupations read through ``MongoOccupationsRepository`` (the "after"). Users are embedded once and
the same vectors are used for both runs, the random county fallback is seeded, and no jobs are
loaded, so any difference comes from the occupation source alone. Writes a Markdown report (for
the PR) and exits 1 on any difference.

``--after``:

* ``documents`` (default): the file ingested in memory exactly as ``scripts/ingest_occupations.py``
  would (seeded from the NPZ, whitened at ingest), read back through the Mongo repository. Offline
  apart from the Gemini call that embeds the users.
* ``mongo``: the live collection of ``--env-file`` / ``MONGO_URL`` + ``MONGO_DB_NAME`` (read only),
  after an ingest of the same file.

Users: ``--users`` (JSON list or JSONL of ``MatchRequest``), else ``--n-users`` synthetic users
built deterministically from the catalogue (skills of a sampled occupation, one of its counties).

Usage (from ``backend/``; needs ``GEMINI_API_KEY`` and the cross-encoder from ``./setup.sh``)::

    python -m scripts.occupation_parity_report --output occupation_parity.md
    python -m scripts.occupation_parity_report --after mongo --env-file .env.shp
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import random
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import numpy as np  # noqa: E402

from app.domain.base import DomainModel  # noqa: E402
from app.domain.occupation import Occupation  # noqa: E402
from app.domain.user import UserProfile  # noqa: E402
from app.occupations import loader  # noqa: E402
from app.occupations.ingest import parse_occupation_file  # noqa: E402
from app.occupations.mongo_repository import MongoOccupationsRepository  # noqa: E402
from app.occupations.repository import (  # noqa: E402
    FileOccupationsRepository,
    IOccupationsRepository,
    load_embeddings_npz,
)
from app.schemas import MatchRequest, MatchResponse  # noqa: E402

_SEED = 20261019


class OccupationRow(DomainModel):
    code: str
    county: str
    final_score: float


class UserParity(DomainModel):
    user_id: str
    province: str
    before: list[OccupationRow]
    after: list[OccupationRow]

    @property
    def same_order(self) -> bool:
        return [(r.code, r.county) for r in self.before] == [
            (r.code, r.county) for r in self.after
        ]

    @property
    def max_score_delta(self) -> float:
        if not self.same_order or not self.before:
            return float("nan") if not self.same_order else 0.0
        return max(
            abs(b.final_score - a.final_score)
            for b, a in zip(self.before, self.after, strict=True)
        )


def _load_requests(path: Path) -> list[MatchRequest]:
    text = path.read_text(encoding="utf-8").strip()
    rows = (
        json.loads(text)
        if text.startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    return [MatchRequest.model_validate(r) for r in rows]


def synthetic_requests(
    entries: Sequence[Mapping[str, Any]], n: int, *, seed: int = _SEED
) -> list[MatchRequest]:
    """``n`` users, each with up to 8 essential skills of a sampled occupation and one of its counties."""
    rng = random.Random(seed)
    candidates = [
        e
        for e in entries
        if ((e.get("skills") or {}).get("essential") or {}).get("labels")
    ]
    requests: list[MatchRequest] = []
    for entry in rng.sample(candidates, min(n, len(candidates))):
        essential = entry["skills"]["essential"]
        pairs = list(zip(essential["uuids"], essential["labels"], strict=False))
        skills = rng.sample(pairs, min(8, len(pairs)))
        counties = [c.get("county") for c in entry.get("counties_data") or []]
        county = rng.choice([c for c in counties if c] or [""])
        requests.append(
            MatchRequest.model_validate(
                {
                    "user_id": f"parity-{entry['occupation']['code']}",
                    "city": county,
                    "province": county,
                    "skills_vector": {
                        "top_skills": [
                            {
                                "preferredLabel": label,
                                "originUUID": uuid,
                                "proficiency": 0.8,
                            }
                            for uuid, label in skills
                        ]
                    },
                }
            )
        )
    return requests


class _InMemoryCollection:
    """Sync side for the ingest, async cursor for ``MongoOccupationsRepository``."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def find(self, filter: Mapping[str, Any], projection: object = None) -> _Cursor:
        docs = sorted(self.docs.values(), key=lambda d: d["position"])
        return _Cursor([copy.copy(d) for d in docs])

    def replace_one(
        self, filter: Mapping[str, Any], replacement: Mapping[str, Any], upsert: bool
    ) -> object:
        self.docs[filter["_id"]] = dict(replacement)
        return None

    def delete_many(self, filter: Mapping[str, Any]) -> object:
        return None


class _Cursor(list[dict[str, Any]]):
    def sort(self, spec: object) -> _Cursor:
        return self

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        async def gen() -> AsyncIterator[dict[str, Any]]:
            for d in self:
                yield d

        return gen()


def documents_repository(json_path: str, npz_path: str) -> IOccupationsRepository:
    """The file ingested in memory as ``scripts/ingest_occupations.py`` would write it."""
    from app.artifacts.repository import get_artifacts_repository
    from app.config import V4_FULL_CONCAT_WHITENING_PATH
    from app.server_dependencies.model_dependencies import get_concat_whitener
    from scripts.ingest_occupations import Whitening, run_ingest

    whitener = get_concat_whitener()
    whitening = (
        Whitening(
            whiten_rows=whitener.whiten_rows,
            artifact_sha256=get_artifacts_repository().file_sha256(
                V4_FULL_CONCAT_WHITENING_PATH
            ),
        )
        if whitener.rescale_target() > 0
        else None
    )
    collection = _InMemoryCollection()
    with open(json_path, encoding="utf-8") as f:
        occupations = parse_occupation_file(json.load(f))
    run_ingest(
        occupations,
        occupations_collection=collection,
        occupations_collection_name="occupations",
        groups_collection=None,
        groups_collection_name="occupation_groups",
        group_documents=None,
        ancestry={},
        seed=load_embeddings_npz(npz_path),
        embed=None,
        whitening=whitening,
        source={"file": Path(json_path).name},
        dry_run=False,
        prune=False,
        now=datetime.now(timezone.utc),
    )
    db = MagicMock()
    db.get_collection.return_value = collection
    return MongoOccupationsRepository(db)


def _corpus(repository: IOccupationsRepository) -> list[Occupation]:
    loader.set_occupations_repository(repository)
    rows, timing = asyncio.run(loader.get_all_occupations_with_timing())
    print(
        f"[parity] {timing.occupation_source}: {timing.n_occupation_entries} occupations, "
        f"{len(rows)} rows",
        file=sys.stderr,
    )
    return loader.attach_occupation_embeddings(rows)


def _occupation_rows(response: MatchResponse) -> list[OccupationRow]:
    return [
        OccupationRow(
            code=r.originUuid or r.uuid,
            county=r.province or "",
            final_score=float(r.final_score or 0.0),
        )
        for r in response.occupation_recommendations
    ]


def _rank(
    users: Sequence[UserProfile], corpus: Sequence[Occupation], user_vectors: np.ndarray
) -> list[MatchResponse]:
    from app.config import MATCH_V4_FINAL_TOP_K, MATCH_V4_RETRIEVE_TOP_K
    from app.matching.get_matching_service import create_matching_service
    from app.matching.service import MatchingService

    service = create_matching_service(MagicMock())
    assert isinstance(service, MatchingService)
    random.seed(_SEED)  # the occupation random-county fallback
    # Same user vectors for both runs (Gemini embeddings are not bit-reproducible).
    with patch.object(service, "embed_users", return_value=user_vectors):
        return service.rank(
            users,
            [],
            corpus,
            retrieve_top_k=MATCH_V4_RETRIEVE_TOP_K,
            final_top_k=MATCH_V4_FINAL_TOP_K,
        )


def _report(
    results: Sequence[UserParity], *, before: str, after: str, n_rows: int
) -> str:
    n_same = sum(r.same_order for r in results)
    lines = [
        "# Occupation source parity",
        "",
        f"- Before: `{before}`",
        f"- After: `{after}`",
        f"- Occupation rows per corpus: {n_rows}",
        f"- Users: {len(results)}; same occupations in the same order: **{n_same}/{len(results)}**",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| user | province | occupations | same order | max Δ final_score | top 3 (before) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        top = ", ".join(row.code for row in r.before[:3])
        lines.append(
            f"| {r.user_id} | {r.province or '—'} | {len(r.before)} / {len(r.after)} | "
            f"{'yes' if r.same_order else '**no**'} | {r.max_score_delta:.2e} | {top} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v4 occupation parity: file vs database.")
    ap.add_argument(
        "--users", type=Path, help="MatchRequest users (JSON list or JSONL)."
    )
    ap.add_argument(
        "--n-users", type=int, default=12, help="Synthetic users (default 12)."
    )
    ap.add_argument("--after", choices=("documents", "mongo"), default="documents")
    ap.add_argument(
        "--env-file", type=Path, help="Env file (MONGO_URL, GEMINI_API_KEY, …)."
    )
    ap.add_argument("--output", type=Path, help="Write the Markdown report here too.")
    args = ap.parse_args(argv)

    if args.env_file is not None:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=True)

    from app.config import OCCUPATION_CONCAT_EMBEDDINGS_PATH, occupation_json_path
    from app.server_dependencies.db_dependencies import MatchingDBProvider
    from app.server_dependencies.model_dependencies import get_gemini_embedding_client

    json_path = occupation_json_path()
    with open(json_path, encoding="utf-8") as f:
        raw_entries = json.load(f)
    requests = (
        _load_requests(args.users)
        if args.users
        else synthetic_requests(raw_entries, args.n_users)
    )
    users = [UserProfile.from_request(r) for r in requests]

    before_repo = FileOccupationsRepository(
        json_path, OCCUPATION_CONCAT_EMBEDDINGS_PATH
    )
    if args.after == "mongo":
        after_repo: IOccupationsRepository = MongoOccupationsRepository(
            asyncio.run(MatchingDBProvider.get_jobs_db())
        )
    else:
        after_repo = documents_repository(json_path, OCCUPATION_CONCAT_EMBEDDINGS_PATH)

    before_corpus = _corpus(before_repo)
    after_corpus = _corpus(after_repo)
    loader.set_occupations_repository(None)

    from app.matching.get_matching_service import create_matching_service
    from app.matching.service import MatchingService

    embedder = create_matching_service(MagicMock())
    assert isinstance(embedder, MatchingService)
    get_gemini_embedding_client().ensure_configured()
    user_vectors = embedder.embed_users(users)

    before_out = _rank(users, before_corpus, user_vectors)
    after_out = _rank(users, after_corpus, user_vectors)
    results = [
        UserParity(
            user_id=u.user_id or "",
            province=u.province,
            before=_occupation_rows(b),
            after=_occupation_rows(a),
        )
        for u, b, a in zip(users, before_out, after_out, strict=True)
    ]
    report = _report(
        results,
        before=before_repo.source,
        after=f"{after_repo.source} ({args.after})",
        n_rows=len(after_corpus),
    )
    print(report)
    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
    return 0 if all(r.same_order for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
