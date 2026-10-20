"""Offline: compute Gemini concat embeddings for occupations -> committed NPZ sidecar.

One embedding per occupation CODE (skills are identical across counties), built from the
``essential ∪ optional`` skill labels in the occupations JSON, via ``gemini-embedding-001`` —
the SAME model/space as the job embeddings used by ``/match_v4``. The text is
``app.occupations.occupation_document.occupation_embedding_text``, the same one
``scripts/ingest_occupations.py`` embeds. The NPZ is read by the file occupation source
(``OCCUPATIONS_SOURCE=file``) and seeds an ingest (``--embeddings-npz``).

Writes:
  * ``occupation_concat_embeddings.npz`` — ``codes`` (str), ``vectors`` (float32 [N, 3072], L2-normed)
  * ``occupation_concat_embeddings.meta.json`` — model, counts, source, timestamp

Prereqs: ``pip install google-genai``, ``GEMINI_API_KEY`` in ``backend/.env``.

Usage::

    cd backend
    python -m tools.embeddings.embed_occupations
    python -m tools.embeddings.embed_occupations --occupations <path> --output <npz>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv

from app.clients.gemini_embedding_client import (
    EMBEDDING_DIM,
    MODEL_NAME,
    embed_text_list,
)
from app.config import OCCUPATION_CONCAT_EMBEDDINGS_PATH, OCCUPATION_JSON_PATH
from app.occupations.occupation_document import occupation_embedding_text
from app.occupations.occupation_file import OCCUPATION_FILE
from app.ranking.retrieval import l2_normalize_rows


def build_occupation_texts(occupations_path: Path) -> tuple[List[str], List[str]]:
    """Return (codes, concat_texts) — one per unique occupation code.

    :raises pydantic.ValidationError: If the file is not a valid occupation file
    """
    entries = OCCUPATION_FILE.validate_json(occupations_path.read_bytes())

    codes: List[str] = []
    texts: List[str] = []
    seen: set[str] = set()
    for entry in entries:
        code = (entry.occupation.code or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        codes.append(code)
        texts.append(occupation_embedding_text(entry))
    return codes, texts


def run(
    occupations_path: Path, npz_out: Path, *, api_key: str, normalize: bool = True
) -> None:
    codes, texts = build_occupation_texts(occupations_path)
    n_empty = sum(1 for t in texts if not t.strip())
    print(
        f"[embed_occupations] {len(codes)} occupations from {occupations_path} "
        f"({n_empty} with no skill labels)",
        file=sys.stderr,
    )

    embeds = embed_text_list(texts, api_key=api_key)
    if embeds.shape[0] != len(codes):
        raise RuntimeError("Gemini embed returned unexpected row count")
    if normalize:
        embeds = l2_normalize_rows(embeds)

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_out,
        codes=np.array(codes, dtype=object),
        vectors=embeds.astype(np.float32),
    )
    meta_out = npz_out.with_suffix(".meta.json")
    meta_out.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "embedding_dim": EMBEDDING_DIM,
                "n_occupations": len(codes),
                "normalized": normalize,
                "source": str(occupations_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[embed_occupations] wrote {npz_out} and {meta_out}", file=sys.stderr)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description="Embed occupations via Gemini concat embeddings."
    )
    ap.add_argument(
        "--occupations", default=OCCUPATION_JSON_PATH, help="Occupations JSON path."
    )
    ap.add_argument(
        "--output", default=OCCUPATION_CONCAT_EMBEDDINGS_PATH, help="Output NPZ path."
    )
    ap.add_argument(
        "--no-normalize", action="store_true", help="Skip L2 normalisation."
    )
    args = ap.parse_args()

    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "[embed_occupations] Set GEMINI_API_KEY in backend/.env or the environment"
        )

    run(
        Path(args.occupations),
        Path(args.output),
        api_key=key,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
