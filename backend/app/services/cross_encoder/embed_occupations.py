"""Offline: compute Gemini concat embeddings for occupations -> committed NPZ sidecar.

One embedding per occupation CODE (skills are identical across counties), built from the
``essential ∪ optional`` skill labels in the occupations JSON, via ``gemini-embedding-001`` —
the SAME model/space as the job embeddings used by ``/match_v4``. Output is loaded at runtime by
``app.database`` to feed the occupation arm of the ``/match_v4`` retrieval.

Writes:
  * ``occupation_concat_embeddings.npz`` — ``codes`` (str), ``vectors`` (float32 [N, 3072], L2-normed)
  * ``occupation_concat_embeddings.meta.json`` — model, counts, source, timestamp

Prereqs: ``pip install google-genai``, ``GEMINI_API_KEY`` in ``backend/.env``.

Usage::

    cd backend
    python -m app.services.cross_encoder.embed_occupations
    python -m app.services.cross_encoder.embed_occupations --occupations <path> --output <npz>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv

from app.config import OCCUPATION_JSON_PATH, OCCUPATION_CONCAT_EMBEDDINGS_PATH
from app.services.cross_encoder.concat_embedding_text import build_concat_embedding_text
from app.services.cross_encoder.gemini_embeddings import (
    EMBEDDING_DIM,
    MODEL_NAME,
    embed_text_list,
    l2_normalize_rows,
)


def _occupation_skill_labels(entry: Dict[str, Any]) -> List[str]:
    """Essential ∪ optional skill labels for one occupation entry (from the JSON)."""
    skills = entry.get("skills", {}) or {}
    out: List[str] = []
    for grp in ("essential", "optional"):
        block = skills.get(grp) or {}
        labels = block.get("labels") if isinstance(block, dict) else None
        if isinstance(labels, list):
            out.extend(str(lab) for lab in labels if lab)
    return out


def build_occupation_texts(occupations_path: Path) -> tuple[List[str], List[str]]:
    """Return (codes, concat_texts) — one per unique occupation code."""
    raw = json.loads(occupations_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{occupations_path}: expected a JSON list of occupation entries")

    codes: List[str] = []
    texts: List[str] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        occ = entry.get("occupation", {}) or {}
        code = str(occ.get("code") or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        text = build_concat_embedding_text(_occupation_skill_labels(entry))
        codes.append(code)
        texts.append(text if text else " ")
    return codes, texts


def run(occupations_path: Path, npz_out: Path, *, api_key: str, normalize: bool = True) -> None:
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
    ap = argparse.ArgumentParser(description="Embed occupations via Gemini concat embeddings.")
    ap.add_argument("--occupations", default=OCCUPATION_JSON_PATH, help="Occupations JSON path.")
    ap.add_argument("--output", default=OCCUPATION_CONCAT_EMBEDDINGS_PATH, help="Output NPZ path.")
    ap.add_argument("--no-normalize", action="store_true", help="Skip L2 normalisation.")
    args = ap.parse_args()

    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise SystemExit("[embed_occupations] Set GEMINI_API_KEY in backend/.env or the environment")

    run(
        Path(args.occupations),
        Path(args.output),
        api_key=key,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
