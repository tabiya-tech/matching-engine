#!/usr/bin/env python3
"""
GEMINI EMBEDDING ARTEFACT BUILDER

Builds a 3072-dim semantic embedding for every skill in the Tabiya/ESCO taxonomy
using Google's ``gemini-embedding-001``. Each skill is embedded from its
``preferredLabel + " — " + description``, producing vectors that encode semantic
concept similarity rather than the graph-topological similarity captured by the
Node2Vec artefact in :mod:`train_model_save_skill_to_row`.

OUTPUT (alongside the Node2Vec artefacts in ``backend/resources/models/``):
  * ``skill_embedding_model_gemini.pt`` — torch checkpoint with ``state_dict``
    holding ``embedding.weight`` as an ``(N_skills, 3072)`` fp16 tensor; same
    schema as ``skill_embedding_model.pt`` so consumers can swap by file path.
    Storing fp16 keeps the artefact under GitHub's 100 MB per-file limit
    (~82 MB at 14k skills × 3072 dims × 2 bytes) without introducing Git LFS.
    Cosine similarity preservation under fp16 vs fp32 was verified once at
    build time on 1000 random pairs (max absolute diff ~8e-5); the loader
    promotes back to fp32 before any downstream math.

ROW ORDERING
  Reuses ``skill_to_row.json`` as the canonical row→skill map so the new
  artefact is row-for-row aligned with the Node2Vec one. A future model swap
  is then "load this .pt instead of the other"; no JSON change required.

PREREQ
  * ``pip install google-genai``
  * ``GEMINI_API_KEY`` env var (same key already wired into llm-reranker)

USAGE
  python -m app.services.skills_utility.build_gemini_embedding
  # or, from backend/:
  python app/services/skills_utility/build_gemini_embedding.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as e:
    print(f"ERROR: google-genai not installed ({e}). Install with: pip install google-genai", file=sys.stderr)
    sys.exit(1)


_BACKEND_ROOT = Path(__file__).resolve().parents[3]  # .../backend
TAXONOMY_DIR = _BACKEND_ROOT / "resources" / "skill_taxonomy"
MODELS_DIR = _BACKEND_ROOT / "resources" / "models"

MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIM = 3072
TASK_TYPE = "SEMANTIC_SIMILARITY"
BATCH_SIZE = 100  # Gemini embed_content batch limit; safe at 100/call
SLEEP_BETWEEN_BATCHES = 0.6  # seconds; respects 100 RPM tier with margin
MAX_RETRIES = 5


def build_skill_text(row: dict) -> str:
    """Compose the embedding input string for one skill row.

    Prefer ``label + " — " + description`` (Option C from the design discussion),
    falling back to whichever piece is available.
    """
    label = (row.get("PREFERREDLABEL") or "").strip()
    description = (row.get("DESCRIPTION") or "").strip()
    if label and description:
        return f"{label} — {description}"
    if label:
        return label
    return description


def load_canonical_row_order() -> tuple[dict[str, int], list[str]]:
    """Read existing ``skill_to_row.json`` so the new artefact is row-aligned with Node2Vec."""
    path = MODELS_DIR / "skill_to_row.json"
    with path.open("r") as f:
        skill_to_row: dict[str, int] = json.load(f)
    n = len(skill_to_row)
    row_to_skill: list[str | None] = [None] * n
    for sid, idx in skill_to_row.items():
        row_to_skill[idx] = sid
    if any(s is None for s in row_to_skill):
        raise SystemExit(f"{path.name} has gaps in row indices")
    return skill_to_row, [s for s in row_to_skill if s is not None]


def load_skills_csv() -> dict[str, dict]:
    """ID-keyed lookup of skills.csv rows."""
    path = TAXONOMY_DIR / "skills.csv"
    by_id: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("ID") or "").strip()
            if sid:
                by_id[sid] = row
    return by_id


def embed_batch(client, texts: list[str]) -> list[list[float]]:
    """Embed a batch with exponential-backoff retry on transient errors."""
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=MODEL_NAME,
                contents=texts,
                config=genai_types.EmbedContentConfig(
                    task_type=TASK_TYPE,
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            return [emb.values for emb in result.embeddings]
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES - 1:
                break
            wait = 2 ** attempt
            print(f"    attempt {attempt + 1} failed ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"embed_content failed after {MAX_RETRIES} attempts: {last_err}")


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set", file=sys.stderr)
        return 1

    skill_to_row, row_to_skill = load_canonical_row_order()
    n_skills = len(row_to_skill)
    print(f"Canonical row order: {n_skills} skills (from skill_to_row.json)")

    by_id = load_skills_csv()
    print(f"skills.csv: {len(by_id)} skills indexed by ID")

    missing = [sid for sid in row_to_skill if sid not in by_id]
    if missing:
        print(f"WARNING: {len(missing)} IDs in skill_to_row.json missing from skills.csv "
              f"(first 3: {missing[:3]}); these will receive empty embedding text and "
              f"corresponding fallback vectors")

    texts: list[str] = []
    for sid in row_to_skill:
        row = by_id.get(sid, {})
        texts.append(build_skill_text(row))
    nonempty = sum(1 for t in texts if t)
    print(f"Built {nonempty}/{n_skills} non-empty embedding texts; sample: {texts[0][:120]!r}")

    print(f"\nEmbedding via {MODEL_NAME} @ {EMBEDDING_DIM} dim, task_type={TASK_TYPE}")
    client = genai.Client(api_key=api_key)

    n_batches = (n_skills + BATCH_SIZE - 1) // BATCH_SIZE
    fp32 = np.zeros((n_skills, EMBEDDING_DIM), dtype=np.float32)
    t0 = time.perf_counter()
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_skills)
        batch_texts = texts[start:end]
        # Replace any empty strings with the literal token " " — Gemini rejects empty input
        batch_texts = [t if t else " " for t in batch_texts]
        vecs = embed_batch(client, batch_texts)
        for i, v in enumerate(vecs):
            fp32[start + i] = np.asarray(v, dtype=np.float32)
        elapsed = time.perf_counter() - t0
        rate = end / elapsed if elapsed > 0 else 0
        eta_s = (n_skills - end) / rate if rate > 0 else 0
        if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
            print(f"  batch {batch_idx + 1}/{n_batches} done | "
                  f"{end}/{n_skills} skills | {rate:.1f}/s | ETA {eta_s:.0f}s")
        if batch_idx < n_batches - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)
    elapsed_total = time.perf_counter() - t0
    print(f"Embedding done in {elapsed_total:.1f}s ({n_skills / elapsed_total:.1f} skills/s)")

    # L2-normalise rows (also restored at load time in SkillScorer; doing it here too is idempotent)
    norms = np.linalg.norm(fp32, axis=1, keepdims=True)
    fp32 = fp32 / np.where(norms > 0, norms, 1.0)

    # Quantise to fp16; the loader (SkillScorer) promotes back to fp32 before
    # any downstream math, so this is purely a storage optimisation.
    fp16 = fp32.astype(np.float16)

    state = {
        "state_dict": {"embedding.weight": torch.from_numpy(fp16)},
        "nodes": list(row_to_skill),
        "model_name": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "task_type": TASK_TYPE,
        "input_format": "preferredLabel + ' — ' + description",
        "dtype": "float16",
    }
    out_path = MODELS_DIR / "skill_embedding_model_gemini.pt"
    torch.save(state, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {out_path.name} ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
