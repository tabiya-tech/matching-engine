"""Embed concat user/job texts from export JSON via Gemini ``gemini-embedding-001``.

Reads :mod:`export_concat_text_users_jobs` output (``concat_embedding_text`` fields),
calls ``google.genai`` in batches, L2-normalises rows by default, and writes:

  * ``<stem>_gemini.npz`` — ``user_ids``, ``user_embeds``, ``job_uuids``, ``job_embeds`` (float32)
  * ``<stem>_gemini_meta.json`` — model, source path, counts, timestamps

With ``--mongo-jobs``, job vectors are also written to Mongo (``MONGO_*`` in ``backend/.env``)
as ``concat_skill_embedding_gemini`` on each matching document in ``MONGO_JOBS_COLLECTION``.
The NPZ is still written (the cosine / CE pipeline loads it offline).

Prereqs: ``pip install google-genai``, ``GEMINI_API_KEY`` in ``backend/.env``.

Usage::

    cd backend
    python -m app.services.cross_encoder.gemini_embeddings \\
        --input experiments/concat_text/njila_users_jobs_concat.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv

MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIM = 3072
TASK_TYPE = "SEMANTIC_SIMILARITY"
# embed_content rejects more than ~100 texts per API call (ClientError / 400).
GEMINI_EMBED_API_MAX_BATCH = 100
DEFAULT_BATCH_SIZE = 100
DEFAULT_SLEEP_S = 0.12
MAX_RETRIES = 5


def _backend_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "backend":
            return parent
    return here.parents[3]


def load_concat_export(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    users = raw.get("users") or []
    jobs = raw.get("jobs") or []
    if not isinstance(users, list) or not isinstance(jobs, list):
        raise ValueError(f"{path}: expected JSON with users[] and jobs[] lists")
    return users, jobs


def _concat_text_nonempty(row: Dict[str, Any]) -> str:
    txt = str(row.get("concat_embedding_text") or "").strip()
    return txt if txt else " "


def _npz_job_key_from_row(row: Dict[str, Any], job_key: Literal["uuid", "fingerprint"]) -> str:
    """Stable row id stored in NPZ ``job_uuids`` (historical name — may hold fingerprint hex)."""

    jk = job_key.strip().lower()
    if jk == "fingerprint":
        fp = str(row.get("job_fingerprint") or "").strip()
        if not fp:
            raise ValueError(
                "job_key=fingerprint requires non-empty job_fingerprint on each job row "
                "(re-run export_concat_text_users_jobs so jobs include job_fingerprint)."
            )
        return fp
    jid = (
        str(row.get("job_uuid") or "")
        or str(row.get("uuid") or "")
        or str(row.get("_id") or "")
    )
    return jid


def _embed_chunk(
    client: Any,
    texts: Sequence[str],
    *,
    model: str,
    embedding_dim: int,
    task_type: str,
) -> np.ndarray:
    from google.genai import types as genai_types

    payload = list(texts)
    last_err: Optional[BaseException] = None
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=model,
                contents=payload,
                config=genai_types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=embedding_dim,
                ),
            )
            rows = []
            for emb in result.embeddings:
                rows.append(np.asarray(emb.values, dtype=np.float32))
            return np.stack(rows, axis=0)
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES - 1:
                break
            wait = 2 ** attempt
            detail = str(e).strip() or repr(e)
            if len(detail) > 280:
                detail = detail[:277] + "…"
            print(
                f"    batch embed attempt {attempt + 1}/{MAX_RETRIES} failed "
                f"({type(e).__name__}): {detail}; sleep {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"Gemini embed_content failed after {MAX_RETRIES} attempts: {last_err}")


def embed_text_list(
    texts: List[str],
    *,
    api_key: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sleep_s: float = DEFAULT_SLEEP_S,
    model: str = MODEL_NAME,
    embedding_dim: int = EMBEDDING_DIM,
    task_type: str = TASK_TYPE,
) -> np.ndarray:
    from google import genai

    if not texts:
        return np.zeros((0, embedding_dim), dtype=np.float32)
    cap = GEMINI_EMBED_API_MAX_BATCH
    if batch_size > cap:
        print(
            f"[gemini_embeddings] batch_size={batch_size} exceeds API max ({cap}); "
            f"using {cap} (see Gemini embed limits).",
            file=sys.stderr,
        )
        batch_size = cap
    client = genai.Client(api_key=api_key.strip())
    n = len(texts)
    safe = [t.strip() if t.strip() else " " for t in texts]
    out = np.zeros((n, embedding_dim), dtype=np.float32)
    t0 = time.perf_counter()
    n_batches = (n + batch_size - 1) // batch_size
    report_every = max(1, (n_batches + 9) // 10)
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n)
        chunk = safe[start:end]
        out[start:end] = _embed_chunk(
            client, chunk, model=model, embedding_dim=embedding_dim, task_type=task_type
        )
        nb = b + 1
        if nb == n_batches or nb % report_every == 0:
            print(f"    … {end}/{n} texts (batch {nb}/{n_batches})", file=sys.stderr)
        if b < n_batches - 1 and sleep_s > 0:
            time.sleep(sleep_s)
    elapsed = time.perf_counter() - t0
    print(f"[gemini_embeddings] embedded {n} texts in {elapsed:.1f}s ({n/elapsed:.1f}/s)", file=sys.stderr)
    return out


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms > 0, norms, 1.0).astype(np.float32)
    return (mat / norms).astype(np.float32)


def run(
    input_path: Path,
    npz_out: Path,
    meta_out: Path,
    *,
    batch_size: int,
    sleep_s: float,
    normalize: bool,
    api_key: Optional[str],
    job_key: Literal["uuid", "fingerprint"] = "uuid",
    mongo_jobs: bool = False,
    mongo_match_field: str = "job_id",
    mongo_bulk_chunk: int = 500,
) -> None:
    inp = input_path.expanduser().resolve()
    users, jobs = load_concat_export(inp)

    user_ids: List[str] = []
    user_texts: List[str] = []
    for row in users:
        user_ids.append(str(row.get("user_id") or ""))
        user_texts.append(_concat_text_nonempty(row))

    job_ids: List[str] = []
    job_texts: List[str] = []
    for row in jobs:
        job_ids.append(_npz_job_key_from_row(row, job_key))
        job_texts.append(_concat_text_nonempty(row))

    key = (api_key or "").strip()
    if not key:
        key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "[gemini_embeddings] Set GEMINI_API_KEY in backend/.env or pass GEMINI_API_KEY in the environment"
        )

    print(
        f"[gemini_embeddings] users={len(user_texts)} jobs={len(job_texts)} model={MODEL_NAME}",
        file=sys.stderr,
    )
    print("[gemini_embeddings] embedding user texts…", file=sys.stderr)
    u_emb = embed_text_list(user_texts, api_key=key, batch_size=batch_size, sleep_s=sleep_s)
    print("[gemini_embeddings] embedding job texts…", file=sys.stderr)
    j_emb = embed_text_list(job_texts, api_key=key, batch_size=batch_size, sleep_s=sleep_s)

    if normalize:
        u_emb = l2_normalize_rows(u_emb)
        j_emb = l2_normalize_rows(j_emb)

    npz_out = npz_out.expanduser().resolve()
    meta_out = meta_out.expanduser().resolve()
    npz_out.parent.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "source_json": str(inp),
        "output_npz": str(npz_out),
        "model": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "task_type": TASK_TYPE,
        "l2_normalized_rows": bool(normalize),
        "n_users": int(u_emb.shape[0]),
        "n_jobs": int(j_emb.shape[0]),
        "batch_size": batch_size,
        "npz_job_key": job_key,
    }

    if mongo_jobs:
        from app.services.two_stage_retrieval.gemini_vs_minilm.sync_gemini_embeddings_to_mongo import (
            bulk_write_concat_gemini_job_embeddings,
            _mongo_client_and_collection,
        )

        nk = str(job_key).strip().lower()
        mfm = (mongo_match_field or "job_id").strip().lower().replace("-", "_")
        if nk == "fingerprint" and mfm != "job_fingerprint":
            print(
                "[gemini_embeddings] warning: --job-key fingerprint but "
                "--mongo-match-field should be job_fingerprint for typical Mongo rows",
                file=sys.stderr,
            )
        if nk == "uuid" and mfm == "job_fingerprint":
            print(
                "[gemini_embeddings] warning: --job-key uuid but --mongo-match-field is job_fingerprint",
                file=sys.stderr,
            )
        print(f"[gemini_embeddings] writing job embeddings to Mongo (match {mfm})…", file=sys.stderr)
        client, coll = _mongo_client_and_collection()
        try:
            mj = np.asarray(job_ids, dtype=object)
            matched, modified, ops = bulk_write_concat_gemini_job_embeddings(
                coll,
                mj,
                j_emb,
                meta,
                match_field=mongo_match_field,
                bulk_chunk=mongo_bulk_chunk,
            )
            print(
                f"[gemini_embeddings] mongo: matched={matched} modified={modified} updates_sent={ops}",
                file=sys.stderr,
            )
            if matched < ops:
                print(
                    "[gemini_embeddings] warning: some job ids did not match a Mongo document "
                    "(embeddings are not upserted; only existing jobs are updated)",
                    file=sys.stderr,
                )
        finally:
            client.close()

    np.savez_compressed(
        npz_out,
        user_ids=np.asarray(user_ids, dtype=object),
        user_embeds=u_emb.astype(np.float32),
        job_uuids=np.asarray(job_ids, dtype=object),
        job_embeds=j_emb.astype(np.float32),
    )

    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[gemini_embeddings] wrote {npz_out}", file=sys.stderr)
    print(f"[gemini_embeddings] wrote {meta_out}", file=sys.stderr)


def default_outputs(input_path: Path) -> Tuple[Path, Path]:
    stem = input_path.resolve().stem
    base = input_path.resolve().parent / "embeddings"
    return base / f"{stem}_gemini.npz", base / f"{stem}_gemini_meta.json"


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv(_backend_root() / ".env", override=False)

    def_inp = _backend_root() / "experiments/concat_text/njila_users_jobs_concat.json"

    p = argparse.ArgumentParser(description="Gemini embeddings for concat user/job texts export JSON.")
    p.add_argument(
        "--input",
        type=Path,
        nargs="?",
        default=None,
        help=f"Concat export JSON (default: {def_inp} if it exists)",
    )
    p.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Output .npz (default: <input_dir>/embeddings/<stem>_gemini.npz).",
    )
    p.add_argument(
        "--output-meta",
        type=Path,
        default=None,
        help="Sidecar meta JSON path (default: next to npz _meta.json).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Texts per embed_content call (capped at {GEMINI_EMBED_API_MAX_BATCH}; default {DEFAULT_BATCH_SIZE}).",
    )
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_S)
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip per-row L2 normalisation.",
    )

    p.add_argument(
        "--job-key",
        choices=("uuid", "fingerprint"),
        default="uuid",
        help=(
            "Which export field identifies each job row in the NPZ job_uuids array: "
            "uuid matches Mongo job_id string form (see sync --mongo-match-field); "
            "fingerprint uses job_fingerprint (stable hex on ranked job docs)."
        ),
    )
    p.add_argument(
        "--mongo-jobs",
        action="store_true",
        help=(
            "After embedding, write job vectors to Mongo as concat_skill_embedding_gemini "
            "(requires MONGO_URL, MONGO_DB_NAME, MONGO_JOBS_COLLECTION in backend/.env). "
            "NPZ/meta are still written for the cosine and cross-encoder pipeline."
        ),
    )
    p.add_argument(
        "--mongo-match-field",
        choices=("job_id", "job_fingerprint"),
        default="job_id",
        help="Mongo field to match export job ids against when --mongo-jobs is set.",
    )
    p.add_argument(
        "--mongo-bulk-chunk",
        type=int,
        default=500,
        help="Mongo bulk_write chunk size when --mongo-jobs is set.",
    )

    args = p.parse_args(argv)
    inp = args.input
    if inp is None:
        inp = def_inp if def_inp.is_file() else None

    cd = Path.cwd()
    if inp is None:
        print(
            "[gemini_embeddings] pass --input path/to/export.json "
            "(or place njila export at experiments/concat_text/njila_users_jobs_concat.json).",
            file=sys.stderr,
        )
        return 2

    inp = Path(inp)
    inp = inp.resolve() if inp.is_absolute() else (cd / inp).resolve()

    if not inp.is_file():
        print(f"[gemini_embeddings] input not found: {inp}", file=sys.stderr)
        return 2

    def_npz, def_meta = default_outputs(inp)
    npz_out = Path(args.output_npz) if args.output_npz else def_npz
    meta_out = Path(args.output_meta) if args.output_meta else def_meta
    try:
        if not Path(npz_out).is_absolute():
            npz_out = (cd / npz_out).resolve()
        if not Path(meta_out).is_absolute():
            meta_out = (cd / meta_out).resolve()
    except Exception:
        pass

    try:
        run(
            input_path=inp,
            npz_out=npz_out,
            meta_out=meta_out,
            batch_size=max(1, int(args.batch_size)),
            sleep_s=float(args.sleep),
            normalize=not args.no_normalize,
            api_key=None,
            job_key=args.job_key,
            mongo_jobs=bool(args.mongo_jobs),
            mongo_match_field=args.mongo_match_field,
            mongo_bulk_chunk=max(1, int(args.mongo_bulk_chunk)),
        )
    except ImportError as e:
        print(f"[gemini_embeddings] {e}; install: pip install google-genai", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
