#!/usr/bin/env python3
"""
WHITENED EMBEDDING ARTEFACT BUILDER

Takes an existing Gemini embedding artefact and produces a whitened sibling
that removes the anisotropic "common direction" of the ESCO skill manifold.

WHY
  Random-pair cosines on Gemini-embedding-001 over ESCO skills cluster tightly
  (mean ≈ 0.76, std ≈ 0.027) because all 13,896 skills share a strong
  work-domain prior in semantic space. Threshold gates set against Node2Vec's
  wide cosine distribution (mean 0.16, std 0.21) are functionally useless on
  this prior — 100% of random pairs exceed the 0.35 gate. Whitening rotates
  and rescales the embedding cloud so each principal axis has unit variance,
  collapsing the common-direction floor and exposing the discriminative
  signal that lives along the lower-variance axes.

METHOD
  1. Load the source artefact's W (N x D, row-normalised unit vectors).
  2. Compute mean μ_W (D-dim) and covariance Σ_W (D x D) over all rows.
  3. Apply Tikhonov shrinkage λI to Σ (λ = SHRINKAGE_LAMBDA · mean(diag(Σ)))
     before eigendecomposition; tail eigenvalues are noisy at our N/D ratio
     (~14k samples / 3072 dims) and unregularised inversion amplifies that
     noise into the whitened representation.
  4. Eigendecompose: Σ_reg = U · diag(D) · U^T.
  5. Apply transform W' = (W − μ_W) · U · diag(1/sqrt(D)).
  6. Row-normalise W' to unit length (cosine downstream is dot product).
  7. Sample 50,000 random pairs to characterise the new floor (μ', σ') for
     threshold recalibration.

OUTPUT
  Sibling artefact at the same path with ``_whitened`` suffix (e.g.
  ``skill_embedding_model_gemini_whitened.pt``). Same schema as the input —
  ``state_dict["embedding.weight"]`` is fp16 — so loaders need no change;
  the embedding dim is unchanged.

  state_dict additionally carries:
    * ``model_name``: e.g. "skill_embedding_model_gemini_whitened"
    * ``whitening``: dict with shrinkage_lambda, eigenvalue stats, and the
      random-pair (mu, std) on whitened cosines, for audit and gate-tuning.

USAGE
  python -m app.services.skills_utility.build_whitened_embedding \\
      --src resources/models/skill_embedding_model_gemini.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

SHRINKAGE_LAMBDA = 2.0  # fraction of mean(diag(Sigma)) used for Tikhonov regularisation.
# Picked by sweeping {0.05, 0.1, ..., 5.0} on five known-good vs three description-smear
# noise pairs; lambda=2.0 maximises (worst_good - best_noise) separation at +0.046, with
# random-pair floor std 0.035. Lower lambda over-amplifies low-variance directions where
# noise drowns signal; higher lambda converges back to plain centering (lower std but
# narrower margin from description-smear collisions).
N_FLOOR_PAIRS = 50_000     # for floor-characterisation (mean, std)
N_TARGET_PAIRS = 1_000_000  # for p99.9 saturation-point estimate (rescale target)
RANDOM_SEED = 42


def _load_W(path: Path) -> tuple[np.ndarray, dict]:
    state = torch.load(path, map_location="cpu")
    W = state["state_dict"]["embedding.weight"].numpy()
    if W.dtype != np.float32:
        W = W.astype(np.float32)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W = W / np.where(norms > 0, norms, 1.0)
    return W, state


def _whiten(W: np.ndarray) -> tuple[np.ndarray, dict]:
    N, D = W.shape
    print(f"  source: {N} rows x {D} dims, dtype={W.dtype}")

    mu = W.mean(axis=0).astype(np.float32)
    Wc = W - mu
    print(f"  computed mean (||mu||_2 = {np.linalg.norm(mu):.4f})")

    t0 = time.perf_counter()
    sigma = (Wc.T @ Wc) / (N - 1)
    print(f"  covariance: {(time.perf_counter()-t0)*1000:.0f} ms ({D}x{D})")

    diag_mean = float(np.mean(np.diag(sigma)))
    lam = SHRINKAGE_LAMBDA * diag_mean
    sigma_reg = sigma + lam * np.eye(D, dtype=np.float32)
    print(f"  shrinkage lambda = {SHRINKAGE_LAMBDA} * diag-mean({diag_mean:.5f}) = {lam:.6f}")

    t0 = time.perf_counter()
    eigvals, eigvecs = np.linalg.eigh(sigma_reg)
    print(f"  eigendecomposition: {(time.perf_counter()-t0)*1000:.0f} ms")
    eigvals = np.clip(eigvals, a_min=lam * 1e-3, a_max=None)  # numerical floor
    inv_sqrt = 1.0 / np.sqrt(eigvals)
    print(
        f"  eigenvalue stats: min={eigvals.min():.6f} max={eigvals.max():.6f} "
        f"top-1/bottom-1 ratio = {eigvals.max()/eigvals.min():.0f}"
    )

    t0 = time.perf_counter()
    W_white = (Wc @ eigvecs) * inv_sqrt
    print(f"  whitened transform applied: {(time.perf_counter()-t0)*1000:.0f} ms")

    norms = np.linalg.norm(W_white, axis=1, keepdims=True)
    W_white = W_white / np.where(norms > 0, norms, 1.0)
    print("  row-normalised whitened embeddings")

    rng = np.random.default_rng(RANDOM_SEED)
    # Floor characterisation (mean / std) on a small sample.
    a = rng.integers(0, N, N_FLOOR_PAIRS)
    b = rng.integers(0, N, N_FLOOR_PAIRS)
    mask = a != b
    a, b = a[mask], b[mask]
    cos_pre = (W[a] * W[b]).sum(axis=1)
    cos_post = (W_white[a] * W_white[b]).sum(axis=1)
    print(
        f"  random-pair cosine floor: pre  mean={cos_pre.mean():.4f} std={cos_pre.std():.4f}"
    )
    print(
        f"                            post mean={cos_post.mean():.4f} std={cos_post.std():.4f}"
    )

    # Saturation-point estimate (target_max_p999): p99.9 over 1M random non-identity pairs
    # of whitened cosines. This is the upper edge of the typical non-identity distribution
    # and serves as the natural rescale target at runtime — divide each rowmax by this and
    # clip at 1.0 to map the empirical [0, target] band into [0, 1] before the GM.
    print(f"  sampling {N_TARGET_PAIRS:,} pairs to estimate target_max_p999 ...")
    big_a = rng.integers(0, N, N_TARGET_PAIRS)
    big_b = rng.integers(0, N, N_TARGET_PAIRS)
    big_mask = big_a != big_b
    big_a, big_b = big_a[big_mask], big_b[big_mask]
    # Chunk the dot products to keep peak memory bounded (~ 1M × 3072 × 4 B ~ 12 GB if naive).
    chunk = 50_000
    cos_target_samples = np.empty(len(big_a), dtype=np.float32)
    for i in range(0, len(big_a), chunk):
        sl = slice(i, i + chunk)
        cos_target_samples[sl] = (W_white[big_a[sl]] * W_white[big_b[sl]]).sum(axis=1)
    target_max_p999 = float(np.quantile(cos_target_samples, 0.999))
    target_max_max = float(cos_target_samples.max())
    print(
        f"  target_max_p999 = {target_max_p999:.4f}  (literal max in sample = {target_max_max:.4f}, "
        f"n_pairs = {len(cos_target_samples):,})"
    )

    meta = {
        "shrinkage_lambda": SHRINKAGE_LAMBDA,
        "diag_mean": diag_mean,
        "eigenvalue_min": float(eigvals.min()),
        "eigenvalue_max": float(eigvals.max()),
        "eigenvalue_top1_to_bottom1": float(eigvals.max() / eigvals.min()),
        "random_pair_floor_pre": {
            "mean": float(cos_pre.mean()),
            "std": float(cos_pre.std()),
        },
        "random_pair_floor_post": {
            "mean": float(cos_post.mean()),
            "std": float(cos_post.std()),
        },
        "n_random_pairs_sampled_floor": int(len(a)),
        "target_max_p999": target_max_p999,
        "target_max_observed": target_max_max,
        "n_random_pairs_sampled_target": int(len(cos_target_samples)),
    }
    return W_white.astype(np.float32), meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        required=True,
        help="path to source embedding artefact (e.g. skill_embedding_model_gemini.pt)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output path; defaults to <src-stem>_whitened.pt next to src",
    )
    args = p.parse_args()

    src = Path(args.src).resolve()
    if not src.exists():
        raise SystemExit(f"source artefact not found: {src}")
    out = Path(args.out).resolve() if args.out else src.with_name(src.stem + "_whitened.pt")
    print(f"src: {src}")
    print(f"out: {out}\n")

    print("Loading + row-normalising source …")
    W, src_state = _load_W(src)

    print("\nWhitening …")
    W_white, meta = _whiten(W)

    new_state = {
        "state_dict": {"embedding.weight": torch.from_numpy(W_white).to(torch.float16)},
        "model_name": src.stem + "_whitened",
        "source_artefact": str(src.name),
        "whitening": meta,
    }
    print(f"\nWriting whitened artefact ({W_white.nbytes/1024/1024:.1f} MB fp32, "
          f"{W_white.nbytes/2/1024/1024:.1f} MB on disk as fp16)")
    torch.save(new_state, out)
    print(f"  -> {out}")
    print(f"  size on disk: {out.stat().st_size/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
