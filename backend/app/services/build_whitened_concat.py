"""Build a whitening transform for the COMBINED (concat) skill-set embedding used by /match_v4's
p_hat skills-fit (Design A). De-anisotropises the concat space so the cosine discriminates and is
count-neutral (see 2026-06-11 skill-eligibility notes, Empirical validation). Mirrors
build_whitened_embedding.py but for the concat (job_embedding) space.

Fits on L2-normalised concat vectors (so the transform applies consistently to the request-time
normalised user concat and the stored job/occupation concat). Saves an .npz with:
    mu      : (d,)   mean of normalised concat vectors
    W       : (d,d)  Sigma^-1/2 (Tikhonov-shrunk)
    target  : float  p99 of whitened cosine over random pairs (rescale-to-[0,1] target)

Usage (from backend/):
    python -m app.services.build_whitened_concat                # fits on data/kenya_jobs_for_pipeline.json
    python -m app.services.build_whitened_concat --jobs <path> --out <npz> --shrinkage 2.0
NOTE: refit on the FULL/live job corpus before production use (this local fit is a starting point).
"""

import argparse
import json
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
DEFAULT_JOBS = REPO / "data" / "kenya_jobs_for_pipeline.json"
DEFAULT_OUT = BACKEND / "resources" / "models" / "concat_whitening_gemini.npz"


def _l2(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n > 0, n, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--shrinkage", type=float, default=2.0, help="Tikhonov lambda (× mean diag)"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    vecs = [
        j["job_embedding"] for j in data if isinstance(j.get("job_embedding"), list)
    ]
    if not vecs:
        raise SystemExit(f"No job_embedding vectors in {args.jobs}")
    X = _l2(np.asarray(vecs, dtype=np.float64))
    n, d = X.shape
    print(f"fitting concat whitening on {n} normalised job concat vectors, dim={d}")

    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / n
    dm = float(np.mean(np.diag(Sigma)))
    Sigma += (
        args.shrinkage * dm * np.eye(d)
    )  # shrinkage (cov underdetermined at d≈3072)
    vals, vecsZ = np.linalg.eigh(Sigma)
    W = (vecsZ * (1.0 / np.sqrt(np.maximum(vals, 1e-12)))) @ vecsZ.T  # Sigma^-1/2

    Xw = _l2(((X - mu) @ W))
    # rescale target = p99 of whitened cosine over random pairs (matches the per-skill convention)
    ia = rng.integers(0, n, size=min(200_000, n * n))
    ib = rng.integers(0, n, size=ia.shape[0])
    mask = ia != ib
    cos = np.sum(Xw[ia[mask]] * Xw[ib[mask]], axis=1)
    target = float(np.percentile(cos, 99))
    print(
        f"random-pair whitened cosine: mean={cos.mean():.4f} p99(target)={target:.4f} max={cos.max():.4f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        mu=mu.astype(np.float32),
        W=W.astype(np.float32),
        target=np.float32(target),
        dim=np.int64(d),
        n_fit=np.int64(n),
    )
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
