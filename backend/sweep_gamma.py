"""Report Phase-2 demotion quality across gamma settings.

For each run (baseline + gamma variants), over opportunity_recommendations and
occupation_recommendations, reports:
  - corr(rank, coverage): how strongly achievability drives ordering (more negative = stronger)
  - mean essential_coverage of top-1 / top-5 / all: are achievable items surfaced at the top?
  - weak@top5: avg # of items with coverage<0.4 in each user's top-5 (lower = fewer unachievable at top)
  - meanfinal@top5: avg final_score of top-5 (watch over-crushing)

Usage: python sweep_gamma.py <label=path> <label=path> ...
       e.g.  python sweep_gamma.py base=runs/p2_baseline/.../r.json g1.0=runs/.../r.json
"""
import json
import sys
from pathlib import Path


def load(p):
    return {r["user_id"]: r for r in json.loads(Path(p).read_text(encoding="utf-8"))}


def pearson(pairs):
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    xs = [a for a, _ in pairs]; ys = [b for _, b in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else None


def cov_of(r):
    c = (r.get("score_breakdown") or {}).get("essential_coverage")
    return 1.0 if c is None else c


def report(label, data, key):
    rank_cov = []
    top1 = []; top5 = []; allc = []; weak5 = []; final5 = []
    for u, rec in data.items():
        rows = sorted(rec.get(key) or [], key=lambda r: r.get("rank") or 999)
        if not rows:
            continue
        for r in rows:
            rank_cov.append((r.get("rank"), cov_of(r)))
            allc.append(cov_of(r))
        top1.append(cov_of(rows[0]))
        t5 = rows[:5]
        top5.append(sum(cov_of(r) for r in t5) / len(t5))
        weak5.append(sum(1 for r in t5 if cov_of(r) < 0.4))
        final5.append(sum((r.get("final_score") or 0) for r in t5) / len(t5))
    def avg(x): return sum(x)/len(x) if x else float("nan")
    pc = pearson(rank_cov)
    print(f"  {label:8s}  corr(rank,cov)={pc:+.3f}  cov@1={avg(top1):.3f}  "
          f"cov@5={avg(top5):.3f}  cov@all={avg(allc):.3f}  weak@5={avg(weak5):.2f}  "
          f"final@5={avg(final5):.3f}")


def main():
    runs = []
    for a in sys.argv[1:]:
        label, path = a.split("=", 1)
        runs.append((label, load(path)))
    for key in ("opportunity_recommendations", "occupation_recommendations"):
        print(f"==== {key} ====")
        for label, data in runs:
            report(label, data, key)
        print()


if __name__ == "__main__":
    main()
