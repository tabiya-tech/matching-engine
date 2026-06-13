"""Compare two /match_v4 runs (Phase-1 baseline vs Phase-2 demotion) to verify Phase 2.

Checks, per user, over opportunity_recommendations (and occupations):
  1. baseline: final == u_hat * p_hat (Phase-1 invariant; demand-tilted for occ)
  2. demote:   final == u_hat * p_hat * coverage**gamma  (p_hat = whitened concat)
  3. ranking shift: which items moved, and whether low-coverage items demoted
  4. corr(rank, coverage) tightens under demotion

Usage:  python compare_p2.py <baseline_response.json> <demote_response.json>
"""
import json
import sys
from pathlib import Path


def load(p):
    return {r["user_id"]: r for r in json.loads(Path(p).read_text(encoding="utf-8"))}


def by_uuid(rows):
    return {str(r.get("uuid")): r for r in rows}


def spearman(pairs):
    # pairs: list of (rank, coverage); rank correlation via simple Pearson on ranks of coverage
    if len(pairs) < 3:
        return None
    import statistics
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    return cov / (sx * sy) if sx and sy else None


def main():
    base = load(sys.argv[1])
    dem = load(sys.argv[2])
    users = sorted(set(base) & set(dem))
    print(f"users compared: {len(users)}\n")

    for key, occ in (("opportunity_recommendations", False), ("occupation_recommendations", True)):
        print(f"==== {key} ====")
        inv_base_bad = inv_dem_bad = 0
        n_items = 0
        all_moves = []
        cov_rank_base = []
        cov_rank_dem = []
        demotions = []  # (user, title, cov, base_rank, dem_rank, base_final, dem_final)
        for u in users:
            b = by_uuid(base[u].get(key) or [])
            d = by_uuid(dem[u].get(key) or [])
            # invariant checks
            for uuid, r in b.items():
                sb = r.get("score_breakdown") or {}
                uh, ph, fs = sb.get("u_hat"), sb.get("p_hat"), r.get("final_score")
                if uh is not None and ph is not None and not occ:
                    if abs(uh * ph - fs) > 0.01:
                        inv_base_bad += 1
            for uuid, r in d.items():
                sb = r.get("score_breakdown") or {}
                uh, ph = sb.get("u_hat"), sb.get("p_hat")
                cov = sb.get("essential_coverage")
                fs = r.get("final_score")
                n_items += 1
                if uh is not None and ph is not None and cov is not None and not occ:
                    expected = uh * ph * cov  # gamma=1
                    if abs(expected - fs) > 0.01:
                        inv_dem_bad += 1
                cov_rank_dem.append((r.get("rank"), cov if cov is not None else 1.0))
            for uuid, r in b.items():
                sb = r.get("score_breakdown") or {}
                cov_rank_base.append((r.get("rank"), sb.get("essential_coverage") if sb.get("essential_coverage") is not None else 1.0))
            # ranking moves
            for uuid in set(b) & set(d):
                rb, rd = b[uuid].get("rank"), d[uuid].get("rank")
                all_moves.append(rd - rb)
                cov = (d[uuid].get("score_breakdown") or {}).get("essential_coverage")
                if rd - rb >= 2 and (cov is not None and cov < 0.5):
                    demotions.append((u, b[uuid].get("opportunity_title") or b[uuid].get("occupation_label"),
                                      cov, rb, rd, b[uuid].get("final_score"), d[uuid].get("final_score")))
        print(f"  items: {n_items}  invariant violations: baseline={inv_base_bad} demote={inv_dem_bad}")
        if all_moves:
            moved = sum(1 for m in all_moves if m != 0)
            print(f"  rank moves: {moved}/{len(all_moves)} items changed rank; "
                  f"max drop={max(all_moves)} max rise={min(all_moves)}")
        print(f"  corr(rank,coverage) baseline={spearman(cov_rank_base)}  demote={spearman(cov_rank_dem)}")
        demotions.sort(key=lambda t: t[4] - t[3], reverse=True)
        for row in demotions[:8]:
            print(f"    demoted: user={row[0][:8]} cov={row[2]:.2f} rank {row[3]}->{row[4]} "
                  f"final {row[5]:.4f}->{row[6]:.4f}  {row[1]}")
        print()


if __name__ == "__main__":
    main()
