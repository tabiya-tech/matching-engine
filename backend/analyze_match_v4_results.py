"""
Analyse a /match_v4 local run: recommendation diversity, the skills↔preferences balance, and
whether recommended jobs/careers are actually achievable for users.

Reads a run directory produced by run_match_v4_local.py (defaults to the most recent under
output_results/match_v4_local/). Pure stdlib — no torch/Gemini/Mongo.

What it checks
--------------
1. DIVERSITY / CONCENTRATION (are we recommending the same things to everyone?)
   - How often each job/career lands in users' top-K (top-1/5/10), the most "ubiquitous" items,
     coverage, and concentration metrics (HHI, Gini, normalised entropy), plus the mean pairwise
     Jaccard overlap between users' top-K sets. High overlap / concentration ⇒ "same to everyone".

2. SKILLS ↔ PREFERENCES BALANCE
   - In v4, final_score = combine(u_hat, p_hat) where u_hat = preference utility and
     p_hat = concat-cosine (whole-profile skills/semantic match). We report the spread of each,
     a score-compression flag (if everything sits in a narrow band the ranking is barely
     differentiated), which factor actually varies within a user (= the ranking driver), and
     which factor is the binding (smaller) one.

3. ACHIEVABILITY (can the user actually do / get this?)
   - Per rec, essential_fit = share of the item's essential skills the user clears the cosine
     threshold on (from matched_skills.essential_skill_matches). We flag recommendations with low
     essential_fit that are still ranked highly (aspirational/unachievable), the is_eligible share,
     and whether better-ranked items are actually more achievable (rank↔essential_fit).

Drill-down
----------
  --explain TOKEN     auto-detects: a user_id → that user's top recs; otherwise an item uuid or a
                      title/label substring → every (user, item) pairing with the score breakdown.
  --user U --item Q   explain one (user, item) pair in full (matched vs unmet essential skills,
                      u_hat/p_hat, preferences).

Usage (from backend/):
    python analyze_match_v4_results.py
    python analyze_match_v4_results.py <run_dir> --top-k 5
    python analyze_match_v4_results.py --explain "Full stack developer"
    python analyze_match_v4_results.py --explain 6a27a1d40c73458a0b453373
    python analyze_match_v4_results.py --user 33FOgG6rG5aRY0PuvaFXqwFLxnp1 --item "developer"
"""

import argparse
import csv
import itertools
import json
import math
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Windows consoles default to cp1252, which can't encode the arrows/symbols used below.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue] — real stdout has it; the except below covers stand-ins that don't
except Exception:
    pass

BACKEND_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_ROOT.parent
DEFAULT_RUN_ROOT = REPO_ROOT / "output_results" / "match_v4_local"

OPP = ("opportunity_recommendations", "opportunity_title", "Opportunities (jobs)")
OCC = ("occupation_recommendations", "occupation_label", "Careers (occupations)")


# ── helpers ────────────────────────────────────────────────────────────────
def _latest_run(root: Path) -> Path:
    runs = [p for p in root.glob("*") if (p / "match_v4_response.json").is_file()]
    if not runs:
        raise SystemExit(f"No runs with match_v4_response.json under {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def _item_id(rec: dict, rec_key: str) -> str:
    # Careers: key by occupation code (originUuid) so the same career across counties collapses.
    if rec_key == OCC[0]:
        return str(rec.get("originUuid") or rec.get("uuid") or "")
    return str(rec.get("uuid") or "")


def _sb(rec: dict, key: str):
    return (rec.get("score_breakdown") or {}).get(key)


def _essential_fit(rec: dict):
    """(met, total, met/total) over the item's essential skills; (0,0,None) if none."""
    em = (rec.get("matched_skills") or {}).get("essential_skill_matches") or []
    total = len(em)
    if not total:
        return 0, 0, None
    met = sum(1 for m in em if m.get("meets_threshold"))
    return met, total, met / total


def _gini(values) -> float:
    xs = sorted(values)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * s) - (n + 1) / n


def _norm_entropy(counts) -> float:
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    ps = [c / total for c in counts]
    h = -sum(p * math.log(p) for p in ps if p > 0)
    return h / math.log(len(counts))


def _mean_pairwise_jaccard(sets) -> float:
    vals = []
    for a, b in itertools.combinations(sets, 2):
        union = len(a | b)
        vals.append(len(a & b) / union if union else 0.0)
    return stats.mean(vals) if vals else 0.0


def _fmt_stats(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return "n/a"
    return (
        f"mean={stats.mean(xs):.3f} sd={(stats.pstdev(xs) if len(xs) > 1 else 0.0):.3f} "
        f"min={min(xs):.3f} max={max(xs):.3f} range={max(xs) - min(xs):.3f}"
    )


# ── 1. diversity / concentration ────────────────────────────────────────────
def analyse_diversity(users, rec_key, label_key, title, k, report):
    n_users = len(users)
    topk_sets = []
    appear = Counter()  # item -> # users with it in top-K
    label_of = {}
    rank_when_appears = defaultdict(list)
    for u in users:
        recs = sorted(u.get(rec_key) or [], key=lambda r: r.get("rank") or 1e9)[:k]
        ids = set()
        for r in recs:
            iid = _item_id(r, rec_key)
            if not iid:
                continue
            ids.add(iid)
            label_of[iid] = r.get(label_key) or iid
            rank_when_appears[iid].append(r.get("rank"))
        topk_sets.append(ids)
        appear.update(ids)

    total_slots = sum(len(s) for s in topk_sets)
    distinct = len(appear)
    counts = list(appear.values())
    ubiquitous = [(iid, c) for iid, c in appear.most_common() if c >= 0.5 * n_users]
    most = appear.most_common(10)

    sect = {
        "n_users": n_users,
        "top_k": k,
        "distinct_items_in_topk": distinct,
        "total_topk_slots": total_slots,
        "coverage_distinct_over_slots": round(distinct / total_slots, 3)
        if total_slots
        else None,
        "mean_pairwise_jaccard_topk": round(_mean_pairwise_jaccard(topk_sets), 3),
        "hhi": round(sum((c / total_slots) ** 2 for c in counts), 4)
        if total_slots
        else None,
        "gini_of_appearance_counts": round(_gini(counts), 3),
        "normalised_entropy": round(_norm_entropy(counts), 3),
        "most_recommended": [
            {
                "id": iid,
                "label": label_of.get(iid, iid),
                "n_users_topk": c,
                "pct_users": round(100 * c / n_users, 1),
                "mean_rank": round(
                    stats.mean([x for x in rank_when_appears[iid] if x]), 1
                ),
            }
            for iid, c in most
        ],
        "n_items_in_majority_of_users": len(ubiquitous),
    }
    report[rec_key] = sect

    print(f"\n=== 1. DIVERSITY — {title} (top-{k}) ===")
    print(
        f"  users={n_users}  distinct items in top-{k}={distinct}  "
        f"(of {total_slots} slots → coverage {sect['coverage_distinct_over_slots']})"
    )
    print(
        f"  mean pairwise Jaccard of top-{k} sets: {sect['mean_pairwise_jaccard_topk']}  "
        f"(0=all different, 1=identical)"
    )
    print(
        f"  concentration: HHI={sect['hhi']}  Gini={sect['gini_of_appearance_counts']}  "
        f"entropy(norm)={sect['normalised_entropy']}  (HHI/Gini↑ & entropy↓ = same-to-everyone)"
    )
    if ubiquitous:
        print(f"  ⚠ {len(ubiquitous)} item(s) appear in ≥50% of users' top-{k}:")
        for iid, c in ubiquitous[:10]:
            print(
                f"      {100 * c / n_users:5.1f}% ({c}/{n_users})  {label_of.get(iid, iid)}"
            )
    print(f"  most-recommended {title.lower()}:")
    for m in sect["most_recommended"][:8]:
        print(
            f"      {m['pct_users']:5.1f}% ({m['n_users_topk']}/{n_users})  "
            f"mean_rank={m['mean_rank']:<4}  {m['label']}"
        )


# ── 2. skills↔preferences balance ───────────────────────────────────────────
def analyse_balance(users, report):
    print("\n=== 2. SKILLS ↔ PREFERENCES BALANCE ===")
    print(
        "  (final_score = combine(u_hat, p_hat); u_hat = preference utility, "
        "p_hat = concat-cosine whole-profile skills match)"
    )
    for rec_key, _label, title in (OPP, OCC):
        u_all, p_all, f_all = [], [], []
        within_u_sd, within_p_sd = [], []
        binding_u = binding_p = 0
        for u in users:
            recs = u.get(rec_key) or []
            us = [v for r in recs if (v := _sb(r, "u_hat")) is not None]
            ps = [v for r in recs if (v := _sb(r, "p_hat")) is not None]
            u_all += us
            p_all += ps
            f_all += [
                r.get("final_score") for r in recs if r.get("final_score") is not None
            ]
            if len(us) > 1:
                within_u_sd.append(stats.pstdev(us))
            if len(ps) > 1:
                within_p_sd.append(stats.pstdev(ps))
            for r in recs:
                uh, ph = _sb(r, "u_hat"), _sb(r, "p_hat")
                if uh is None or ph is None:
                    continue
                if uh <= ph:
                    binding_u += 1
                else:
                    binding_p += 1
        n = len(u_all)
        mean_u_sd = stats.mean(within_u_sd) if within_u_sd else 0.0
        mean_p_sd = stats.mean(within_p_sd) if within_p_sd else 0.0
        driver = (
            "u_hat (preferences)" if mean_u_sd > mean_p_sd else "p_hat (skills/cosine)"
        )
        compressed = []
        if u_all and (max(u_all) - min(u_all)) < 0.10:
            compressed.append("u_hat")
        if p_all and (max(p_all) - min(p_all)) < 0.10:
            compressed.append("p_hat")
        sect = {
            "n_recs": n,
            "u_hat": _fmt_stats(u_all),
            "p_hat": _fmt_stats(p_all),
            "final_score": _fmt_stats(f_all),
            "mean_within_user_sd_u_hat": round(mean_u_sd, 4),
            "mean_within_user_sd_p_hat": round(mean_p_sd, 4),
            "ranking_driver": driver,
            "binding_factor_u_hat_le_p_hat": binding_u,
            "binding_factor_p_hat_lt_u_hat": binding_p,
            "compressed_factors": compressed,
        }
        report.setdefault("balance", {})[rec_key] = sect
        print(f"\n  {title}:  ({n} recs)")
        print(f"    u_hat (preferences): {sect['u_hat']}")
        print(f"    p_hat (skills cos) : {sect['p_hat']}")
        print(
            f"    within-user spread → u_hat sd={mean_u_sd:.4f}  p_hat sd={mean_p_sd:.4f}  "
            f"→ ranking driven by {driver}"
        )
        print(
            f"    binding (smaller) factor: u_hat in {binding_u}, p_hat in {binding_p} of recs"
        )
        if compressed:
            print(
                f"    ⚠ LOW SPREAD in {', '.join(compressed)} (range < 0.10) — rankings barely "
                f"differentiated; small score gaps may not be meaningful"
            )


# ── 3. achievability ─────────────────────────────────────────────────────────
def _ess_cos(rec):
    """Continuous essential-skill fit: mean best cosine over the item's essential skills."""
    v = _sb(rec, "total_skill_utility")
    if v is None:
        v = (_sb(rec, "skill_components") or {}).get("ess")
    return v


def analyse_achievability(users, report, k, min_ess_cos=0.78):
    print(
        "\n=== 3. ACHIEVABILITY (does the user actually have the essential skills?) ==="
    )
    for rec_key, label_key, title in (OPP, OCC):
        binary_fits, elig = [], []
        ess_all, rank_ess_pairs, topk_low = [], [], 0
        low_examples = []
        for u in users:
            for r in sorted(u.get(rec_key) or [], key=lambda r: r.get("rank") or 1e9):
                _met, _total, bfit = _essential_fit(r)
                if bfit is not None:
                    binary_fits.append(bfit)
                elig.append(1 if r.get("is_eligible") else 0)
                ec = _ess_cos(r)
                if ec is not None:
                    ess_all.append(ec)
                    rank_ess_pairs.append((r.get("rank"), ec))
                    if (r.get("rank") or 1e9) <= k and ec < min_ess_cos:
                        topk_low += 1
                        if len(low_examples) < 30:
                            low_examples.append(
                                {
                                    "user_id": u.get("user_id"),
                                    "rank": r.get("rank"),
                                    "label": r.get(label_key),
                                    "essential_cos": round(ec, 3),
                                    "u_hat": _sb(r, "u_hat"),
                                    "p_hat": _sb(r, "p_hat"),
                                    "final_score": r.get("final_score"),
                                }
                            )
        corr = None
        if len(rank_ess_pairs) > 2:
            try:
                corr = round(
                    stats.correlation(
                        [p[0] for p in rank_ess_pairs], [p[1] for p in rank_ess_pairs]
                    ),
                    3,
                )  # py3.10+
            except Exception:
                corr = None
        bin_mean = stats.mean(binary_fits) if binary_fits else None
        sect = {
            "binary_essential_fit_mean": round(bin_mean, 3)
            if bin_mean is not None
            else None,
            "binary_essential_fit_saturated": bool(
                bin_mean is not None and bin_mean > 0.99
            ),
            "pct_eligible": round(100 * sum(elig) / len(elig), 1) if elig else None,
            "essential_cosine_continuous": _fmt_stats(ess_all),
            "corr_rank_vs_essential_cosine": corr,
            "min_ess_cos_threshold": min_ess_cos,
            "n_topk_recs_below_threshold": topk_low,
            "low_essential_cos_examples": sorted(
                low_examples, key=lambda e: e["essential_cos"]
            ),
        }
        report.setdefault("achievability", {})[rec_key] = sect
        print(f"\n  {title}:")
        print(
            f"    binary essential_fit (share of essential skills meeting threshold): mean="
            f"{sect['binary_essential_fit_mean']}, eligible={sect['pct_eligible']}%"
        )
        if sect["binary_essential_fit_saturated"]:
            print(
                "      ⚠ SATURATED at ~1.0 — the meets_threshold gate passes virtually every "
                "essential skill, so binary eligibility is uninformative. Use the continuous cosine ↓"
            )
        print(
            f"    essential cosine (continuous, mean best-match per essential skill): "
            f"{sect['essential_cosine_continuous']}"
        )
        print(
            f"    rank↔essential_cosine corr: {corr}  (≈0 ⇒ ranking is NOT driven by how well "
            f"the user's skills fit; recs can be preference-led, not skill-achievable)"
        )
        print(
            f"    ⚠ {topk_low} top-{k} recs with essential cosine < {min_ess_cos} "
            f"(weak skill fit despite being recommended). Weakest examples:"
        )
        for e in sect["low_essential_cos_examples"][:10]:
            print(
                f"        {e['user_id'][:12]}.. rank{e['rank']:<3} ess_cos={e['essential_cos']} "
                f"u_hat={e['u_hat']} p_hat={e['p_hat']}  {e['label']}"
            )


# ── drill-down ────────────────────────────────────────────────────────────
def _load_user_top_skills(run_dir: Path) -> dict:
    """user_id -> user_top_skills string, from the opportunities vectors CSV (if present)."""
    out = {}
    p = run_dir / "recommendations_with_vectors.csv"
    if p.is_file():
        for row in csv.DictReader(open(p, encoding="utf-8")):
            out.setdefault(row["user_id"], row.get("user_top_skills", ""))
    return out


def _explain_pair(user, rec, rec_key, label_key, user_skills):
    label = rec.get(label_key) or rec.get("uuid")
    met, total, fit = _essential_fit(rec)
    print(f"\n  ── {user.get('user_id')}  ×  {label}  (rank {rec.get('rank')}) ──")
    print(
        f"    final_score={rec.get('final_score')}  u_hat(pref)={_sb(rec, 'u_hat')}  "
        f"p_hat(skills cos)={_sb(rec, 'p_hat')}  total_skill_utility={_sb(rec, 'total_skill_utility')}"
    )
    print(
        f"    is_eligible={rec.get('is_eligible')}  essential_fit={fit if fit is None else round(fit, 2)} "
        f"({met}/{total} essential skills met)  demand={_sb(rec, 'demand_label')}"
    )
    if user_skills:
        print(f"    user's top skills: {user_skills[:240]}")
    em = sorted(
        (rec.get("matched_skills") or {}).get("essential_skill_matches") or [],
        key=lambda m: float(m.get("similarity") or 0),
        reverse=True,
    )
    unmet = [m for m in em if not m.get("meets_threshold")]
    print(
        f"    essential skills MET ({total - len(unmet)}): "
        + ", ".join(
            f"{m.get('job_skill_label')}~{m.get('best_user_skill_label')}({m.get('similarity')})"
            for m in em
            if m.get("meets_threshold")
        )[:400]
        or "(none)"
    )
    print(
        f"    essential skills NOT met ({len(unmet)}): "
        + (", ".join(str(m.get("job_skill_label")) for m in unmet[:25]) or "(none)")
    )
    prefs = rec.get("matched_preferences") or []
    if prefs:
        print(
            "    matched preferences: "
            + ", ".join(
                f"{p.get('attribute')}={p.get('job_value_label') or p.get('job_value')}"
                f"{'✓' if p.get('matched') else '✗'}"
                for p in prefs
            )
        )


def explain(users, token, run_dir, k):
    user_skills = _load_user_top_skills(run_dir)
    by_id = {u.get("user_id"): u for u in users}
    if token in by_id:
        u = by_id[token]
        print(f"\n=== EXPLAIN user {token} — top {k} of each ===")
        for rec_key, label_key, title in (OPP, OCC):
            print(f"\n  {title}:")
            for r in sorted(u.get(rec_key) or [], key=lambda r: r.get("rank") or 1e9)[
                :k
            ]:
                met, total, fit = _essential_fit(r)
                print(
                    f"    rank{r.get('rank'):<3} final={r.get('final_score')} "
                    f"u_hat={_sb(r, 'u_hat')} p_hat={_sb(r, 'p_hat')} "
                    f"ess_fit={fit if fit is None else round(fit, 2)}({met}/{total}) "
                    f"elig={r.get('is_eligible')}  {r.get(label_key)}"
                )
        return

    # Otherwise treat token as an item: match uuid/originUuid exactly OR title/label substring.
    tl = token.lower()
    hits = []
    for u in users:
        for rec_key, label_key, _t in (OPP, OCC):
            for r in u.get(rec_key) or []:
                ids = {str(r.get("uuid") or ""), str(r.get("originUuid") or "")}
                label = str(r.get(label_key) or "")
                if token in ids or tl in label.lower():
                    hits.append((u, r, rec_key, label_key))
    if not hits:
        print(
            f"\nNo user, item uuid, or title/label matched '{token}'. "
            f"Tip: pass a user_id, a job/occupation uuid, or part of a title."
        )
        return
    print(f"\n=== EXPLAIN item '{token}' — {len(hits)} (user, item) pairing(s) ===")
    for u, r, rec_key, label_key in hits[:25]:
        _explain_pair(u, r, rec_key, label_key, user_skills.get(u.get("user_id"), ""))


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Analyse a /match_v4 local run.")
    ap.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Run dir (default: latest under output_results/match_v4_local/)",
    )
    ap.add_argument(
        "--top-k", type=int, default=5, help="K for diversity/top-K checks (default 5)"
    )
    ap.add_argument(
        "--explain",
        metavar="TOKEN",
        default=None,
        help="Explain a user_id, an item uuid, or a title/label substring",
    )
    ap.add_argument(
        "--user", default=None, help="With --item, explain one (user, item) pair"
    )
    ap.add_argument(
        "--item",
        default=None,
        help="Item uuid or title/label substring (use with --user)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Write the JSON report here (default: <run_dir>/analysis_report.json)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run(DEFAULT_RUN_ROOT)
    resp_path = run_dir / "match_v4_response.json"
    if not resp_path.is_file():
        raise SystemExit(f"No match_v4_response.json in {run_dir}")
    users = json.loads(resp_path.read_text(encoding="utf-8"))
    print(f"Run: {run_dir}\nUsers: {len(users)}")

    # Targeted pair explanation.
    if args.user and args.item:
        by_id = {u.get("user_id"): u for u in users}
        u = by_id.get(args.user)
        if not u:
            raise SystemExit(f"user_id {args.user} not found")
        skills = _load_user_top_skills(run_dir).get(args.user, "")
        il = args.item.lower()
        found = False
        for rec_key, label_key, _t in (OPP, OCC):
            for r in u.get(rec_key) or []:
                if (
                    args.item in {str(r.get("uuid")), str(r.get("originUuid"))}
                    or il in str(r.get(label_key) or "").lower()
                ):
                    _explain_pair(u, r, rec_key, label_key, skills)
                    found = True
        if not found:
            print(f"Item '{args.item}' is not in {args.user}'s recommendations.")
        return

    if args.explain:
        explain(users, args.explain, run_dir, args.top_k)
        return

    report = {"run_dir": str(run_dir), "n_users": len(users), "top_k": args.top_k}
    analyse_diversity(users, *OPP, args.top_k, report)
    analyse_diversity(users, *OCC, args.top_k, report)
    analyse_balance(users, report)
    analyse_achievability(users, report, args.top_k)

    out = Path(args.out) if args.out else (run_dir / "analysis_report.json")
    out.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote machine-readable report → {out}")


if __name__ == "__main__":
    main()
