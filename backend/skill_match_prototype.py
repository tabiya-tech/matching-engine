"""
STANDALONE, READ-ONLY experiment (touches no production module / pipeline).

Answers two questions for the skill-eligibility plan:
  (1) How much of the per-skill "saturation" comes from (a) embedding anisotropy, (b) the
      max-over-all-user-skills aggregation, and (c) generic/transversal skills?
  (2) Does an assignment-based *whitened* coverage actually separate same-occupation from
      cross-occupation skill sets — i.e. do we even need a cross-encoder?

Method — split-half cross-occupation self-match (no exact-id shortcut):
  For each occupation A, resolve its essential-skill labels to embedding rows and split them into
  two DISJOINT halves A1 (the "graduate"/user) and A2 (the "job essentials"). Then for occupations
  A,B we score how well A1 covers B2. The diagonal (A1→A2, disjoint skills of the SAME occupation)
  should be HIGH if the matcher recognises genuine within-field relatedness; the off-diagonal
  (A1→B2, different occupation) should be LOW. Splitting into disjoint halves removes the trivial
  exact-overlap (=1.0) inflation, so this purely tests the embedding + aggregation.

Configs compared (same occupations/skills throughout):
  raw+max            : current behaviour (raw Gemini cosine, max over user skills)
  whitened+max       : isolates the whitening (space) contribution
  whitened+assign    : adds one-to-one greedy assignment (isolates max-over-user inflation)
  whitened+assign+dw : adds generic-skill down-weighting (isolates generic-skill inflation)

Relatedness proxy for off-diagonal: ISCO major group = first digit of the occupation code.
  "related"   = different occupation, same major group (should stay moderate)
  "unrelated" = different major group (should collapse to ~0)

Run from backend/:  KMP_DUPLICATE_LIB_OK=TRUE python skill_match_prototype.py
"""

import os
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("MONGO_DB_NAME", "test")

from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parent
REPO_ROOT = BACKEND.parent
load_dotenv(BACKEND / ".env")
sys.path.insert(0, str(BACKEND))

try:  # Windows consoles default to cp1252 and can't encode the arrows/symbols below
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# NB (Windows/miniforge): torch MUST be imported before numpy, or torch's DLLs fail to load
# (OSError WinError 127 / shm.dll). Importing it first also fixes the order for app modules
# (which do `import numpy` then `import torch`) since torch is then already initialised.
import torch  # noqa: E402  (import order is deliberate)
import numpy as np  # noqa: E402

import app.config as C  # noqa: E402
from app.server_dependencies.model_dependencies import build_skill_matcher  # noqa: E402
from app.services.offline_io import user_from_record  # noqa: E402


def _l2(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n > 0, n, 1.0)


def _load_whitened_W(path, dim_check):
    st = torch.load(path, map_location="cpu")
    W = st["state_dict"]["embedding.weight"].numpy()
    if W.dtype != np.float32:
        W = W.astype(np.float32)
    target = float(st.get("whitening", {}).get("target_max_p999") or 0.0)
    return _l2(W), target


def _auc(pos, neg):
    """P(score(pos) > score(neg)) via rank statistic (Mann-Whitney). 0.5 = no separation."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    # (good enough without explicit tie-handling for this diagnostic)
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _greedy_assign_scores(S, n_cols):
    """Greedy one-to-one: each row (A1 skill) used once. Return assigned sim per col (B2 skill),
    0 if unmatched. S is |A1| x |B2|."""
    if S.size == 0:
        return np.zeros(n_cols)
    flat = [(S[i, j], i, j) for i in range(S.shape[0]) for j in range(S.shape[1])]
    flat.sort(reverse=True)
    used_r, used_c = set(), set()
    out = np.zeros(n_cols)
    for v, i, j in flat:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        out[j] = v
        if len(used_c) == n_cols or len(used_r) == S.shape[0]:
            break
    return out


def _make_scorer(raw_W, white_W, target, used_rows):
    """Returns score(rows_a, rows_b, space, mode) over global embedding-row ids (read-only)."""
    pos = {int(g): i for i, g in enumerate(used_rows)}
    raw_u = raw_W[used_rows]
    white_u = white_W[used_rows]

    def rescale(c):
        return (
            np.minimum(1.0, np.maximum(0.0, c) / target)
            if target > 0
            else np.maximum(0.0, c)
        )

    def score(rows_a, rows_b, space, mode):
        ia = [pos[int(r)] for r in rows_a if int(r) in pos]
        ib = [pos[int(r)] for r in rows_b if int(r) in pos]
        if not ia or not ib:
            return np.nan
        U = raw_u if space == "raw" else white_u
        S = U[np.array(ia)] @ U[np.array(ib)].T
        if space == "white":
            S = rescale(S)
        per_b = S.max(axis=0) if mode == "max" else _greedy_assign_scores(S, len(ib))
        return float(per_b.mean())

    return score


def load_users(matcher, path, min_skills):
    """Resolve each user's top_skills to embedding rows; track label/originUUID resolution."""
    users = []
    stat = dict(total=0, label=0, uuid=0, unresolved=0, users=0)
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        u = json.loads(line)
        rows = []
        for s in (u.get("skills_vector") or {}).get("top_skills") or []:
            stat["total"] += 1
            lab = s.get("preferredLabel") or s.get("label")
            sid = matcher._resolve_label(str(lab) if lab else None)
            if sid is not None:
                stat["label"] += 1
            else:
                sid = matcher._resolve_origin_uuid(
                    s.get("originUUID") or s.get("origin_uuid") or s.get("originUuid")
                )
                if sid is not None:
                    stat["uuid"] += 1
                else:
                    stat["unresolved"] += 1
            if sid is not None and sid in matcher.skill_to_row:
                rows.append(matcher.skill_to_row[sid])
        stat["users"] += 1
        rows = sorted(set(rows))
        if len(rows) >= min_skills:
            users.append((str(u.get("user_id") or ""), rows))
    return users, stat


def load_jobs_rows(matcher, path, sample_n, rng, min_ess=3):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    jobs = []
    for j in data:
        rows = []
        for s in j.get("essential_skills") or []:
            lab = s.get("label")
            sid = matcher._resolve_label(str(lab) if lab else None)
            if sid is None:
                sid = matcher._resolve_origin_uuid(
                    s.get("originUuid") or s.get("origin_uuid") or s.get("id")
                )
            if sid is not None and sid in matcher.skill_to_row:
                rows.append(matcher.skill_to_row[sid])
        rows = sorted(set(rows))
        if len(rows) >= min_ess:
            jobs.append(
                (str(j.get("uuid") or ""), j.get("opportunity_title") or "", rows)
            )
    if len(jobs) > sample_n:
        idx = rng.choice(len(jobs), sample_n, replace=False)
        jobs = [jobs[i] for i in idx]
    return jobs


def analyze_real_users(matcher, raw_W, white_W, target, name, path, jobs, rng, args):
    users, stat = load_users(matcher, path, args.min_skills)
    print(f"\n############## REAL USERS: {name} ##############")
    tot = max(stat["total"], 1)
    print(
        f"  skill entries={stat['total']}  resolved-by-label={100 * stat['label'] / tot:.1f}%  "
        f"recovered-by-originUUID={100 * stat['uuid'] / tot:.1f}%  unresolved={100 * stat['unresolved'] / tot:.1f}%"
    )
    print(
        f"  users total={stat['users']}  with >= {args.min_skills} resolved skills={len(users)}"
    )
    if len(users) < 2 or not jobs:
        print("  not enough data; skipping")
        return

    used = sorted(
        {int(r) for _, rows in users for r in rows}
        | {int(r) for _, _, rows in jobs for r in rows}
    )
    score = _make_scorer(raw_W, white_W, target, used)
    configs = [
        ("raw+max", "raw", "max"),
        ("whitened+max", "white", "max"),
        ("whitened+assign", "white", "assign"),
    ]

    halves = {}
    for uid, rows in users:
        r = rows[:]
        rng.shuffle(r)
        h = len(r) // 2
        halves[uid] = (r[:h], r[h:])
    uids = [u for u, _ in users]

    print(
        "  -- Analysis 1: user skill split-half — self (h1→h2) vs other users (h1→other h2) --"
    )
    print(f"    {'config':<18}{'self':>8}{'cross-user':>12}{'AUC self|cross':>16}")
    for cname, space, mode in configs:
        self_s, cross_s = [], []
        for uid, _ in users:
            h1, h2 = halves[uid]
            s = score(h1, h2, space, mode)
            if not np.isnan(s):
                self_s.append(s)
            for v in rng.choice(uids, min(8, len(uids)), replace=False):
                if v == uid:
                    continue
                cs = score(h1, halves[v][1], space, mode)
                if not np.isnan(cs):
                    cross_s.append(cs)
        print(
            f"    {cname:<18}{np.mean(self_s):>8.3f}{np.mean(cross_s):>12.3f}{_auc(self_s, cross_s):>16.3f}"
        )

    print(
        f"  -- Analysis 2: user → job coverage across {len(jobs)} sampled jobs (de-saturation) --"
    )
    print(f"    {'config':<18}{'mean cov':>10}{'within-user std':>16}")
    for cname, space, mode in configs:
        allcov, within = [], []
        for uid, rows in users:
            cs = [score(rows, jr, space, mode) for _, _, jr in jobs]
            cs = [c for c in cs if not np.isnan(c)]
            if cs:
                allcov += cs
                within.append(float(np.std(cs)))
        print(f"    {cname:<18}{np.mean(allcov):>10.3f}{np.mean(within):>16.3f}")
    print(
        "    (raw+max ~flat/high = saturated; whitened+assign lower mean + higher within-user std = discriminating)"
    )

    umap = dict(users)
    pick = (
        "FeVg9sIDZUd2bGIrvCUOcz31dum1"
        if "FeVg9sIDZUd2bGIrvCUOcz31dum1" in umap
        else max(users, key=lambda x: len(x[1]))[0]
    )
    urows = umap[pick]
    scored = sorted(
        ((score(urows, jr, "white", "assign"), title) for _, title, jr in jobs),
        reverse=True,
    )
    print(
        f"  -- Drill-down {pick[:14]}.. ({len(urows)} skills) whitened+assign coverage over jobs --"
    )
    print("     top:    " + " | ".join(f"{s:.2f} {t[:34]}" for s, t in scored[:3]))
    print("     bottom: " + " | ".join(f"{s:.2f} {t[:34]}" for s, t in scored[-3:]))


def _fit_logistic_1d(x, y, iters=800, lr=0.5):
    """One-feature logistic fit (Platt). Returns (A, B) for p = sigmoid(A*sim + B)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mu, sd = x.mean(), x.std() + 1e-9
    xs = (x - mu) / sd
    a = b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * xs + b)))
        a -= lr * np.mean((p - y) * xs)
        b -= lr * np.mean(p - y)
    return a / sd, b - a * mu / sd  # de-standardise


def fit_platt(matcher, white_W, target, rng, n=4000):
    """Global soft-coverage calibration: positives = within-occupation skill pairs, negatives =
    random skill pairs; fit one logistic on the whitened+rescaled cosine. Returns (A, B)."""
    occ_raw = json.loads(Path(C.OCCUPATION_JSON_PATH).read_text(encoding="utf-8"))
    occ_rows = []
    for e in occ_raw:
        labels = ((e.get("skills") or {}).get("essential") or {}).get("labels") or []
        rows = []
        for lab in labels:
            sid = matcher._resolve_label(str(lab) if lab else None)
            if sid is not None and sid in matcher.skill_to_row:
                rows.append(matcher.skill_to_row[sid])
        rows = sorted(set(rows))
        if len(rows) >= 2:
            occ_rows.append(rows)

    def resc(c):
        return min(1.0, max(0.0, c) / target) if target > 0 else max(0.0, c)

    N = white_W.shape[0]
    pos = []
    for _ in range(n):
        r = occ_rows[rng.integers(len(occ_rows))]
        i, j = rng.choice(len(r), 2, replace=False)
        pos.append(resc(float(white_W[r[i]] @ white_W[r[j]])))
    neg = [
        resc(float(white_W[rng.integers(N)] @ white_W[rng.integers(N)]))
        for _ in range(n)
    ]
    x = np.array(pos + neg)
    y = np.array([1] * len(pos) + [0] * len(neg))
    A, B = _fit_logistic_1d(x, y)
    print(
        f"  Platt fit: A={A:.2f} B={B:.2f}  (pos mean={np.mean(pos):.3f}, neg mean={np.mean(neg):.3f}, "
        f"AUC={_auc(pos, neg):.3f})",
        file=sys.stderr,
    )
    return A, B


def analyze_count(matcher, raw_W, white_W, target, name, users_path, jobs, AB, args):
    """Test count-dependence of: coverage (whitened+assign mean), the combined skills_fit (mean-pool
    PROXY), and their product — each with and WITHOUT whitened de-dup of near-duplicate user skills."""
    users, _ = load_users(matcher, users_path, args.min_skills)
    if len(users) < 2 or not jobs:
        return
    used = sorted(
        {int(r) for _, rows in users for r in rows}
        | {int(r) for _, _, rows in jobs for r in rows}
    )
    posmap = {g: i for i, g in enumerate(used)}
    white_u = white_W[used]
    d = args.dedup_thresh

    def resc(c):
        return (
            np.minimum(1.0, np.maximum(0.0, c) / target)
            if target > 0
            else np.maximum(0.0, c)
        )

    def pool(idx):
        if not idx:
            return None
        v = white_u[idx].mean(axis=0)
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v

    def dedup(idx):
        """Greedy: drop a skill if its whitened cosine to an already-kept skill is >= d (near-dup)."""
        kept = []
        for i in idx:
            if all(float(white_u[i] @ white_u[k]) < d for k in kept):
                kept.append(i)
        return kept

    jobib = {
        u: [posmap[int(r)] for r in rows if int(r) in posmap] for u, _, rows in jobs
    }
    jobvec = {u: pool(jobib[u]) for u in jobib}

    cnt = []
    covF, covD, sfF, sfD, prF, prD = [], [], [], [], [], []
    collapse = []
    for uid, rows in users:
        ia = [posmap[int(r)] for r in rows if int(r) in posmap]
        iad = dedup(ia)
        collapse.append((len(ia), len(iad)))
        uvF, uvD = pool(ia), pool(iad)
        for u, _, _ in jobs:
            ib = jobib[u]
            if not ia or not ib:
                continue
            cf = float(
                _greedy_assign_scores(resc(white_u[ia] @ white_u[ib].T), len(ib)).mean()
            )
            cd = (
                float(
                    _greedy_assign_scores(
                        resc(white_u[iad] @ white_u[ib].T), len(ib)
                    ).mean()
                )
                if iad
                else np.nan
            )
            sf = (
                float(resc(uvF @ jobvec[u]))
                if (uvF is not None and jobvec[u] is not None)
                else np.nan
            )
            sd = (
                float(resc(uvD @ jobvec[u]))
                if (uvD is not None and jobvec[u] is not None)
                else np.nan
            )
            cnt.append(len(rows))
            covF.append(cf)
            covD.append(cd)
            sfF.append(sf)
            sfD.append(sd)
            prF.append(sf * cf if not np.isnan(sf) else np.nan)
            prD.append(sd * cd if not (np.isnan(sd) or np.isnan(cd)) else np.nan)

    def corr(ys):
        xs = np.array(cnt, float)
        ys = np.array(ys, float)
        m = ~np.isnan(ys)
        if m.sum() < 3 or np.std(xs[m]) == 0 or np.std(ys[m]) == 0:
            return float("nan")
        return float(np.corrcoef(xs[m], ys[m])[0, 1])

    of = np.mean([a for a, _ in collapse])
    od = np.mean([b for _, b in collapse])
    print(
        f"\n====== COUNT-DEPENDENCE + DEDUP: {name}  (dedup whitened-cos>= {d}; mean #skills {of:.0f}→{od:.0f}) ======"
    )
    print(f"  corr(#orig user skills, metric)        {'no-dedup':>10}{'dedup':>10}")
    print(
        f"    coverage (whitened+assign mean)      {corr(covF):>10.3f}{corr(covD):>10.3f}"
    )
    print(
        f"    combined skills_fit (pool proxy)     {corr(sfF):>10.3f}{corr(sfD):>10.3f}"
    )
    print(
        f"    product (skills_fit × coverage)      {corr(prF):>10.3f}{corr(prD):>10.3f}"
    )
    print("    (closer to 0 = less count-driven)")


def validate_real_concat(matcher, white_W, target, name, users_path, jobs_path, args):
    """Validate the two-factor self-correction on the REAL Gemini concat skills_fit (not the mean-pool
    proxy): user concat embedded live via Gemini; job concat = stored job_embedding. Reports the
    count-correlation of the real skills_fit, the per-skill coverage, and their product."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass
    from app.clients.gemini_embedding_client import embed_text_list
    from app.ranking.retrieval import l2_normalize_rows, user_concat_embedding_text

    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        print(
            f"[{name}] no GEMINI_API_KEY — skipping real-concat validation",
            file=sys.stderr,
        )
        return

    raws = [
        json.loads(user) for user in open(users_path, encoding="utf-8") if user.strip()
    ]
    users = []
    for u in raws:
        rows = []
        for s in (u.get("skills_vector") or {}).get("top_skills") or []:
            lab = s.get("preferredLabel") or s.get("label")
            sid = matcher._resolve_label(str(lab) if lab else None)
            if sid is None:
                sid = matcher._resolve_origin_uuid(
                    s.get("originUUID") or s.get("origin_uuid") or s.get("originUuid")
                )
            if sid is not None and sid in matcher.skill_to_row:
                rows.append(matcher.skill_to_row[sid])
        rows = sorted(set(rows))
        n_skills = len((u.get("skills_vector") or {}).get("top_skills") or [])
        if len(rows) >= args.min_skills:
            users.append((u, rows, n_skills))
    if len(users) < 2:
        print(f"[{name}] <2 users; skipping", file=sys.stderr)
        return

    print(
        f"[{name}] embedding {len(users)} user concat texts via Gemini ...",
        file=sys.stderr,
    )
    texts = [
        user_concat_embedding_text(user_from_record(u)) or " " for u, _, _ in users
    ]
    Uraw = embed_text_list(texts, api_key=api_key, batch_size=100, sleep_s=0.12).astype(
        np.float64
    )
    U = l2_normalize_rows(Uraw.astype(np.float32)).astype(np.float64)
    dim = U.shape[1]

    rng = np.random.default_rng(args.seed)
    data = json.loads(Path(jobs_path).read_text(encoding="utf-8"))
    jraw_all, jobs = [], []
    for j in data:
        emb = j.get("job_embedding")
        if not (isinstance(emb, list) and len(emb) == dim):
            continue
        jraw_all.append(np.asarray(emb, dtype=np.float64))
        rows = []
        for s in j.get("essential_skills") or []:
            lab = s.get("label")
            sid = matcher._resolve_label(str(lab) if lab else None)
            if sid is None:
                sid = matcher._resolve_origin_uuid(s.get("originUuid") or s.get("id"))
            if sid is not None and sid in matcher.skill_to_row:
                rows.append(matcher.skill_to_row[sid])
        rows = sorted(set(rows))
        if len(rows) >= 3:
            jobs.append((np.asarray(emb, dtype=np.float64), rows))
    if not jobs:
        print(f"[{name}] no jobs with usable job_embedding; skipping", file=sys.stderr)
        return
    if len(jobs) > args.n_jobs:
        idx = rng.choice(len(jobs), args.n_jobs, replace=False)
        jobs = [jobs[i] for i in idx]
    Jraw = np.stack([e for e, _ in jobs])
    Jc = l2_normalize_rows(Jraw.astype(np.float32)).astype(np.float64)

    # Fit a concat whitening transform (Σ^-1/2 with Tikhonov shrinkage) on all job concat vectors +
    # the user concat vectors — the plan's Design A de-compression, tested on the real signal.
    Xfit = np.vstack([np.stack(jraw_all), Uraw])
    mu = Xfit.mean(axis=0)
    Xc = Xfit - mu
    Sigma = (Xc.T @ Xc) / Xc.shape[0]
    dm = float(np.mean(np.diag(Sigma)))
    Sigma += 2.0 * dm * np.eye(dim)  # shrinkage (cov underdetermined at dim=3072)
    vals, vecs = np.linalg.eigh(Sigma)
    Winv = (vecs * (1.0 / np.sqrt(np.maximum(vals, 1e-12)))) @ vecs.T  # Σ^-1/2

    def whiten(M):
        return l2_normalize_rows(((M - mu) @ Winv).astype(np.float32)).astype(
            np.float64
        )

    Uw = whiten(Uraw)
    Jw = whiten(Jraw)

    used = sorted(
        {int(r) for _, rows, _ in users for r in rows}
        | {int(r) for _, rows in jobs for r in rows}
    )
    posmap = {g: i for i, g in enumerate(used)}
    white_u = white_W[used]

    def resc(c):
        return (
            np.minimum(1.0, np.maximum(0.0, c) / target)
            if target > 0
            else np.maximum(0.0, c)
        )

    cnt, fit_r, fit_w, cov, prod_r, prod_w = [], [], [], [], [], []
    for ui, (u, urows, n_skills) in enumerate(users):
        ia = [posmap[int(r)] for r in urows]
        for ji, (emb, jrows) in enumerate(jobs):
            ib = [posmap[int(r)] for r in jrows]
            if not ia or not ib:
                continue
            sr = float(U[ui] @ Jc[ji])  # RAW Gemini concat cosine
            sw = float(Uw[ui] @ Jw[ji])  # WHITENED concat cosine
            cv = float(
                _greedy_assign_scores(resc(white_u[ia] @ white_u[ib].T), len(ib)).mean()
            )
            cnt.append(n_skills)
            fit_r.append(sr)
            fit_w.append(sw)
            cov.append(cv)
            prod_r.append(sr * cv)
            prod_w.append(sw * cv)

    def corr(ys):
        xs = np.array(cnt, float)
        ys = np.array(ys, float)
        m = ~np.isnan(ys)
        if m.sum() < 3 or np.std(xs[m]) == 0 or np.std(ys[m]) == 0:
            return float("nan")
        return float(np.corrcoef(xs[m], ys[m])[0, 1])

    print(
        f"\n====== REAL CONCAT VALIDATION: {name} ({len(cnt)} pairs, {len(users)} users) ======"
    )
    print(f"  corr(#skills, per-skill coverage)            = {corr(cov):+.3f}")
    print(
        f"  corr(#skills, RAW concat skills_fit)         = {corr(fit_r):+.3f}   "
        f"(spread mean={np.mean(fit_r):.3f} sd={np.std(fit_r):.3f})"
    )
    print(
        f"  corr(#skills, WHITENED concat skills_fit)    = {corr(fit_w):+.3f}   "
        f"(spread mean={np.mean(fit_w):.3f} sd={np.std(fit_w):.3f})"
    )
    print(f"  corr(#skills, product RAW concat × coverage) = {corr(prod_r):+.3f}")
    # whitened cosine is centred ~0; as a p_hat factor it must be rescaled to [0,1] (p99 target).
    fw = np.array(fit_w)
    tw = float(np.percentile(fw, 99))
    tw = tw if tw > 1e-6 else 1.0
    sf01 = np.clip(fw / tw, 0.0, 1.0)
    prod01 = (sf01 * np.array(cov)).tolist()
    print(
        f"  corr(#skills, product WHITENED→[0,1] × cov)  = {corr(prod01):+.3f}   "
        f"(rescaled skills_fit∈[0,1], p99 target={tw:.3f}; proxy predicted ~+0.3)"
    )


def calibrate_threshold(matcher, white_W, target, rng, n=12000):
    """Calibrate the per-skill rescaled-whitened threshold tau: sweep tau, reporting FPR on RANDOM
    skill pairs (the reliable side) and recall on WITHIN-OCCUPATION pairs (a NOISY positive — many
    co-listed skills aren't truly similar, so recall is a soft lower bound). Recommend tau at a target
    random-pair FPR."""
    occ_raw = json.loads(Path(C.OCCUPATION_JSON_PATH).read_text(encoding="utf-8"))
    occ_rows = []
    for e in occ_raw:
        labels = ((e.get("skills") or {}).get("essential") or {}).get("labels") or []
        rows = []
        for lab in labels:
            sid = matcher._resolve_label(str(lab) if lab else None)
            if sid is not None and sid in matcher.skill_to_row:
                rows.append(matcher.skill_to_row[sid])
        rows = sorted(set(rows))
        if len(rows) >= 2:
            occ_rows.append(rows)

    def resc(c):
        return min(1.0, max(0.0, c) / target) if target > 0 else max(0.0, c)

    N = white_W.shape[0]
    pos_l, neg_l = [], []
    for _ in range(n):
        r = occ_rows[rng.integers(len(occ_rows))]
        a, b = rng.choice(len(r), 2, replace=False)
        pos_l.append(resc(float(white_W[r[a]] @ white_W[r[b]])))
        neg_l.append(resc(float(white_W[rng.integers(N)] @ white_W[rng.integers(N)])))
    pos = np.array(pos_l)
    neg = np.array(neg_l)

    print(
        f"\n====== CALIBRATE per-skill tau (rescaled whitened; target={target:.3f}; n={n}) ======"
    )
    print(
        f"  positive (within-occupation, NOISY) mean={pos.mean():.3f} p25={np.percentile(pos, 25):.3f} "
        f"p50={np.percentile(pos, 50):.3f} p75={np.percentile(pos, 75):.3f}"
    )
    print(
        f"  negative (random pairs)            mean={neg.mean():.3f} p95={np.percentile(neg, 95):.3f} "
        f"p99={np.percentile(neg, 99):.3f} max={neg.max():.3f}"
    )
    print(f"  {'tau':>5} {'FPR(rand)':>10} {'recall(occ)':>12} {'Youden J':>9}")
    best = (0.0, -1.0)
    for tau in np.arange(0.0, 1.0001, 0.05):
        fpr = float((neg >= tau).mean())
        rec = float((pos >= tau).mean())
        if rec - fpr > best[1]:
            best = (float(tau), rec - fpr)
        if tau in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            print(f"  {tau:>5.2f} {fpr:>10.3f} {rec:>12.3f} {rec - fpr:>9.3f}")
    t5 = float(np.percentile(neg, 95))
    t1 = float(np.percentile(neg, 99))
    print(
        f"  RECOMMEND: FPR~5% -> tau={t5:.3f} (occ-recall {float((pos >= t5).mean()):.2f}) | "
        f"FPR~1% -> tau={t1:.3f} (occ-recall {float((pos >= t1).mean()):.2f}) | Youden-J peak tau={best[0]:.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["users", "occ", "both", "concat", "calibrate"],
        default="users",
    )
    ap.add_argument("--n-occ", type=int, default=180, help="sample size of occupations")
    ap.add_argument(
        "--min-ess",
        type=int,
        default=6,
        help="min resolved essentials to include an occupation",
    )
    ap.add_argument(
        "--min-skills",
        type=int,
        default=6,
        help="min resolved skills to include a real user",
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=150,
        help="sampled jobs for the user→job coverage test",
    )
    ap.add_argument(
        "--dedup-thresh",
        type=float,
        default=0.5,
        help="whitened-cosine ≥ this ⇒ near-duplicate user skill (collapsed)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("Loading matcher (raw) + whitened matrix ...", file=sys.stderr)
    matcher = build_skill_matcher()  # raw W + resolution maps (read-only)
    raw_W = matcher.W.astype(np.float32)  # already L2-normalised
    white_W, target = _load_whitened_W(
        Path(C.EMBEDDING_MODEL_PATH).parent
        / "skill_embedding_model_gemini_whitened.pt",
        raw_W.shape,
    )
    print(
        f"raw_W {raw_W.shape}  white_W {white_W.shape}  whiten target_max_p999={target:.4f}",
        file=sys.stderr,
    )

    if args.mode == "calibrate":
        calibrate_threshold(matcher, white_W, target, rng)
        return

    if args.mode == "concat":
        for ds in ("kenya", "njila"):
            p = REPO_ROOT / "data" / f"{ds}_match_input.jsonl"
            if p.is_file():
                validate_real_concat(
                    matcher,
                    white_W,
                    target,
                    ds,
                    p,
                    REPO_ROOT / "data" / "kenya_jobs_for_pipeline.json",
                    args,
                )
            else:
                print(f"[{ds}] dataset missing: {p}", file=sys.stderr)
        return

    if args.mode in ("users", "both"):
        jobs = load_jobs_rows(
            matcher,
            REPO_ROOT / "data" / "kenya_jobs_for_pipeline.json",
            args.n_jobs,
            rng,
        )
        print(f"jobs loaded for coverage test: {len(jobs)}", file=sys.stderr)
        AB = fit_platt(
            matcher, white_W, target, rng
        )  # one global soft-coverage calibration
        for ds in ("kenya", "njila"):
            p = REPO_ROOT / "data" / f"{ds}_match_input.jsonl"
            if p.is_file():
                analyze_real_users(
                    matcher, raw_W, white_W, target, ds, p, jobs, rng, args
                )
                analyze_count(matcher, raw_W, white_W, target, ds, p, jobs, AB, args)
            else:
                print(f"[{ds}] dataset missing: {p}", file=sys.stderr)
        if args.mode == "users":
            return

    # --- Load occupations directly from the taxonomy JSON (no Mongo) ---
    occ_raw = json.loads(Path(C.OCCUPATION_JSON_PATH).read_text(encoding="utf-8"))
    occs = []  # (code, major_group, [row indices of resolved essential skills])
    for e in occ_raw:
        o = e.get("occupation", {})
        code = str(o.get("code") or "")
        ess = (e.get("skills") or {}).get("essential") or {}
        labels = ess.get("labels") or []
        rows = []
        for lab in labels:
            sid = matcher._resolve_label(str(lab) if lab else None)
            if sid is not None and sid in matcher.skill_to_row:
                rows.append(matcher.skill_to_row[sid])
        rows = sorted(set(rows))
        if code and len(rows) >= args.min_ess:
            major = code[0] if code[:1].isdigit() else "?"
            occs.append((code, major, rows))

    if len(occs) > args.n_occ:
        idx = rng.choice(len(occs), args.n_occ, replace=False)
        occs = [occs[i] for i in idx]
    print(
        f"occupations used: {len(occs)} (>= {args.min_ess} resolved essentials)",
        file=sys.stderr,
    )

    # split each occupation's essentials into disjoint halves A1 (user) / A2 (job)
    A1, A2 = {}, {}
    for k, (code, major, rows) in enumerate(occs):
        r = rows[:]
        rng.shuffle(r)
        h = len(r) // 2
        A1[k] = np.array(r[:h])
        A2[k] = np.array(r[h:])

    # distinct skills used -> compact index + precompute genericness
    used = sorted(
        {int(x) for k in range(len(occs)) for x in np.concatenate([A1[k], A2[k]])}
    )
    pos_of = {g: i for i, g in enumerate(used)}
    raw_u = raw_W[used]
    white_u = white_W[used]
    # genericness = mean cosine to a random sample of the FULL taxonomy (per space)
    R = rng.choice(raw_W.shape[0], min(1500, raw_W.shape[0]), replace=False)
    _gen_raw = (raw_u @ raw_W[R].T).mean(axis=1)
    gen_white = (white_u @ white_W[R].T).mean(axis=1)
    # down-weight weight in [0,1]: low for the most generic skills (whitened genericness)
    g = gen_white
    lo, hi = np.percentile(g, 50), np.percentile(g, 95)
    dw = np.clip(
        1.0 - (g - lo) / (hi - lo + 1e-9), 0.0, 1.0
    )  # 1 below median, →0 by p95

    def rescale(
        c,
    ):  # whitened cosine -> [0,1] like production (monotonic; AUC-invariant)
        return (
            np.minimum(1.0, np.maximum(0.0, c) / target)
            if target > 0
            else np.maximum(0.0, c)
        )

    # score one ordered pair A1[a] -> A2[b] under a config; returns mean per-essential coverage
    def score(a, b, space, mode, downweight):
        ia = np.array([pos_of[int(x)] for x in A1[a]])
        ib = np.array([pos_of[int(x)] for x in A2[b]])
        if len(ia) == 0 or len(ib) == 0:
            return np.nan
        U = raw_u if space == "raw" else white_u
        S = U[ia] @ U[ib].T  # |A1| x |B2| cosine
        if space == "white":
            S = rescale(S)
        if mode == "max":
            per_b = S.max(
                axis=0
            )  # best user skill per job essential (inflation source)
        else:  # one-to-one greedy assignment (rows=A1 used once)
            per_b = _greedy_assign_scores(S, len(ib))
        if downweight:
            w = dw[ib]
            return float((per_b * w).sum() / (w.sum() + 1e-9))
        return float(per_b.mean())

    configs = [
        ("raw+max", "raw", "max", False),
        ("whitened+max", "white", "max", False),
        ("whitened+assign", "white", "assign", False),
        ("whitened+assign+dw", "white", "assign", True),
    ]

    n = len(occs)
    majors = [occs[k][1] for k in range(n)]
    results = {}
    for name, space, mode, dw_on in configs:
        diag, related, unrelated = [], [], []
        # to measure max-inflation vs user-skill count
        cnt_unrel = []  # (|A1|, unrelated score) for raw+max & whitened variants
        for a in range(n):
            for b in range(n):
                s = score(a, b, space, mode, dw_on)
                if np.isnan(s):
                    continue
                if a == b:
                    diag.append(s)
                elif majors[a] != "?" and majors[a] == majors[b]:
                    related.append(s)
                    cnt_unrel.append((len(A1[a]), s, "rel"))
                else:
                    unrelated.append(s)
                    cnt_unrel.append((len(A1[a]), s, "unrel"))
        results[name] = dict(
            diag=np.array(diag),
            related=np.array(related),
            unrelated=np.array(unrelated),
            cnt=cnt_unrel,
        )

    # ---------- report ----------
    print(
        "\n================ (2) DOES IT SEPARATE? split-half cross-occupation ================"
    )
    print(
        f"{'config':<20} {'diag(A1→A2)':>12} {'related':>9} {'unrelated':>10} "
        f"{'sep(d−u)':>9} {'AUC d|u':>8} {'AUC rel|unrel':>13}"
    )
    for name, *_ in configs:
        r = results[name]
        d, rel, un = r["diag"].mean(), r["related"].mean(), r["unrelated"].mean()
        print(
            f"{name:<20} {d:>12.3f} {rel:>9.3f} {un:>10.3f} {d - un:>9.3f} "
            f"{_auc(r['diag'], r['unrelated']):>8.3f} {_auc(r['related'], r['unrelated']):>13.3f}"
        )
    print(
        "  diag should be HIGH (within-field, disjoint skills), unrelated LOW. "
        "AUC≈1 ⇒ separates (bi-encoder enough); AUC≈0.5–0.6 ⇒ need cross-encoder."
    )

    print(
        "\n================ (1) WHERE DOES THE SATURATION COME FROM? ================"
    )
    base = results["raw+max"]["unrelated"].mean()
    wmax = results["whitened+max"]["unrelated"].mean()
    wasn = results["whitened+assign"]["unrelated"].mean()
    wdw = results["whitened+assign+dw"]["unrelated"].mean()
    print("  spurious UNRELATED coverage by config:")
    print(
        f"    raw+max               = {base:.3f}   (current behaviour — the saturation)"
    )
    print(
        f"    whitened+max          = {wmax:.3f}   (Δ {base - wmax:+.3f}  ← whitening / anisotropy)"
    )
    print(
        f"    whitened+assign       = {wasn:.3f}   (Δ {wmax - wasn:+.3f}  ← max-over-user-skills inflation)"
    )
    print(
        f"    whitened+assign+dw    = {wdw:.3f}   (Δ {wasn - wdw:+.3f}  ← generic-skill inflation)"
    )
    tot = base - wdw
    if tot > 1e-6:
        print(f"  contribution share of the total {base:.3f}→{wdw:.3f} drop:")
        print(
            f"    whitening {100 * (base - wmax) / tot:4.0f}%   max→assignment {100 * (wmax - wasn) / tot:4.0f}%   "
            f"generic-downweight {100 * (wasn - wdw) / tot:4.0f}%"
        )

    # max-inflation vs user-skill count (correlation), under max vs assignment (whitened)
    def corr_count(cfg):
        c = [(n_, s) for (n_, s, tag) in results[cfg]["cnt"] if tag == "unrel"]
        if len(c) < 3:
            return float("nan")
        a = np.array([x[0] for x in c], float)
        b = np.array([x[1] for x in c], float)
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print(
        f"\n  corr(#user skills, UNRELATED coverage):  whitened+max = {corr_count('whitened+max'):+.3f}   "
        f"whitened+assign = {corr_count('whitened+assign'):+.3f}"
    )
    print(
        "    (positive under max ⇒ more skills manufacture more spurious coverage; assignment should flatten it)"
    )

    # most-generic skills (interpretability)
    order = np.argsort(-gen_white)
    labels_by_row = {
        v: matcher.skill_labels.get(k, k) for k, v in matcher.skill_to_row.items()
    }
    print("\n  most 'generic' skills (highest mean whitened similarity to everything):")
    for oi in order[:12]:
        row = used[oi]
        print(f"    g={gen_white[oi]:+.3f}  {labels_by_row.get(row, row)}")

    print("\nDone. (read-only; no production files touched)", file=sys.stderr)


if __name__ == "__main__":
    main()
