"""Embed ``cosine_match_*_results.json`` into a standalone HTML dashboard.

Same layout pattern as ``bm25_scoring/build_bm25_dashboard.py``:
sidebar users, expandable job rows, score bar keyed on ``mean_best_cosine``.

Usage::

    cd backend
    python -m app.services.cosine_similarity.build_cosine_dashboard \\
        --input ./path/to/cosine_results.json \\
        --output ./path/to/cosine_dashboard.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _compact_rec(raw: Dict[str, Any], *, max_rows: int) -> Dict[str, Any]:
    pj_in = raw.get("per_job_skill") or []
    rows: List[Dict[str, str]] = []
    for row in pj_in[:max_rows]:
        if not isinstance(row, dict):
            continue
        rows.append({
            "jl": str(row.get("job_skill_label") or "")[:180],
            "ul": str(row.get("best_user_skill_label") or "")[:180],
            "c": float(row.get("cosine_similarity") or 0.0),
        })
    full_n = len(pj_in)
    return {
        "r": raw.get("rank"),
        "uuid": str(raw.get("job_uuid") or ""),
        "t": str(raw.get("job_title") or "")[:280],
        "e": str(raw.get("employer") or "")[:180],
        "l": str(raw.get("location") or "")[:200],
        "s": float(raw.get("mean_best_cosine") or 0.0),
        "mn": float(raw.get("min_best_cosine") or 0.0),
        "nu": int(raw.get("n_user_skills_embedded") or 0),
        "nj": int(raw.get("n_job_skills_embedded") or 0),
        "pj": rows,
        "n_pj": full_n,
        "pj_trunc": full_n > max_rows,
    }


def _compact_user(row: Dict[str, Any], *, max_rows: int, max_labels: int) -> Dict[str, Any]:
    labs = [str(x) for x in (row.get("resolved_user_skill_labels") or [])]
    return {
        "uid": str(row.get("user_id") or ""),
        "city": str(row.get("city") or ""),
        "prov": str(row.get("province") or ""),
        "ns": int(row.get("n_resolved_user_skills") or 0),
        "skills": labs[:max_labels],
        "skills_trunc": len(labs) > max_labels,
        "n_lab": len(labs),
        "recs": [_compact_rec(r, max_rows=max_rows) for r in (row.get("recommendations") or [])],
    }


def build_html(
    payload_raw: Dict[str, Any],
    *,
    max_per_job_skills: int = 80,
    max_user_skill_labels: int = 120,
) -> str:
    cfg = payload_raw.get("config") or {}
    idx = payload_raw.get("index_stats") or {}
    meta = {
        "n_users": payload_raw.get("n_users"),
        "n_jobs": payload_raw.get("n_jobs"),
        "jobs_source": cfg.get("jobs_source"),
        "top_k_stored": cfg.get("top_k"),
        "users_path": cfg.get("users_path"),
        "mongo_filter_by_users": cfg.get("mongo_filter_by_users"),
        "scorer": cfg.get("scorer"),
        "embedding_model_path": cfg.get("embedding_model_path"),
        "embedding_dim": idx.get("embedding_dim"),
        "max_per_job_skills_in_output": cfg.get("max_per_job_skills_in_output"),
    }
    compact = {
        "meta": meta,
        "users": [
            _compact_user(
                u, max_rows=max_per_job_skills, max_labels=max_user_skill_labels
            )
            for u in payload_raw.get("results") or []
        ],
    }
    data_json = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    safe = data_json.replace("</", "<\\/")

    tpl = (
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cosine skill match dashboard</title>
<style>
%(css)s
</style>
</head>
<body>
<header>
  <h1>Cosine skill match dashboard</h1>
  <span class="pill" id="pillMeta"></span>
  <button id="keyBtn">\u2139 Key</button>
  <span class="stats" id="hdrstats"></span>
</header>
<main>
  <aside>
    <input type="search" id="usearch" placeholder="Search user_id, city, province\u2026">
    <ul class="ulist" id="ulist"></ul>
  </aside>
  <section class="main" id="mainpanel"></section>
</main>

<div class="modalbg" id="modalbg"><div class="modal" id="modal"></div></div>

<script>
const PAYLOAD = %(data)s;

let activeUid = null;
let topN = 50;

%(js)s
</script>
</body>
</html>"""
        % {"css": _CSS, "data": safe, "js": _JS}
    )
    return tpl.strip() + "\n"


_CSS = r"""
  * { box-sizing: border-box; }
  body { font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin: 0; color: #1a1a1a; background: #f6f7f8; }
  header { background: #111827; color: #fff; padding: 10px 16px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  header h1 { font-size: 14px; font-weight: 600; margin: 0 12px 0 0; }
  header .pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 500; background: #374151; color: #d1d5db; }
  header button { font: inherit; padding: 4px 10px; border-radius: 4px; border: 1px solid #4b5563; background: #1f2937; color: #fff; cursor: pointer; }
  header button:hover { background: #374151; }
  .stats { color: #9ca3af; font-size: 11px; flex: 1; text-align: right; }
  main { display: grid; grid-template-columns: 300px 1fr; height: calc(100vh - 46px); }
  aside { background: #fff; border-right: 1px solid #e5e7eb; overflow-y: auto; }
  aside input { width: calc(100% - 24px); padding: 6px 10px; margin: 10px 12px; border: 1px solid #d1d5db; border-radius: 4px; font: inherit; }
  .ulist { list-style: none; margin: 0; padding: 0; }
  .ulist li { padding: 8px 14px; cursor: pointer; border-bottom: 1px solid #f0f0f0; font-size: 12px; }
  .ulist li:hover { background: #f9fafb; }
  .ulist li.active { background: #eff6ff; border-left: 3px solid #2563eb; padding-left: 11px; }
  .ulist .uname { font-weight: 600; color: #111827; font-size: 11px; word-break: break-all; }
  .ulist .umeta { color: #6b7280; font-size: 10px; margin-top: 2px; }
  section.main { overflow-y: auto; padding: 14px 18px; }
  .userhead { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
  .userhead h2 { margin: 0 0 4px 0; font-size: 15px; word-break: break-all; }
  .userhead .meta { color: #6b7280; font-size: 11px; margin-bottom: 8px; }
  .skills { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; max-height: 140px; overflow-y: auto; }
  .skill { background: #eef2ff; color: #3730a3; border-radius: 10px; padding: 1px 8px; font-size: 10px; white-space: nowrap; max-width: 100%; overflow: hidden; text-overflow: ellipsis; }
  .controls { display: flex; gap: 14px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; font-size: 11px; color: #4b5563; }
  .controls label { display: flex; align-items: center; gap: 6px; }
  .controls input[type=number] { width: 56px; padding: 4px 6px; border: 1px solid #d1d5db; border-radius: 4px; }
  .col { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; min-width: 0; }
  .col h3 { margin: 0 0 10px 0; font-size: 13px; color: #374151; }
  .empty { color: #9ca3af; padding: 28px 0; text-align: center; font-style: italic; font-size: 12px; }
  .row { padding: 8px 10px; border-bottom: 1px solid #f3f4f6; cursor: pointer; border-radius: 4px; }
  .row:hover { background: #fafbfc; }
  .row .top { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; flex-wrap: wrap; }
  .row .title { font-weight: 600; color: #111827; font-size: 13px; }
  .score { font-variant-numeric: tabular-nums; font-weight: 700; color: #2563eb; flex-shrink: 0; font-size: 13px; }
  .row.zero-score .score { color: #b45309; }
  .submeta { color: #6b7280; font-size: 11px; margin-top: 2px; }
  .scorebar { position: relative; height: 4px; background: #e5e7eb; border-radius: 2px; margin-top: 6px; overflow: hidden; }
  .scorebar > span { display: block; height: 100%; background: linear-gradient(90deg,#60a5fa,#2563eb); border-radius: 2px; }
  .detail { display: none; margin-top: 8px; padding-top: 8px; border-top: 1px dashed #e5e7eb; font-size: 11px; }
  .row.expanded { background: #fffbeb; border: 1px solid #fde68a; }
  .row.expanded .detail { display: block; }
  code.jobid { font-size: 10px; color: #6b7280; }
  details.summary-line summary { cursor: pointer; font-weight: 500; color: #1f2937; }
  details.summary-line summary:hover { color: #2563eb; }
  .components { display: flex; gap: 12px; flex-wrap: wrap; font-size: 10px; color: #4b5563; margin-top: 4px; }
  .components .chip { background: #f3f4f6; padding: 1px 6px; border-radius: 4px; font-variant-numeric: tabular-nums; }
  .components .chip b { color: #111827; }
  table.pm { border-collapse: collapse; width: 100%; font-size: 10px; margin-top: 6px; }
  table.pm th, table.pm td { border: 1px solid #e5e7eb; padding: 4px 6px; text-align: left; vertical-align: top; }
  table.pm th { background: #f9fafb; color: #4b5563; font-weight: 600; }
  table.pm .c { font-variant-numeric: tabular-nums; text-align: right; width: 56px; }
  .modalbg { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 100; padding: 28px; overflow-y: auto; }
  .modalbg.open { display: block; }
  .modal { max-width: 720px; margin: 0 auto; background: #fff; border-radius: 8px; padding: 22px 26px; box-shadow: 0 25px 50px rgba(0,0,0,0.25); position: relative; }
  .modal .closebtn { position: absolute; top: 16px; right: 18px; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 4px 12px; border-radius: 4px; cursor: pointer; }
  .modal h2 { margin: 0 0 12px 0; font-size: 17px; }
  .modal p, .modal li { color: #374151; font-size: 12px; line-height: 1.5; }
  .modal code { background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 11px; }
"""


_JS = r"""
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

const usersArr = PAYLOAD.users || [];
const uidMap = new Map(usersArr.map(u => [u.uid, u]));
const META = PAYLOAD.meta || {};

function fmt(n, d) {
  if (n == null || isNaN(n)) return '\u2014';
  return Number(n).toFixed(d == null ? 4 : d);
}

function topScoreOf(u) {
  return (u.recs && u.recs[0]) ? u.recs[0].s : 0;
}

function renderHeader() {
  const m = META;
  document.getElementById('pillMeta').textContent =
    `${m.n_users ?? '?'} users \u00b7 ${m.jobs_source === 'mongo' ? 'Mongo' : 'file'} \u00b7 ${m.n_jobs ?? '?'} jobs \u00b7 top-${m.top_k_stored ?? '?'}`;
  const shortMod = (m.embedding_model_path || '').split(/[\\/]/).pop() || '?';
  document.getElementById('hdrstats').textContent =
    `${m.scorer ?? 'cosine'} \u00b7 dim=${m.embedding_dim ?? '?'} \u00b7 model=${shortMod}`;
}

function renderUserList(q) {
  const f = (q || '').toLowerCase().trim();
  const ul = document.getElementById('ulist');
  ul.innerHTML = '';
  let n = 0;
  for (const u of usersArr) {
    if (f && !(u.uid + ' ' + (u.city || '') + ' ' + (u.prov || '')).toLowerCase().includes(f)) continue;
    if (++n > 2000) break;
    const li = document.createElement('li');
    if (u.uid === activeUid) li.classList.add('active');
    const top = topScoreOf(u);
    li.innerHTML =
      '<div class="uname">' + esc((u.uid || '').substring(0, 36)) + '</div>' +
      '<div class="umeta">' + esc(u.city || '?') + ' \u00b7 ' + esc(u.prov || '?') + ' \u00b7 ' + u.ns + ' skills \u00b7 top mean: <b>' + fmt(top, 3) + '</b></div>';
    li.onclick = () => { activeUid = u.uid; renderUserList(f); renderMain(); };
    ul.appendChild(li);
  }
  if (!ul.children.length) {
    ul.innerHTML = '<li class="empty" style="cursor:default;border:none;padding:18px;">No users match filter.</li>';
  }
}

function renderPerJobTable(rc) {
  const pj = rc.pj || [];
  if (!pj.length) {
    return '<div style="color:#9ca3af;font-style:italic;margin-top:6px;font-size:11px">No job skills embedded (unresolved labels or empty job).</div>';
  }
  const ts = rc.pj_trunc ? ` (showing ${pj.length} of ${rc.n_pj})` : '';
  const rows = pj.map(r =>
    '<tr><td>' + esc(r.jl) + '</td><td>' + esc(r.ul) + '</td><td class="c">' + fmt(r.c, 3) + '</td></tr>'
  ).join('');
  return `<div style="font-size:10px;color:#6b7280;margin:8px 0 4px">Per job requirement: best user skill cosine${ts}</div>` +
    '<table class="pm"><thead><tr><th>Job skill</th><th>Closest user skill</th><th>cos</th></tr></thead><tbody>' + rows + '</tbody></table>';
}

function renderRecRow(rc, maxCombined) {
  const zc = rc.s <= 0 ? ' zero-score' : '';
  const barPct = maxCombined > 0 ? Math.min(100, (rc.s / maxCombined) * 100) : 0;
  return `
<div class="row${zc}" data-uid="">
  <div class="top">
    <div>
      <span style="color:#9ca3af;font-weight:600">#${rc.r}</span>
      <span class="title">${esc(rc.t)}</span>
    </div>
    <span class="score">${fmt(rc.s, 3)}</span>
  </div>
  <div class="submeta">${esc(rc.e)} \u00b7 ${esc(rc.l)}</div>
  <div class="scorebar"><span style="width:${barPct.toFixed(1)}%"></span></div>
  <div class="components">
    <span class="chip">min best cosine: <b>${fmt(rc.mn, 3)}</b></span>
    <span class="chip">user skills in mat: <b>${rc.nu}</b></span>
    <span class="chip">job skills in mat: <b>${rc.nj}</b></span>
  </div>
  <div class="detail">
    <code class="jobid">job_id: ${esc(rc.uuid)}</code>
    <details open class="summary-line"><summary>Skill alignment (${rc.n_pj})</summary>
      ${renderPerJobTable(rc)}
    </details>
  </div>
</div>`;
}

function renderMain() {
  const panel = document.getElementById('mainpanel');
  if (!activeUid) {
    panel.innerHTML = '<div class="empty">Pick a user from the sidebar.</div>';
    return;
  }
  const u = uidMap.get(activeUid);
  if (!u) {
    panel.innerHTML = '<div class="empty">User not found.</div>';
    return;
  }
  const labSuffix = u.skills_trunc ? ` (${u.skills.length} shown of ${u.n_lab})` : '';
  const skillPills = (u.skills || []).map(s => `<span class="skill">${esc(String(s).substring(0, 100))}</span>`).join('');
  panel.innerHTML = `
    <div class="userhead">
      <h2>${esc(u.uid)}</h2>
      <div class="meta">${esc(u.city || '?')} \u00b7 ${esc(u.prov || '?')} \u00b7 ${u.ns} resolved skills${labSuffix} \u00b7 ${u.recs?.length || 0} recommendations in file</div>
      <div style="font-size:10px;color:#9ca3af;margin:0 0 4px">User skills embedded for cosine:</div>
      <div class="skills">${skillPills || '<span style="color:#9ca3af;font-style:italic">no skills</span>'}</div>
    </div>
    <div class="controls">
      <label>Show top <input type="number" id="topNinp" min="1" max="500" value="${topN}"> jobs</label>
    </div>
    <div class="col"><h3>Recommendations (mean best cosine across job requirements)</h3><div id="recsWrap"></div></div>
  `;
  document.getElementById('topNinp').onchange = e => {
    topN = Math.max(1, Math.min(500, parseInt(e.target.value, 10) || 50));
    renderRecs();
  };
  renderRecs();
}

function renderRecs() {
  const u = uidMap.get(activeUid);
  const wrap = document.getElementById('recsWrap');
  if (!wrap || !u) return;
  const slice = (u.recs || []).slice(0, topN);
  if (!slice.length) {
    wrap.innerHTML = '<div class="empty">No recommendations.</div>';
    return;
  }
  let maxCombined = 0;
  for (const r of slice) if (r.s > maxCombined) maxCombined = r.s;
  wrap.innerHTML = slice.map(r => renderRecRow(r, maxCombined)).join('');
  wrap.querySelectorAll('.row').forEach(r => {
    r.onclick = ev => {
      if (ev.target.closest('details') || ev.target.closest('summary') || ev.target.closest('table')) return;
      r.classList.toggle('expanded');
    };
  });
}

function openKey() {
  document.getElementById('modal').innerHTML =
    `<button type="button" class="closebtn" onclick="document.getElementById('modalbg').classList.remove('open')">Close</button>` +
    `<h2>Cosine skill matching</h2>` +
    `<p>Job skills listed (essential + optional order, de-duplicated) are compared against <strong>every</strong> resolved user skill. For each job skill we take <strong>row-wise maximum</strong> cosine. The headline score is the <strong>mean</strong> of those maxima (tie-break: higher min-best-cosine in the runner).</p>` +
    `<p>This is embedding-only — no BM25 or index-overlap ranking here.</p>` +
    `<p style="margin-top:12px;color:#6b7280;font-size:11px">${esc(JSON.stringify({ usersPath: META.users_path }))}</p>`;
  document.getElementById('modalbg').classList.add('open');
}

renderHeader();
document.getElementById('keyBtn').onclick = openKey;
document.getElementById('modalbg').onclick = e => {
  if (e.target === document.getElementById('modalbg')) document.getElementById('modalbg').classList.remove('open');
};
document.getElementById('usearch').oninput = e => renderUserList(e.target.value);
if (usersArr.length) { activeUid = usersArr[0].uid; }
renderUserList('');
renderMain();
"""


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Embed cosine match JSON in HTML.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--max-per-job-skills", type=int, default=80,
        help="Max per-job skill rows embedded per recommendation",
    )
    p.add_argument(
        "--max-user-skill-labels", type=int, default=120,
        help="Max user skill label pills in embedded payload",
    )
    args = p.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    html_doc = build_html(
        payload,
        max_per_job_skills=args.max_per_job_skills,
        max_user_skill_labels=args.max_user_skill_labels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_doc, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
