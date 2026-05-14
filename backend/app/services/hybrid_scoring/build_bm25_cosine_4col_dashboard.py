"""Build standalone HTML from ``run_bm25_cosine_hybrid`` JSON (**fused column only**).

Embeds ``column_fused_weighted_minmax`` only: alpha * pool-normalised cosine plus (1-alpha) * pool-normalised BM25 (see runner ``run_bm25_cosine_hybrid``).

Usage::

    cd backend
    python -m app.services.hybrid_scoring.build_bm25_cosine_4col_dashboard \\
        --input scripts/results_bm25_cosine_hybrid_njila_full_corpus.json \\
        --output scripts/results_bm25_cosine_hybrid_njila_full_corpus_dashboard.html

Truncation flags keep embed size small on large batches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _compact_fused_row(raw: Dict[str, Any], *, max_msk: int) -> Dict[str, Any]:
    msk = [str(x) for x in (raw.get("matched_skills") or [])][:max_msk]
    mcs = [str(x) for x in (raw.get("matched_skills_cosine") or [])][:max_msk]
    return {
        "r": raw.get("rank"),
        "uuid": str(raw.get("job_uuid") or ""),
        "t": str(raw.get("job_title") or "")[:280],
        "e": str(raw.get("employer") or "")[:180],
        "l": str(raw.get("location") or "")[:200],
        "fus": float(raw.get("fusion_score") or raw.get("weighted_minmax_fusion") or 0.0),
        "bn": raw.get("bm25_norm_within_candidates"),
        "cn": raw.get("cos_norm_within_candidates"),
        "mbr": raw.get("mean_best_cosine_raw"),
        "bm_raw": raw.get("bm25_score_raw"),
        "msk": msk,
        "mcs": mcs,
    }


def _compact_user(
    row: Dict[str, Any],
    *,
    max_rows: int,
    max_msk: int,
) -> Dict[str, Any]:
    labs = row.get("user_skill_labels_for_display") or row.get("resolved_user_skill_labels") or []
    if not labs:
        labs = []
    fu = row.get("column_fused_weighted_minmax") or []
    return {
        "uid": str(row.get("user_id") or ""),
        "city": str(row.get("city") or ""),
        "prov": str(row.get("province") or ""),
        "labs": [str(x)[:120] for x in labs[:200]],
        "fused": [_compact_fused_row(r, max_msk=max_msk) for r in fu[:max_rows]],
    }


def build_html(
    payload: Dict[str, Any],
    *,
    max_rows: int,
    max_msk: int,
    max_users: Optional[int],
) -> str:
    cfg = payload.get("config") or {}
    idx = payload.get("index_stats") or {}
    meta = {
        "n_jobs": payload.get("n_jobs"),
        "jobs_source": cfg.get("jobs_source"),
        "users_path": cfg.get("users_path"),
        "alpha_on_cosine": cfg.get("alpha_on_cosine_skill"),
        "variant_bm25": cfg.get("variant_bm25"),
        "skills_weight": cfg.get("skills_weight"),
        "text_weight": cfg.get("text_weight"),
        "embedding_dim": idx.get("embedding_dim"),
        "max_rows_compact": max_rows,
        "max_users_embedded": max_users,
        "fusion": cfg.get("fusion"),
    }

    rows = payload.get("results") or []
    if max_users is not None:
        rows = rows[: max(0, int(max_users))]

    compact = {
        "meta": meta,
        "users": [_compact_user(r, max_rows=max_rows, max_msk=max_msk) for r in rows],
    }
    compact["meta"]["n_users"] = len(compact["users"])

    data_json = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    safe = data_json.replace("</", "<\\/")

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>Fused BM25 × cosine dashboard</title>\n"
        "<style>\n"
        f"{_CSS}\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<header>\n"
        "  <h1>Fused min-max (recommended)</h1>\n"
        '  <span class="pill" id="pillMeta"></span>\n'
        '  <button id="keyBtn">\u2139 Key</button>\n'
        '  <span class="stats" id="hdrstats"></span>\n'
        "</header>\n"
        "<main>\n"
        "  <aside>\n"
        '    <input type="search" id="usearch" placeholder="Search user_id, city, province\u2026">\n'
        '    <ul class="ulist" id="ulist"></ul>\n'
        "  </aside>\n"
        '  <section class="main" id="mainpanel"></section>\n'
        "</main>\n\n"
        '<div class="modalbg" id="modalbg"><div class="modal" id="modal"></div></div>\n\n'
        "<script>\n"
        "const PAYLOAD = "
        f"{safe};\n"
        f"{_JS}\n"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )


_CSS = r"""
  * { box-sizing: border-box; }
  body { font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin: 0; color: #1a1a1a; background: #f6f7f8; }
  header { background: #111827; color: #fff; padding: 10px 16px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  header h1 { font-size: 14px; font-weight: 600; margin: 0 12px 0 0; }
  header .pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 500; background: #374151; color: #d1d5db; }
  header button { font: inherit; padding: 4px 10px; border-radius: 4px; border: 1px solid #4b5563; background: #1f2937; color: #fff; cursor: pointer; }
  header button:hover { background: #374151; }
  .stats { color: #9ca3af; font-size: 11px; flex: 1; text-align: right; }
  main { display: grid; grid-template-columns: 280px 1fr; height: calc(100vh - 46px); }
  aside { background: #fff; border-right: 1px solid #e5e7eb; overflow-y: auto; }
  aside input { width: calc(100% - 24px); padding: 6px 10px; margin: 10px 12px; border: 1px solid #d1d5db; border-radius: 4px; font: inherit; }
  .ulist { list-style: none; margin: 0; padding: 0; }
  .ulist li { padding: 8px 14px; cursor: pointer; border-bottom: 1px solid #f0f0f0; font-size: 12px; }
  .ulist li:hover { background: #f9fafb; }
  .ulist li.active { background: #eff6ff; border-left: 3px solid #2563eb; padding-left: 11px; }
  .ulist .uname { font-weight: 600; color: #111827; font-size: 11px; word-break: break-all; }
  .ulist .umeta { color: #6b7280; font-size: 10px; margin-top: 2px; }
  section.main { overflow-y: auto; padding: 14px 16px; }
  .userhead { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
  .userhead h2 { margin: 0 0 4px 0; font-size: 15px; word-break: break-all; }
  .userhead .meta { color: #6b7280; font-size: 11px; margin-bottom: 8px; }
  .skills { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; max-height: 100px; overflow-y: auto; }
  .skill { background: #eef2ff; color: #3730a3; border-radius: 10px; padding: 1px 8px; font-size: 10px; white-space: nowrap; }
  .controls { display: flex; gap: 14px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; font-size: 11px; color: #4b5563; }
  .controls label { display: flex; align-items: center; gap: 6px; }
  .controls input[type=number] { width: 56px; padding: 4px 6px; border: 1px solid #d1d5db; border-radius: 4px; }
  .col { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; min-width: 0; }
  .col h3 { margin: 0 0 10px 0; font-size: 13px; color: #374151; font-weight: 600; line-height: 1.35; }
  .colwrap { max-height: calc(100vh - 220px); min-height: 240px; overflow-y: auto; }
  .empty { color: #9ca3af; padding: 18px 0; text-align: center; font-style: italic; font-size: 11px; }
  .row { padding: 6px 8px; border-bottom: 1px solid #f3f4f6; cursor: pointer; border-radius: 4px; }
  .row:hover { background: #fafbfc; }
  .row .top { display: flex; justify-content: space-between; gap: 6px; align-items: baseline; flex-wrap: wrap; }
  .row .title { font-weight: 600; color: #111827; font-size: 12px; }
  .score { font-variant-numeric: tabular-nums; font-weight: 700; color: #2563eb; flex-shrink: 0; font-size: 12px; }
  .submeta { color: #6b7280; font-size: 10px; margin-top: 2px; line-height: 1.35; }
  .chip { font-size: 9px; background: #f3f4f6; padding: 1px 5px; border-radius: 4px; color: #4b5563; margin-right: 4px; }
  .detail { display: none; margin-top: 6px; padding-top: 6px; border-top: 1px dashed #e5e7eb; font-size: 10px; }
  .row.expanded { background: #fffbeb; border: 1px solid #fde68a; }
  .row.expanded .detail { display: block; }
  code.jobid { font-size: 9px; color: #6b7280; }
  .matchsec { margin-top: 8px; }
  .matchsec .lbl { font-size: 9px; font-weight: 600; color: #374151; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 3px; }
  ul.compact { margin: 4px 0 0 14px; padding: 0; }
  ul.compact li { margin: 2px 0; color: #374151; }
  .modalbg { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 100; padding: 28px; overflow-y: auto; }
  .modalbg.open { display: block; }
  .modal { max-width: 720px; margin: 0 auto; background: #fff; border-radius: 8px; padding: 22px 26px; box-shadow: 0 25px 50px rgba(0,0,0,0.25); position: relative; }
  .modal .closebtn { position: absolute; top: 16px; right: 18px; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 4px 12px; border-radius: 4px; cursor: pointer; }
  .modal h2 { margin: 0 0 12px 0; font-size: 17px; }
  .modal p, .modal li { color: #374151; font-size: 12px; line-height: 1.5; }
"""

_JS = r"""
let activeUid = null;
let topN = 50;

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function fmt(v, n) {
  const x = Number(v);
  if (!Number.isFinite(x)) return '\u2014';
  return x.toFixed(n == null ? 3 : n);
}

const META = PAYLOAD.meta || {};
const usersArr = PAYLOAD.users || [];
const uidMap = new Map(usersArr.map(u => [u.uid, u]));

function topFusedHint(u) {
  const fu = u.fused && u.fused[0];
  if (!fu) return '\u2014';
  return fmt(fu.fus, 3);
}

function renderHeader() {
  document.getElementById('pillMeta').textContent =
    `${META.n_users ?? '?'} users (embedded max) \u00b7 ${META.jobs_source === 'mongo' ? 'Mongo' : 'file'} \u00b7 ${META.n_jobs ?? '?'} jobs \u00b7 fused column only \u00b7 \u2264${META.max_rows_compact ?? '?'} rows`;
  document.getElementById('hdrstats').textContent =
    `alpha on cosine = ${META.alpha_on_cosine ?? '?'} \u00b7 BM25 variant = ${META.variant_bm25 ?? '?'} \u00b7 embed dim ${META.embedding_dim ?? '?'}`;
}

function matchedList(items) {
  if (!items || !items.length) return '<span style="color:#9ca3af">none</span>';
  return `<ul class="compact">${items.map(x => `<li>${esc(String(x).substring(0, 140))}</li>`).join('')}</ul>`;
}

function renderFusedRow(r) {
  const chips =
    `<span class="chip">bm_norm=${fmt(r.bn, 3)}</span>` +
    `<span class="chip">cos_norm=${fmt(r.cn, 3)}</span>` +
    `<span class="chip">cos_raw=${fmt(r.mbr, 3)}</span>` +
    `<span class="chip">BM25_raw=${fmt(r.bm_raw, 2)}</span>`;
  const bm25Sec =
    `<div class="matchsec"><div class="lbl">Taxonomy phrase overlap (BM25)</div>${matchedList(r.msk)}</div>`;
  const cosSec =
    `<div class="matchsec"><div class="lbl">Embedding alignment (skills cosine)</div>` +
    `${matchedList(r.mcs)}</div>`;
  return `
<div class="row">
  <div class="top"><div><span style="color:#9ca3af;font-weight:600">#${r.r}</span> <span class="title">${esc(r.t)}</span></div>
    <span class="score">${fmt(r.fus, 4)}</span></div>
  <div class="submeta">${esc(r.e)} \u00b7 ${esc(r.l)} ${chips}</div>
  <div class="detail"><code class="jobid">${esc(r.uuid)}</code>${bm25Sec}${cosSec}</div>
</div>`;
}

function setupWrap(wrap) {
  wrap.querySelectorAll('.row').forEach(r => {
    r.onclick = ev => {
      if (ev.target.closest('details') || ev.target.closest('summary')) return;
      r.classList.toggle('expanded');
    };
  });
}

function renderMain() {
  const panel = document.getElementById('mainpanel');
  if (!activeUid) {
    panel.innerHTML = '<div class="empty">Pick a user.</div>';
    return;
  }
  const u = uidMap.get(activeUid);
  if (!u) {
    panel.innerHTML = '<div class="empty">User missing.</div>';
    return;
  }
  const sk = (u.labs || []).slice(0, 200).map(s => `<span class="skill">${esc(s)}</span>`).join('');
  panel.innerHTML = `
    <div class="userhead">
      <h2>${esc(u.uid)}</h2>
      <div class="meta">${esc(u.city)} \u00b7 ${esc(u.prov)} \u00b7 ${u.labs.length} labels embedded</div>
      <div class="skills">${sk || '<span style="color:#9ca3af;font-style:italic">no labels</span>'}</div>
    </div>
    <div class="controls">
      <label>Show top <input type="number" id="topNinp" min="1" max="500" value="${topN}"> fused rows</label>
    </div>
    <div class="col"><h3>column_fused_weighted_minmax (pool min-max fusion, recommended ranking)</h3><div class="colwrap" id="wFu"></div></div>`;

  document.getElementById('topNinp').onchange = e => {
    topN = Math.max(1, Math.min(500, parseInt(e.target.value, 10) || 50));
    renderMain();
  };

  const el = document.getElementById('wFu');
  const slice = (u.fused || []).slice(0, topN);
  if (!el) return;
  if (!slice.length) el.innerHTML = '<div class="empty">no fused recommendations</div>';
  else el.innerHTML = slice.map(renderFusedRow).join('');
  setupWrap(el);
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
    li.innerHTML =
      '<div class="uname">' + esc((u.uid || '').substring(0, 36)) + '</div>' +
      '<div class="umeta">' + esc(u.city || '?') + ' \u00b7 ' + esc(u.prov || '?') +
      ' \u00b7 top fused score: <b>' + topFusedHint(u) + '</b></div>';
    li.onclick = () => { activeUid = u.uid; renderUserList(f); renderMain(); };
    ul.appendChild(li);
  }
  if (!ul.children.length) {
    ul.innerHTML = '<li class="empty" style="cursor:default;border:none;padding:18px;">No users.</li>';
  }
}

function openKey() {
  document.getElementById('modal').innerHTML =
    '<button type="button" class="closebtn" onclick="document.getElementById(\'modalbg\').classList.remove(\'open\')">Close</button>' +
    '<h2>Fused BM25 × cosine column</h2>' +
    '<p>This dashboard embeds only <code>column_fused_weighted_minmax</code> from <code>run_bm25_cosine_hybrid</code> JSON.</p>' +
    '<p>Jobs are ranked by weighted min-max fusion within the BM25∩cosine candidate pool: '
    + 'alpha&nbsp;&times;&nbsp;normalised cosine + (1\u2212alpha)&nbsp;&times;&nbsp;normalised BM25.</p>'
    + '<p>See <code>config.alpha_on_cosine_skill</code> in the JSON source.</p>';
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
    parser = argparse.ArgumentParser(
        description="Build standalone HTML dashboard from bm25_cosine_hybrid JSON output.",
    )
    parser.add_argument("--input", type=Path, required=True, help="results_*.json from run_bm25_cosine_hybrid")
    parser.add_argument("--output", type=Path, required=True, help="path to write .html")
    parser.add_argument(
        "--max-rows-per-column",
        type=int,
        default=80,
        help="Max fused recommendation rows embedded per user (default 80)",
    )
    parser.add_argument(
        "--max-matched-skills",
        type=int,
        default=25,
        help="Matched skill strings per job (default 25)",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Only embed first N users (omit for all in JSON)",
    )
    args = parser.parse_args(argv)

    inp = args.input.expanduser()
    if not inp.is_file():
        print(f"[4col_dashboard] input not found: {inp}", file=sys.stderr)
        return 2
    payload = json.loads(inp.read_text(encoding="utf-8"))
    res = payload.get("results") or []
    if not res:
        print("[4col_dashboard] JSON has no results[] — wrong file?", file=sys.stderr)
        return 2
    first = res[0]
    if "column_fused_weighted_minmax" not in first:
        print("[4col_dashboard] results[0] missing column_fused_weighted_minmax — pass output from run_bm25_cosine_hybrid", file=sys.stderr)
        return 2

    html_doc = build_html(
        payload,
        max_rows=max(1, args.max_rows_per_column),
        max_msk=max(1, args.max_matched_skills),
        max_users=args.max_users,
    )
    out = args.output.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    print(f"[4col_dashboard] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
