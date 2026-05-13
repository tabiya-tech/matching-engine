"""Build a self-contained HTML dashboard from ``bm25_*_results.json``.

Mirrors the look-and-feel of ``index_based_matching/index_match_*_dashboard.html``
(dark header, sidebar user list + search, expandable job rows), but visualises
BM25:

- Combined per-job score with a horizontal bar.
- Raw component scores (skills-only, full-text) for hybrid runs.
- ``matched_skills`` / ``matched_skills_detail`` — taxonomy overlaps (phrase
  tokens plus optional user vs job wording for each overlap).

Usage::

    cd backend
    python -m app.services.bm25_scoring.build_bm25_dashboard \\
        --input ./path/to/bm25_catalog_results.json \\
        --output ./path/to/bm25_dashboard.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _compact_rec(raw: Dict[str, Any], *, max_matched_skills: int) -> Dict[str, Any]:
    msk_full = [str(x) for x in (raw.get("matched_skills") or [])]
    msk_take = msk_full[:max_matched_skills]
    ov_compact: List[Dict[str, str]] = []
    for d in raw.get("matched_skills_detail") or []:
        if not isinstance(d, dict):
            continue
        ov_compact.append({
            "p": str(d.get("phrase_token") or ""),
            "u": str(d.get("user_label") or "")[:200],
            "j": str(d.get("job_label") or "")[:200],
        })
    ov_full_n = len(ov_compact)
    ov_take = ov_compact[:max_matched_skills]
    return {
        "r": raw.get("rank"),
        "uuid": str(raw.get("job_uuid") or ""),
        "t": str(raw.get("job_title") or "")[:280],
        "e": str(raw.get("employer") or "")[:180],
        "l": str(raw.get("location") or "")[:200],
        "s": float(raw.get("bm25_score") or 0.0),
        "ssr": (
            float(raw["bm25_skills_score_raw"])
            if "bm25_skills_score_raw" in raw
            else None
        ),
        "tsr": (
            float(raw["bm25_text_score_raw"])
            if "bm25_text_score_raw" in raw
            else None
        ),
        "sr": (
            float(raw["bm25_score_raw"])
            if "bm25_score_raw" in raw
            else None
        ),
        "msk": [s[:120] for s in msk_take],
        "n_msk": len(msk_full),
        "msk_trunc": len(msk_full) > max_matched_skills,
        "ov": ov_take,
        "n_ov": ov_full_n,
        "ov_trunc": ov_full_n > max_matched_skills,
    }


def _compact_user(row: Dict[str, Any], *, max_matched_skills: int) -> Dict[str, Any]:
    return {
        "uid": str(row.get("user_id") or ""),
        "city": str(row.get("city") or ""),
        "prov": str(row.get("province") or ""),
        "nq": int(row.get("n_query_tokens") or 0),
        "q": [str(t) for t in (row.get("query_tokens") or [])],
        "recs": [
            _compact_rec(r, max_matched_skills=max_matched_skills)
            for r in (row.get("recommendations") or [])
        ],
    }


def build_html(payload_raw: Dict[str, Any], *, max_matched_skills: int = 120) -> str:
    cfg = payload_raw.get("config") or {}
    idx_stats = payload_raw.get("index_stats") or {}
    meta = {
        "n_users": payload_raw.get("n_users"),
        "n_jobs": payload_raw.get("n_jobs"),
        "variant": cfg.get("variant"),
        "k1": cfg.get("k1"),
        "b": cfg.get("b"),
        "skills_weight": cfg.get("skills_weight"),
        "text_weight": cfg.get("text_weight"),
        "jobs_source": cfg.get("jobs_source"),
        "top_k_stored": cfg.get("top_k"),
        "users_path": cfg.get("users_path"),
        "mongo_filter_by_users": cfg.get("mongo_filter_by_users"),
        "include_programme_context": cfg.get("include_programme_context"),
        "skills_vocab_size": idx_stats.get("skills_vocab_size"),
        "full_vocab_size": idx_stats.get("full_vocab_size"),
        "skills_avg_doc_len": idx_stats.get("skills_avg_doc_len"),
        "full_avg_doc_len": idx_stats.get("full_avg_doc_len"),
    }
    compact = {
        "meta": meta,
        "users": [
            _compact_user(u, max_matched_skills=max_matched_skills)
            for u in payload_raw.get("results") or []
        ],
    }
    data_json = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    # Avoid prematurely closing the embedding <script>: </ -> <\/ in any string.
    safe = data_json.replace("</", "<\\/")

    tpl = (
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BM25 match dashboard</title>
<style>
%(css)s
</style>
</head>
<body>
<header>
  <h1>BM25 match dashboard</h1>
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
  .skills { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; max-height: 120px; overflow-y: auto; }
  .skill { background: #eef2ff; color: #3730a3; border-radius: 10px; padding: 1px 8px; font-size: 10px; white-space: nowrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  table.ov { border-collapse: collapse; width: 100%; font-size: 10px; margin-top: 6px; }
  table.ov th, table.ov td { border: 1px solid #e5e7eb; padding: 4px 6px; text-align: left; vertical-align: top; }
  table.ov th { background: #f9fafb; color: #4b5563; font-weight: 600; }
  table.ov col.phr { width: 28%; }
  table.ov .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #3730a3; }
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
const IS_HYBRID = META.variant === 'hybrid';

function fmt(n, d) {
  if (n == null || isNaN(n)) return '\u2014';
  return Number(n).toFixed(d == null ? 3 : d);
}

function topScoreOf(u) {
  return (u.recs && u.recs[0]) ? u.recs[0].s : 0;
}

function renderHeader() {
  const m = META;
  const variantPill = m.variant === 'hybrid'
    ? `hybrid (skills ${m.skills_weight ?? '?'} / text ${m.text_weight ?? '?'})`
    : 'single (full-text)';
  document.getElementById('pillMeta').textContent =
    `${m.n_users ?? '?'} users \u00b7 ${m.jobs_source === 'mongo' ? 'Mongo' : 'file'} \u00b7 ${m.n_jobs ?? '?'} jobs \u00b7 top-${m.top_k_stored ?? '?'} \u00b7 ${variantPill}`;
  document.getElementById('hdrstats').textContent =
    `rank_bm25 k1=${m.k1 ?? '?'} b=${m.b ?? '?'} \u00b7 vocab: skills=${m.skills_vocab_size ?? '?'}, full=${m.full_vocab_size ?? '?'} \u00b7 avg|d|: skills=${m.skills_avg_doc_len ?? '?'}, full=${m.full_avg_doc_len ?? '?'}`;
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
      '<div class="umeta">' + esc(u.city || '?') + ' \u00b7 ' + esc(u.prov || '?') + ' \u00b7 ' + u.nq + ' tokens \u00b7 top score: <b>' + fmt(top, 3) + '</b></div>';
    li.onclick = () => { activeUid = u.uid; renderUserList(f); renderMain(); };
    ul.appendChild(li);
  }
  if (!ul.children.length) {
    ul.innerHTML = '<li class="empty" style="cursor:default;border:none;padding:18px;">No users match filter.</li>';
  }
}

function renderMatchedSkills(rc) {
  const arr = rc.msk || [];
  const ov = rc.ov || [];
  if (!arr.length && !ov.length) {
    return '<div style="color:#9ca3af;font-style:italic;font-size:11px;margin-top:6px">No taxonomy skill phrases overlap between this user\u2019s declared skills and this job\u2019s essential \u222a optional skills.</div>';
  }
  const suffix = rc.msk_trunc ? ` (showing phrase pills ${arr.length} of ${rc.n_msk})` : '';
  let pillsHtml = '';
  if (arr.length) {
    pillsHtml =
      '<div style="font-size:10px;color:#6b7280;margin-bottom:4px">Matched phrase tokens (same string on both sides after normalisation) \u2014 '
      + rc.n_msk + ' total' + suffix + '</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:4px">' +
      arr.map(x => `<span class="skill">${esc(String(x))}</span>`).join('') +
      '</div>';
  }
  let tableHtml = '';
  if (ov.length) {
    const ts = rc.ov_trunc ? ` (showing ${ov.length} of ${rc.n_ov})` : '';
    const rows = ov.map(r =>
      '<tr>' +
      `<td class="mono">${esc(r.p)}</td>` +
      `<td>${esc(r.u)}</td>` +
      `<td>${esc(r.j)}</td>` +
      '</tr>'
    ).join('');
    tableHtml =
      `<div style="font-size:10px;color:#6b7280;margin:10px 0 4px">How each side phrases that overlap${ts}</div>` +
      '<table class="ov"><colgroup><col class="phr"/><col/><col/></colgroup>' +
      '<thead><tr><th>Phrase token</th><th>On user profile</th><th>On job posting</th></tr></thead>' +
      `<tbody>${rows}</tbody></table>` +
      `<div style="font-size:9px;color:#9ca3af;margin-top:4px">Both columns refer to the same underlying skill key; wording can differ slightly before underscore normalisation.</div>`;
  }
  return `<div style="margin-top:6px">${pillsHtml}${tableHtml}</div>`;
}

function renderRecRow(rc, maxCombined) {
  const zc = rc.s <= 0 ? ' zero-score' : '';
  const barPct = maxCombined > 0 ? Math.min(100, (rc.s / maxCombined) * 100) : 0;

  let compsHtml = '';
  if (IS_HYBRID) {
    compsHtml = `<div class="components">
      <span class="chip">skills raw: <b>${fmt(rc.ssr, 2)}</b></span>
      <span class="chip">text raw: <b>${fmt(rc.tsr, 2)}</b></span>
    </div>`;
  } else if (rc.sr != null) {
    compsHtml = `<div class="components"><span class="chip">raw BM25: <b>${fmt(rc.sr, 2)}</b></span></div>`;
  }

  const scoreLabel = IS_HYBRID ? fmt(rc.s, 3) : fmt(rc.s, 2);
  const mskN = rc.n_msk != null ? rc.n_msk : (rc.msk || []).length;

  return `
<div class="row${zc}" data-uid="">
  <div class="top">
    <div>
      <span style="color:#9ca3af;font-weight:600">#${rc.r}</span>
      <span class="title">${esc(rc.t)}</span>
    </div>
    <span class="score">${scoreLabel}</span>
  </div>
  <div class="submeta">${esc(rc.e)} \u00b7 ${esc(rc.l)}</div>
  <div class="scorebar"><span style="width:${barPct.toFixed(1)}%"></span></div>
  ${compsHtml}
  <div class="detail">
    <code class="jobid">job_id: ${esc(rc.uuid)}</code>
    <details open class="summary-line"><summary>Matched taxonomy skills (${mskN})</summary>
      ${renderMatchedSkills(rc)}
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
  const tokenPills = (u.q || []).slice(0, 200).map(t =>
    `<span class="skill">${esc(String(t).substring(0, 60))}</span>`
  ).join('');
  const ttitle = IS_HYBRID
    ? 'Recommendations (hybrid BM25: skills + full text, min-max blended)'
    : 'Recommendations (BM25 over full-text document)';
  panel.innerHTML = `
    <div class="userhead">
      <h2>${esc(u.uid)}</h2>
      <div class="meta">${esc(u.city || '?')} \u00b7 ${esc(u.prov || '?')} \u00b7 ${u.nq} BM25 query tokens \u00b7 ${u.recs?.length || 0} recommendations in file</div>
      <div style="font-size:10px;color:#9ca3af;margin:0 0 4px">Skill phrases sent to BM25${META.include_programme_context ? ' (+ programme words)' : ''}:</div>
      <div class="skills">${tokenPills || '<span style="color:#9ca3af;font-style:italic">no tokens</span>'}</div>
    </div>
    <div class="controls">
      <label>Show top <input type="number" id="topNinp" min="1" max="500" value="${topN}"> jobs</label>
    </div>
    <div class="col"><h3>${esc(ttitle)}</h3><div id="recsWrap"></div></div>
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
      if (ev.target.closest('details') || ev.target.closest('summary')) return;
      r.classList.toggle('expanded');
    };
  });
}

function openKey() {
  const m = META;
  document.getElementById('modal').innerHTML =
    `<button type="button" class="closebtn" onclick="document.getElementById('modalbg').classList.remove('open')">Close</button>` +
    `<h2>BM25 matching</h2>` +
    `<p>Job documents are indexed by <code>rank_bm25.BM25Okapi</code>. Job skills become phrase tokens (e.g. <code>apply_teaching_strategies</code>); title, employer, location, description are split into words. Programme / institution strings are <strong>not</strong> merged into documents \u2014 only optionally into the <em>query</em> when enabled for that run.</p>` +
    `<p><strong>k1</strong> controls term-frequency saturation, <strong>b</strong> controls length normalisation (0 = ignore length, 1 = full).</p>` +
    `<p><strong>Variant:</strong> <code>${esc(m.variant)}</code>. ` +
    (m.variant === 'hybrid'
      ? `Two BM25 indexes \u2014 one over skill phrases only, one over title + employer + location + skills + description. Each user's two score vectors are min-max normalised per-user and blended <code>${m.skills_weight ?? '?'}</code> skills + <code>${m.text_weight ?? '?'}</code> text.`
      : `One BM25 index over title + employer + location + skills + description.`) + `</p>` +
    `<p><strong>Matched taxonomy skills</strong> lists only overlaps: skills the user declares <em>and</em> the job lists as essential \u222a optional \u2014 after phrase normalisation (\u2264 one token per skill). The optional table repeats the overlap with separate \u201cuser profile\u201d vs \u201cjob posting\u201d wording.</p>` +
    `<p><strong>BM25 query</strong>: by default this is skill phrase tokens only. <strong>Programme / institution / school year</strong> loose words ${m.include_programme_context === true ? '<b>were added</b> to the query (\u2260 matched skills).' : '<b>were not</b> added to the query (\u2260 matched skills).'}</p>` +
    `<p style="margin-top:12px;color:#6b7280;font-size:11px">Embedded from rank_bm25 output \u00b7 ${esc(JSON.stringify({ users: m.users_path, src: m.jobs_source }))}</p>`;
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
    p = argparse.ArgumentParser(
        description="Embed BM25 results into a standalone HTML dashboard."
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Path to *_results.json from bm25library")
    p.add_argument("--output", type=Path, required=True,
                   help="Output .html path")
    p.add_argument("--max-matched-skills", type=int, default=120,
                   help="Max matched taxonomy skill strings to embed per job")
    args = p.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    html_doc = build_html(payload, max_matched_skills=args.max_matched_skills)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_doc, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
