"""Build self-contained HTML dashboard from skills-graph match JSON."""

from __future__ import annotations

import json
from pathlib import Path


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Skills Graph Match Dashboard</title>
<style>
  * { box-sizing: border-box; }
  body { font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin: 0; color: #1a1a1a; background: #f6f7f8; }
  header { background: #111827; color: #fff; padding: 10px 16px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  header h1 { font-size: 14px; font-weight: 600; margin: 0 12px 0 0; }
  header select { font: inherit; padding: 4px 8px; border-radius: 4px; border: 1px solid #374151; background: #1f2937; color: #fff; }
  header label { color: #d1d5db; font-size: 11px; margin-right: 4px; }
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
  .ulist .uname { font-weight: 600; color: #111827; }
  .ulist .umeta { color: #6b7280; font-size: 10px; margin-top: 2px; }
  section.main { overflow-y: auto; padding: 14px 18px; }
  .userhead { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; }
  .userhead h2 { margin: 0 0 4px 0; font-size: 16px; }
  .userhead .meta { color: #6b7280; font-size: 11px; margin-bottom: 6px; }
  .controls { display: flex; gap: 14px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; font-size: 11px; color: #4b5563; }
  .controls label { display: flex; align-items: center; gap: 4px; }
  .controls input[type=number] { width: 50px; padding: 3px 5px; }
  .alphas { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .col { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 10px; min-width: 0; }
  .col h3 { margin: 0 0 6px 0; font-size: 12px; color: #374151; line-height: 1.3; }
  .col .empty { color: #9ca3af; padding: 12px 0; text-align: center; font-style: italic; font-size: 11px; }
  .row { padding: 6px 0; border-bottom: 1px solid #f3f4f6; cursor: pointer; }
  .row:hover { background: #fafbfc; }
  .row .top { display: flex; justify-content: space-between; gap: 6px; align-items: baseline; }
  .row .title { font-weight: 600; color: #111827; font-size: 12px; }
  .row .score { color: #2563eb; font-variant-numeric: tabular-nums; font-weight: 600; flex-shrink: 0; font-size: 12px; }
  .row .submeta { color: #6b7280; font-size: 10px; margin-top: 1px; }
  .row.expanded { background: #fffbeb; padding: 8px 10px; border: 1px solid #fde68a; border-radius: 4px; }
  .detail { display: none; margin-top: 6px; padding: 6px 0; border-top: 1px dashed #e5e7eb; font-size: 11px; }
  .row.expanded .detail { display: block; }
  .scoregrid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 3px 10px; margin: 4px 0; padding: 5px; background: #f9fafb; border-radius: 4px; font-size: 10px; font-variant-numeric: tabular-nums; }
  .scoregrid .label { color: #6b7280; }
  .scoregrid .value { color: #111827; font-weight: 500; }
  .matchlist { margin: 6px 0; padding-left: 0; list-style: none; font-size: 10px; }
  .matchlist li { padding: 1px 0; color: #374151; }
  .matchlist .ok .checkmark { color: #047857; font-weight: 600; }
  .matchlist .bad .checkmark { color: #b91c1c; font-weight: 600; }
  .matchlist .lbl-u { background: #eef2ff; color: #3730a3; padding: 0 5px; border-radius: 7px; font-size: 9px; font-weight: 600; margin-right: 3px; }
  .matchlist .lbl-j { background: #fef3c7; color: #92400e; padding: 0 5px; border-radius: 7px; font-size: 9px; font-weight: 600; margin-right: 3px; }
  .matchlist .skill-u { color: #3730a3; }
  .matchlist .skill-j { color: #92400e; }
  .matchlist .sim { color: #6b7280; font-variant-numeric: tabular-nums; margin-left: 4px; }
  .skillpill { display: inline-block; background: #f3f4f6; color: #374151; padding: 1px 6px; border-radius: 7px; font-size: 9px; margin: 1px; }
  details summary { cursor: pointer; padding: 1px 0; color: #4b5563; font-size: 10px; user-select: none; }
  .skills { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
  .skill { background: #eef2ff; color: #3730a3; border-radius: 10px; padding: 1px 8px; font-size: 10px; white-space: nowrap; }
  .modalbg { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 100; padding: 28px; overflow-y: auto; }
  .modalbg.open { display: block; }
  .modal { max-width: 920px; margin: 0 auto; background: #fff; border-radius: 8px; padding: 24px 28px; box-shadow: 0 25px 50px rgba(0,0,0,0.25); }
  .modal h2 { margin: 0 0 8px 0; font-size: 18px; }
  .modal p { margin: 6px 0; line-height: 1.5; color: #374151; font-size: 12px; }
  .modal .closebtn { float: right; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 4px 12px; border-radius: 4px; cursor: pointer; }
  .modal code { background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>Skills Graph Match</h1>
  <label>A:</label><select id="selA"></select>
  <label>B:</label><select id="selB"></select>
  <button id="swapBtn">⇄ Swap</button>
  <button id="keyBtn">📖 Key</button>
  <span class="stats" id="hdrstats"></span>
</header>
<main>
  <aside>
    <input type="search" id="usearch" placeholder="Search user_id or programme…">
    <ul class="ulist" id="ulist"></ul>
  </aside>
  <section class="main" id="mainpanel">
    <div class="empty" style="text-align:center;padding:40px;color:#9ca3af;font-style:italic">Pick a user from the sidebar.</div>
  </section>
</main>
<div class="modalbg" id="modalbg"><div class="modal" id="modal"></div></div>
<script>
const META = __META_JSON__;
const DATA = __DATA_JSON__;
let selA = "exact_match";
let selB = "final";
let activeUid = null;
let topN = 10;

function escape(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function dataset(key) { return DATA[key] || []; }

function buildSelectors() {
  const optsHtml = Object.keys(META).map(k =>
    `<option value="${escape(k)}">${escape(META[k].label)}</option>`
  ).join('');
  document.getElementById('selA').innerHTML = optsHtml;
  document.getElementById('selB').innerHTML = optsHtml;
  document.getElementById('selA').value = selA;
  document.getElementById('selB').value = selB;
}

function buildUserList() {
  const A = dataset(selA);
  const items = A.map(u => ({ uid: u.uid, name: u.name || u.uid, city: u.city, prov: u.prov, ns: u.ns, mapped: u.mapped }));
  items.sort((x, y) => (x.name || '').localeCompare(y.name || ''));
  return items;
}

function renderUserList(filter) {
  const items = buildUserList();
  const f = (filter || '').toLowerCase();
  const filtered = f ? items.filter(u =>
    (u.name||'').toLowerCase().includes(f) || (u.uid||'').toLowerCase().includes(f) || (u.city||'').toLowerCase().includes(f)
  ) : items;
  const ul = document.getElementById('ulist');
  ul.innerHTML = '';
  for (const u of filtered) {
    const li = document.createElement('li');
    if (u.uid === activeUid) li.classList.add('active');
    li.innerHTML =
      `<div class="uname">${escape((u.name||'').substring(0,36) || u.uid.substring(0,16))}</div>` +
      `<div class="umeta">${escape(u.uid.substring(0,22))} · ${escape(u.city||'?')} · ${u.ns} skills · ${u.mapped} mapped</div>`;
    li.onclick = () => { activeUid = u.uid; renderUserList(filter); renderMain(); };
    ul.appendChild(li);
  }
}

function renderHeaderStats() {
  document.getElementById('hdrstats').textContent =
    `${META[selA]?.label || selA} vs ${META[selB]?.label || selB} · ${dataset(selA).length} users · ${META.jobs || '?'} jobs`;
}

function getUser(key, uid) {
  return dataset(key).find(u => u.uid === uid);
}

function recsFor(user, method) {
  if (!user) return [];
  if (method === 'exact_match') return user.recs_exact || [];
  if (method === 'graph_dijkstra') return user.recs_dijkstra || [];
  if (method === 'final') return user.recs_final || [];
  return [];
}

function renderMain() {
  const panel = document.getElementById('mainpanel');
  if (!activeUid) {
    panel.innerHTML = '<div class="empty" style="text-align:center;padding:40px;color:#9ca3af;font-style:italic">Pick a user from the sidebar.</div>';
    return;
  }
  const u = getUser(selA, activeUid) || getUser(selB, activeUid);
  const skillsHtml = (u.user_skills || []).map(s => `<span class="skill">${escape(s)}</span>`).join('');
  panel.innerHTML = `
    <div class="userhead">
      <h2>${escape(u.name || u.uid)}</h2>
      <div class="meta">user_id: <code>${escape(u.uid)}</code> · ${escape(u.city||'?')} · ${escape(u.prov||'?')} · ${u.ns} skills · ${u.mapped}/${u.ns} mapped to taxonomy</div>
      <div class="skills">${skillsHtml}</div>
    </div>
    <div class="controls">
      <label>Top <input type="number" id="topN" min="1" max="128" value="${topN}"></label>
    </div>
    <div class="alphas">
      <div class="col"><h3>${escape(META[selA].label)}</h3><div id="recsA"></div></div>
      <div class="col"><h3>${escape(META[selB].label)}</h3><div id="recsB"></div></div>
    </div>`;
  document.getElementById('topN').onchange = (e) => { topN = parseInt(e.target.value, 10) || 10; renderRecs(); };
  renderRecs();
}

function renderRecs() {
  const a = getUser(selA, activeUid);
  const b = getUser(selB, activeUid);
  document.getElementById('recsA').innerHTML = a ? renderRecList(recsFor(a, selA), selA) : '<div class="empty">not in dataset</div>';
  document.getElementById('recsB').innerHTML = b ? renderRecList(recsFor(b, selB), selB) : '<div class="empty">not in dataset</div>';
  document.querySelectorAll('.row').forEach(r => {
    r.onclick = (e) => { if (e.target.closest('details')) return; r.classList.toggle('expanded'); };
  });
}

function renderRecList(recs, method) {
  const filt = recs.slice(0, topN);
  if (!filt.length) return '<div class="empty">no matches</div>';
  return filt.map(r => renderRow(r, method)).join('');
}

function renderRow(r, method) {
  const score = (r.f ?? 0).toFixed(4);
  const graphHeader = `${r.em ?? 0} exact · avg ${Number(r.ad ?? 0).toFixed(2)} hops`;
  let submeta = `${escape(r.e)} · ${escape(r.l||'?')}`;
  if (method === 'exact_match') submeta += ` · ${r.mc||0}/${r.jsc||r.ns||'?'} skills matched`;
  if (method === 'graph_dijkstra') submeta += ` · min ${r.mind??'?'} · max ${r.maxd??'?'} dist · reach ${r.reach||0}/${r.tjs||'?'}`;
  if (method === 'final') {
    if (r.src === 'exact') {
      submeta += ` · ✓ exact match · ${r.mc||0}/${r.jsc||'?'} skills` + (r.gr ? ` · graph agrees #${r.gr}` : '');
    } else {
      submeta += ` · ~ related via graph · graph #${r.gr??'?'} · avg dist ${Number(r.raw_dist ?? r.ad ?? 0).toFixed(2)}`;
    }
  }

  const matches = (r.ms || []).map(m => {
    const u = m[0], j = m[1], sim = m[2], ok = m[3], dist = m[4];
    if (method === 'graph_dijkstra') {
      const distTxt = dist === 0 ? 'exact (0 dist)' : (dist > 0 ? `dist ${dist}` : 'no path');
      return `<li class="${ok?'ok':'bad'}"><span class="checkmark">${ok?'✓':'~'}</span> <span class="lbl-u">USER</span><span class="skill-u">${escape(u)}</span> ↔ <span class="lbl-j">JOB</span><span class="skill-j">${escape(j)}</span> <span class="sim">${distTxt}</span></li>`;
    }
    if (method === 'final') {
      return `<li class="ok"><span class="checkmark">✓</span> <span class="lbl-j">JOB</span><span class="skill-j">${escape(j)}</span> ↔ <span class="lbl-u">USER</span><span class="skill-u">${escape(u)}</span></li>`;
    }
    return `<li class="ok"><span class="checkmark">✓</span> <span class="lbl-j">JOB</span><span class="skill-j">${escape(j)}</span> ↔ <span class="lbl-u">USER</span><span class="skill-u">${escape(u)}</span> <span class="sim">${Number(sim).toFixed(3)}</span></li>`;
  }).join('');

  const jobEss = (r.je || []).map(s => `<span class="skillpill">${escape(s)}</span>`).join('');
  const jobOpt = (r.jo || []).map(s => `<span class="skillpill">${escape(s)}</span>`).join('');

  let scoregrid = '';
  if (method === 'graph_dijkstra') {
    scoregrid = `<div class="scoregrid">
      <div><span class="label">rank by:</span> <span class="value">exact ↓, avg_dist ↑</span></div>
      <div><span class="label">exact:</span> <span class="value">${r.em??0}/${r.tjs??'?'}</span></div>
      <div><span class="label">avg_dist:</span> <span class="value">${r.ad??'?'}</span></div>
      <div><span class="label">min_dist:</span> <span class="value">${r.mind??'?'}</span></div>
      <div><span class="label">max_dist:</span> <span class="value">${r.maxd??'?'}</span></div>
      <div><span class="label">reachable:</span> <span class="value">${r.reach??0}</span></div>
    </div>`;
  } else if (method !== 'final') {
    scoregrid = `<div class="scoregrid">
      <div><span class="label">coverage:</span> <span class="value">${score}</span></div>
      <div><span class="label">matched:</span> <span class="value">${r.mc??0}</span></div>
      <div><span class="label">job skills:</span> <span class="value">${r.jsc??r.ns??'?'}</span></div>
    </div>`;
  }

  const scoreHeader = method === 'graph_dijkstra' ? graphHeader : (method === 'final' ? '' : score);

  return `
    <div class="row">
      <div class="top"><div><span style="color:#9ca3af">#${r.r}</span> <span class="title">${escape(r.t)}</span></div>${scoreHeader ? `<span class="score">${scoreHeader}</span>` : ''}</div>
      <div class="submeta">${submeta}</div>
      <div class="detail">
        ${scoregrid}
        ${matches ? `<details open><summary>Skill matches</summary><ul class="matchlist">${matches}</ul></details>` : ''}
        ${jobEss ? `<details><summary>Job essential skills (${(r.je||[]).length})</summary><div>${jobEss}</div></details>` : ''}
        ${jobOpt ? `<details><summary>Job optional skills (${(r.jo||[]).length})</summary><div>${jobOpt}</div></details>` : ''}
      </div>
    </div>`;
}

function openKey() {
  document.getElementById('modal').innerHTML = `
    <button class="closebtn" onclick="document.getElementById('modalbg').classList.remove('open')">Close</button>
    <h2>Key — skills graph matching</h2>
    <p>Users: <code>data/njila_users.jsonl</code> · Jobs: <code>data/ranked_jobs_v2.json</code> · Taxonomy: <code>backend/resources/skill_taxonomy</code></p>
    <p><strong>Exact match</strong> (<code>exact_match.py</code>) — label overlap; ≥2 matches and ≥10% job skill coverage.</p>
    <p><strong>Graph Dijkstra</strong> (<code>graph_dijkstra.py</code>) — weighted shortest path per job skill from any user skill. Ranked by exact node hits (desc), then avg distance (asc).</p>
    <p><strong>Final</strong> (<code>final.py</code>) — production ranker. Sequential: exact block first, then Dijkstra on remaining jobs only. Rank order only (no combined score shown).</p>
    <p>Built by <code>dashboard/run_njila_dashboard.py</code> → <code>dashboard/output/final_dashboard.html</code>.</p>`;
  document.getElementById('modalbg').classList.add('open');
}

document.getElementById('selA').onchange = (e) => { selA = e.target.value; renderHeaderStats(); renderUserList(document.getElementById('usearch').value); renderMain(); };
document.getElementById('selB').onchange = (e) => { selB = e.target.value; renderHeaderStats(); renderUserList(document.getElementById('usearch').value); renderMain(); };
document.getElementById('swapBtn').onclick = () => { [selA, selB] = [selB, selA]; document.getElementById('selA').value = selA; document.getElementById('selB').value = selB; renderHeaderStats(); renderUserList(document.getElementById('usearch').value); renderMain(); };
document.getElementById('keyBtn').onclick = openKey;
document.getElementById('modalbg').onclick = (e) => { if (e.target === document.getElementById('modalbg')) document.getElementById('modalbg').classList.remove('open'); };
document.getElementById('usearch').oninput = (e) => renderUserList(e.target.value);
buildSelectors();
if (DATA.final && DATA.final.length) activeUid = DATA.final[0].uid;
renderHeaderStats();
renderUserList();
renderMain();
</script>
</body>
</html>
"""


def build_html(payload: dict) -> str:
    meta = payload["meta"]
    data = payload["data"]
    html = HTML_TEMPLATE.replace("__META_JSON__", json.dumps(meta, ensure_ascii=False))
    html = html.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    return html


def write_dashboard(json_path: Path, html_path: Path) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    html_path.write_text(build_html(payload), encoding="utf-8")
