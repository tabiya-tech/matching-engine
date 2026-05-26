"""Three-column HTML: skills only | hybrid attrs only | hybrid attrs + BWS work activities.

Requires JSON from ``run_matching`` with ``recommendations_attrs_only`` and ``recommendations``.

Usage::

    cd backend
    python3 -m app.services.gemini_ce_preference_matching.build_dashboard_with_wa \\
        --input output/results_gemini_ce_hybrid_v1.json \\
        --output output/dashboards/results_gemini_ce_hybrid_v1_with_wa.html \\
        --top-k 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .build_dashboard import (
    BRAND,
    SCRIPT_NAME,
    _HTML_STYLE,
    _b64_utf8,
    _compact_row,
)
from .scoring import work_activity_match_for_dashboard

SCRIPT_NAME_WA = "app.services.gemini_ce_preference_matching.build_dashboard_with_wa"
BRAND_WA = f"{BRAND} — with work activities comparison"

_EXTRA_STYLE = """
  .triple { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; align-items: start; }
  @media (max-width: 1200px) { .triple { grid-template-columns: 1fr; } }
  .score.wa { color: #059669; font-size: 10px; font-weight: 600; }
  .chip.wa { background: #ecfdf5; color: #047857; }
  .prefbreak { margin-top: 6px; color: #4b5563; font-size: 10px; line-height: 1.4; }
"""


def build_triple_payload(results: Dict[str, Any], *, top_k: int) -> Dict[str, Any]:
    cfg = results.get("config") or {}
    users_out: List[Dict[str, Any]] = []
    for r in results.get("results") or []:
        if not isinstance(r, dict):
            continue
        uid = str(r.get("user_id") or "").strip()
        user_skills = list(r.get("user_concat_skills") or r.get("resolved_user_skill_labels") or [])
        ce_rows = [
            _compact_row(row, include_uxp=False)
            for row in (r.get("cross_encoder_recommendations") or [])[:top_k]
            if isinstance(row, dict)
        ]
        attrs_rows = [
            _compact_row_wa(row)
            for row in (r.get("recommendations_attrs_only") or r.get("recommendations") or [])[:top_k]
            if isinstance(row, dict)
        ]
        full_rows = [
            _compact_row_wa(row)
            for row in (r.get("recommendations") or [])[:top_k]
            if isinstance(row, dict)
        ]
        users_out.append({
            "uid": uid,
            "city": r.get("city"),
            "prov": r.get("province"),
            "user_skills": user_skills,
            "has_skills": bool(r.get("has_user_skills", len(user_skills) > 0)),
            "user_bws": r.get("user_bws") or {"has_bws": False, "rows": []},
            "ce_final": ce_rows,
            "attrs_only": attrs_rows,
            "with_wa": full_rows,
        })
    return {
        "meta": {
            "brand": BRAND_WA,
            "n_users": len(users_out),
            "n_jobs": results.get("n_jobs"),
            "rows_per_column": top_k,
            "final_formula": cfg.get("final_formula") or "u_hat * p_hat",
            "preference_scorer_mode": cfg.get("preference_scorer_mode"),
            "preference_module": cfg.get("preference_module"),
            "stage3_variants": cfg.get("stage3_variants")
            or ["attrs_only", "attrs_plus_work_activities"],
            "cross_encoder_model": cfg.get("cross_encoder_model"),
        },
        "topN_default": top_k,
        "users": users_out,
    }


def _compact_row_wa(raw: Dict[str, Any]) -> Dict[str, Any]:
    row = _compact_row(raw, include_uxp=True)
    row["S_attrs"] = raw.get("S_attrs")
    row["S_wa"] = raw.get("S_wa")
    wa_block = work_activity_match_for_dashboard(raw.get("preference_details"))
    row["wa_match"] = wa_block
    row["n_work_activities"] = wa_block.get("n_work_activities")
    row["wa_score_sum"] = wa_block.get("S_wa")
    return row


def _triple_js(b64: str) -> str:
    return r"""
<script>
const PAYLOAD = JSON.parse(atob('__B64__'));
let usersArr = [];
let activeUid = null;
let topN = PAYLOAD.topN_default || 10;
const SCRIPT_REF = '__SCRIPT__';
const BRAND = (PAYLOAD.meta || {}).brand || '';

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function fmt(v, n) {
  const x = Number(v);
  if (!Number.isFinite(x)) return '\u2014';
  return x.toFixed(n == null ? 3 : n);
}
function uidMapFrom(arr) { return new Map(arr.map(u => [u.uid, u])); }
function jobSkillsHtml(skills) {
  const labs = skills || [];
  if (!labs.length) {
    return '<div class="matchsec"><div class="lbl">Job skills (concat text)</div>'
      + '<p style="color:#9ca3af;font-size:10px;margin:4px 0 0;">No skills on job record.</p></div>';
  }
  return '<div class="matchsec"><div class="lbl">Job skills (concat text)</div>'
    + '<ul class="compact">' + labs.map(s => '<li>' + esc(s) + '</li>').join('') + '</ul></div>';
}
function prefBreakHtml(r) {
  const parts = [];
  if (r.S_attrs != null) parts.push('S_attrs ' + fmt(r.S_attrs, 3));
  if (r.S_wa != null) parts.push('S_wa ' + fmt(r.S_wa, 3));
  if (r.n_work_activities != null) parts.push(r.n_work_activities + ' job WAs');
  if (!parts.length) return '';
  return '<div class="prefbreak">' + parts.join(' \u00b7 ') + '</div>';
}
function pHatHeadline(r) {
  return fmt(r.concat_cosine_similarity, 3);
}
function renderCeRow(r) {
  const chips =
    (r.rank_cosine != null ? '<span class="chip">was #' + r.rank_cosine + ' cosine</span>' : '') +
    (r.cross_encoder_logit != null ? '<span class="chip">CE logit ' + fmt(r.cross_encoder_logit, 2) + '</span>' : '');
  return '<div class="row">' +
    '<div class="top"><div><span style="color:#9ca3af;font-weight:600">#' + r.rank + '</span> '
    + '<span class="title">' + esc(r.job_title) + '</span></div>'
    + '<span class="score ce">p_hat ' + pHatHeadline(r) + '</span></div>' +
    '<div class="submeta">' + esc(r.employer) + ' \u00b7 ' + esc(r.location) + ' ' + chips + '</div>' +
    '<div class="detail"><code class="jobid">' + esc(r.job_uuid) + '</code>' + jobSkillsHtml(r.job_concat_skills) + '</div>' +
    '</div>';
}
function renderPrefRow(r, label) {
  const waChip = (r.S_wa != null && Number(r.S_wa) > 0)
    ? '<span class="chip wa">S_wa ' + fmt(r.S_wa, 3) + '</span>' : '';
  const chips =
    '<span class="chip">p_hat ' + fmt(r.p_hat, 3) + '</span>' +
    (r.rank_cross_encoder != null ? '<span class="chip">was #' + r.rank_cross_encoder + ' CE</span>' : '') +
    waChip;
  const uh = (r.u_hat != null && Number.isFinite(Number(r.u_hat)))
    ? '<span class="score uh">u_hat ' + fmt(r.u_hat, 3) + '</span>' : '';
  return '<div class="row uxp">' +
    '<div class="top"><div><span style="color:#9ca3af;font-weight:600">#' + r.rank + '</span> '
    + '<span class="title">' + esc(r.job_title) + '</span></div>'
    + '<div class="scores"><span class="score uxp">final ' + fmt(r.final_score, 3) + '</span>' + uh + '</div></div>' +
    '<div class="submeta">' + esc(r.employer) + ' \u00b7 ' + esc(r.location) + ' ' + chips + '</div>' +
    '<div class="detail"><code class="jobid">' + esc(r.job_uuid) + '</code>' + prefBreakHtml(r) + jobSkillsHtml(r.job_concat_skills) + '</div>' +
    '</div>';
}
function setupWrap(wrap) {
  wrap.querySelectorAll('.row').forEach(row => { row.onclick = () => row.classList.toggle('expanded'); });
}
function topFinal(rows) {
  const r = (rows || [])[0];
  return r ? fmt(r.final_score, 3) : '\u2014';
}
function renderMain() {
  const panel = document.getElementById('mainpanel');
  const uidMap = uidMapFrom(usersArr);
  if (!activeUid || !uidMap.get(activeUid)) {
    panel.innerHTML = '<div class="empty">Pick a user.</div>';
    return;
  }
  const u = uidMap.get(activeUid);
  const labs = u.user_skills || [];
  const sk = labs.slice(0, 400).map(s => '<span class="skill">' + esc(s) + '</span>').join('');
  panel.innerHTML =
    '<div class="userhead"><h2>' + esc(u.uid) + '</h2>' +
    '<div class="meta">' + esc(u.city) + ' \u00b7 ' + esc(u.prov) + ' \u00b7 ' + labs.length + ' skills in user concat text</div>' +
    '<div class="skills">' + (sk || '<span style="color:#9ca3af;font-style:italic">no skills</span>') + '</div></div>' +
    '<div class="controls"><label>Rows per column <input type="number" id="topNinp" min="1" max="500" value="' + topN + '"></label>'
    + '<span style="color:#6b7280">Col1: skills \u00b7 Col2: DCE attrs only \u00b7 Col3: attrs + BWS work activities</span></div>' +
    '<div class="triple">' +
    '<div class="col"><h3>1 — Skills only (p_hat)</h3><div class="colwrap" id="wCe"></div></div>' +
    '<div class="col"><h3>2 — Preferences (attrs only)</h3><div class="colwrap" id="wAttrs"></div></div>' +
    '<div class="col"><h3>3 — Preferences + work activities (BWS)</h3><div class="colwrap" id="wWa"></div></div></div>';
  document.getElementById('topNinp').onchange = e => {
    topN = Math.max(1, Math.min(500, parseInt(e.target.value, 10) || 10));
    renderMain();
  };
  const wCe = document.getElementById('wCe');
  const wAttrs = document.getElementById('wAttrs');
  const wWa = document.getElementById('wWa');
  const ceSlice = (u.ce_final || []).slice(0, topN);
  const aSlice = (u.attrs_only || []).slice(0, topN);
  const wSlice = (u.with_wa || []).slice(0, topN);
  wCe.innerHTML = ceSlice.length ? ceSlice.map(renderCeRow).join('') : '<div class="empty">none</div>';
  wAttrs.innerHTML = aSlice.length ? aSlice.map(r => renderPrefRow(r, 'attrs')).join('') : '<div class="empty">none</div>';
  wWa.innerHTML = wSlice.length ? wSlice.map(r => renderPrefRow(r, 'wa')).join('') : '<div class="empty">none</div>';
  setupWrap(wCe); setupWrap(wAttrs); setupWrap(wWa);
}
function renderUserList(q) {
  const f = (q || '').toLowerCase().trim();
  const ul = document.getElementById('ulist');
  ul.innerHTML = '';
  let n = 0;
  for (const u of usersArr) {
    if (f && !(u.uid + ' ' + (u.city || '') + ' ' + (u.prov || '')).toLowerCase().includes(f)) continue;
    if (++n > 4000) break;
    const li = document.createElement('li');
    if (u.uid === activeUid) li.classList.add('active');
    li.innerHTML = '<div class="uname">' + esc((u.uid || '').substring(0, 36)) + '</div>' +
      '<div class="umeta">' + esc(u.city || '?') + ' \u00b7 ' + esc(u.prov || '?') +
      ' \u00b7 attrs <b>' + topFinal(u.attrs_only) + '</b> \u00b7 +WA <b>' + topFinal(u.with_wa) + '</b></div>';
    li.onclick = () => { activeUid = u.uid; renderUserList(f); renderMain(); };
    ul.appendChild(li);
  }
}
function init() {
  usersArr = (PAYLOAD.users || []).map(u => ({
    uid: u.uid, city: u.city, prov: u.prov, user_skills: u.user_skills || [],
    ce_final: u.ce_final || [], attrs_only: u.attrs_only || [], with_wa: u.with_wa || [],
  }));
  topN = PAYLOAD.topN_default || 10;
  const m = PAYLOAD.meta || {};
  document.getElementById('pillMeta').textContent =
    (m.n_users || usersArr.length) + ' users \u00b7 3 columns';
  document.getElementById('hdrstats').textContent =
    (m.preference_scorer_mode || 'hybrid_v1') + ' \u00b7 attrs vs attrs+BWS';
  activeUid = usersArr.length ? usersArr[0].uid : null;
  renderUserList('');
  renderMain();
}
function openKey() {
  document.getElementById('modal').innerHTML =
    '<button type="button" class="closebtn" onclick="document.getElementById(\'modalbg\').classList.remove(\'open\')">Close</button>' +
    '<h2>' + esc(BRAND) + ' \u2014 key</h2>' +
    '<p><b>Column 1</b> — Same as dual dashboard: Gemini concat cosine + CE, no preferences.</p>' +
    '<p><b>Column 2</b> — hybrid_v1 Part A only (DCE attributes, gain/cost ladders). S_wa = 0.</p>' +
    '<p><b>Column 3</b> — Part A + Part B: user <code>bws_scores</code> (work_activity_id) \u00d7 job <code>onet_work_activities</code>, mean aggregation.</p>' +
    '<p>Re-run <code>run_matching</code> after this change so JSON includes <code>recommendations_attrs_only</code>.</p>' +
    '<p>Users without BWS: columns 2 and 3 may match. Jobs need enriched <code>onet_work_activities</code> on Mongo.</p>' +
    '<p>Built by <code>' + esc(SCRIPT_REF) + '</code>.</p>';
  document.getElementById('modalbg').classList.add('open');
}
document.getElementById('keyBtn').onclick = openKey;
document.getElementById('modalbg').onclick = e => {
  if (e.target === document.getElementById('modalbg')) document.getElementById('modalbg').classList.remove('open');
};
document.getElementById('usearch').oninput = e => renderUserList(e.target.value);
init();
</script>
""".replace("__B64__", b64).replace("__SCRIPT__", SCRIPT_NAME_WA)


def render_triple_page(b64: str) -> str:
    title = f"{BRAND_WA}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{_HTML_STYLE}{_EXTRA_STYLE}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <span class="pill" id="pillMeta"></span>
  <button id="keyBtn" type="button">ℹ Key</button>
  <span class="stats" id="hdrstats"></span>
</header>
<main>
  <aside>
    <input type="search" id="usearch" placeholder="Search user_id, city, province…">
    <ul class="ulist" id="ulist"></ul>
  </aside>
  <section class="main" id="mainpanel"></section>
</main>
<div class="modalbg" id="modalbg"><div class="modal" id="modal"></div></div>
{_triple_js(b64)}
</body>
</html>
"""


def build_html(payload: Dict[str, Any], *, top_k: int = 50) -> str:
    triple = build_triple_payload(payload, top_k=top_k)
    return render_triple_page(_b64_utf8(triple))


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build 3-column dashboard: skills | attrs-only prefs | attrs + BWS WAs."
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=None)
    args = p.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cfg = payload.get("config") or {}
    top_k = args.top_k if args.top_k is not None else int(cfg.get("final_top_k") or 50)
    top_k = max(1, min(500, top_k))

    html = build_html(payload, top_k=top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    b64_len = len(_b64_utf8(build_triple_payload(payload, top_k=top_k))) // 1024
    print(f"Wrote {args.output} (~{b64_len} KiB embedded payload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
