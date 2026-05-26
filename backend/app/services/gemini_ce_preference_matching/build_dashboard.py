"""Standalone HTML dashboard (Ethiopia-style): match_v3 CE vs u_hat × p_hat.

Column 1: Gemini concat cosine + CE (headline = cosine; u_hat chip).
Column 2: u_hat × p_hat rerank (headline = final; u_hat and p_hat shown).
Expanded rows: concat skill lists only (no per-skill pairs).

Usage::

    cd backend
    python3 -m app.services.gemini_ce_preference_matching.build_dashboard \\
        --input output/results_gemini_ce_preference.json \\
        --output output/dashboards/results_gemini_ce_preference_dual.html \\
        --top-k 50
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, List

from app.services.gemini_ce_preference_matching.scoring import (
    preference_details_for_dashboard,
    work_activity_match_for_dashboard,
)

SCRIPT_NAME = "app.services.gemini_ce_preference_matching.build_dashboard"
BRAND = "Kenya — p_hat (skills) vs preferences + BWS"

_HTML_STYLE = """
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
  .skills { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; max-height: 140px; overflow-y: auto; }
  .skill { background: #eef2ff; color: #3730a3; border-radius: 10px; padding: 1px 8px; font-size: 10px; white-space: nowrap; }
  .prefhead { font-size: 10px; font-weight: 600; color: #374151; text-transform: uppercase; letter-spacing: 0.04em; margin: 10px 0 4px 0; }
  .preffactors { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; max-height: 100px; overflow-y: auto; }
  .preffactor { background: #fef3c7; color: #92400e; border-radius: 10px; padding: 2px 8px; font-size: 10px; white-space: nowrap; }
  .bwsfactor { background: #ecfdf5; color: #047857; border-radius: 10px; padding: 2px 8px; font-size: 10px; white-space: nowrap; }
  .bwsfactor.neg { background: #fef2f2; color: #b91c1c; }
  .bwsfactor.pos { background: #ecfdf5; color: #047857; }
  .chip.wa { background: #ecfdf5; color: #047857; }
  .chip.sa { background: #fffbeb; color: #b45309; }
  .prefbreak { margin-top: 4px; color: #6b7280; font-size: 10px; }
  table.wa { width: 100%; border-collapse: collapse; font-size: 10px; margin-top: 6px; }
  table.wa th, table.wa td { border: 1px solid #e5e7eb; padding: 3px 5px; text-align: left; vertical-align: top; }
  table.wa th { background: #f0fdf4; font-weight: 600; }
  .warn { color: #b45309; font-size: 11px; margin: 6px 0; }
  table.pref { width: 100%; border-collapse: collapse; font-size: 10px; margin-top: 6px; }
  table.pref th, table.pref td { border: 1px solid #e5e7eb; padding: 3px 5px; text-align: left; vertical-align: top; }
  table.pref th { background: #f9fafb; font-weight: 600; }
  .umeta .warnbadge { color: #b45309; font-weight: 600; }
  .controls { display: flex; gap: 14px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; font-size: 11px; color: #4b5563; }
  .controls label { display: flex; align-items: center; gap: 6px; }
  .controls input[type=number] { width: 56px; padding: 4px 6px; border: 1px solid #d1d5db; border-radius: 4px; }
  .dual { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }
  @media (max-width: 960px) { .dual { grid-template-columns: 1fr; } }
  .col { background: #fff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px; min-width: 0; }
  .col h3 { margin: 0 0 10px 0; font-size: 13px; color: #374151; font-weight: 600; line-height: 1.35; }
  .colwrap { max-height: calc(100vh - 240px); min-height: 240px; overflow-y: auto; }
  .empty { color: #9ca3af; padding: 18px 0; text-align: center; font-style: italic; font-size: 11px; }
  .row { padding: 6px 8px; border-bottom: 1px solid #f3f4f6; cursor: pointer; border-radius: 4px; }
  .row:hover { background: #fafbfc; }
  .row .top { display: flex; justify-content: space-between; gap: 6px; align-items: baseline; flex-wrap: wrap; }
  .row .title { font-weight: 600; color: #111827; font-size: 12px; }
  .scores { display: flex; flex-direction: column; align-items: flex-end; gap: 2px; flex-shrink: 0; }
  .score { font-variant-numeric: tabular-nums; font-weight: 700; font-size: 12px; }
  .score.ce { color: #2563eb; }
  .score.uxp { color: #7c3aed; }
  .score.uh { color: #b45309; font-size: 10px; font-weight: 600; }
  .submeta { color: #6b7280; font-size: 10px; margin-top: 2px; line-height: 1.35; }
  .chip { font-size: 9px; background: #f3f4f6; padding: 1px 5px; border-radius: 4px; color: #4b5563; margin-right: 4px; }
  .chip.uh { background: #fffbeb; color: #b45309; }
  .detail { display: none; margin-top: 6px; padding-top: 6px; border-top: 1px dashed #e5e7eb; font-size: 10px; }
  .row.expanded { background: #ecfdf5; border: 1px solid #a7f3d0; }
  .row.expanded.prefcol { background: #f5f3ff; border: 1px solid #ddd6fe; }
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


def _compact_row(raw: Dict[str, Any], *, include_uxp: bool) -> Dict[str, Any]:
    sb = raw.get("score_breakdown") or {}
    row: Dict[str, Any] = {
        "rank": raw.get("rank"),
        "job_uuid": str(raw.get("job_uuid") or ""),
        "job_title": raw.get("job_title"),
        "employer": raw.get("employer"),
        "location": raw.get("location"),
        "concat_cosine_similarity": raw.get("concat_cosine_similarity"),
        "cross_encoder_logit": raw.get("cross_encoder_logit"),
        "rank_cosine": raw.get("rank_cosine"),
        "job_concat_skills": list(raw.get("job_concat_skills") or []),
    }
    if include_uxp:
        row["u_hat"] = raw.get("u_hat") if raw.get("u_hat") is not None else sb.get("u_hat")
        row["p_hat"] = raw.get("p_hat") if raw.get("p_hat") is not None else sb.get("p_hat")
        row["final_score"] = raw.get("final_score") if raw.get("final_score") is not None else sb.get("final_score")
        row["rank_cross_encoder"] = raw.get("rank_cross_encoder")
        details = raw.get("preference_details") or raw.get("preference_match_rows")
        row["pref_match"] = preference_details_for_dashboard(details)
        row["wa_match"] = work_activity_match_for_dashboard(details)
        if raw.get("work_activity_match"):
            row["wa_match"] = raw.get("work_activity_match")
        row["S_attrs"] = raw.get("S_attrs")
        row["S_wa"] = raw.get("S_wa")
    return row


def build_dual_payload(results: Dict[str, Any], *, top_k: int) -> Dict[str, Any]:
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
        uxp_rows = [
            _compact_row(row, include_uxp=True)
            for row in (r.get("recommendations") or [])[:top_k]
            if isinstance(row, dict)
        ]
        users_out.append({
            "uid": uid,
            "city": r.get("city"),
            "prov": r.get("province"),
            "user_skills": user_skills,
            "has_skills": bool(r.get("has_user_skills", len(user_skills) > 0)),
            "skip_reason": r.get("skip_reason"),
            "user_prefs": list(r.get("user_preference_factors") or []),
            "user_bws": r.get("user_bws") or {"has_bws": False, "rows": []},
            "ce_final": ce_rows,
            "uxp": uxp_rows,
        })
    return {
        "meta": {
            "brand": BRAND,
            "n_users": len(users_out),
            "n_jobs": results.get("n_jobs"),
            "rows_per_column": top_k,
            "final_formula": cfg.get("final_formula") or "u_hat * p_hat",
            "preference_scorer_mode": cfg.get("preference_scorer_mode"),
            "preference_module": cfg.get("preference_module"),
            "stage1_scorer": cfg.get("stage1_scorer"),
            "cross_encoder_model": cfg.get("cross_encoder_model"),
        },
        "topN_default": top_k,
        "users": users_out,
    }


def _b64_utf8(obj: Dict[str, Any]) -> str:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.standard_b64encode(raw).decode("ascii")


def _dual_js(b64: str) -> str:
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
    return '<div class="matchsec"><div class="lbl">Job skills (concat)</div>'
      + '<p style="color:#9ca3af;font-size:10px;margin:4px 0 0;">None listed on job.</p></div>';
  }
  const show = labs.slice(0, 40);
  const more = labs.length > show.length ? ' <span style="color:#9ca3af">+' + (labs.length - show.length) + ' more</span>' : '';
  return '<div class="matchsec"><div class="lbl">Job skills (concat)</div>'
    + '<ul class="compact">' + show.map(s => '<li>' + esc(s) + '</li>').join('') + '</ul>' + more + '</div>';
}
function prefMatchTable(rows) {
  const rs = rows || [];
  if (!rs.length) {
    return '<div class="matchsec"><div class="lbl">Preference match</div>'
      + '<p style="color:#9ca3af;font-size:10px;margin:4px 0 0;">No attribute details (job may lack llm_job_attributes).</p></div>';
  }
  const trs = rs.map(p => {
    const raw = p.job_value || '\u2014';
    const resolved = p.job_level_resolved || '';
    let offer = '<code style="font-size:9px">' + esc(raw) + '</code>';
    if (resolved && resolved !== raw) {
      offer += '<br><span style="color:#6b7280;font-size:9px">scored as ' + esc(resolved) + '</span>';
    }
    return '<tr><td>' + esc(p.attr_label) + '</td><td>' + fmt(p.user_weight, 2)
      + '</td><td>' + offer + '</td><td>' + fmt(p.encoded_value, 2)
      + '</td><td>' + fmt(p.contribution, 3) + '</td></tr>';
  }).join('');
  return '<div class="matchsec"><div class="lbl">Preference match (importance \u00d7 job level id from Mongo)</div>'
    + '<table class="pref"><thead><tr><th>Attribute</th><th>You care</th><th>Job level (raw)</th><th>Fit V</th><th>Contrib</th></tr></thead><tbody>'
    + trs + '</tbody></table></div>';
}
function userPrefsHtml(factors) {
  const fs = factors || [];
  if (!fs.length) return '';
  return '<div class="prefhead">Your importance (preference_vector)</div><div class="preffactors">'
    + fs.map(f => '<span class="preffactor" title="' + esc(f.attribute) + '">'
    + esc(f.label) + ' ' + fmt(f.importance, 2) + '</span>').join('') + '</div>';
}
function userBwsHtml(bws) {
  const box = bws || {};
  if (!box.has_bws || !(box.rows || []).length) {
    return '<div class="prefhead">BWS (work activities)</div>'
      + '<p style="color:#9ca3af;font-size:10px;margin:4px 0 8px;">No O*NET work-activity BWS on this user — Part B uses S_wa = 0.</p>';
  }
  const chips = (box.rows || []).map(r => {
    const v = Number(r.bws);
    const cls = v > 0 ? 'bwsfactor pos' : (v < 0 ? 'bwsfactor neg' : 'bwsfactor');
    const sign = v > 0 ? '+' : '';
    return '<span class="' + cls + '" title="' + esc(r.wa_code) + '">' + esc(r.wa_code) + ' ' + sign + fmt(v, 1) + '</span>';
  }).join('');
  return '<div class="prefhead">BWS (work activities you prefer / avoid)</div><div class="preffactors">' + chips + '</div>';
}
function waMatchTable(wa) {
  const block = wa || {};
  const rs = block.rows || [];
  const summary = (block.S_wa != null)
    ? '<div class="prefbreak">S_wa (mean BWS\u00d7importance\u00d7level) = <b>' + fmt(block.S_wa, 3) + '</b>'
      + ' \u00b7 ' + (block.n_work_activities || 0) + ' job activities</div>' : '';
  if (!rs.length) {
    return '<div class="matchsec"><div class="lbl">Work activities (BWS)</div>'
      + summary
      + '<p style="color:#9ca3af;font-size:10px;margin:4px 0 0;">No onet_work_activities on this job.</p></div>';
  }
  const trs = rs.map(w => {
    const bws = Number(w.user_bws);
    const bwsCls = bws > 0 ? 'pos' : (bws < 0 ? 'neg' : '');
    return '<tr><td><code style="font-size:9px">' + esc(w.wa_code) + '</code></td>'
      + '<td>' + esc((w.wa_label || '').substring(0, 48)) + ((w.wa_label || '').length > 48 ? '\u2026' : '') + '</td>'
      + '<td class="' + bwsCls + '">' + (bws > 0 ? '+' : '') + fmt(bws, 1) + '</td>'
      + '<td>' + fmt(w.wa_importance, 2) + '</td><td>' + fmt(w.wa_level, 2) + '</td>'
      + '<td>' + fmt(w.wa_contribution, 3) + '</td></tr>';
  }).join('');
  return '<div class="matchsec"><div class="lbl">Work activities (BWS \u00d7 job WA)</div>' + summary
    + '<table class="wa"><thead><tr><th>Code</th><th>Activity</th><th>Your BWS</th><th>Job I/5</th><th>Job L/7</th><th>Contrib</th></tr></thead><tbody>'
    + trs + '</tbody></table></div>';
}
function prefPartsHtml(r) {
  const parts = [];
  if (r.S_attrs != null) parts.push('S_attrs ' + fmt(r.S_attrs, 3));
  if (r.S_wa != null && Number(r.S_wa) !== 0) parts.push('S_wa ' + fmt(r.S_wa, 3));
  if (!parts.length) return '';
  return '<div class="prefbreak">u_hat parts: ' + parts.join(' \u00b7 ') + ' \u2192 sigmoid(S_attrs + S_wa)</div>';
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
    '<div class="detail"><code class="jobid">' + esc(r.job_uuid) + '</code>'
    + jobSkillsHtml(r.job_concat_skills) + '</div>' +
    '</div>';
}
function prefScoreStack(r) {
  const parts = [];
  if (r.u_hat != null && Number.isFinite(Number(r.u_hat))) {
    parts.push('<span class="score uh">u_hat ' + fmt(r.u_hat, 3) + '</span>');
  }
  if (r.p_hat != null && Number.isFinite(Number(r.p_hat))) {
    parts.push('<span class="score ce" style="font-size:11px;font-weight:600">p_hat ' + fmt(r.p_hat, 3) + '</span>');
  }
  if (r.final_score != null && Number.isFinite(Number(r.final_score))) {
    parts.push('<span class="score uxp">final ' + fmt(r.final_score, 3) + '</span>');
  }
  return parts.length ? '<div class="scores">' + parts.join('') + '</div>' : '';
}
function renderPrefRow(r) {
  const waChip = (r.S_wa != null && Number(r.S_wa) !== 0)
    ? '<span class="chip wa">S_wa ' + fmt(r.S_wa, 3) + '</span>' : '';
  const saChip = (r.S_attrs != null) ? '<span class="chip sa">S_attrs ' + fmt(r.S_attrs, 3) + '</span>' : '';
  const chips =
    (r.rank_cross_encoder != null ? '<span class="chip">was #' + r.rank_cross_encoder + ' skills col</span>' : '') +
    saChip + waChip;
  return '<div class="row prefcol">' +
    '<div class="top"><div><span style="color:#9ca3af;font-weight:600">#' + r.rank + '</span> '
    + '<span class="title">' + esc(r.job_title) + '</span></div>'
    + prefScoreStack(r) + '</div>' +
    '<div class="submeta">' + esc(r.employer) + ' \u00b7 ' + esc(r.location) + ' ' + chips + '</div>' +
    '<div class="detail"><code class="jobid">' + esc(r.job_uuid) + '</code>'
    + prefPartsHtml(r) + prefMatchTable(r.pref_match) + waMatchTable(r.wa_match) + '</div>' +
    '</div>';
}
function setupWrap(wrap) {
  wrap.querySelectorAll('.row').forEach(row => { row.onclick = () => row.classList.toggle('expanded'); });
}
function topCeHint(u) {
  const r = (u.ce_final || [])[0];
  return r ? pHatHeadline(r) : '\u2014';
}
function topUxpHint(u) {
  const r = (u.uxp || [])[0];
  return r ? fmt(r.final_score, 3) : '\u2014';
}
function topUhatHint(u) {
  const r = (u.uxp || [])[0];
  return r && r.u_hat != null ? fmt(r.u_hat, 3) : '\u2014';
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
  const warn = (!u.has_skills || u.skip_reason)
    ? '<p class="warn">No user skills in concat text — skill retrieval skipped'
      + (u.skip_reason ? ' (' + esc(u.skip_reason) + ').' : '.') + '</p>' : '';
  panel.innerHTML =
    '<div class="userhead"><h2>' + esc(u.uid) + '</h2>' +
    '<div class="meta">' + esc(u.city) + ' \u00b7 ' + esc(u.prov) + ' \u00b7 ' + labs.length + ' skills (concat)</div>' +
    warn + userPrefsHtml(u.user_prefs) + userBwsHtml(u.user_bws) +
    '<div class="prefhead">Your skills (concat text)</div>' +
    '<div class="skills">' + (sk || '<span style="color:#9ca3af;font-style:italic">none</span>') + '</div></div>' +
    '<div class="controls"><label>Rows per column <input type="number" id="topNinp" min="1" max="500" value="' + topN + '"></label>'
    + '<span style="color:#6b7280">Left = p_hat only \u00b7 Right = preferences + work activities (ranked by u_hat \u00d7 p_hat)</span></div>' +
    '<div class="dual">' +
    '<div class="col"><h3>Column 1 — Skills (p_hat, CE order)</h3><div class="colwrap" id="wCe"></div></div>' +
    '<div class="col"><h3>Column 2 — Preferences + BWS / work activities</h3><div class="colwrap" id="wPref"></div></div></div>';
  document.getElementById('topNinp').onchange = e => {
    topN = Math.max(1, Math.min(500, parseInt(e.target.value, 10) || 10));
    renderMain();
  };
  const wCe = document.getElementById('wCe');
  const wPref = document.getElementById('wPref');
  const ceSlice = (u.ce_final || []).slice(0, topN);
  const prefSlice = (u.uxp || []).slice(0, topN);
  wCe.innerHTML = ceSlice.length ? ceSlice.map(renderCeRow).join('') : '<div class="empty">none</div>';
  wPref.innerHTML = prefSlice.length ? prefSlice.map(renderPrefRow).join('') : '<div class="empty">none</div>';
  setupWrap(wCe);
  setupWrap(wPref);
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
    const skBadge = u.has_skills ? ((u.user_skills || []).length + ' sk') : '<span class="warnbadge">0 sk</span>';
    li.innerHTML = '<div class="uname">' + esc((u.uid || '').substring(0, 36)) + '</div>' +
      '<div class="umeta">' + esc(u.city || '?') + ' \u00b7 ' + esc(u.prov || '?') + ' \u00b7 ' + skBadge +
      ' \u00b7 p_hat <b>' + topCeHint(u) + '</b> \u00b7 u_hat <b>' + topUhatHint(u) + '</b></div>';
    li.onclick = () => { activeUid = u.uid; renderUserList(f); renderMain(); };
    ul.appendChild(li);
  }
}
function init() {
  usersArr = (PAYLOAD.users || []).map(u => ({
    uid: u.uid,
    city: u.city,
    prov: u.prov,
    user_skills: u.user_skills || [],
    has_skills: u.has_skills !== false,
    skip_reason: u.skip_reason,
    user_prefs: u.user_prefs || [],
    user_bws: u.user_bws || { has_bws: false, rows: [] },
    ce_final: u.ce_final || [],
    uxp: u.uxp || [],
  }));
  topN = PAYLOAD.topN_default || 10;
  const m = PAYLOAD.meta || {};
  document.getElementById('pillMeta').textContent =
    (m.n_users || usersArr.length) + ' users \u00b7 ' + (m.n_jobs ?? '?') + ' jobs \u00b7 col1 p_hat \u00b7 col2 prefs+WA';
  document.getElementById('hdrstats').textContent =
    (m.preference_scorer_mode || 'preferences') + ' \u00b7 ' + (m.final_formula || 'u_hat * p_hat') + ' \u00b7 CE ' + (m.cross_encoder_model || '?');
  activeUid = usersArr.length ? usersArr[0].uid : null;
  renderUserList('');
  renderMain();
}
function openKey() {
  document.getElementById('modal').innerHTML =
    '<button type="button" class="closebtn" onclick="document.getElementById(\'modalbg\').classList.remove(\'open\')">Close</button>' +
    '<h2>' + esc(BRAND) + ' \u2014 dashboard key</h2>' +
    '<p><b>Column 1:</b> <b>p_hat</b> headline; expand for <b>job concat skills</b>.</p>' +
    '<p><b>Column 2:</b> Top scores: <b>u_hat</b>, <b>p_hat</b>, <b>final</b>; expand for attribute + BWS tables.</p>' +
    '<p>Users with zero concat skills: column 1 empty.</p>' +
    '<p>Requires <code>PREFERENCE_SCORER_MODE=hybrid_v1</code> when running <code>run_matching</code>.</p>' +
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
""".replace("__B64__", b64).replace("__SCRIPT__", SCRIPT_NAME)


def render_dual_page(b64: str) -> str:
    title = f"{BRAND}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{_HTML_STYLE}
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
{_dual_js(b64)}
</body>
</html>
"""


def build_html(payload: Dict[str, Any], *, top_k: int = 50) -> str:
    dual = build_dual_payload(payload, top_k=top_k)
    return render_dual_page(_b64_utf8(dual))


def main() -> int:
    p = argparse.ArgumentParser(description="Build static dual-column dashboard.")
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
    b64_len = len(_b64_utf8(build_dual_payload(payload, top_k=top_k))) // 1024
    print(f"Wrote {args.output} (~{b64_len} KiB embedded payload)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
