import React from 'react';

/** Matches API `SkillGapRecommendation` from /match. */
export const SkillGapCard = ({ recommendation }) => {
  const rec = recommendation ?? {};
  const label = rec.skill_label || 'Skill';
  const skillId = rec.skill_id ?? '';
  const proximity = rec.proximity_score;
  const unlocks = rec.job_unlock_count;
  const combined = rec.combined_score;
  const reasoning = rec.reasoning ?? '';

  const scoreDisplay =
    combined != null && !Number.isNaN(Number(combined))
      ? Math.round(Number(combined) * 100)
      : proximity != null && !Number.isNaN(Number(proximity))
        ? Math.round(Number(proximity) * 100)
        : null;

  return (
    <div className="bg-white rounded-3xl shadow-xl shadow-amber-500/5 border border-amber-100 p-6 flex gap-8 mb-6 hover:translate-y-[-4px] transition-all duration-300">
      <div className="flex flex-col items-center justify-center bg-amber-50 w-28 h-28 rounded-3xl border-2 border-amber-100 shrink-0">
        <div className="text-3xl font-black text-amber-700">
          {scoreDisplay != null ? `${scoreDisplay}%` : '—'}
        </div>
        <div className="text-[10px] font-black text-amber-600 uppercase tracking-widest mt-1 text-center px-1 leading-tight">
          Priority
        </div>
      </div>

      <div className="flex-grow pt-1 min-w-0">
        <div className="flex items-start gap-2 flex-wrap mb-2">
          <span className="text-[10px] font-black uppercase tracking-wide text-amber-800 bg-amber-100 px-2 py-0.5 rounded shrink-0">
            Skill development
          </span>
          <h3 className="text-2xl font-black text-slate-800 leading-tight">{label}</h3>
        </div>

        {skillId && (
          <p className="text-xs font-mono text-slate-400 mb-3 truncate" title={skillId}>
            {skillId}
          </p>
        )}

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">
          <MetricPill
            label="Proximity"
            value={
              proximity != null && !Number.isNaN(Number(proximity))
                ? `${(Number(proximity) * 100).toFixed(0)}%`
                : '—'
            }
            hint="Similarity to roles requiring this skill"
          />
          <MetricPill
            label="Jobs unlocked"
            value={unlocks != null ? String(unlocks) : '—'}
            hint="Estimated postings this skill opens"
          />
          <MetricPill
            label="Combined score"
            value={
              combined != null && !Number.isNaN(Number(combined))
                ? `${(Number(combined) * 100).toFixed(0)}%`
                : '—'
            }
            hint="Ranking signal from the engine"
          />
        </div>

        {reasoning && (
          <p className="text-sm text-slate-600 leading-relaxed border-t border-amber-100/80 pt-4">
            {reasoning}
          </p>
        )}
      </div>
    </div>
  );
};

function MetricPill({ label, value, hint }) {
  return (
    <div
      className="rounded-xl bg-amber-50/90 border border-amber-100 px-3 py-2"
      title={hint}
    >
      <div className="text-[9px] font-black uppercase tracking-wide text-amber-700/80 mb-0.5">
        {label}
      </div>
      <div className="font-mono text-lg font-bold text-slate-800">{value}</div>
    </div>
  );
}
