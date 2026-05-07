import React from 'react';
import { ScoreBreakdownDetails } from './ScoreBreakdownDetails';

/** Matches API `OccupationRecommendation` from /match. */
export const OccupationCard = ({ occupation }) => {
  const final_score = occupation?.final_score ?? 0;
  const score_breakdown = occupation?.score_breakdown ?? {};
  const title = occupation?.occupation_label || 'Occupation';
  const province = occupation?.province;
  const is_eligible = occupation?.is_eligible ?? false;
  const justification = occupation?.justification || '';
  const description = occupation?.occupation_description || '';

  return (
    <div className="bg-white rounded-3xl shadow-xl shadow-teal-500/5 border border-teal-50 p-6 flex gap-8 mb-6 hover:translate-y-[-4px] transition-all duration-300">
      <div className="flex flex-col items-center justify-center bg-teal-50/80 w-28 h-28 rounded-3xl border-2 border-teal-100/50">
        <div className="text-3xl font-black text-teal-700">
          {(final_score * 100).toFixed(0)}%
        </div>
        <div className="text-[10px] font-black text-teal-500 uppercase tracking-widest mt-1">
          Match
        </div>
      </div>

      <div className="flex-grow pt-2 min-w-0">
        <div className="flex justify-between items-start mb-4">
          <div>
            <div className="flex items-center gap-3 mb-1 flex-wrap">
              <span className="text-[10px] font-black uppercase tracking-wide text-teal-600 bg-teal-100 px-2 py-0.5 rounded">
                Occupation
              </span>
              <h3 className="text-2xl font-black text-slate-800 leading-none">
                {title}
              </h3>
              <span
                className={`px-2 py-0.5 rounded text-[9px] font-black uppercase ${
                  is_eligible
                    ? 'bg-emerald-100 text-emerald-600'
                    : 'bg-rose-100 text-rose-600'
                }`}
              >
                {is_eligible ? 'Qualified' : 'Skill gap'}
              </span>
            </div>
            {province != null && province !== '' && (
              <span className="text-slate-400 font-bold text-sm">📍 {province}</span>
            )}
            {description && (
              <p className="text-slate-600 text-sm mt-2 leading-relaxed line-clamp-4">
                {description}
              </p>
            )}
            {justification && (
              <p className="text-slate-600 text-sm mt-2 leading-relaxed">
                {justification}
              </p>
            )}
          </div>
        </div>

        <ScoreBreakdownDetails score_breakdown={score_breakdown} variant="teal" />
      </div>
    </div>
  );
};
