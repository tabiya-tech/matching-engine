// src/components/MatchCard.jsx
import React from 'react';

export const MatchCard = ({ job }) => {
  // Use optional chaining and default values to prevent "undefined" errors
  const final_score = job?.final_score || 0;
  const score_breakdown = job?.score_breakdown || {};
  const opportunity_title = job?.opportunity_title || "Untitled Opportunity";
  const location = job?.location || "Location not specified";
  const is_eligible = job?.is_eligible ?? false;
  const justification = job?.justification || "";
  const contract_type = job?.contract_type || "";

  return (
    <div className="bg-white rounded-3xl shadow-xl shadow-indigo-500/5 border border-white p-6 flex gap-8 mb-6 hover:translate-y-[-4px] transition-all duration-300">
      {/* Radial Score Gauge */}
      <div className="flex flex-col items-center justify-center bg-slate-50 w-28 h-28 rounded-3xl border-2 border-indigo-50/50">
        <div className="text-3xl font-black text-indigo-600">
          {(final_score * 100).toFixed(0)}%
        </div>
        <div className="text-[10px] font-black text-indigo-400 uppercase tracking-widest mt-1">
          Match
        </div>
      </div>

      <div className="flex-grow pt-2">
        <div className="flex justify-between items-start mb-4">
          <div>
            <div className="flex items-center gap-3 mb-1">
               <h3 className="text-2xl font-black text-slate-800 leading-none">
                {opportunity_title}
              </h3>
              {/* Added a dynamic eligibility tag */}
              <span className={`px-2 py-0.5 rounded text-[9px] font-black uppercase ${is_eligible ? 'bg-emerald-100 text-emerald-600' : 'bg-rose-100 text-rose-600'}`}>
                {is_eligible ? 'Qualified' : 'Skill Gap'}
              </span>
              {contract_type && (
                <span className="px-2 py-0.5 rounded text-[9px] font-black uppercase bg-indigo-100 text-indigo-600">
                  {contract_type}
                </span>
              )}
            </div>
            <span className="text-slate-400 font-bold text-sm">📍 {location}</span>
            {justification && (
              <p className="text-slate-600 text-sm mt-2 leading-relaxed">
                {justification}
              </p>
            )}
          </div>
        </div>

        {/* Skill Indicators - Updated to match the new API schema */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
          <SkillIndicator 
            label="Skill Alignment" 
            val={score_breakdown.total_skill_utility || 0} 
          />
          <SkillIndicator 
            label="User Preference" 
            val={score_breakdown.preference_score || 0} 
          />
          <SkillIndicator 
            label="Market Demand" 
            val={score_breakdown.demand_score || 0} 
          />
        </div>
      </div>
    </div>
  );
};

const SkillIndicator = ({ label, val }) => (
  <div>
    <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase mb-2">
      <span>{label}</span>
      <span>{(val * 100).toFixed(0)}%</span>
    </div>
    <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
      <div 
        className="h-full bg-indigo-500 rounded-full transition-all duration-500" 
        style={{ width: `${Math.min(val * 100, 100)}%` }} 
      />
    </div>
  </div>
);