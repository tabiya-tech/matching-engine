import React from 'react';
import { MatchCard } from '../components/MatchCard';
import { OccupationCard } from '../components/OccupationCard';
import { SkillGapCard } from '../components/SkillGapCard';
import { SearchableSelect } from '../components/SearchableSelect'; // Import here

export const JobseekerView = ({
  users,
  onMatch,
  data,
  isLoading,
  selectedUserId,
}) => {
  const getUserId = (u) => String(u?.user_id ?? u?.youth_id ?? "");

  const handleSelect = (user) => {
    const id = getUserId(user);
    onMatch({ ...user, user_id: id }); // App persists selectedUserId when match runs
  };

  const currentUserMatch = data.find((u) => String(u?.user_id ?? "") === selectedUserId);

  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
        <label className="block text-sm font-black text-slate-400 uppercase mb-2 text-indigo-600">
          Search Profile ID
        </label>
        {/* New Searchable Dropdown */}
        <SearchableSelect
          options={users}
          onSelect={handleSelect}
          placeholder="Type to search User ID (e.g. user_001)..."
          labelKey="user_id"
          committedLabel={selectedUserId}
        />
      </div>

      <div className="grid gap-4">
        {isLoading ? (
          <div className="p-20 text-center animate-pulse text-slate-400 font-black uppercase tracking-tighter">
            AI Engine Calculating...
          </div>
        ) : (
          <>
            {selectedUserId && !currentUserMatch && (
              <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 text-slate-600">
                No matches returned for <span className="font-mono">{selectedUserId}</span>.
              </div>
            )}

            {currentUserMatch?.occupation_recommendations &&
              currentUserMatch.occupation_recommendations.length > 0 && (
                <div className="space-y-3">
                  <h3 className="text-sm font-black text-slate-500 uppercase tracking-wider">
                    Occupation recommendations
                  </h3>
                  {currentUserMatch.occupation_recommendations.map((occ, idx) => (
                    <OccupationCard
                      key={occ.uuid || occ.originUuid || `occ-${idx}`}
                      occupation={occ}
                    />
                  ))}
                </div>
              )}

            {currentUserMatch?.opportunity_recommendations &&
              currentUserMatch.opportunity_recommendations.length > 0 && (
                <div className="space-y-3">
                  <h3 className="text-sm font-black text-slate-500 uppercase tracking-wider">
                    Job opportunities
                  </h3>
                  {currentUserMatch.opportunity_recommendations.map((job) => (
                    <MatchCard key={job.uuid} job={job} />
                  ))}
                </div>
              )}

            {currentUserMatch?.skill_gap_recommendations &&
              currentUserMatch.skill_gap_recommendations.length > 0 && (
                <div className="space-y-3">
                  <h3 className="text-sm font-black text-slate-500 uppercase tracking-wider">
                    Skill development recommendations
                  </h3>
                  {currentUserMatch.skill_gap_recommendations.map((rec, idx) => (
                    <SkillGapCard
                      key={rec.skill_id || `gap-${idx}`}
                      recommendation={rec}
                    />
                  ))}
                </div>
              )}
          </>
        )}
      </div>
    </div>
  );
};