import React, { useState } from 'react';
import { MatchCard } from '../components/MatchCard';
import { SearchableSelect } from '../components/SearchableSelect'; // Import here

export const JobseekerView = ({ users, onMatch, data, isLoading }) => {
  const [selectedUserId, setSelectedUserId] = useState("");

  const getUserId = (u) => String(u?.user_id ?? u?.youth_id ?? "");

  const handleSelect = (user) => {
    const id = getUserId(user);
    setSelectedUserId(id);
    onMatch({ ...user, user_id: id }); // Ensure payload always includes user_id
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

            {currentUserMatch?.skill_gap_recommendations && currentUserMatch.skill_gap_recommendations.length > 0 && (
              <div className="bg-amber-50 border border-amber-200 rounded-2xl p-6 mb-4">
                <h3 className="text-lg font-black text-amber-800 mb-3">💡 Skill Development Recommendations</h3>
                <div className="space-y-2">
                  {currentUserMatch.skill_gap_recommendations.map((rec, idx) => (
                    <div key={idx} className="text-sm text-amber-700">
                      <span className="font-bold">•</span> {JSON.stringify(rec)}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {currentUserMatch?.opportunity_recommendations?.map(job => (
              <MatchCard key={job.uuid} job={job} />
            ))}
          </>
        )}
      </div>
    </div>
  );
};