import React, { useState, useMemo } from 'react'

export const EmployerView = ({ data, isLoading }) => {
  const [selectedJob, setSelectedJob] = useState(null)

  // 1. Pivot the data: Take multiple API results and group them by Job UUID
  const jobsMap = useMemo(() => {
    const map = {}

    // 'data' is now an array of responses from your /match endpoint
    data.forEach((userMatchResponse) => {
      const userId = userMatchResponse.user_id

      userMatchResponse.opportunity_recommendations.forEach((rec) => {
        if (!map[rec.uuid]) {
          map[rec.uuid] = {
            title: rec.opportunity_title,
            location: rec.location,
            candidates: [],
          }
        }

        // Push this candidate into the specific job's list
        map[rec.uuid].candidates.push({
          user_id: userId,
          score: rec.final_score,
          eligible: rec.is_eligible,
          breakdown: rec.score_breakdown,
        })
      })
    })
    return map
  }, [data])

  const jobKeys = Object.keys(jobsMap)

  // Default to the first job if nothing is selected
  const currentJobUuid = selectedJob || jobKeys[0]
  const currentJob = jobsMap[currentJobUuid]

  // Loading State
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center p-20 space-y-4">
        <div className="w-12 h-12 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
        <p className="font-black text-slate-400 uppercase tracking-widest">
          Running Global Match Engine...
        </p>
      </div>
    )
  }

  // Empty State
  if (jobKeys.length === 0) {
    return (
      <div className="p-10 text-center bg-slate-50 rounded-2xl border-2 border-dashed border-slate-200">
        <p className="text-slate-500 font-bold">No candidate matches found.</p>
        <p className="text-sm text-slate-400">
          Run a match from the Jobseeker view to populate this list.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Job Selection Header */}
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
        <label className="block text-sm font-black text-slate-400 uppercase mb-2">
          Select Active Job Opening
        </label>
        <select
          value={currentJobUuid}
          onChange={(e) => setSelectedJob(e.target.value)}
          className="w-full bg-slate-50 border border-slate-200 p-3 rounded-xl font-bold text-slate-700 outline-none focus:ring-2 focus:ring-indigo-500"
        >
          {jobKeys.map((uuid) => (
            <option key={uuid} value={uuid}>
              {jobsMap[uuid].title} ({jobsMap[uuid].location})
            </option>
          ))}
        </select>
      </div>

      {/* Candidates Table */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-4 border-b border-slate-100 bg-slate-50/50">
          <h3 className="font-black text-slate-700 uppercase text-sm">
            Ranked Candidates for {currentJob.title}
          </h3>
        </div>
        <table className="w-full text-left">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr>
              <th className="p-4 text-xs font-black text-slate-400 uppercase">
                Candidate ID
              </th>
              <th className="p-4 text-xs font-black text-slate-400 uppercase text-center">
                Eligibility
              </th>
              <th className="p-4 text-xs font-black text-slate-400 uppercase text-right">
                Match Score
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {currentJob.candidates
              .sort((a, b) => b.score - a.score)
              .map((cand) => (
                <tr
                  key={cand.user_id}
                  className="hover:bg-slate-50 transition-colors"
                >
                  <td className="p-4">
                    <div className="font-bold text-slate-700">
                      {cand.user_id}
                    </div>
                    <div className="text-[10px] text-slate-400 font-medium">
                      Skills:{' '}
                      {(cand.breakdown.total_skill_utility * 100).toFixed(0)}% |
                      Prefs:{' '}
                      {(cand.breakdown.preference_score * 100).toFixed(0)}%
                    </div>
                  </td>
                  <td className="p-4 text-center">
                    <span
                      className={`px-3 py-1 rounded-full text-[10px] font-black uppercase ${
                        cand.eligible
                          ? 'bg-emerald-100 text-emerald-600'
                          : 'bg-rose-100 text-rose-600'
                      }`}
                    >
                      {cand.eligible ? 'Qualified' : 'Gap Identified'}
                    </span>
                  </td>
                  <td className="p-4 text-right">
                    <span className="font-black text-indigo-600 text-xl">
                      {(cand.score * 100).toFixed(0)}%
                    </span>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
