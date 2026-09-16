import React, { useMemo } from 'react'

export const EmployerView = ({ data }) => {
  // Pivot logic to group by Job instead of User
  const jobsMap = useMemo(() => {
    const map = {}
    data.forEach((user) => {
      user.opportunity_recommendations.forEach((rec) => {
        if (!map[rec.uuid])
          map[rec.uuid] = { title: rec.opportunity_title, candidates: [] }
        map[rec.uuid].candidates.push({
          id: user.user_id,
          score: rec.final_score,
        })
      })
    })
    return map
  }, [data])

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-6 text-slate-800">
        Talent Pipeline
      </h2>
      {Object.entries(jobsMap).map(([id, job]) => (
        <div
          key={id}
          className="mb-8 bg-white p-6 rounded-2xl border border-slate-200"
        >
          <h3 className="text-lg font-bold text-indigo-900 mb-4">
            {job.title}
          </h3>
          <div className="space-y-2">
            {job.candidates
              .sort((a, b) => b.score - a.score)
              .slice(0, 5)
              .map((c) => (
                <div
                  key={c.id}
                  className="flex justify-between p-2 bg-slate-50 rounded-lg"
                >
                  <span className="font-medium">{c.id}</span>
                  <span className="font-black text-indigo-600">
                    {(c.score * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      ))}
    </div>
  )
}
