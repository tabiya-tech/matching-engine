import React, { useMemo } from 'react'

export const PolicyView = ({ data }) => {
  // 1. Calculate real-time market intelligence from the API data
  const stats = useMemo(() => {
    if (!data || data.length === 0)
      return { avgFit: 0, skillGap: 0, topSectors: 'N/A' }

    // Calculate Average Fit Score across all matched users
    const allScores = data.flatMap((user) =>
      user.opportunity_recommendations.map((r) => r.final_score),
    )
    const avgFit = allScores.length
      ? allScores.reduce((a, b) => a + b, 0) / allScores.length
      : 0

    // Calculate Skill Gap (Percentage of 'is_eligible' being false)
    const totalRecs = data.flatMap((user) => user.opportunity_recommendations)
    const gaps = totalRecs.filter((r) => !r.is_eligible).length
    const gapPercentage = totalRecs.length ? gaps / totalRecs.length : 0

    return {
      avgFit: (avgFit * 100).toFixed(1) + '%',
      skillGap: (gapPercentage * 100).toFixed(1) + '%',
      count: data.length,
    }
  }, [data])

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="md:col-span-3 bg-indigo-900 text-white p-10 rounded-3xl shadow-xl">
        <h2 className="text-3xl font-black mb-2 uppercase tracking-tight">
          Market Talent Intelligence
        </h2>
        <p className="text-indigo-200 font-medium">
          Aggregating live matching data across <strong>{stats.count}</strong>{' '}
          active youth profiles.
        </p>
      </div>

      <StatCard
        label="Avg placement fit"
        val={stats.avgFit}
        desc="Mean vector similarity score"
      />
      <StatCard
        label="Talent Gap Rate"
        val={stats.skillGap}
        desc="Candidates missing essential skills"
      />
      <StatCard
        label="System Status"
        val="Live"
        desc="FastAPI Match Engine Connected"
      />
    </div>
  )
}

const StatCard = ({ label, val, desc }) => (
  <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm hover:border-indigo-300 transition-colors">
    <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">
      {label}
    </div>
    <div className="text-3xl font-black text-slate-800 mb-1">{val}</div>
    <div className="text-xs text-slate-500 font-medium">{desc}</div>
  </div>
)
