import React from 'react';

function pct1(x) {
  if (x == null || Number.isNaN(Number(x))) return '—';
  return `${(Number(x) * 100).toFixed(1)}%`;
}

function Bar({ label, val, color }) {
  const v = Math.min(Math.max(Number(val) || 0, 0), 1);
  return (
    <div>
      <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase mb-1 gap-2">
        <span className="leading-tight">{label}</span>
        <span className="shrink-0">{pct1(v)}</span>
      </div>
      <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${v * 100}%` }} />
      </div>
    </div>
  );
}

function MiniStat({ label, value }) {
  const n = value != null && value !== '' ? Number(value) : null;
  const text =
    n != null && !Number.isNaN(n)
      ? n >= 0 && n <= 1
        ? `${n.toFixed(3)} (${pct1(n)})`
        : n.toFixed(3)
      : '—';
  return (
    <div className="text-xs">
      <span className="text-slate-400">{label}</span>
      <span className="ml-1 font-mono text-slate-700">{text}</span>
    </div>
  );
}

function PhatStat({ label, value }) {
  const n = value != null ? Number(value) : null;
  const display = n != null && !Number.isNaN(n) ? pct1(n) : '—';
  return (
    <div className="rounded-lg bg-white/80 border border-slate-100 px-2 py-1.5">
      <div className="text-[9px] font-black uppercase tracking-wide text-slate-400">{label}</div>
      <div className="font-mono text-sm text-slate-800">{display}</div>
    </div>
  );
}

function Glossary({ title, items }) {
  return (
    <div className="rounded-xl bg-white/50 border border-slate-100/90 px-3 py-2.5">
      {title && (
        <p className="text-[10px] font-black uppercase tracking-wide text-slate-500 mb-2">{title}</p>
      )}
      <dl className="space-y-2 text-[11px] leading-snug text-slate-600">
        {items.map(({ k, text }) => (
          <div key={k}>
            <dt className="inline font-semibold text-slate-800">{k}</dt>
            <dd className="inline text-slate-600"> — {text}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** Collapsible explanations only — scores stay outside */
function ExplainScoresDropdown({ variant, children }) {
  const accent =
    variant === 'teal'
      ? 'text-teal-800 hover:text-teal-900 border-teal-100'
      : 'text-indigo-800 hover:text-indigo-900 border-indigo-100';
  return (
    <details className={`group rounded-xl border bg-white/30 ${variant === 'teal' ? 'border-teal-100/80' : 'border-indigo-100/80'} open:bg-white/50`}>
      <summary
        className={`cursor-pointer list-none px-3 py-2.5 text-xs font-bold ${accent} flex items-center gap-2 select-none rounded-xl`}
      >
        <span className="inline-block text-[10px] transition-transform duration-200 group-open:rotate-90">
          ▸
        </span>
        What do these metrics mean?
      </summary>
      <div className="px-3 pb-3 pt-1 space-y-3 border-t border-slate-100/90">{children}</div>
    </details>
  );
}

const MULT_MODEL_INTRO =
  'Multiplicative mode combines seeker-side utility (u_hat) with recruiter/market propensity (p_hat). The headline match score follows u_hat × p_hat (after rounding).';

const MULT_GLOSS_UP = [
  {
    k: 'u_hat',
    text: 'Seeker-side utility from preferences and job attributes (including optional BWS / work-activity alignment when present).',
  },
  {
    k: 'p_hat',
    text: 'Recruiter-side success propensity: feasibility gate, essential-skill fit, readiness, and market signals folded into one propensity score.',
  },
];

const MULT_GLOSS_PHAT_PARTS = [
  {
    k: 'Gate',
    text: 'Hard feasibility: whether enough essential skills clear the similarity gate (shown as strength of the gate signal).',
  },
  {
    k: 'Essential fit',
    text: 'Aggregate similarity of required essential skills to the youth’s skills (geometric mean of best-match similarities).',
  },
  {
    k: 'Recruiter readiness',
    text: 'Blend of optional-skill alignment and skill-group overlap when the posting lists those dimensions.',
  },
  {
    k: 'Market opportunity',
    text: 'Labour-market / demand signal derived from the posting’s demand attributes.',
  },
];

const MULT_GLOSS_SKILL_U = [
  {
    k: 'U_final',
    text: 'Combined skill-location utility after weights and soft penalty — this is the scalar skill utility feeding the multiplicative pipeline.',
  },
  {
    k: 'Preference (legacy scalar)',
    text: 'Legacy display scalar related to preference scoring (see u_hat for the primary preference utility in this mode).',
  },
  {
    k: 'Penalty applied',
    text: 'Soft penalty from the share of essential skills that sit below the eligibility threshold.',
  },
];

const MULT_GLOSS_SKILL_COMPONENTS = [
  {
    k: 'loc',
    text: 'Location utility: stronger when city matches, weaker for province-only or distant placements.',
  },
  {
    k: 'ess',
    text: 'Essential skills: average best-match cosine similarity between required skills and the youth’s profile.',
  },
  {
    k: 'opt',
    text: 'Optional skills: centroid similarity between optional requirements and the youth’s skill mix.',
  },
  {
    k: 'grp',
    text: 'Skill groups: recall — fraction of the job’s competency groups covered by the youth’s groups.',
  },
];

const ADD_MODEL_INTRO =
  'Additive mode combines skill utility, preference score, and market demand using configured weights (not shown here). Each bar is one ingredient before weighting.';

const ADD_GLOSS_TOP_BARS = [
  {
    k: 'Skill utility (U)',
    text: 'Overall U after blending loc / ess / opt / grp and subtracting the gap penalty — same geometric core as multiplicative, but mixed linearly with pref and demand.',
  },
  {
    k: 'Preference',
    text: 'Normalized alignment between the youth’s preference vector and job attributes.',
  },
  {
    k: 'Market demand',
    text: 'Score mapped from the posting’s demand label (labour-market outlook).',
  },
];

const ADD_GLOSS_SKILL_COMPONENTS = [
  {
    k: 'loc',
    text: 'Location match strength (city vs province vs baseline). Feeds U before it is weighted with preference and demand.',
  },
  {
    k: 'ess',
    text: 'Essential skill proximity — embedding similarity for required skills.',
  },
  {
    k: 'opt',
    text: 'Optional skill alignment via centroid similarity.',
  },
  {
    k: 'grp',
    text: 'Skill group overlap — how many of the job’s competency groups the youth covers.',
  },
  {
    k: 'Penalty',
    text: 'Penalty applied for essential skills that fall below the gate threshold (gap share).',
  },
];

export function ScoreBreakdownDetails({ score_breakdown, variant = 'indigo' }) {
  const sb = score_breakdown ?? {};
  const pc = sb.p_hat_components ?? {};
  const sk = sb.skill_components ?? {};

  const mult =
    sb.u_hat != null &&
    sb.p_hat != null &&
    !Number.isNaN(Number(sb.u_hat)) &&
    !Number.isNaN(Number(sb.p_hat));

  const barColor = variant === 'teal' ? 'bg-teal-500' : 'bg-indigo-500';
  const border = variant === 'teal' ? 'border-teal-100' : 'border-indigo-100';
  const softBg = variant === 'teal' ? 'bg-teal-50/60' : 'bg-indigo-50/60';

  const product =
    mult && sb.u_hat != null && sb.p_hat != null
      ? Number(sb.u_hat) * Number(sb.p_hat)
      : null;

  return (
    <div className={`mt-6 rounded-2xl border ${border} ${softBg} p-4 space-y-4`}>
      <h4 className="text-[10px] font-black uppercase tracking-wider text-slate-500">
        Score breakdown
      </h4>

      {mult ? (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Bar
              label="u_hat — seeker utility (preferences / attributes)"
              val={sb.u_hat}
              color={barColor}
            />
            <Bar
              label="p_hat — success propensity (feasibility / market)"
              val={sb.p_hat}
              color={barColor}
            />
          </div>

          <p className="text-[11px] text-slate-700 leading-snug font-mono">
            u_hat × p_hat
            {product != null && !Number.isNaN(product) && (
              <span className="text-slate-500 font-sans">
                {' '}
                → {pct1(product)}
              </span>
            )}
          </p>

          <div>
            <p className="text-[10px] font-black uppercase text-slate-400 mb-2">
              p_hat components
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <PhatStat label="Gate" value={pc.gate} />
              <PhatStat label="Essential fit" value={pc.essential_fit} />
              <PhatStat label="Recruiter readiness" value={pc.recruiter_readiness} />
              <PhatStat label="Market opportunity" value={pc.market_opportunity} />
            </div>
          </div>

          <div className="border-t border-slate-200/80 pt-3 space-y-2">
            <p className="text-[10px] font-black uppercase text-slate-400">Skill utility (U)</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <MiniStat label="U_final" value={sb.total_skill_utility} />
              <MiniStat label="Preference (legacy scalar)" value={sb.preference_score} />
              <MiniStat label="Penalty applied" value={sb.skill_penalty_applied} />
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-3 gap-y-1 pt-1">
              <MiniStat label="loc" value={sk.loc} />
              <MiniStat label="ess" value={sk.ess} />
              <MiniStat label="opt" value={sk.opt} />
              <MiniStat label="grp" value={sk.grp} />
            </div>
          </div>

          {(sb.demand_label != null && sb.demand_label !== '') ||
          sb.demand_score != null ? (
            <p className="text-xs text-slate-600 border-t border-slate-200/80 pt-3">
              {sb.demand_label != null && sb.demand_label !== '' && (
                <>
                  Demand label:{' '}
                  <span className="font-semibold">{sb.demand_label}</span>
                  {sb.demand_score != null && (
                    <span className="text-slate-500 ml-2">
                      (mapped score {pct1(sb.demand_score)})
                    </span>
                  )}
                </>
              )}
              {(sb.demand_label == null || sb.demand_label === '') &&
                sb.demand_score != null && (
                  <span>Mapped demand score: {pct1(sb.demand_score)}</span>
                )}
            </p>
          ) : null}

          <ExplainScoresDropdown variant={variant}>
            <p className="text-[11px] text-slate-600 leading-relaxed border-l-2 border-slate-300 pl-3">
              {MULT_MODEL_INTRO}
            </p>
            <Glossary title="Multiplicative — u_hat & p_hat" items={MULT_GLOSS_UP} />
            <Glossary title="Multiplicative — what each p_hat part means" items={MULT_GLOSS_PHAT_PARTS} />
            <Glossary title="Multiplicative — skill utility scalars" items={MULT_GLOSS_SKILL_U} />
            <Glossary title="Multiplicative — loc / ess / opt / grp" items={MULT_GLOSS_SKILL_COMPONENTS} />
          </ExplainScoresDropdown>
        </>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Bar label="Skill utility (U)" val={sb.total_skill_utility} color={barColor} />
            <Bar label="Preference" val={sb.preference_score} color={barColor} />
            <Bar label="Market demand" val={sb.demand_score} color={barColor} />
          </div>

          {sb.demand_label != null && sb.demand_label !== '' && (
            <p className="text-xs text-slate-600">
              Demand label: <span className="font-semibold">{sb.demand_label}</span>
            </p>
          )}
          <div className="border-t border-slate-200/80 pt-3 space-y-2">
            <p className="text-[10px] font-black uppercase text-slate-400">Skill components</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-3 gap-y-1">
              <MiniStat label="loc" value={sk.loc} />
              <MiniStat label="ess" value={sk.ess} />
              <MiniStat label="opt" value={sk.opt} />
              <MiniStat label="grp" value={sk.grp} />
              <MiniStat label="Penalty" value={sb.skill_penalty_applied} />
            </div>
          </div>

          <ExplainScoresDropdown variant={variant}>
            <p className="text-[11px] text-slate-600 leading-relaxed border-l-2 border-slate-300 pl-3">
              {ADD_MODEL_INTRO}
            </p>
            <Glossary title="Additive — the three weighted inputs" items={ADD_GLOSS_TOP_BARS} />
            <Glossary title="Additive — loc / ess / opt / grp / penalty" items={ADD_GLOSS_SKILL_COMPONENTS} />
          </ExplainScoresDropdown>
        </>
      )}
    </div>
  );
}
