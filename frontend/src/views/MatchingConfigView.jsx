import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchMatchingConfig, saveMatchingConfig } from '../api/client';

function formatSource(sources, key) {
  const s = sources?.[key];
  if (!s) return '—';
  return s === 'mongodb' ? 'MongoDB' : s === 'default' ? 'default' : s;
}

export function MatchingConfigView() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [effective, setEffective] = useState({});
  const [sources, setSources] = useState({});
  const [updatedAt, setUpdatedAt] = useState(null);
  const [draft, setDraft] = useState({});

  const keys = useMemo(() => Object.keys(effective).sort(), [effective]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchMatchingConfig();
      setEffective(data.effective ?? {});
      setSources(data.sources ?? {});
      setUpdatedAt(data.updated_at ?? null);
      setDraft(data.effective ?? {});
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const onChange = (key, value) => {
    setDraft((d) => ({ ...d, [key]: value }));
  };

  const onSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const data = await saveMatchingConfig(draft);
      setEffective(data.effective ?? {});
      setSources(data.sources ?? {});
      setUpdatedAt(data.updated_at ?? null);
      setDraft(data.effective ?? {});
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setSaving(false);
    }
  };

  const onReset = () => {
    setDraft({ ...effective });
    setError(null);
  };

  const dirty = useMemo(() => {
    return keys.some((k) => String(draft[k]) !== String(effective[k]));
  }, [keys, draft, effective]);

  if (loading) {
    return (
      <div className="bg-white p-12 rounded-2xl border border-slate-200 text-center text-slate-400 font-semibold">
        Loading configuration…
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="text-xl font-black text-tabiya-navy tracking-tight">
              Matching configuration
            </h2>
            <p className="text-sm text-slate-500 mt-1 max-w-2xl">
              Tunables are stored in MongoDB and merged with code defaults. Saving
              updates the effective values used by{' '}
              <span className="font-mono text-xs bg-slate-100 px-1 rounded">/match</span>.
            </p>
            {updatedAt && (
              <p className="text-xs text-slate-400 mt-2">
                Last persisted: {updatedAt}
              </p>
            )}
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={load}
              disabled={saving}
              className="px-4 py-2 rounded-full text-sm font-bold border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              Reload
            </button>
            <button
              type="button"
              onClick={onReset}
              disabled={saving || !dirty}
              className="px-4 py-2 rounded-full text-sm font-bold border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              Discard edits
            </button>
            <button
              type="button"
              onClick={onSave}
              disabled={saving || !dirty}
              className="px-5 py-2 rounded-full text-sm font-black bg-tabiya-navy text-white shadow-md hover:shadow-lg disabled:opacity-50"
            >
              {saving ? 'Saving…' : 'Save to MongoDB'}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 rounded-2xl p-4 text-sm font-medium whitespace-pre-wrap">
          {error}
        </div>
      )}

      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-left text-xs font-black uppercase tracking-wide text-slate-500">
                <th className="px-4 py-3 w-2/5">Key</th>
                <th className="px-4 py-3">Value</th>
                <th className="px-4 py-3 w-40">Source</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((key) => (
                <tr key={key} className="border-b border-slate-100 last:border-0">
                  <td className="px-4 py-2 align-top">
                    <span className="font-mono text-xs text-tabiya-navy">{key}</span>
                  </td>
                  <td className="px-4 py-2">
                    <input
                      type="text"
                      value={draft[key] ?? ''}
                      onChange={(e) => onChange(key, e.target.value)}
                      className="w-full max-w-md rounded-lg border border-slate-200 px-3 py-1.5 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-tabiya-green/40"
                    />
                  </td>
                  <td className="px-4 py-2 text-xs text-slate-500">
                    {formatSource(sources, key)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
