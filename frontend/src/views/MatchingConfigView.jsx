import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  fetchMatchingConfig,
  saveMatchingConfig,
  fetchMongoConfig,
  saveMongoConfig,
} from '../api/client';

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
  const [mongoCfg, setMongoCfg] = useState({});
  const [mongoDraft, setMongoDraft] = useState({});

  const keys = useMemo(() => Object.keys(effective).sort(), [effective]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [data, mongoData] = await Promise.all([
        fetchMatchingConfig(),
        fetchMongoConfig(),
      ]);
      setEffective(data.effective ?? {});
      setSources(data.sources ?? {});
      setUpdatedAt(data.updated_at ?? null);
      setDraft(data.effective ?? {});
      setMongoCfg(mongoData ?? {});
      setMongoDraft(mongoData ?? {});
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
    setMongoDraft({ ...mongoCfg });
    setError(null);
  };

  const dirty = useMemo(() => {
    return keys.some((k) => String(draft[k]) !== String(effective[k]));
  }, [keys, draft, effective]);
  const mongoDirty = useMemo(() => {
    return Object.keys(mongoCfg).some((k) => String(mongoDraft[k] ?? '') !== String(mongoCfg[k] ?? ''));
  }, [mongoCfg, mongoDraft]);

  const onMongoChange = (key, value) => {
    setMongoDraft((d) => ({ ...d, [key]: value }));
  };

  const onSaveMongo = async () => {
    setSaving(true);
    setError(null);
    try {
      const data = await saveMongoConfig(mongoDraft);
      const applied = data.mongo_config ?? {};
      setMongoCfg(applied);
      setMongoDraft(applied);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setSaving(false);
    }
  };

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
              disabled={saving || (!dirty && !mongoDirty)}
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

      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
          <h3 className="text-xs font-black uppercase tracking-wide text-slate-500">
            Mongo DB routing (jobs only)
          </h3>
          <button
            type="button"
            onClick={onSaveMongo}
            disabled={saving || !mongoDirty}
            className="px-4 py-1.5 rounded-full text-xs font-black bg-tabiya-navy text-white disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save DB config'}
          </button>
        </div>
        <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-3">
          {Object.keys(mongoDraft).map((key) => (
            <label key={key} className="block">
              <span className="block mb-1 font-mono text-[11px] text-slate-600">{key}</span>
              <input
                type="text"
                value={mongoDraft[key] ?? ''}
                onChange={(e) => onMongoChange(key, e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-1.5 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-tabiya-green/40"
              />
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}
