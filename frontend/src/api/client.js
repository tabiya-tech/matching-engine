/** Backend API helpers (Vite env: `VITE_API_BASE`, `VITE_API_KEY`). */

const STAGING_UI_HOST = 'matching-ui-staging-896947954616.us-central1.run.app';
const STAGING_API_BASE = 'https://matching-service-staging-896947954616.us-central1.run.app';
const envApiBase = import.meta.env.VITE_API_BASE && String(import.meta.env.VITE_API_BASE).trim();
const isHostedStaging = typeof window !== 'undefined' && window.location.hostname === STAGING_UI_HOST;

export const API_BASE = envApiBase || (isHostedStaging ? STAGING_API_BASE : 'http://127.0.0.1:8000');

export function apiHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  // Backend uses FastAPI APIKeyHeader: missing x-api-key → 403 {"detail":"Not authenticated"}.
  // Keep a default test key so staging frontends still work if build-time env is empty.
  const key =
    (import.meta.env.VITE_API_KEY && String(import.meta.env.VITE_API_KEY).trim()) ||
    'local-dev';
  if (key) {
    headers['x-api-key'] = key;
  }
  return headers;
}

export async function fetchMatchingConfig() {
  const res = await fetch(`${API_BASE}/config`, { headers: apiHeaders() });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `GET /config failed (${res.status})`);
  }
  return res.json();
}

export async function saveMatchingConfig(body) {
  const res = await fetch(`${API_BASE}/config`, {
    method: 'POST',
    headers: apiHeaders(),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `POST /config failed (${res.status})`);
  }
  return res.json();
}

export async function postMatch(usersPayload) {
  const res = await fetch(`${API_BASE}/match`, {
    method: 'POST',
    headers: apiHeaders(),
    body: JSON.stringify(usersPayload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `POST /match failed (${res.status})`);
  }
  return res.json();
}

export async function fetchTestUsers() {
  const res = await fetch(`${API_BASE}/test-users`, { headers: apiHeaders() });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `GET /test-users failed (${res.status})`);
  }
  const data = await res.json();
  return Array.isArray(data) ? data : (data.users ?? []);
}

export async function fetchMongoConfig() {
  const res = await fetch(`${API_BASE}/config/mongo`, { headers: apiHeaders() });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `GET /config/mongo failed (${res.status})`);
  }
  return res.json();
}

export async function saveMongoConfig(body) {
  const res = await fetch(`${API_BASE}/config/mongo`, {
    method: 'POST',
    headers: apiHeaders(),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `POST /config/mongo failed (${res.status})`);
  }
  return res.json();
}
