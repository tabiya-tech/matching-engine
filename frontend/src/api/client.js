/** Backend API helpers (Vite env: `VITE_API_BASE`, `VITE_API_KEY`). */

export const API_BASE =
  import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000';

export function apiHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  // Backend uses FastAPI APIKeyHeader: missing x-api-key → 403 {"detail":"Not authenticated"}.
  // VITE_API_KEY must be set for production builds; local dev falls back so /config and /match work.
  const key =
    import.meta.env.VITE_API_KEY ||
    (import.meta.env.DEV ? 'local-dev' : '');
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
