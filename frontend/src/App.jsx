import React, { useState, useEffect } from 'react';
import { Layout } from './components/Layout';
import { JobseekerView } from './views/JobseekerView';
import { EmployerView } from './views/EmployerView';
import { PolicyView } from './views/PolicyView';
import { MatchingConfigView } from './views/MatchingConfigView';
import { fetchTestUsers, postMatch } from './api/client';

function App() {
  const [view, setView] = useState('jobseeker');
  const [users, setUsers] = useState([]);
  const [liveRecommendations, setLiveRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  /** Persisted across tab switches — JobseekerView used to reset this on unmount. */
  const [selectedJobseekerUserId, setSelectedJobseekerUserId] = useState('');

  // 1. Load test users from Mongo API; fallback to local file for resilience.
  useEffect(() => {
    const canonicalizeUser = (u) => {
      const uid = String(u?.user_id ?? u?.youth_id ?? '').trim();
      return { ...u, user_id: uid };
    };

    const normalizeUsers = (raw) => {
      if (Array.isArray(raw)) return raw.map(canonicalizeUser);
      if (raw && typeof raw === 'object') {
        if (Array.isArray(raw.users)) return raw.users.map(canonicalizeUser);
        if (Array.isArray(raw.test_users)) return raw.test_users.map(canonicalizeUser);
      }
      return [];
    };

    const loadInitialData = async () => {
      try {
        const mongoUsers = await fetchTestUsers();
        setUsers(normalizeUsers(mongoUsers));
      } catch (error) {
        try {
          const response = await fetch('/data/test_users.json');
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const text = await response.text();
          const raw = text.trim() ? JSON.parse(text) : [];
          setUsers(normalizeUsers(raw));
          console.warn('Mongo test-users unavailable, fallback to local file.', error);
        } catch (fallbackError) {
          console.error('Error loading test users (mongo + fallback file):', fallbackError);
        }
      }
    };
    loadInitialData();
  }, []);

  // 2. Function to trigger the live Matching API
  const handleMatch = async (selectedUser) => {
    const uid = String(
      selectedUser.user_id ?? selectedUser.youth_id ?? ''
    ).trim();
    setSelectedJobseekerUserId(uid);

    setLoading(true);
    try {
      // Transform the user data to match the schema
      const requestPayload = {
        user_id: uid,
        city: selectedUser.city,
        province: selectedUser.province,
        skills_vector: selectedUser.skills_vector || { top_skills: [] },
        skill_groups_origin_uuids: selectedUser.skill_groups_origin_uuids || [],
        preference_vector: selectedUser.preference_vector
      };

      const results = await postMatch([requestPayload]);
      const result = Array.isArray(results) ? results[0] : results;
      if (!result?.user_id) throw new Error('Invalid match response');

      // Update global results: replace if user already matched, else add
      setLiveRecommendations(prev => {
        const otherUsers = prev.filter(r => r.user_id !== result.user_id);
        return [...otherUsers, result];
      });
    } catch (error) {
      console.error("API Match failed:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout currentView={view} setView={setView}>
      {view === 'jobseeker' && (
        <JobseekerView
          users={users}
          onMatch={handleMatch}
          data={liveRecommendations}
          isLoading={loading}
          selectedUserId={selectedJobseekerUserId}
        />
      )}
      {view === 'employer' && (
        <EmployerView data={liveRecommendations} isLoading={loading} />
      )}
      {view === 'policy' && (
        <PolicyView data={liveRecommendations} />
      )}
      {view === 'config' && <MatchingConfigView />}
    </Layout>
  );
}

export default App;