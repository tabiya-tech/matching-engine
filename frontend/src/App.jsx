import React, { useState, useEffect } from 'react';
import { Layout } from './components/Layout';
import { JobseekerView } from './views/JobseekerView';
import { EmployerView } from './views/EmployerView';
import { PolicyView } from './views/PolicyView';

function App() {
  const [view, setView] = useState('jobseeker');
  const [users, setUsers] = useState([]);
  const [liveRecommendations, setLiveRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);

  // 1. Load supply data on startup from public/data
  useEffect(() => {
    const loadInitialData = async () => {
      const fetchJsonl = async (url) => {
        const response = await fetch(url);
        const text = await response.text();
        // Parse JSONL: split by lines and filter out empty strings
        return text.trim().split('\n').filter(line => line).map(line => JSON.parse(line));
      };

      try {
        const supplyData = await fetchJsonl('/data/supply.jsonl');
        setUsers(supplyData);
      } catch (error) {
        console.error("Error loading initial data files:", error);
      }
    };
    loadInitialData();
  }, []);

  // 2. Function to trigger the live Matching API
  const handleMatch = async (selectedUser) => {
    setLoading(true);
    try {
      // Transform the user data to match the schema
      const requestPayload = {
        user_id: selectedUser.user_id,
        city: selectedUser.city,
        province: selectedUser.province,
        skills_vector: selectedUser.skills_vector || { top_skills: [] },
        skill_groups_origin_uuids: selectedUser.skill_groups_origin_uuids || [],
        preference_vector: selectedUser.preference_vector
      };

      const response = await fetch('http://127.0.0.1:8000/match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload)
      });

      if (!response.ok) throw new Error("Backend connection failed");

      const result = await response.json();

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
        />
      )}
      {view === 'employer' && (
        <EmployerView data={liveRecommendations} isLoading={loading} />
      )}
      {view === 'policy' && (
        <PolicyView data={liveRecommendations} />
      )}
    </Layout>
  );
}

export default App;