# Tabiya Matching Engine 
A real-time matching service that connects Jobseekers (Supply) with Employment Opportunities (Demand) using AI-driven scoring based on Skills match, Personal Preferences, and Demand Signals.

## Architecture Overview

The system consists of three primary layers:

- **Backend (FastAPI):** Manages the matching logic and serves recommendations via a REST API.
- **Scoring Engine:** Services that calculate similarity and utility across skills, preferences, and demand data.
- **Frontend (React):** An interactive dashboard for visualizing matches and market insights.

## Installation & Setup

### 1. Backend Setup
1.  Navigate to the `/backend` directory.
2.  Create a virtual environment:  
    `python -m venv venv`
3.  Activate the environment:
    * **Mac/Linux:** `source venv/bin/activate`
    * **Windows:** `venv\Scripts\activate`
4.  Install dependencies:  
    `pip install -r requirements.txt`
5.  Run the API server:  
    `uvicorn app.main:app --reload`
    * *Interactive API documentation is available at `http://127.0.0.1:8000/docs`*.

### 2. Frontend Setup
1.  Navigate to the `/frontend` directory.
2.  Install dependencies:  
    `npm install`
    `npm install -D @tailwindcss/vite`
3.  Run the development server:  
    `npm run dev`
    * *Note: Ensure the backend is running so the frontend can fetch live matches.*

## API Endpoints

The service exposes a primary endpoint for matching. Once the backend is running, visit `http://127.0.0.1:8000/docs` for the interactive Swagger documentation.

### `POST /match`
Calculates matches for one or more users.

- **Example**:
```bash
curl -X POST "http://127.0.0.1:8000/match" -H "Content-Type: application/json" -d '[{"user_id":"u1","city":"Accra","province":"Greater Accra","skills_vector":{"top_skills":[{"preferredLabel":"Software Engineer","originUUID":"sk-123","proficiency":1.0}]},"skill_groups_origin_uuids":["grp-456"],"preference_vector":{"earnings_per_month":500,"task_content":1.0,"physical_demand":0.5,"work_flexibility":0.8,"social_interaction":0.5,"career_growth":0.9,"social_meaning":0.7}}]'
```

## Deployment (Google Cloud Run)

1. Enable required APIs: `gcloud services enable run.googleapis.com cloudbuild.googleapis.com`
2. Copy and customize the env template: `cp backend/configs/template.env.yaml backend/deployment.env.yaml`
3. Build and deploy from the backend directory:
   ```bash
   cd backend
   ./build-and-deploy.sh <project-id> deployment.env.yaml
   ```

## Project Structure

```text
├── backend
│   ├── app
│   │   ├── services/       # Scorer logic (Skill, Preference, Demand)
│   │   ├── main.py         # FastAPI routes & CORS configuration
│   │   ├── schemas.py      # Pydantic data models
│   │   └── config.py       # Global weights and score mappings
│   └── data/               # Model weights (.pt) and skill mappings
├── frontend
│   ├── public/data/        # supply.jsonl and demand.jsonl (Local DB)
│   ├── src
│   │   ├── components/     # UI elements (MatchCard, SearchableSelect)
│   │   ├── views/          # Jobseeker, Employer, and Policy dashboards
│   │   └── App.jsx         # API Orchestration and state management

```

# The Scoring Logic (S_total)

The system calculates a dynamic match score by synthesizing three distinct analytical components:

## 1. Skill Score (S_skill)

This component represents the semantic alignment between a candidate's expertise and a job's requirements.

- **Embedding Model:** Uses a Skip-gram (Node2Vec) approach trained on the Tabiya skill taxonomy.  
- **Vector Similarity:** Maps skills to a 64-dimensional space where related skills are clustered together.  
- **Similarity Engine:** Calculates the cosine similarity between user skills and job requirements.  


## 2. Preference Score (S_pref)

A weighted utility function that measures how well a job's attributes satisfy a user's stated career desires.

- **Factors:** Evaluates variables such as Expected Salary, Preferred Location, and Contract Type (e.g., Full-time vs. Internship).  
- **Utility Mapping:** Converts categories into a value between 0 and 1.  


## 3. Demand Score (S_demand)

A market-intelligence adjustment that optimizes placement by considering labor market demand signals.

- **Market Signals:** Boosts scores for roles in growth sectors or lowers them for over-saturated categories to improve the chance of a successful hire.  


## Skill Gap Analysis

As part of the match response, the system provides a **Skill Gap Analysis**. This identifies specific skills from the Tabiya taxonomy that, if acquired, would significantly increase a user's match score across the current job market.

## Final Calculation

The final match score ($S_{total}$) is a weighted linear combination of the three components:

$$S_{total} = (w_1 \cdot S_{skill}) + (w_2 \cdot S_{pref}) + (w_3 \cdot S_{demand})$$

> **Note:** Weights are configurable in `backend/app/config.py`, allowing policy-makers to prioritize different dimensions (e.g., favoring skill alignment over personal preferences or demand signals).
