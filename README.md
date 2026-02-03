# Tabiya Matching Engine 
A real-time matching service that connects Youth (Supply) with Employment Opportunities (Demand) using Skills match, preference weightings, and market demand signals.

---

## System Architecture

This project is built using a **Service-Oriented Architecture (SOA)**:

* **Backend:** FastAPI service providing on-demand matching calculations.
* **AI Engine:** Modular scoring services for Skills (Vector Similarity), Preferences (Weighted Utility), and Demand (Market Signals).
* **Frontend:** React (Vite) dashboard for Jobseekers, Employers, and Policy Makers.



---

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

---

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

---

# The Scoring Logic (S_total)

The system calculates a dynamic match score by synthesizing three distinct analytical components:

---

## 1. Skill Score (S_skill)

This component represents the semantic alignment between a candidate's expertise and a job's requirements.

- **Embedding Model:** Utilizes a Skip-gram (Node2Vec) architecture trained on the Tabiya skill taxonomy via the Gensim library.  
- **Vector Similarity:** Maps skills to a 64-dimensional vector space where related skills (e.g., "Python" and "Data Analysis") are mathematically clustered based on hierarchical (parent-child) and relational graph data.  
- **Similarity Engine:** Performs real-time inference using normalized PyTorch tensors to calculate the cosine similarity between a user's `top_skills` and a job's `essential_skills`.  

---

## 2. Preference Score (S_pref)

A weighted utility function that measures how well a job's attributes satisfy a user's stated career desires.

- **Factors:** Evaluates variables such as Expected Salary, Preferred Location, and Contract Type (e.g., Full-time vs. Internship).  
- **Utility Mapping:** Translates categorical preferences into a numerical utility value between 0 and 1.  

---

## 3. Demand Score (S_demand)

A market-intelligence adjustment that optimizes placement by considering labor market signals.

- **Market Signals:** Boosts matches for roles in high-growth sectors or dampens scores for over-saturated job categories to improve the likelihood of a successful hire.  

---

## Final Calculation

The system aggregates these components into a single, actionable score using a weighted linear combination:


S_total = (w_1 * S_skill) + (w_2 * S_pref) + (w_3 * S_demand)


**Note:** The weights (`w1`, `w2`, `w3`) are configurable in `backend/app/config.py`, allowing the system to prioritize skill alignment over preferences or market signals depending on policy requirements.
