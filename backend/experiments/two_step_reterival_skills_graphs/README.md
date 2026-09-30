# Two-step skills-graph job matching

Standalone experiment: match Njila users to jobs using a **skill taxonomy graph** with a sequential two-step ranker.

```
exact label overlap  →  weighted graph distance (Dijkstra)
     (precision)              (recall on the rest)
```

| Ranker | Module | Role |
|--------|--------|------|
| **Final** (production) | `final.py` | Exact matches first, then up to 15 graph recommendations from jobs not already retrieved |
| **Exact match** (baseline) | `exact_match.py` | Label overlap only: ≥2 skills and ≥10% job skill coverage |
| **Graph Dijkstra** (baseline) | `graph_dijkstra.py` | Weighted shortest path per job skill from any mapped user skill |

## Data

| File | Description |
|------|-------------|
| `data/njila_users.jsonl` | Njila user profiles (one JSON object per line) |
| `data/ranked_jobs_v2.json` | Job pool with mapped skills |
| `backend/resources/skill_taxonomy/` | Shared repo taxonomy (CSV hierarchy) |

## Layout

```
two_step_reterival_skills_graphs/
├── README.md
├── requirements.txt
├── paths.py                # Shared data paths
├── main.py                 # CLI: final ranker for one user
├── final.py                # Production ranker
├── exact_match.py          # Step 1 baseline
├── graph_dijkstra.py       # Step 2 baseline
├── registry.py             # run_final(), run_all(), dashboard helpers
├── graph_engine/
├── data/
└── dashboard/
    ├── build_dashboard.py
    ├── run_njila_dashboard.py
    └── output/             # Generated (gitignored)
```

### `graph_engine/`

| File | Purpose |
|------|---------|
| `build_graph.py` | Taxonomy CSV → weighted NetworkX graph |
| `context.py` | `load_context()`, map user/job skills to nodes |
| `job_loader.py` | `Job` types, load `ranked_jobs_v2.json` |
| `user_profile.py` | Parse Njila users from JSONL |
| `rec_format.py` | Recommendation dict shape for rankers + dashboard |
| `models.py` | Taxonomy node/edge enums and CSV row types |

## Setup

```bash
pip install -r requirements.txt
```

## Commands

### One user (production ranker)

```bash
python3 main.py <user_id>
```

### Full Njila batch + comparison dashboard

```bash
python3 dashboard/run_njila_dashboard.py
```

Writes `dashboard/output/final_dashboard.json` and `.html` (regenerate locally; not committed).

Open the HTML in a browser. Default columns: **Exact match** vs **Final**. Use the dropdowns to compare any method.

### Regenerate HTML only

```bash
python3 -c "
from pathlib import Path
from dashboard.build_dashboard import write_dashboard
write_dashboard(
    Path('dashboard/output/final_dashboard.json'),
    Path('dashboard/output/final_dashboard.html'),
)
"
```

## Final ranker

1. **Block A — exact:** All jobs passing exact-match thresholds, in exact-match order (`src=exact`). Optional “graph agrees #N” badge.
2. **Block B — graph:** Dijkstra on jobs not in Block A, capped at 15 (`src=graph`).

No job appears twice. The dashboard **Final** column shows rank order and badges only — no combined score.

## Graph scoring

- Edge weight: `1 + ABSTRACTION_ALPHA × (max_depth - level)` (`ABSTRACTION_ALPHA=1.0` in `graph_engine/build_graph.py`).
- Per job skill: minimum weighted distance from any user skill node (0 = same node).
- Sort key: `(-exact_node_matches, avg_distance)`.
