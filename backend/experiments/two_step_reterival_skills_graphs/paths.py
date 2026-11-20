from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
USERS_PATH = DATA_DIR / "njila_users.jsonl"
JOBS_PATH = DATA_DIR / "ranked_jobs_v2.json"
