"""Step 1 ranker: exact label overlap (no graph traversal)."""

from __future__ import annotations

from graph_engine.job_loader import Job
from graph_engine.rec_format import job_rec
from graph_engine.user_profile import UserProfile

NAME = "exact_match"
LABEL = "Exact label overlap (≥2 skills, ≥10% coverage)"

MIN_MATCH_COUNT = 2
MIN_COVERAGE = 0.10


def rank(user: UserProfile, jobs: list[Job], *, top_n: int = 30) -> list[dict]:
    user_labels = {s.preferred_label.lower().strip() for s in user.skills}
    results: list[tuple[float, Job, list[str]]] = []

    for job in jobs:
        job_labels = [sk.label.lower().strip() for sk in job.skills if sk.matched]
        if not job_labels:
            continue
        matched_job = [jl for jl in job_labels if jl in user_labels]
        if not matched_job:
            continue
        match_count = len(matched_job)
        job_skill_count = len(job_labels)
        if (
            match_count < MIN_MATCH_COUNT
            or match_count / job_skill_count < MIN_COVERAGE
        ):
            continue
        score = match_count / job_skill_count
        results.append((score, job, matched_job))

    results.sort(key=lambda x: x[0], reverse=True)
    recs: list[dict] = []
    for rank_idx, (score, job, matched_job) in enumerate(results[:top_n], 1):
        ms = [[m, m, 1.0, True] for m in sorted(set(matched_job))]
        recs.append(
            job_rec(
                job,
                rank_idx,
                score,
                ms=ms,
                mc=len(matched_job),
                jsc=len([s for s in job.skills if s.matched]),
            )
        )
    return recs
