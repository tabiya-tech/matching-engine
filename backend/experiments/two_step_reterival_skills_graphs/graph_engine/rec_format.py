from __future__ import annotations

from graph_engine.job_loader import Job


def job_rec(job: Job, rank: int, score: float, **extra) -> dict:
    rec = {
        "r": rank,
        "jid": job.job_id,
        "t": job.title,
        "e": job.employer,
        "l": job.city or "?",
        "f": round(float(score), 4),
        "je": [s.label for s in job.essential_skills if s.matched],
        "jo": [s.label for s in job.optional_skills if s.matched],
        "ns": len([s for s in job.skills if s.matched]),
    }
    rec.update(extra)
    return rec
