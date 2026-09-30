"""Production ranker: exact matches first, then graph on remaining jobs."""

from __future__ import annotations

import exact_match
import graph_dijkstra
from graph_engine.job_loader import Job
from graph_engine.user_profile import UserProfile

NAME = "final"
LABEL = "Final (exact first, then graph on the rest)"
NEEDS_GRAPH = True
GRAPH_BLOCK_LIMIT = 15


def rank(
    user: UserProfile,
    jobs: list[Job],
    *,
    graph,
    user_nodes,
    user_dist_maps=None,
    top_n: int = 30,
) -> list[dict]:
    exact_recs = exact_match.rank(user, jobs, top_n=len(jobs))
    exact_ids = {r["jid"] for r in exact_recs}

    graph_recs = graph_dijkstra.rank(
        user,
        jobs,
        graph=graph,
        user_nodes=user_nodes,
        user_dist_maps=user_dist_maps,
        top_n=len(jobs),
    )
    graph_rank = {r["jid"]: i for i, r in enumerate(graph_recs)}

    graph_block: list[dict] = []
    for r in graph_recs:
        if r["jid"] in exact_ids:
            continue
        graph_block.append(r)
        if len(graph_block) >= GRAPH_BLOCK_LIMIT:
            break

    recs: list[dict] = []
    rank_idx = 1

    for r in exact_recs:
        rec = dict(r)
        rec["r"] = rank_idx
        rec["src"] = "exact"
        if r["jid"] in graph_rank:
            rec["gr"] = graph_rank[r["jid"]] + 1
        recs.append(rec)
        rank_idx += 1
        if rank_idx > top_n:
            return recs

    for r in graph_block:
        rec = dict(r)
        rec["r"] = rank_idx
        rec["src"] = "graph"
        rec["gr"] = graph_rank[r["jid"]] + 1
        rec["raw_dist"] = r.get("ad")
        recs.append(rec)
        rank_idx += 1
        if rank_idx > top_n:
            break

    return recs
