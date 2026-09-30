"""Step 2 ranker: weighted shortest-path distance (Dijkstra) on the taxonomy graph."""

from __future__ import annotations

import networkx as nx

from graph_engine.job_loader import Job
from graph_engine.rec_format import job_rec
from graph_engine.user_profile import UserProfile

NAME = "graph_dijkstra"
LABEL = "Graph Dijkstra (weighted taxonomy distance)"
NEEDS_GRAPH = True


def compute_user_dist_maps(
    graph: nx.MultiDiGraph,
    user_nodes: list[tuple[str, str]],
) -> dict[str, dict[str, float]]:
    """For each user skill node, shortest weighted distance to every reachable node."""
    user_dist_maps: dict[str, dict[str, float]] = {}
    for nid, _ in user_nodes:
        user_dist_maps[nid] = nx.single_source_dijkstra_path_length(
            graph, nid, weight="weight"
        )
    return user_dist_maps


def rank(
    _user: UserProfile,
    jobs: list[Job],
    *,
    graph: nx.MultiDiGraph,
    user_nodes: list[tuple[str, str]],
    user_dist_maps: dict[str, dict[str, float]] | None = None,
    top_n: int = 30,
) -> list[dict]:
    if user_dist_maps is None:
        user_dist_maps = compute_user_dist_maps(graph, user_nodes)

    ranked: list[tuple[tuple[int, float], Job, dict]] = []
    for job in jobs:
        job_skill_ids = [
            sk.graph_node_id
            for sk in job.skills
            if sk.matched and sk.graph_node_id and graph.has_node(sk.graph_node_id)
        ]
        if not job_skill_ids:
            continue

        per_skill_min: list[tuple[str, float, str | None]] = []
        for js_id in job_skill_ids:
            js_label = next(
                (sk.label for sk in job.skills if sk.graph_node_id == js_id), js_id
            )
            best_dist = -1.0
            best_user: str | None = None
            for unid, ulbl in user_nodes:
                d = user_dist_maps[unid].get(js_id)
                if d is None:
                    continue
                if best_dist < 0 or d < best_dist:
                    best_dist = d
                    best_user = ulbl
            if best_dist >= 0:
                per_skill_min.append((js_label, best_dist, best_user))

        if not per_skill_min:
            continue

        exact_matches = sum(1 for _, d, _ in per_skill_min if d == 0)
        skill_mins = [d for _, d, _ in per_skill_min]
        avg_dist = sum(skill_mins) / len(skill_mins)
        min_dist = min(skill_mins)
        max_dist = max(skill_mins)
        reachable = len(per_skill_min)

        sort_key = (-exact_matches, avg_dist)
        ms = []
        for js_label, dist, user_lbl in per_skill_min:
            ok = dist == 0
            ms.append([user_lbl or "?", js_label, round(dist, 2), ok, round(dist, 2)])

        ranked.append(
            (
                sort_key,
                job,
                {
                    "avg_dist": round(avg_dist, 3),
                    "min_dist": round(min_dist, 2),
                    "max_dist": round(max_dist, 2),
                    "exact_matches": exact_matches,
                    "reachable": reachable,
                    "total_job_skills": len(job_skill_ids),
                    "ms": ms,
                },
            )
        )

    ranked.sort(key=lambda x: x[0])
    recs: list[dict] = []
    for rank_idx, (_, job, meta) in enumerate(ranked[:top_n], 1):
        recs.append(
            job_rec(
                job,
                rank_idx,
                meta["avg_dist"],
                ms=meta["ms"],
                ad=meta["avg_dist"],
                mind=meta["min_dist"],
                maxd=meta["max_dist"],
                em=meta["exact_matches"],
                reach=meta["reachable"],
                tjs=meta["total_job_skills"],
            )
        )
    return recs
