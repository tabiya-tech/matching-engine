"""Run production ranker and dashboard baselines."""

from __future__ import annotations

import exact_match
import final
import graph_dijkstra
from graph_engine.context import MatchContext, map_user_to_graph
from graph_engine.user_profile import UserProfile

RECS_FIELD = {
    final.NAME: "recs_final",
    exact_match.NAME: "recs_exact",
    graph_dijkstra.NAME: "recs_dijkstra",
}

ALL = (final, exact_match, graph_dijkstra)


def _graph_context(user: UserProfile, ctx: MatchContext):
    graph = ctx.tax
    user_nodes = map_user_to_graph(user, ctx.label_index, graph)
    user_dist_maps = (
        graph_dijkstra.compute_user_dist_maps(graph, user_nodes) if user_nodes else {}
    )
    return graph, user_nodes, user_dist_maps


def run_final(user: UserProfile, ctx: MatchContext, *, top_n: int = 30) -> list[dict]:
    graph, user_nodes, user_dist_maps = _graph_context(user, ctx)
    return final.rank(
        user,
        ctx.jobs,
        graph=graph,
        user_nodes=user_nodes,
        user_dist_maps=user_dist_maps,
        top_n=top_n,
    )


def run_all(user: UserProfile, ctx: MatchContext, *, top_n: int = 30) -> dict:
    graph, user_nodes, user_dist_maps = _graph_context(user, ctx)
    results = {"mapped_skills": len(user_nodes)}
    for mod in ALL:
        if getattr(mod, "NEEDS_GRAPH", False):
            results[mod.NAME] = mod.rank(
                user,
                ctx.jobs,
                graph=graph,
                user_nodes=user_nodes,
                user_dist_maps=user_dist_maps,
                top_n=top_n,
            )
        else:
            results[mod.NAME] = mod.rank(user, ctx.jobs, top_n=top_n)
    return results


def user_dashboard_record(raw: dict, user: UserProfile, rankings: dict) -> dict:
    record = {
        "uid": user.user_id,
        "name": raw.get("programme_name") or user.user_id,
        "city": user.city,
        "prov": user.province,
        "ns": len(user.skills),
        "mapped": rankings["mapped_skills"],
        "user_skills": [s.preferred_label for s in user.skills],
    }
    for name, field in RECS_FIELD.items():
        record[field] = rankings.get(name, [])
    return record


def dashboard_meta(n_users: int) -> dict:
    return {mod.NAME: {"label": mod.LABEL, "n_users": n_users} for mod in ALL}
