from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import networkx as nx
import pandas as pd

from graph_engine.models import EdgeType, NodeType, Skill, SkillGroup

logger = logging.getLogger(__name__)

ABSTRACTION_ALPHA = 1.0


def _load_csv_rows(path: Path) -> list[dict]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return [row.to_dict() for _, row in df.iterrows()]


def _compute_depths(G: nx.MultiDiGraph) -> dict[str, int]:
    roots = [
        n
        for n in G.nodes
        if not any(
            d.get("type") == EdgeType.PARENT_OF for _, _, d in G.in_edges(n, data=True)
        )
    ]
    depth: dict[str, int] = {r: 0 for r in roots}
    dq = deque(roots)
    while dq:
        n = dq.popleft()
        for _, child, d in G.out_edges(n, data=True):
            if d.get("type") == EdgeType.PARENT_OF and child not in depth:
                depth[child] = depth[n] + 1
                dq.append(child)
    return depth


def _assign_edge_weights(G: nx.MultiDiGraph) -> None:
    depth = _compute_depths(G)
    max_depth = max(depth.values()) if depth else 0
    for _u, _v, _key, data in G.edges(keys=True, data=True):
        level = min(depth.get(_u, max_depth), depth.get(_v, max_depth))
        data["weight"] = 1.0 + ABSTRACTION_ALPHA * (max_depth - level)


def build_taxonomy_graph(
    skills_csv: Path,
    groups_csv: Path,
    hierarchy_csv: Path,
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    for row in _load_csv_rows(skills_csv):
        try:
            skill = Skill.from_csv_row(row)
            G.add_node(skill.id, type=NodeType.SKILL, label=skill.preferred_label)
        except Exception as e:
            logger.warning("Skipping skill row: %s", e)

    for row in _load_csv_rows(groups_csv):
        try:
            group = SkillGroup.from_csv_row(row)
            G.add_node(group.id, type=NodeType.SKILL_GROUP, label=group.preferred_label)
        except Exception as e:
            logger.warning("Skipping skill group row: %s", e)

    for row in _load_csv_rows(hierarchy_csv):
        parent_id = row["PARENTID"].strip()
        child_id = row["CHILDID"].strip()
        if G.has_node(parent_id) and G.has_node(child_id):
            G.add_edge(parent_id, child_id, type=EdgeType.PARENT_OF)
            G.add_edge(child_id, parent_id, type=EdgeType.CHILD_OF)

    _assign_edge_weights(G)
    logger.info(
        "Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G
