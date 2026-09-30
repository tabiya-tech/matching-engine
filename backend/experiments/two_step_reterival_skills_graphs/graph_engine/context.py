"""Load taxonomy graph, jobs, and map user/job skills to graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from graph_engine.build_graph import build_taxonomy_graph
from graph_engine.job_loader import Job, load_ranked_jobs_v2
from graph_engine.models import NodeType
from graph_engine.user_profile import UserProfile

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TAXONOMY_DIR = _BACKEND_ROOT / "resources" / "skill_taxonomy"


@dataclass
class MatchContext:
    tax: nx.MultiDiGraph
    label_index: dict[str, str]
    jobs: list[Job]


def skill_label_index(graph: nx.MultiDiGraph) -> dict[str, str]:
    index: dict[str, str] = {}
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("type") != NodeType.SKILL:
            continue
        label = str(attrs.get("label", "")).lower().strip()
        if label:
            index[label] = nid
    return index


def map_jobs_to_graph(
    jobs: list[Job], label_index: dict[str, str], graph: nx.MultiDiGraph
) -> None:
    for job in jobs:
        for skill in job.skills:
            node_id = label_index.get(skill.label.lower().strip())
            if node_id and graph.has_node(node_id):
                skill.graph_node_id = node_id
                skill.matched = True


def map_user_to_graph(
    user: UserProfile, label_index: dict[str, str], graph: nx.MultiDiGraph
) -> list[tuple[str, str]]:
    user_nodes: list[tuple[str, str]] = []
    for skill in user.skills:
        label = skill.preferred_label.lower().strip()
        node_id = label_index.get(label)
        if node_id and graph.has_node(node_id):
            user_nodes.append((node_id, skill.preferred_label))
    return user_nodes


def load_context(taxonomy_dir: Path, jobs_path: Path) -> MatchContext:
    graph = build_taxonomy_graph(
        taxonomy_dir / "skills.csv",
        taxonomy_dir / "skill_groups.csv",
        taxonomy_dir / "skill_hierarchy.csv",
    )
    jobs = load_ranked_jobs_v2(jobs_path)
    label_index = skill_label_index(graph)
    map_jobs_to_graph(jobs, label_index, graph)
    return MatchContext(tax=graph, label_index=label_index, jobs=jobs)
