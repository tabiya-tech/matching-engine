#!/usr/bin/env python3
"""
SKILL EMBEDDING TRAINING ENGINE (GENSIM VERSION)

DESCRIPTION:
This script builds a mathematical representation of a skill taxonomy by converting 
hierarchical and relational data into high-dimensional vectors (embeddings). 
It follows a three-stage pipeline:
  1. GRAPH CONSTRUCTION: Builds a network where skills are nodes and relationships 
     (parent-child or related) are edges.
  2. RANDOM WALKS: Simulates "navigators" moving through the network to discover 
     contextual neighborhoods for each skill.
  3. EMBEDDING TRAINING: Uses the Word2Vec (Skip-gram) algorithm to assign similar 
     vectors to skills that frequently appear together in the same walks.

WHY GENSIM?
We opted for Gensim over the torch-geometric Node2Vec implementation to resolve 
environment installation errors on Mac ARM64 (M1/M2/M3) chips. While the original 
code requires complex C++ extensions (torch-cluster/pyg-lib), this version uses 
pure-Python random walks and Gensim's optimized CPU-based Word2Vec, delivering 
mathematically equivalent results for skill proximity.

OUTPUTS:
  - skill_embedding_model.pt: A PyTorch checkpoint containing the vector matrix.
  - skill_to_row.json: A mapping of Skill IDs/UUIDs to their specific matrix row.
"""

import os
import json
import torch
import pandas as pd
import networkx as nx
import random
from pathlib import Path
from gensim.models import Word2Vec

_BACKEND_ROOT = Path(__file__).resolve().parents[3]  # .../backend
TAXONOMY_DIR = str(_BACKEND_ROOT / "resources" / "skill_taxonomy")
MODELS_DIR = str(_BACKEND_ROOT / "resources" / "models")

# =============================================================================
# 1) DATA LOADING
# =============================================================================

def load_taxonomy(taxonomy_path: str):
    """Reads taxonomy CSVs from the specified directory."""
    skills = pd.read_csv(os.path.join(taxonomy_path, 'skills.csv'))
    hierarchy = pd.read_csv(os.path.join(taxonomy_path, 'skill_hierarchy.csv'))
    
    # Safely handle the optional relations file if it doesn't exist
    rel_path = os.path.join(taxonomy_path, 'skill_to_skill_relations.csv')
    if os.path.exists(rel_path):
        relations = pd.read_csv(rel_path)
    else:
        relations = pd.DataFrame(columns=['REQUIRINGID', 'REQUIREDID'])
    return skills, hierarchy, relations

# =============================================================================
# 2) GRAPH CONSTRUCTION
# =============================================================================

def build_unweighted_skill_graph(skills_df, hierarchy_df, relations_df):
    """Creates an undirected graph representing the skill network."""
    G = nx.Graph()
    
    # Node order defines the final embedding row order
    node_list = [str(x) for x in skills_df['ID'].dropna().unique().tolist()]
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}

    # Add all skills as nodes in the graph
    for node_id in node_list:
        G.add_node(node_id) 

    # Add hierarchy edges (is-a relationships)
    for _, row in hierarchy_df.iterrows():
        p, c = str(row.get('PARENTID')), str(row.get('CHILDID'))
        if p in G and c in G:
            G.add_edge(p, c)

    # Add cross-relation edges (related-to relationships)
    for _, row in relations_df.iterrows():
        a, b = str(row.get('REQUIRINGID')), str(row.get('REQUIREDID'))
        if a in G and b in G:
            G.add_edge(a, b)
        
    return G, node_to_idx, node_list

# =============================================================================
# 3) MODEL TRAINING (Node2Vec Logic via Gensim)
# =============================================================================

def train_and_save(graph, node_list, artifacts_path):
    """Performs random walks and trains the Word2Vec model."""
    
    def generate_random_walks(G, num_walks, walk_length):
        """Standard random walk generator to mimic Node2Vec behavior."""
        walks = []
        nodes = list(G.nodes())
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                while len(walk) < walk_length:
                    cur = walk[-1]
                    neighbors = list(G.neighbors(cur))
                    if neighbors:
                        walk.append(random.choice(neighbors))
                    else:
                        break
                walks.append(walk)
        return walks

    print("Step 1: Generating random walks...")
    # Parameters matched to original Node2Vec configuration
    walks = generate_random_walks(graph, num_walks=20, walk_length=30)

    print("Step 2: Training Skip-Gram embeddings...")
    # sg=1 (Skip-gram) is the objective used by Node2Vec
    model = Word2Vec(
        sentences=walks, 
        vector_size=64, 
        window=10, 
        min_count=1, 
        sg=1, 
        workers=4,
        epochs=50 # Number of training iterations
    )

    # Convert Gensim vectors into the Torch format expected by the Utility engine
    ordered_weights = [model.wv[node] for node in node_list]
    state = {
        "state_dict": {"embedding.weight": torch.tensor(ordered_weights)},
        "nodes": list(node_list)
    }
    
    # Save artifacts
    os.makedirs(artifacts_path, exist_ok=True)
    torch.save(state, os.path.join(artifacts_path, "skill_embedding_model.pt"))
    print(f"Artifacts successfully saved to: {artifacts_path}")

# =============================================================================
# 4) EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Execute training pipeline
    s_df, h_df, r_df = load_taxonomy(TAXONOMY_DIR)
    G, n2idx, n_list = build_unweighted_skill_graph(s_df, h_df, r_df)
    train_and_save(G, n_list, MODELS_DIR)

    # Export mapping for inference lookup
    with open(os.path.join(MODELS_DIR, "skill_to_row.json"), "w") as f:
        json.dump(n2idx, f)