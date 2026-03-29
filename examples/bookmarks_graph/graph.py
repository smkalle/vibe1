"""NetworkX graph builder and community detection."""

import pickle
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import networkx as nx

GRAPH_PATH = Path.home() / ".bookmarks_graph.gpickle"


def build_graph(
    all_ids: List[str],
    get_neighbors_fn: Callable[[str], List[dict]],
    progress_callback: Callable[[int], None] = None,
) -> nx.Graph:
    """Build undirected weighted graph from ChromaDB neighbor queries."""
    G = nx.Graph()
    G.add_nodes_from(all_ids)
    for i, tweet_id in enumerate(all_ids):
        for nb in get_neighbors_fn(tweet_id):
            if not G.has_edge(tweet_id, nb["id"]):
                G.add_edge(tweet_id, nb["id"], weight=nb["similarity"])
        if progress_callback:
            progress_callback(i + 1)
    return G


def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """Return {tweet_id: community_id} mapping.

    Uses python-louvain if installed, falls back to NetworkX greedy modularity.
    """
    if G.number_of_nodes() == 0:
        return {}
    try:
        from community import best_partition  # python-louvain
        return best_partition(G)
    except ImportError:
        pass
    # Fallback: NetworkX greedy modularity communities
    comms = nx.community.greedy_modularity_communities(G)
    return {node: i for i, comm in enumerate(comms) for node in comm}


def save_graph(G: nx.Graph):
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)


def load_graph() -> nx.Graph:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError("Graph not built yet. Run: bookmarks build")
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def get_neighbors_in_graph(
    G: nx.Graph, tweet_id: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """Return [(neighbor_id, similarity)] sorted descending."""
    if tweet_id not in G:
        return []
    neighbors = [(nb, G[tweet_id][nb]["weight"]) for nb in G.neighbors(tweet_id)]
    return sorted(neighbors, key=lambda x: x[1], reverse=True)[:top_n]


def graph_stats(G: nx.Graph) -> dict:
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "connected_components": nx.number_connected_components(G),
        "avg_degree": round(sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1), 2),
        "density": round(nx.density(G), 6),
    }
