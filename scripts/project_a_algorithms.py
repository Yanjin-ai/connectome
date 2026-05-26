from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


DATA = Path("data/real")
EDGES_PATH = DATA / "edges_filtered.csv"
NODE_TYPES_PATH = DATA / "node_types.csv"
SM_NODES_PATH = DATA / "sm_subgraph_nodes.csv"
SM_EDGES_PATH = DATA / "sm_subgraph_edges.csv"

BETWEENNESS_PATH = DATA / "betweenness_results.csv"
COMMUNITY_PATH = DATA / "community_results.csv"
GRC_PATH = DATA / "grc_result.txt"
BETWEENNESS_FIG_PATH = DATA / "fig5_betweenness.png"
COMMUNITY_FIG_PATH = DATA / "fig6_communities.png"


def load_node_types() -> dict:
    return pd.read_csv(NODE_TYPES_PATH).set_index("neuron_id")["type"].to_dict()


def expand_small_subgraph(G: nx.DiGraph, node_type: dict, base_nodes: set) -> set:
    expanded = set(base_nodes)
    frontier = set(base_nodes)

    for _ in range(2):
        next_frontier = set()
        for node in frontier:
            if node not in G:
                continue
            next_frontier.update(G.predecessors(node))
            next_frontier.update(G.successors(node))
        expanded.update(n for n in next_frontier if node_type.get(n, "inter") == "inter")
        frontier = next_frontier

    return expanded


def build_sensorimotor_subgraph(node_type: dict) -> nx.DiGraph:
    edges = pd.read_csv(EDGES_PATH)
    G = nx.from_pandas_edgelist(
        edges,
        source="pre",
        target="post",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )

    sensor_ids = {n for n, t in node_type.items() if t == "sensor"}
    motor_ids = {n for n, t in node_type.items() if t == "motor"}

    inter_with_sensor_input = {
        v
        for u, v in G.edges()
        if node_type.get(u, "inter") == "sensor" and node_type.get(v, "inter") == "inter"
    }
    inter_with_motor_output = {
        u
        for u, v in G.edges()
        if node_type.get(u, "inter") == "inter" and node_type.get(v, "inter") == "motor"
    }

    sm_nodes = sensor_ids | motor_ids | (inter_with_sensor_input & inter_with_motor_output)
    G_sm = G.subgraph(sm_nodes).copy()
    print_sensorimotor_summary(G_sm, node_type)

    if G_sm.number_of_nodes() < 200:
        print("WARNING Subgraph too small - check classification")
        sm_nodes = expand_small_subgraph(G, node_type, sensor_ids | motor_ids)
        G_sm = G.subgraph(sm_nodes).copy()
        print("Expanded subgraph to include inter-neurons within 2 hops of any sensor or motor node")
        print_sensorimotor_summary(G_sm, node_type)

    if G_sm.number_of_nodes() > 50_000:
        print("WARNING Subgraph large - betweenness will be slow")
        top_nodes = sorted(G_sm.degree(), key=lambda x: x[1], reverse=True)[:5000]
        G_sm = G_sm.subgraph([n for n, _ in top_nodes]).copy()
        print("Subgraph truncated to top-5000 by degree")
        print_sensorimotor_summary(G_sm, node_type)

    pd.Series(list(G_sm.nodes()), name="neuron_id").to_csv(SM_NODES_PATH, index=False)
    nx.to_pandas_edgelist(G_sm).to_csv(SM_EDGES_PATH, index=False)
    print("Saved sm_subgraph_nodes.csv and sm_subgraph_edges.csv")
    return G_sm


def print_sensorimotor_summary(G_sm: nx.DiGraph, node_type: dict) -> None:
    counts = pd.Series([node_type.get(n, "inter") for n in G_sm.nodes()]).value_counts()
    print(f"Sensorimotor subgraph: {G_sm.number_of_nodes()} nodes, {G_sm.number_of_edges()} edges")
    print(
        "  sensor: "
        f"{counts.get('sensor', 0)}  inter: {counts.get('inter', 0)}  motor: {counts.get('motor', 0)}"
    )


def load_or_build_sm_graph(node_type: dict) -> nx.DiGraph:
    if not SM_NODES_PATH.exists() or not SM_EDGES_PATH.exists():
        return build_sensorimotor_subgraph(node_type)

    sm_nodes = pd.read_csv(SM_NODES_PATH)["neuron_id"].tolist()
    sm_edges = pd.read_csv(SM_EDGES_PATH)
    G_sm = nx.from_pandas_edgelist(
        sm_edges,
        source="source",
        target="target",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )
    G_sm.add_nodes_from(sm_nodes)
    print(f"G_sm loaded: {G_sm.number_of_nodes()} nodes, {G_sm.number_of_edges()} edges")
    return G_sm


def run_betweenness(G_sm: nx.DiGraph, node_type: dict) -> None:
    t0 = time.time()
    print("Starting betweenness centrality (k=300)...")

    k_val = 300
    if G_sm.number_of_nodes() > 20_000:
        k_val = 100
        print(f"Large subgraph: using k={k_val}")

    bc = nx.betweenness_centrality(G_sm, normalized=True, weight="weight", k=k_val, seed=42)
    print(f"Done in {time.time() - t0:.0f}s")

    bc_df = pd.DataFrame(
        {
            "neuron_id": list(bc.keys()),
            "betweenness": list(bc.values()),
            "type": [node_type.get(n, "inter") for n in bc.keys()],
        }
    )
    bc_df.to_csv(BETWEENNESS_PATH, index=False)
    print(f"Saved {BETWEENNESS_PATH}")
    print("Top-10 bottleneck neurons:")
    print(bc_df.sort_values("betweenness", ascending=False).head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for typ in ["sensor", "inter", "motor"]:
        values = bc_df.loc[bc_df["type"] == typ, "betweenness"]
        if len(values):
            ax.hist(values, bins=50, alpha=0.7, label=typ)
    ax.set_yscale("log")
    ax.set_xlabel("Betweenness centrality")
    ax.set_ylabel("Neuron count")
    ax.set_title("Betweenness centrality by type")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BETWEENNESS_FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {BETWEENNESS_FIG_PATH}")


def run_louvain(G_sm: nx.DiGraph, node_type: dict) -> None:
    G_sm_undir = G_sm.to_undirected()
    t0 = time.time()
    print("Starting Louvain community detection...")

    communities = nx.community.louvain_communities(G_sm_undir, seed=42, weight="weight")
    print(f"Done in {time.time() - t0:.0f}s - {len(communities)} communities found")

    node_to_comm = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            node_to_comm[n] = cid

    comm_df = pd.DataFrame(
        {
            "neuron_id": list(node_to_comm.keys()),
            "community": list(node_to_comm.values()),
            "type": [node_type.get(n, "inter") for n in node_to_comm.keys()],
        }
    )
    comm_df.to_csv(COMMUNITY_PATH, index=False)
    print(f"Saved {COMMUNITY_PATH}")

    comm_sizes = sorted([len(c) for c in communities], reverse=True)
    print(f"Number of communities: {len(communities)}")
    print(f"Top-5 sizes: {comm_sizes[:5]}")
    print(f"Smallest community size: {comm_sizes[-1]}")

    largest = sorted(enumerate(communities), key=lambda x: len(x[1]), reverse=True)[:5]
    for cid, comm in largest:
        types = pd.Series([node_type.get(n, "inter") for n in comm])
        fractions = types.value_counts(normalize=True)
        print(
            f"Community {cid} size {len(comm)} fractions: "
            f"sensor={fractions.get('sensor', 0):.3f}, "
            f"inter={fractions.get('inter', 0):.3f}, "
            f"motor={fractions.get('motor', 0):.3f}"
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(comm_sizes, bins=50, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Community size")
    ax.set_ylabel("Community count")
    ax.set_title("Community size distribution")
    fig.tight_layout()
    fig.savefig(COMMUNITY_FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {COMMUNITY_FIG_PATH}")


def run_grc(G_sm: nx.DiGraph) -> None:
    t0 = time.time()
    print("Starting Global Reaching Centrality...")

    fallback_triggered = False
    try:
        if G_sm.number_of_nodes() > 10_000:
            print(f"WARNING: {G_sm.number_of_nodes()} nodes - GRC may take 20-60 min")
        grc = nx.global_reaching_centrality(G_sm, weight="weight", normalized=True)
        print(f"GRC = {grc:.4f}  (computed in {time.time() - t0:.0f}s)")
    except Exception as exc:
        print(f"GRC failed: {exc}")
        print("FALLBACK: using feedforward fraction as hierarchy proxy")
        grc = None
        fallback_triggered = True

    with open(GRC_PATH, "w") as f:
        f.write(f"GRC: {grc}\n")
        f.write(f"Fallback used: {fallback_triggered}\n")
        f.write(f"Nodes in subgraph: {G_sm.number_of_nodes()}\n")
    print(f"Saved {GRC_PATH}")
    if grc is None:
        print("GRC interpretation: exact hierarchy estimate unavailable; use FF fraction proxy.")
    else:
        print("GRC interpretation: larger values indicate more globally hierarchical reachability.")


def main() -> None:
    node_type = load_node_types()
    G_sm = load_or_build_sm_graph(node_type)
    run_betweenness(G_sm, node_type)
    run_louvain(G_sm, node_type)
    run_grc(G_sm)


if __name__ == "__main__":
    main()
