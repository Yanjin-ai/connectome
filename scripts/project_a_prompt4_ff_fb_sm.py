from pathlib import Path

import networkx as nx
import pandas as pd


DATA = Path("data/real")
EDGES_PATH = DATA / "edges_filtered.csv"
NODE_TYPES_PATH = DATA / "node_types.csv"
FIG_PATH = DATA / "fig3_ff_fb_balance.png"
SM_NODES_PATH = DATA / "sm_subgraph_nodes.csv"
SM_EDGES_PATH = DATA / "sm_subgraph_edges.csv"

LAYER_RANK = {"sensor": 0, "inter": 1, "motor": 2}


def require_inputs() -> None:
    missing = [str(path) for path in (EDGES_PATH, NODE_TYPES_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input file(s):\n  "
            + "\n  ".join(missing)
            + "\nRun Project A Prompt 2 and Prompt 3 first, or place these CSVs in data/real/."
        )


def validate_columns(edges: pd.DataFrame, node_types: pd.DataFrame) -> None:
    required_edges = {"pre", "post", "weight"}
    required_types = {"neuron_id", "type"}
    missing_edges = required_edges - set(edges.columns)
    missing_types = required_types - set(node_types.columns)
    if missing_edges:
        raise ValueError(f"{EDGES_PATH} is missing columns: {sorted(missing_edges)}")
    if missing_types:
        raise ValueError(f"{NODE_TYPES_PATH} is missing columns: {sorted(missing_types)}")


def print_direction_balance(G: nx.DiGraph, node_type: dict) -> tuple[list[str], list[float]]:
    feedforward = feedback = lateral = 0.0
    ff_count = fb_count = lat_count = 0

    for u, v, d in G.edges(data=True):
        w = d["weight"]
        r_u = LAYER_RANK.get(node_type.get(u, "inter"), 1)
        r_v = LAYER_RANK.get(node_type.get(v, "inter"), 1)

        if r_v > r_u:
            feedforward += w
            ff_count += 1
        elif r_v < r_u:
            feedback += w
            fb_count += 1
        else:
            lateral += w
            lat_count += 1

    weights = [feedforward, feedback, lateral]
    counts = [ff_count, fb_count, lat_count]
    labels = ["feedforward", "feedback", "lateral"]

    total_weight = sum(weights)
    total_count = sum(counts)
    weight_pct = [100 * value / total_weight if total_weight else 0 for value in weights]
    count_pct = [100 * value / total_count if total_count else 0 for value in counts]

    print("FF/FB/lateral by weight:")
    for label, value, pct in zip(labels, weights, weight_pct):
        print(f"  {label:12s}: {value:,.0f} ({pct:5.2f}%)")

    print("FF/FB/lateral by edge count:")
    for label, value, pct in zip(labels, counts, count_pct):
        print(f"  {label:12s}: {value:,} ({pct:5.2f}%)")

    return labels, weight_pct


def save_balance_plot(labels: list[str], weight_pct: list[float]) -> None:
    import matplotlib.pyplot as plt

    colors = ["cornflowerblue", "tomato", "goldenrod"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, weight_pct, color=colors)
    ax.set_ylabel("Synaptic weight (%)")
    ax.set_title("FF / FB / Lateral balance")
    ax.set_ylim(0, max(weight_pct) * 1.18 if weight_pct else 1)

    for bar, pct in zip(bars, weight_pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_PATH}")


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


def build_sensorimotor_subgraph(G: nx.DiGraph, node_type: dict) -> nx.DiGraph:
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

    sm_inter = inter_with_sensor_input & inter_with_motor_output
    sm_nodes = sensor_ids | motor_ids | sm_inter
    G_sm = G.subgraph(sm_nodes).copy()

    print_sensorimotor_summary(G_sm, node_type)

    if G_sm.number_of_nodes() < 200:
        print("WARNING Subgraph too small — check classification")
        sm_nodes = expand_small_subgraph(G, node_type, sensor_ids | motor_ids)
        G_sm = G.subgraph(sm_nodes).copy()
        print("Expanded subgraph to include inter-neurons within 2 hops of any sensor or motor node")
        print_sensorimotor_summary(G_sm, node_type)

    if G_sm.number_of_nodes() > 50_000:
        print("WARNING Subgraph large — betweenness will be slow")
        top_nodes = sorted(G_sm.degree(), key=lambda x: x[1], reverse=True)[:5000]
        G_sm = G_sm.subgraph([n for n, _ in top_nodes]).copy()
        print("Subgraph truncated to top-5000 by degree")
        print_sensorimotor_summary(G_sm, node_type)

    return G_sm


def print_sensorimotor_summary(G_sm: nx.DiGraph, node_type: dict) -> None:
    counts = pd.Series([node_type.get(n, "inter") for n in G_sm.nodes()]).value_counts()
    print(f"Sensorimotor subgraph: {G_sm.number_of_nodes()} nodes, {G_sm.number_of_edges()} edges")
    print(
        "  sensor: "
        f"{counts.get('sensor', 0)}  inter: {counts.get('inter', 0)}  motor: {counts.get('motor', 0)}"
    )


def main() -> None:
    require_inputs()

    edges = pd.read_csv(EDGES_PATH)
    node_types = pd.read_csv(NODE_TYPES_PATH)
    validate_columns(edges, node_types)

    node_type = node_types.set_index("neuron_id")["type"].to_dict()
    G = nx.from_pandas_edgelist(
        edges,
        source="pre",
        target="post",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )

    labels, weight_pct = print_direction_balance(G, node_type)
    save_balance_plot(labels, weight_pct)

    G_sm = build_sensorimotor_subgraph(G, node_type)
    pd.Series(list(G_sm.nodes()), name="neuron_id").to_csv(SM_NODES_PATH, index=False)
    nx.to_pandas_edgelist(G_sm).to_csv(SM_EDGES_PATH, index=False)
    print("Saved sm_subgraph_nodes.csv and sm_subgraph_edges.csv")


if __name__ == "__main__":
    main()
