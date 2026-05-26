from __future__ import annotations

import gc
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa


DATA = Path("data/real")
CONNECTIONS = DATA / "proofread_connections_783.feather"
SCHEMA_REPORT = DATA / "schema_report.txt"
FIG_OUT = DATA / "fig1_degree_distribution.png"
EDGES_OUT = DATA / "edges_filtered.csv"
NODES_OUT = DATA / "all_nodes.csv"
DEGREES_OUT = DATA / "degree_sequences.npz"
SYN_THRESHOLD = 5


def read_or_create_schema_report() -> str:
    if SCHEMA_REPORT.exists():
        return SCHEMA_REPORT.read_text()

    if not CONNECTIONS.exists():
        raise FileNotFoundError(f"Missing required input: {CONNECTIONS}")

    with pa.memory_map(str(CONNECTIONS), "r") as source:
        schema = pa.ipc.open_file(source).schema
    columns = [field.name for field in schema]
    pre_col, post_col, syn_col = infer_columns_from_names(columns)
    report = [
        "=== proofread_connections_783.feather ===",
        f"Shape: column count = {len(columns)}",
        f"Columns: {columns}",
        "",
        f"ASSUMED COLUMNS: pre={pre_col!r}, post={post_col!r}, syn={syn_col!r}",
        "",
    ]
    SCHEMA_REPORT.write_text("\n".join(report))
    print(f"schema_report.txt was missing; created {SCHEMA_REPORT} from Feather schema.")
    return SCHEMA_REPORT.read_text()


def infer_columns_from_names(columns: list[str]) -> tuple[str, str, str]:
    lower = {column.lower(): column for column in columns}
    pre = lower.get("pre_pt_root_id") or lower.get("pre_neuron_id") or columns[0]
    post = lower.get("post_pt_root_id") or lower.get("post_neuron_id") or columns[1]
    for candidate in ("syn_count", "synapse_count", "weight", "n_syn"):
        if candidate in lower:
            return pre, post, lower[candidate]
    syn_like = [
        column
        for column in columns
        if "syn" in column.lower() and column not in {pre, post}
    ]
    if syn_like:
        return pre, post, syn_like[0]
    return pre, post, columns[2]


def identify_columns(report_text: str) -> tuple[str, str, str]:
    assumed = re.search(
        r"ASSUMED COLUMNS:\s*pre=(['\"])(?P<pre>.+?)\1,\s*"
        r"post=(['\"])(?P<post>.+?)\3,\s*syn=(['\"])(?P<syn>.+?)\5",
        report_text,
    )
    if assumed:
        return assumed.group("pre"), assumed.group("post"), assumed.group("syn")

    columns_match = re.search(r"Columns:\s*\[(?P<cols>.*?)\]", report_text, re.S)
    if not columns_match:
        raise ValueError("Could not parse columns from schema_report.txt")
    columns = re.findall(r"['\"]([^'\"]+)['\"]", columns_match.group("cols"))
    if len(columns) < 3:
        raise ValueError("schema_report.txt does not list at least three columns")
    return columns[0], columns[1], columns[2]


def stats(values: list[int]) -> tuple[float, float, int]:
    arr = np.asarray(values)
    return float(np.median(arr)), float(np.mean(arr)), int(np.max(arr))


def plot_degree_distributions(in_degrees: list[int], out_degrees: list[int], nodes: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(f"FlyWire connectome degree distribution (n={nodes} neurons)")

    for ax, vals, title in [
        (axes[0], in_degrees, "In-degree"),
        (axes[1], out_degrees, "Out-degree"),
    ]:
        positive = np.asarray([v for v in vals if v > 0])
        if positive.size == 0:
            ax.text(0.5, 0.5, "No positive degrees", ha="center", va="center")
            continue
        bins = np.logspace(0, np.log10(positive.max()), 40)
        ax.hist(positive, bins=bins, color="#3b6ea8", alpha=0.82, edgecolor="white")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Neuron count")
        ax.grid(True, which="both", alpha=0.25)

    fig.savefig(FIG_OUT, dpi=180)
    plt.close(fig)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    report_text = read_or_create_schema_report()
    pre_col, post_col, syn_col = identify_columns(report_text)
    print(f"Identified pre-synaptic neuron ID column: {pre_col}")
    print(f"Identified post-synaptic neuron ID column: {post_col}")
    print(f"Identified synapse count column: {syn_col}")

    print(f"Loading {CONNECTIONS} columns: {pre_col}, {post_col}, {syn_col}...")
    conn = pd.read_feather(CONNECTIONS, columns=[pre_col, post_col, syn_col])
    edges = conn[conn[syn_col] >= SYN_THRESHOLD][[pre_col, post_col, syn_col]].copy()
    edges.columns = ["pre", "post", "weight"]
    del conn
    gc.collect()
    unique_neurons = pd.unique(edges[["pre", "post"]].to_numpy().ravel()).size
    print(f"Edges after filtering (synapse_count >= {SYN_THRESHOLD}): {len(edges):,}")
    print(f"Unique neurons after filtering: {unique_neurons:,}")
    edges.to_csv(EDGES_OUT, index=False)

    graph_edges = edges
    start = time.monotonic()
    try:
        G = nx.from_pandas_edgelist(
            graph_edges,
            source="pre",
            target="post",
            edge_attr="weight",
            create_using=nx.DiGraph(),
        )
        elapsed = time.monotonic() - start
        if elapsed > 15 * 60:
            print("FALLBACK: using 500k edge sample")
            graph_edges = edges.sample(500_000, random_state=42)
            G = nx.from_pandas_edgelist(
                graph_edges,
                source="pre",
                target="post",
                edge_attr="weight",
                create_using=nx.DiGraph(),
            )
    except MemoryError:
        print("FALLBACK: using 500k edge sample")
        graph_edges = edges.sample(500_000, random_state=42)
        G = nx.from_pandas_edgelist(
            graph_edges,
            source="pre",
            target="post",
            edge_attr="weight",
            create_using=nx.DiGraph(),
        )

    nodes = G.number_of_nodes()
    graph_edge_count = G.number_of_edges()
    density = nx.density(G)
    print(f"G.number_of_nodes(): {nodes:,}")
    print(f"G.number_of_edges(): {graph_edge_count:,}")
    print(f"nx.density(G): {density:.8g}")
    del edges
    del graph_edges
    gc.collect()

    print("Computing degree sequences...")
    in_degrees = [degree for _, degree in G.in_degree()]
    out_degrees = [degree for _, degree in G.out_degree()]
    in_median, in_mean, in_max = stats(in_degrees)
    out_median, out_mean, out_max = stats(out_degrees)
    print(f"In-degree median/mean/max: {in_median:.2f} / {in_mean:.2f} / {in_max:,}")
    print(f"Out-degree median/mean/max: {out_median:.2f} / {out_mean:.2f} / {out_max:,}")

    print("Computing degree assortativity...")
    r = nx.degree_assortativity_coefficient(G)
    print(f"Degree assortativity coefficient: {r:.6g}")
    if r > 0:
        print("Interpretation: r > 0, so hubs tend to connect to hubs.")
    elif r < 0:
        print("Interpretation: r < 0, so hubs tend to connect to low-degree nodes.")
    else:
        print("Interpretation: r = 0, so there is no clear degree assortativity pattern.")

    nodes_list = list(G.nodes)
    del G
    gc.collect()

    pd.Series(nodes_list, name="neuron_id").to_csv(NODES_OUT, index=False)
    np.savez_compressed(DEGREES_OUT, in_degree=in_degrees, out_degree=out_degrees, nodes=nodes)

    print("Plotting degree distributions...")
    plot_degree_distributions(in_degrees, out_degrees, nodes)
    print(f"Saved {FIG_OUT}")

    print("Saved edges_filtered.csv and all_nodes.csv")


if __name__ == "__main__":
    main()
