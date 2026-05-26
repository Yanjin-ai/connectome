from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DATA = Path("data/real")
NODES_IN = DATA / "all_nodes.csv"
IN_COUNTS = Path("/private/tmp/flywire_in_counts.txt")
OUT_COUNTS = Path("/private/tmp/flywire_out_counts.txt")
DEGREES_OUT = DATA / "degree_sequences.npz"
FIG_OUT = DATA / "fig1_degree_distribution.png"


def read_count_file(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open() as handle:
        for line in handle:
            count, neuron_id = line.strip().split(maxsplit=1)
            counts[neuron_id] = int(count)
    return counts


def plot_degree_distributions(in_degrees: np.ndarray, out_degrees: np.ndarray, nodes: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(f"FlyWire connectome degree distribution (n={nodes} neurons)")

    for ax, vals, title in [
        (axes[0], in_degrees, "In-degree"),
        (axes[1], out_degrees, "Out-degree"),
    ]:
        positive = vals[vals > 0]
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
    with NODES_IN.open() as handle:
        next(handle)
        nodes = [line.strip() for line in handle if line.strip()]

    in_counts = read_count_file(IN_COUNTS)
    out_counts = read_count_file(OUT_COUNTS)
    in_degrees = np.array([in_counts.get(node, 0) for node in nodes], dtype=np.int64)
    out_degrees = np.array([out_counts.get(node, 0) for node in nodes], dtype=np.int64)

    np.savez_compressed(DEGREES_OUT, in_degree=in_degrees, out_degree=out_degrees, nodes=len(nodes))
    plot_degree_distributions(in_degrees, out_degrees, len(nodes))

    print(f"Saved {DEGREES_OUT}")
    print(f"Saved {FIG_OUT}")


if __name__ == "__main__":
    main()
