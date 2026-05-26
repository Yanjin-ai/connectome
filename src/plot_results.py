import sys
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEEDS = [42, 123, 456, 789, 1024]
ARCHS = {
    "connectome": ("Connectome GNN", "#2196F3"),
    "random": ("Random Graph", "#FF9800"),
    "mlp": ("MLP", "#9E9E9E"),
}
SOLVE_THRESHOLD = 450


data = {}
for arch in ARCHS:
    curves = []
    for seed in SEEDS:
        p = Path(f"results/{arch}_seed{seed}.npy")
        if p.exists():
            curves.append(np.load(p))
    if curves:
        ts = curves[0][0]
        means = np.stack([c[1] for c in curves if len(c[0]) == len(ts)])
        data[arch] = {"ts": ts, "mean": means.mean(0), "std": means.std(0), "all": means}
    else:
        print(f"WARNING: no data for {arch}")


metrics = {}
for arch, d in data.items():
    ts, all_r = d["ts"], d["all"]
    solve_steps = []
    for r in all_r:
        idx = np.where(r >= SOLVE_THRESHOLD)[0]
        solve_steps.append(ts[idx[0]] if len(idx) else ts[-1])
    last_30 = int(0.7 * len(ts))
    asym_means = all_r[:, last_30:].mean(axis=1)
    stab_stds = all_r[:, last_30:].std(axis=1)
    metrics[arch] = {
        "solve_steps": np.array(solve_steps),
        "asym_reward": asym_means,
        "stability": stab_stds,
    }


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Project B: Connectome GNN vs Baselines (CartPole-v1, 5 seeds)",
    fontsize=13,
    fontweight="bold",
)

ax = axes[0, 0]
for arch, (label, color) in ARCHS.items():
    if arch not in data:
        continue
    d = data[arch]
    ax.plot(d["ts"], d["mean"], color=color, lw=2, label=label)
    ax.fill_between(
        d["ts"],
        d["mean"] - d["std"],
        d["mean"] + d["std"],
        color=color,
        alpha=0.15,
    )
ax.axhline(
    SOLVE_THRESHOLD,
    color="black",
    lw=1,
    ls="--",
    alpha=0.5,
    label=f"Solved ({SOLVE_THRESHOLD})",
)
ax.set_xlabel("Training timesteps")
ax.set_ylabel("Mean eval reward")
ax.set_title("A. Learning curves (mean ± std over 5 seeds)")
ax.legend(fontsize=9)
ax.set_ylim(0, 520)

ax = axes[0, 1]
positions = range(len(ARCHS))
for pos, (arch, (label, color)) in zip(positions, ARCHS.items()):
    if arch not in metrics:
        continue
    s = metrics[arch]["solve_steps"]
    ax.bar(pos, s.mean(), color=color, alpha=0.8, width=0.6)
    ax.errorbar(pos, s.mean(), yerr=s.std(), color="black", capsize=5, lw=2)
ax.set_xticks(list(positions))
ax.set_xticklabels([ARCHS[a][0] for a in ARCHS], fontsize=10)
ax.set_ylabel("Timesteps to reach reward ≥ 450")
ax.set_title("B. Sample efficiency (lower = better)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))

ax = axes[1, 0]
for pos, (arch, (label, color)) in zip(positions, ARCHS.items()):
    if arch not in metrics:
        continue
    v = metrics[arch]["asym_reward"]
    ax.bar(pos, v.mean(), color=color, alpha=0.8, width=0.6)
    ax.errorbar(pos, v.mean(), yerr=v.std(), color="black", capsize=5, lw=2)
ax.set_xticks(list(positions))
ax.set_xticklabels([ARCHS[a][0] for a in ARCHS], fontsize=10)
ax.set_ylabel("Mean reward (last 30% of training)")
ax.set_title("C. Asymptotic performance")
ax.set_ylim(0, 520)

ax = axes[1, 1]
for pos, (arch, (label, color)) in zip(positions, ARCHS.items()):
    if arch not in metrics:
        continue
    v = metrics[arch]["stability"]
    ax.bar(pos, v.mean(), color=color, alpha=0.8, width=0.6)
    ax.errorbar(pos, v.mean(), yerr=v.std(), color="black", capsize=5, lw=2)
ax.set_xticks(list(positions))
ax.set_xticklabels([ARCHS[a][0] for a in ARCHS], fontsize=10)
ax.set_ylabel("Reward std (last 30% of training)")
ax.set_title("D. Policy stability (lower = better)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
Path("results").mkdir(exist_ok=True)
plt.savefig("results/fig_project_b_comparison.png", dpi=150)
plt.close(fig)
print("Saved: results/fig_project_b_comparison.png")

print("\n" + "=" * 65)
print(f"{'Architecture':<18} {'SolveSteps':>12} {'FinalReward':>12} {'Stability':>10}")
print("=" * 65)
for arch, (label, _) in ARCHS.items():
    if arch not in metrics:
        continue
    m = metrics[arch]
    ss = m["solve_steps"]
    ar = m["asym_reward"]
    st = m["stability"]
    print(
        f"{label:<18} {ss.mean() / 1000:>8.0f}k±{ss.std() / 1000:.0f}k  "
        f"{ar.mean():>8.1f}±{ar.std():.1f}  {st.mean():>8.1f}±{st.std():.1f}"
    )
print("=" * 65)
