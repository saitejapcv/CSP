import matplotlib.pyplot as plt
 
 
def plot_placement(cluster, jobs, assignment, path="placement.png"):
    nodes = sorted(cluster["nodes"], key=lambda n: (n["zone"], n["id"]))
    cpu = {j["id"]: j["cpu"] for j in jobs}
    fig, ax = plt.subplots(figsize=(1.6 * len(nodes) + 1, 5))
    for x, n in enumerate(nodes):
        ax.bar(x, n["cpu"], fill=False, edgecolor="black", linewidth=2)
        bottom = 0
        for job, nid in assignment.items():
            if nid == n["id"]:
                ax.bar(x, cpu[job], bottom=bottom, edgecolor="white")
                ax.text(x, bottom + cpu[job] / 2, f"{job} ({cpu[job]})", ha="center", va="center", fontsize=8)
                bottom += cpu[job]
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels([f"{n['id']}\n{n['zone']}{' GPU' if n['gpu'] else ''}" for n in nodes])
    ax.set_ylabel("CPU")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
 
 
def _plot_pairs(rows, names, ylabel, path):
    x = list(range(len(rows)))
    fig, ax = plt.subplots()
    ax.bar([i - 0.2 for i in x], [r[1] for r in rows], 0.4, label=names[0])
    ax.bar([i + 0.2 for i in x], [r[2] for r in rows], 0.4, label=names[1])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)   # head-room so the legend never hides a bar
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
 
 
def plot_node_counts(rows, path="minimax_vs_alphabeta.png"):
    _plot_pairs(rows, ("Minimax", "Alpha-Beta"), "nodes expanded", path)
 
 
def plot_offline_vs_online(rows, path="offline_vs_online.png"):
    _plot_pairs(rows, ("Offline (CSP)", "Online worst case"), "jobs placed", path)
 


# ----------------------------------------------------------------------------
# Plots that read the benchmark CSVs (added during integration)
# ----------------------------------------------------------------------------
import csv
from collections import defaultdict
from statistics import mean

_COLORS = {"bt": "#7a7a7a", "bt_mrv": "#d98b1f", "ac3_bt": "#2a9d6f", "ac3_bt_mrv": "#1f5fbf",
           "firstfit": "#c0392b"}


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_solver_comparison(csv_path, path="solver_comparison.png"):
    """Mean nodes expanded and mean runtime for each Backtracking configuration."""
    rows = [r for r in _read_csv(csv_path) if r["config"] != "firstfit"]
    configs = [c for c in _COLORS if any(r["config"] == c for r in rows)]
    nodes = [mean(float(r["nodes_expanded"]) for r in rows if r["config"] == c) for c in configs]
    secs = [1000 * mean(float(r["total_time"]) for r in rows if r["config"] == c) for c in configs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, label in ((axes[0], nodes, "mean nodes expanded"),
                            (axes[1], secs, "mean runtime (ms)")):
        bars = ax.bar(configs, vals, color=[_COLORS[c] for c in configs])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Backtracking configurations (same seeded instances)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pruning(csv_path, path="pruning.png"):
    """Percent of domain values removed by node consistency and by AC-3, per instance kind."""
    by_kind = defaultdict(list)
    for r in _read_csv(csv_path):
        by_kind[r["kind"]].append(r)
    kinds = list(by_kind)
    nc = [mean(float(r["pct_removed_by_nc"]) for r in by_kind[k]) for k in kinds]
    total = [mean(float(r["pct_removed_total"]) for r in by_kind[k]) for k in kinds]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(kinds, nc, label="node consistency", color="#2a9d6f")
    ax.bar(kinds, [t - n for t, n in zip(total, nc)], bottom=nc, label="AC-3 (extra)", color="#1f5fbf")
    ax.set_ylabel("% of domain values removed")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_success_rates(csv_path, path="firstfit_vs_csp.png"):
    """Success rate of first-fit and of AC-3 + Backtracking as capacity gets tighter."""
    groups = defaultdict(list)
    for r in _read_csv(csv_path):
        groups[(r["config"], float(r["utilization"]))].append(r["status"] == "solved")
    fig, ax = plt.subplots(figsize=(6, 4))
    for config, label in (("firstfit", "first-fit (greedy)"), ("ac3_bt_mrv", "AC-3 + Backtracking + MRV")):
        pts = sorted((u, 100 * mean(v)) for (c, u), v in groups.items() if c == config)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label,
                color=_COLORS[config])
    ax.set_xlabel("utilisation (job CPU / node CPU)")
    ax.set_ylabel("instances solved (%)")
    ax.set_ylim(-5, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
