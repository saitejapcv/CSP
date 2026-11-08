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
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
 
 
def plot_node_counts(rows, path="minimax_vs_alphabeta.png"):
    _plot_pairs(rows, ("Minimax", "Alpha-Beta"), "nodes expanded", path)
 
 
def plot_offline_vs_online(rows, path="offline_vs_online.png"):
    _plot_pairs(rows, ("Offline (CSP)", "Online worst case"), "jobs placed", path)
 
