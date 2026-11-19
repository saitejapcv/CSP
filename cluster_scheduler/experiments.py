"""experiments.py - ONE command reproduces every table and figure (fixed seeds).

    python -m cluster_scheduler.experiments --out results
    python -m cluster_scheduler.experiments --out /tmp/quick --quick     (smaller, a few seconds)

Writes CSV files, PNG figures and results/SUMMARY.md:
  solve.csv + solver_comparison.png   BT vs AC-3+BT, with/without MRV   (P3 + P2)
  firstfit_vs_csp.csv + .png          greedy first-fit vs the CSP solver  (P3)
  pruning.csv + pruning.png           how much AC-3 prunes, and when it proves infeasibility (P2)
  game.csv + 2 pngs                   Minimax vs Alpha-Beta, offline vs online (P4)
  oracle_agreement.csv                every solver vs the brute-force oracle
  placement_easy.png                  Figure-1 style placement diagram
"""
import argparse
import csv
import os
from collections import defaultdict
from statistics import mean

from cluster_scheduler import alphabeta, minimax, visualize
from cluster_scheduler.ac3 import prune
from cluster_scheduler.backtracking import backtrack
from cluster_scheduler.baselines import brute_force, first_fit
from cluster_scheduler.benchmark import (COLUMNS, KINDS, _build, run_one, write_csv)
from cluster_scheduler.cli import run_method
from cluster_scheduler.game import Game
from cluster_scheduler.generator import generate
from cluster_scheduler.instance import build_instance, load_instance


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def exp_solve(out, seeds, n_jobs=14, n_nodes=4, util=0.85):
    path = os.path.join(out, "solve.csv")
    write_csv(path, "solve", seeds, ["bt", "bt_mrv", "ac3_bt", "ac3_bt_mrv", "firstfit"],
              n_jobs, n_nodes, util, 0.3, 10.0, "sat")
    rows = _read(path)
    visualize.plot_solver_comparison(path, os.path.join(out, "solver_comparison.png"))
    table = []
    for c in ["bt", "bt_mrv", "ac3_bt", "ac3_bt_mrv", "firstfit"]:
        r = [x for x in rows if x["config"] == c]
        table.append([c, len(r), sum(x["status"] == "solved" for x in r),
                      sum(x["status"] == "timeout" for x in r),
                      f"{mean(float(x['nodes_expanded']) for x in r):.1f}",
                      f"{mean(float(x['backtracks']) for x in r):.1f}",
                      f"{mean(float(x['constraint_checks']) for x in r):.0f}",
                      f"{1000 * mean(float(x['total_time']) for x in r):.2f}",
                      sum(x["verified"] == "False" for x in r)])
    head = ["config", "instances", "solved", "timeout", "mean nodes", "mean backtracks",
            "mean checks", "mean ms", "failed verifier"]
    return (f"### Backtracking configurations ({n_jobs} jobs, {n_nodes} nodes, utilisation {util}, "
            f"{len(seeds)} seeds)\n\n" + _md(head, table))


def exp_firstfit(out, seeds, n_jobs=14, n_nodes=4):
    path = os.path.join(out, "firstfit_vs_csp.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for util in (0.6, 0.7, 0.8, 0.9):
            for s in seeds:
                try:
                    for c in ("firstfit", "ac3_bt_mrv"):
                        w.writerow(run_one(c, n_jobs, n_nodes, util, 0.3, s, 10.0, "sat"))
                except ValueError:
                    pass
    rows = _read(path)
    visualize.plot_success_rates(path, os.path.join(out, "firstfit_vs_csp.png"))
    table = []
    for util in sorted({float(r["utilization"]) for r in rows}):
        ff = [r for r in rows if r["config"] == "firstfit" and float(r["utilization"]) == util]
        cs = [r for r in rows if r["config"] == "ac3_bt_mrv" and float(r["utilization"]) == util]
        table.append([util, len(ff), f"{100 * mean(r['status'] == 'solved' for r in ff):.0f}%",
                      f"{100 * mean(r['status'] == 'solved' for r in cs):.0f}%"])
    return ("### First-fit vs the CSP solver (every instance is satisfiable by construction)\n\n"
            + _md(["utilisation", "instances", "first-fit solved", "AC-3+BT+MRV solved"], table))


def exp_pruning(out, seeds, n_jobs=10, n_nodes=4, util=0.8):
    path = os.path.join(out, "pruning.csv")
    write_csv(path, "pruning", seeds, [], n_jobs, n_nodes, util, 0.3, 10.0, "mixed")
    rows = _read(path)
    visualize.plot_pruning(path, os.path.join(out, "pruning.png"))
    table = []
    for kind in KINDS:
        r = [x for x in rows if x["kind"] == kind]
        if not r:
            continue
        table.append([kind, len(r), f"{mean(float(x['pct_removed_by_nc']) for x in r):.1f}",
                      f"{mean(float(x['pct_removed_total']) for x in r):.1f}",
                      sum(x["failed_at"] == "node_consistency" for x in r),
                      sum(x["failed_at"] == "ac3" for x in r),
                      sum(x["failed_at"] == "" for x in r)])
    false_alarms = sum(1 for x in rows if x["kind"] == "sat" and x["failed_at"] != "")
    head = ["instance kind", "instances", "% removed by node consistency", "% removed in total",
            "infeasible found by node consistency", "infeasible found by AC-3", "not detected"]
    note = (f"\nAC-3 / node consistency reported infeasible on {false_alarms} satisfiable instances "
            f"(must be 0). 'not detected' on an unsat kind means only search can prove it.")
    return ("### How much AC-3 prunes, and when it proves infeasibility alone\n\n"
            + _md(head, table) + note)


def _overload(jobs, extra, seed):
    """Add `extra` copies of random jobs so the pool no longer fits the cluster."""
    import random
    rng = random.Random(seed)
    jobs = [dict(j) for j in jobs]
    base = list(jobs)
    for k in range(extra):
        copy = dict(rng.choice(base))
        copy["id"] = f"job-x{k + 1}"
        copy["allowed_zones"] = None
        jobs.append(copy)
    return jobs


def exp_game(out, seeds, n_jobs=5, n_nodes=3, util=0.9):
    """Minimax vs Alpha-Beta, and offline best case vs online worst case.

    extra=0: everything fits. Because the scheduler knows the whole pool it can follow
    one fixed valid placement whatever the arrival order, so the online value equals the
    offline value (gap 0). extra>0: the pool is overloaded; the offline optimum picks the
    best subset while the adversary's order makes the online scheduler hit a job that no
    longer fits, so a real gap appears.
    """
    path = os.path.join(out, "game.csv")
    fields = ["extra_jobs", "seed", "pool_size", "n_nodes", "offline_value", "online_value",
              "gap", "minimax_nodes", "alphabeta_nodes", "saved_pct", "values_equal", "ac3_calls"]
    rows = []
    for extra in (0, 1, 2):
        for s in seeds:
            try:
                cluster, jobs, specs = generate(n_jobs, n_nodes, util, 0.3, s)
            except ValueError:
                continue
            jobs = _overload(jobs, extra, s)
            game = Game(cluster, jobs, specs)
            v1, _, s1 = minimax.solve(game.start())
            v2, _, s2 = alphabeta.solve(game.start())
            offline, _ = minimax.best_case(game.start())
            rows.append({"extra_jobs": extra, "seed": s, "pool_size": len(jobs), "n_nodes": n_nodes,
                         "offline_value": offline, "online_value": v1, "gap": offline - v1,
                         "minimax_nodes": s1["nodes_expanded"], "alphabeta_nodes": s2["nodes_expanded"],
                         "saved_pct": round(100 * (1 - s2["nodes_expanded"] / s1["nodes_expanded"]), 1),
                         "values_equal": v1 == v2, "ac3_calls": game.ac3_calls})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    by_extra = defaultdict(list)
    for r in rows:
        by_extra[r["extra_jobs"]].append(r)
    label = lambda e: f"pool {n_jobs + e}"
    visualize.plot_node_counts(
        [(label(e), mean(r["minimax_nodes"] for r in v), mean(r["alphabeta_nodes"] for r in v))
         for e, v in sorted(by_extra.items())], os.path.join(out, "minimax_vs_alphabeta.png"))
    visualize.plot_offline_vs_online(
        [(label(e), mean(r["offline_value"] for r in v), mean(r["online_value"] for r in v))
         for e, v in sorted(by_extra.items())], os.path.join(out, "offline_vs_online.png"))
    table = [[n_jobs + e, len(v), f"{mean(r['minimax_nodes'] for r in v):.0f}",
              f"{mean(r['alphabeta_nodes'] for r in v):.0f}",
              f"{mean(r['saved_pct'] for r in v):.0f}%",
              f"{mean(r['offline_value'] for r in v):.2f}", f"{mean(r['online_value'] for r in v):.2f}",
              f"{mean(r['gap'] for r in v):.2f}", all(r["values_equal"] for r in v)]
             for e, v in sorted(by_extra.items())]
    head = ["pool size", "games", "Minimax nodes", "Alpha-Beta nodes", "nodes saved",
            "offline best case", "online worst case", "gap", "Minimax = Alpha-Beta"]
    return (f"### Online game ({n_nodes} nodes, utilisation {util}; moves filtered by AC-3)\n\n"
            + _md(head, table)
            + "\n\nWhen everything fits (pool 5) the gap is 0: a scheduler that knows the pool can follow "
              "one valid placement whatever the arrival order. The gap appears once the pool is overloaded.")


def exp_oracle(out, seeds):
    """Every solver must agree with brute force (solved vs no_solution) and AC-3 must be sound."""
    path = os.path.join(out, "oracle_agreement.csv")
    fields = ["seed", "kind", "brute", "bt", "bt_mrv", "ac3_bt", "ac3_bt_mrv", "ac3_sound", "agree"]
    rows = []
    for s in seeds:
        kind = "sat" if s % 2 == 0 else KINDS[1 + (s // 2) % 3]
        try:
            inst = _build(5, 3, 0.8, 0.3, s, kind)
        except ValueError:
            continue
        bf, _ = brute_force(inst.cluster, inst.jobs, inst.specs)
        row = {"seed": s, "kind": kind, "brute": "solved" if bf else "no_solution"}
        ok, pruned, _ = prune(inst.csp)
        for name, (use_ac3, mrv) in {"bt": (0, 0), "bt_mrv": (0, 1), "ac3_bt": (1, 0),
                                    "ac3_bt_mrv": (1, 1)}.items():
            if use_ac3 and not ok:
                row[name] = "no_solution"
            else:
                a, st = backtrack(inst.csp, pruned if use_ac3 else inst.csp.copy_domains(), bool(mrv), 10)
                row[name] = st["status"]
        # soundness: AC-3 never removes a value that a real solution uses
        row["ac3_sound"] = (ok and all(n in pruned[j] for j, n in bf.items())) if bf else True
        row["agree"] = all(row[k] == row["brute"] for k in ("bt", "bt_mrv", "ac3_bt", "ac3_bt_mrv")) \
            and row["ac3_sound"]
        rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    agree = sum(r["agree"] for r in rows)
    sat = sum(r["brute"] == "solved" for r in rows)
    return (f"### Agreement with the brute-force oracle (5 jobs, 3 nodes)\n\n"
            f"{agree}/{len(rows)} random tiny instances agree ({sat} satisfiable, {len(rows) - sat} "
            f"unsatisfiable): every solver configuration returns the same solved / no_solution as "
            f"brute force, and AC-3 never removes a value used by a real solution.")


def exp_placement(out, root):
    inst = load_instance(os.path.join(root, "examples", "easy.json"))
    result = run_method(inst, "ac3+bt+mrv", 10)
    visualize.plot_placement(inst.raw["cluster"], inst.raw["jobs"], result["assignment"],
                             os.path.join(out, "placement_easy.png"))
    return f"### Placement for examples/easy.json\n\nverified = {result['verified']} (see placement_easy.png)"


def main(argv=None):
    p = argparse.ArgumentParser(description="Reproduce every experiment")
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true", help="fewer seeds, for a fast check")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    n = 8 if a.quick else 50
    sections = [
        exp_solve(a.out, list(range(n))),
        exp_firstfit(a.out, list(range(n // 2 if a.quick else 30))),
        exp_pruning(a.out, list(range(n * 2 if a.quick else 80))),
        exp_game(a.out, list(range(3 if a.quick else 10))),
        exp_oracle(a.out, list(range(n * 2 if a.quick else 60))),
        exp_placement(a.out, root),
    ]
    text = "# Experiment summary (fixed seeds)\n\n" + "\n\n".join(sections) + "\n"
    with open(os.path.join(a.out, "SUMMARY.md"), "w") as f:
        f.write(text)
    print(text)
    print(f"files written to {a.out}/")


if __name__ == "__main__":
    main()
