"""cli.py (P3) - run the whole pipeline from JSON and verify the answer.

    python -m cluster_scheduler.cli --instance examples/easy.json --method ac3+bt+mrv
    python -m cluster_scheduler.cli --cluster c.json --jobs j.json [--constraints s.json]
    python -m cluster_scheduler.cli --generate --n-jobs 8 --n-nodes 4 --seed 1 --method firstfit

Methods: bt | bt+mrv | ac3+bt | ac3+bt+mrv | firstfit | brute
Pipeline: load JSON -> build CSP -> (node consistency + AC-3) -> search -> verify -> write JSON.

Exit code: 0 = finished (solved / no_solution / timeout),
           1 = a returned placement FAILED verification (a bug),
           2 = bad input.
"""

import argparse
import json
import sys

from cluster_scheduler.ac3 import prune
from cluster_scheduler.backtracking import backtrack
from cluster_scheduler.baselines import brute_force, first_fit
from cluster_scheduler.generator import generate
from cluster_scheduler.instance import build_instance, load_instance, load_parts
from cluster_scheduler.verifier import verify

METHODS = {
    "bt":         {"ac3": False, "mrv": False},
    "bt+mrv":     {"ac3": False, "mrv": True},
    "ac3+bt":     {"ac3": True,  "mrv": False},
    "ac3+bt+mrv": {"ac3": True,  "mrv": True},
    "firstfit":   {},
    "brute":      {},
}


def run_method(inst, method, time_limit=None):
    """Run one method on an Instance. Returns a JSON-serialisable result dict."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {', '.join(METHODS)}")

    pruning = None
    assignment = None

    if method == "firstfit":
        assignment, stats = first_fit(inst.csp)
    elif method == "brute":
        assignment, stats = brute_force(inst.cluster, inst.jobs, inst.specs)
    else:
        cfg = METHODS[method]
        domains = inst.csp.copy_domains()
        if cfg["ac3"]:
            ok, domains, pruning = prune(inst.csp, domains)
            if not ok:   # AC-3 alone proved there is no placement
                stats = {"nodes_expanded": 0, "backtracks": 0, "constraint_checks": 0,
                         "time": 0.0, "status": "no_solution"}
                return _result(inst, method, None, stats, pruning)
        assignment, stats = backtrack(inst.csp, domains, cfg["mrv"], time_limit)

    return _result(inst, method, assignment, stats, pruning)


def _result(inst, method, assignment, stats, pruning):
    result = {"method": method, "status": stats["status"],
              "assignment": assignment, "stats": stats}
    if pruning is not None:
        result["pruning"] = pruning
    if assignment is not None:
        ok, violations = verify(inst.cluster, inst.jobs, assignment, inst.specs)
        result["verified"] = ok
        result["violations"] = violations
    return result


def _parse(argv):
    p = argparse.ArgumentParser(prog="python -m cluster_scheduler.cli",
                                description="Cloud Cluster Scheduler (CSP solver)")
    src = p.add_argument_group("input (choose one)")
    src.add_argument("--instance", help="one JSON file with cluster, jobs and constraints")
    src.add_argument("--cluster", help="cluster JSON (zones + nodes)")
    src.add_argument("--jobs", help="jobs JSON (list)")
    src.add_argument("--constraints", help="constraint specs JSON (list), optional")
    src.add_argument("--generate", action="store_true", help="use the seeded generator")
    gen = p.add_argument_group("generator options (with --generate)")
    gen.add_argument("--n-jobs", type=int, default=8)
    gen.add_argument("--n-nodes", type=int, default=4)
    gen.add_argument("--utilisation", type=float, default=0.8)
    gen.add_argument("--density", type=float, default=0.3)
    gen.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="ac3+bt+mrv", choices=list(METHODS))
    p.add_argument("--time-limit", type=float, default=10.0, help="seconds (search only)")
    p.add_argument("--out", help="write the result JSON here")
    p.add_argument("--quiet", action="store_true", help="do not print the placement")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse(argv)
    try:
        if a.instance:
            inst = load_instance(a.instance)
        elif a.cluster and a.jobs:
            inst = load_parts(a.cluster, a.jobs, a.constraints)
        elif a.generate:
            cluster, jobs, specs = generate(a.n_jobs, a.n_nodes, a.utilisation, a.density, a.seed)
            inst = build_instance(cluster, jobs, specs)
        else:
            print("error: give --instance, or --cluster and --jobs, or --generate", file=sys.stderr)
            return 2
        result = run_method(inst, a.method, a.time_limit)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=2)

    if not a.quiet:
        print(f"method : {result['method']}")
        print(f"status : {result['status']}")
        s = result["stats"]
        print("stats  : " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in s.items() if k != "status"))
        if "pruning" in result:
            pr = result["pruning"]
            print(f"pruning: arcs={pr['arcs_processed']} revisions={pr['revisions']} "
                  f"values_pruned={pr['values_pruned']} failed_at={pr['failed_at']}")
        if result["assignment"]:
            for job, node in result["assignment"].items():
                print(f"  {job:>8} -> {node}")
            print(f"verified: {result['verified']}")
            for v in result["violations"]:
                print("  VIOLATION:", v)

    if result.get("verified") is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
