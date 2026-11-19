"""baselines.py (P3) - the two comparison points for the CSP solver.

first_fit(csp)      greedy, never undoes a choice (shows where greedy fails)
brute_force(...)    enumerates every assignment; the oracle used by the tests
"""

from itertools import product
from time import perf_counter

from cluster_scheduler.backtracking import Checker
from cluster_scheduler.verifier import verify


def first_fit(csp, domains=None):
    """Place jobs in order on the first node that fits and respects the rules.

    Never backtracks. Returns (assignment | None, stats). If some job has no node
    left the whole run fails: stats["status"] = "no_solution" and stats["failed_job"]
    names the job. NOTE: for first_fit that only means "greedy failed", which is
    not a proof that no valid placement exists.
    """
    start = perf_counter()
    checker = Checker(csp)
    if domains is None:
        domains = csp.domains

    order = {nid: i for i, nid in enumerate(checker.nodes)}
    failed = None

    for var in csp.variables:
        placed = False
        for node in sorted(domains[var], key=lambda n: order.get(n, len(order))):
            if checker.ok(var, node):
                checker.assign(var, node)
                placed = True
                break
        if not placed:
            failed = var
            break

    stats = {
        "nodes_expanded": len(checker.assignment),
        "backtracks": 0,
        "constraint_checks": checker.checks,
        "time": perf_counter() - start,
        "status": "solved" if failed is None else "no_solution",
        "failed_job": failed,
    }
    return (dict(checker.assignment) if failed is None else None), stats


def brute_force(cluster, jobs, specs=(), max_jobs=8):
    """Oracle for tiny instances: try every assignment, judge with verify().

    Uses only P1's independent verifier and the raw data, so it shares no code
    with the CSP or with Backtracking. Returns (assignment | None, stats).
    """
    if len(jobs) > max_jobs:
        raise ValueError(f"brute_force is limited to {max_jobs} jobs (got {len(jobs)})")

    start = perf_counter()
    node_ids = [n.id for n in cluster.nodes]
    job_ids = [j.id for j in jobs]
    checked = 0
    found = None

    for combo in product(node_ids, repeat=len(job_ids)):
        checked += 1
        assignment = dict(zip(job_ids, combo))
        if verify(cluster, jobs, assignment, specs)[0]:
            found = assignment
            break

    stats = {
        "assignments_checked": checked,
        "time": perf_counter() - start,
        "status": "solved" if found else "no_solution",
    }
    return found, stats
