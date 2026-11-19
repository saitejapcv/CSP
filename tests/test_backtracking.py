"""Tests for backtracking.py / baselines.py: solutions verified, agreement with the
brute-force oracle, undo correctness, exact capacity, timeout handling."""
import pytest

from cluster_scheduler.ac3 import prune
from cluster_scheduler.backtracking import _Search, backtrack
from cluster_scheduler.baselines import brute_force, first_fit
from cluster_scheduler.benchmark import _build
from cluster_scheduler.instance import build_instance, load_instance
from cluster_scheduler.verifier import verify
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _node(i, zone, cpu, gpu=False):
    return {"id": i, "zone": zone, "cpu": cpu, "ram": cpu * 4, "gpu": gpu}


def _job(i, cpu, gpu=False, zones=None):
    return {"id": i, "service": "s", "cpu": cpu, "ram": cpu * 4, "needs_gpu": gpu, "allowed_zones": zones}


CONFIGS = [(False, False), (False, True), (True, False), (True, True)]


def solve(inst, use_ac3, use_mrv, time_limit=10):
    domains = inst.csp.copy_domains()
    if use_ac3:
        ok, domains, _ = prune(inst.csp, domains)
        if not ok:
            return None, {"status": "no_solution"}
    return backtrack(inst.csp, domains, use_mrv, time_limit)


@pytest.mark.parametrize("use_ac3,use_mrv", CONFIGS)
@pytest.mark.parametrize("name", ["easy", "tight"])
def test_every_configuration_solves_satisfiable_examples_and_passes_verifier(name, use_ac3, use_mrv):
    inst = load_instance(EXAMPLES / f"{name}.json")
    assignment, stats = solve(inst, use_ac3, use_mrv)
    assert stats["status"] == "solved"
    ok, errors = verify(inst.cluster, inst.jobs, assignment, inst.specs)
    assert ok, errors


@pytest.mark.parametrize("use_ac3,use_mrv", CONFIGS)
@pytest.mark.parametrize("name", ["unsat", "unsat_ac3"])
def test_every_configuration_reports_no_solution_on_unsat_examples(name, use_ac3, use_mrv):
    inst = load_instance(EXAMPLES / f"{name}.json")
    assignment, stats = solve(inst, use_ac3, use_mrv)
    assert assignment is None and stats["status"] == "no_solution"


@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_oracle_on_random_tiny_instances(seed):
    kind = "sat" if seed % 2 == 0 else ["unsat_pigeonhole", "unsat_gpu", "unsat_zone_conflict"][(seed // 2) % 3]
    inst = _build(5, 3, 0.8, 0.3, seed, kind)
    oracle, _ = brute_force(inst.cluster, inst.jobs, inst.specs)
    for use_ac3, use_mrv in CONFIGS:
        assignment, stats = solve(inst, use_ac3, use_mrv)
        assert stats["status"] == ("solved" if oracle else "no_solution")
        if assignment:
            assert verify(inst.cluster, inst.jobs, assignment, inst.specs)[0]


@pytest.mark.parametrize("seed", range(15))
def test_ac3_never_removes_a_value_used_by_a_real_solution(seed):
    inst = _build(5, 3, 0.8, 0.3, seed, "sat")
    oracle, _ = brute_force(inst.cluster, inst.jobs, inst.specs)
    ok, pruned, _ = prune(inst.csp)
    assert ok
    assert all(node in pruned[job] for job, node in oracle.items())


def test_domains_passed_in_are_never_modified():
    inst = load_instance(EXAMPLES / "unsat.json")
    domains = inst.csp.copy_domains()
    snapshot = {k: set(v) for k, v in domains.items()}
    backtrack(inst.csp, domains, use_mrv=True)
    assert domains == snapshot


@pytest.mark.parametrize("use_mrv", [False, True])
def test_undo_restores_every_running_total(use_mrv):
    """After a complete failed search all assignments are undone and totals are back to 0."""
    inst = load_instance(EXAMPLES / "unsat.json")
    s = _Search(inst.csp, inst.csp.copy_domains(), use_mrv, None)
    assert s._recurse() is False
    assert s.assignment == {}
    assert all(v == 0 for v in s.used_cpu.values()) and all(v == 0 for v in s.used_ram.values())
    assert s.stats["backtracks"] > 0


def test_assign_then_unassign_is_exact():
    inst = load_instance(EXAMPLES / "easy.json")
    s = _Search(inst.csp, inst.csp.copy_domains(), False, None)
    s.assign("db", "N2")
    assert s.used_cpu["N2"] == 8
    s.unassign("db")
    assert s.used_cpu["N2"] == 0 and s.used_ram["N2"] == 0 and "db" not in s.assignment


def test_exact_capacity_is_enforced_even_though_pairs_fit():
    """Three 4-CPU jobs on one 10-CPU node: every PAIR fits (8 <= 10) so AC-3 stays quiet,
    but the triple does not (12 > 10). Only the exact running-total check catches it."""
    inst = build_instance({"zones": ["ZA"], "nodes": [_node("N1", "ZA", 10)]},
                          [_job("a", 4), _job("b", 4), _job("c", 4)], [])
    ok, pruned, _ = prune(inst.csp)
    assert ok                                    # pairwise decomposition cannot see it
    assignment, stats = backtrack(inst.csp, pruned)
    assert assignment is None and stats["status"] == "no_solution"
    # and the n-ary constraint agrees
    assert not inst.csp.is_consistent("c", "N1", {"a": "N1", "b": "N1"})


def test_mrv_picks_the_most_constrained_job_first_with_degree_tiebreak():
    inst = build_instance({"zones": ["ZA", "ZB"], "nodes": [_node("N1", "ZA", 16, True), _node("N2", "ZB", 16),
                                                             _node("N3", "ZB", 16)]},
                          [_job("free", 2), _job("gpu", 2, gpu=True), _job("other", 2)],
                          [{"type": "anti_affinity", "jobs": ["other", "free"]}])
    s = _Search(inst.csp, inst.csp.copy_domains(), True, None)
    var, legal = s._select()
    assert var == "gpu" and legal == ["N1"]       # one legal node beats three
    s2 = _Search(inst.csp, inst.csp.copy_domains(), True, None)
    s2.assign("gpu", "N1")
    var, _ = s2._select()
    assert var in ("free", "other")               # tie on legal values -> degree (both have 1 neighbour)


def test_mrv_never_expands_more_than_plain_bt_on_these_seeds():
    total_plain = total_mrv = 0
    for seed in range(15):
        inst = _build(12, 4, 0.85, 0.3, seed, "sat")
        total_plain += solve(inst, False, False)[1]["nodes_expanded"]
        total_mrv += solve(inst, False, True)[1]["nodes_expanded"]
    assert total_mrv < total_plain


def test_timeout_is_reported_as_timeout_never_as_no_solution():
    inst = load_instance(EXAMPLES / "easy.json")
    assignment, stats = backtrack(inst.csp, inst.csp.copy_domains(), True, time_limit=0.0)
    assert assignment is None
    assert stats["status"] == "timeout"


def test_stats_have_the_agreed_keys_and_statuses():
    inst = load_instance(EXAMPLES / "easy.json")
    _, stats = backtrack(inst.csp, inst.csp.copy_domains())
    for key in ("nodes_expanded", "backtracks", "constraint_checks", "time", "status"):
        assert key in stats
    assert stats["status"] in ("solved", "no_solution", "timeout")


def test_ac3_then_backtracking_needs_fewer_nodes_than_plain_backtracking():
    plain = pruned = 0
    for seed in range(15):
        inst = _build(12, 4, 0.85, 0.3, seed, "sat")
        plain += solve(inst, False, False)[1]["nodes_expanded"]
        pruned += solve(inst, True, False)[1]["nodes_expanded"]
    assert pruned < plain


# ------------------------------------ baselines ------------------------------------
def test_first_fit_fails_on_the_tight_example_but_the_csp_solver_succeeds():
    inst = load_instance(EXAMPLES / "tight.json")
    ff, st = first_fit(inst.csp)
    assert ff is None and st["status"] == "no_solution" and st["failed_job"] == "c"
    assignment, bt = solve(inst, True, True)
    assert bt["status"] == "solved"


def test_first_fit_never_returns_an_invalid_placement():
    for seed in range(30):
        inst = _build(10, 4, 0.8, 0.3, seed, "sat")
        ff, _ = first_fit(inst.csp)
        if ff is not None:
            assert verify(inst.cluster, inst.jobs, ff, inst.specs)[0]


def test_brute_force_refuses_large_instances():
    inst = _build(10, 4, 0.8, 0.3, 0, "sat")
    with pytest.raises(ValueError):
        brute_force(inst.cluster, inst.jobs, inst.specs)
