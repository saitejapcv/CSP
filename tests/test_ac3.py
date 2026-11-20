"""tests/test_ac3.py  (P2)

Self-contained tests: uses a tiny FakeCSP so you can run them BEFORE P1 delivers csp.py.
Once P1's real CSP exists, add one test that runs ac3 on P1's examples/*.json.

Run:  pytest tests/test_ac3.py -q      (from the repo root, with ac3.py importable)
"""

import copy
from itertools import product

import pytest

from cluster_scheduler.ac3 import node_consistency, revise, ac3, prune

ZONE = {"N1": "A", "N2": "A", "N3": "B", "N4": "B"}
NODES = list(ZONE)


# --------------------------- fake constraints / CSP ---------------------------
class Unary:
    def __init__(self, var, ok):
        self.scope = (var,)
        self._ok = ok

    def is_satisfied(self, assignment):
        v = assignment.get(self.scope[0])
        return True if v is None else self._ok(v)


class Binary:
    def __init__(self, x, y, pred):
        self.scope = (x, y)
        self._pred = pred  # pred(node_x, node_y) -> bool

    def allows(self, x, nx, y, ny):
        if (x, y) == self.scope:
            return self._pred(nx, ny)
        if (y, x) == self.scope:
            return self._pred(ny, nx)
        return True

    def is_satisfied(self, assignment):
        a, b = assignment.get(self.scope[0]), assignment.get(self.scope[1])
        return True if a is None or b is None else self._pred(a, b)


class FakeCSP:
    def __init__(self, variables, constraints):
        self.variables = list(variables)
        self.constraints = constraints
        self.domains = {v: set(NODES) for v in variables}

    def copy_domains(self):
        return {v: set(d) for v, d in self.domains.items()}

    def neighbors(self, x):
        out = set()
        for c in self.constraints:
            if len(c.scope) == 2 and x in c.scope:
                out |= set(c.scope) - {x}
        return out

    def constraints_between(self, x, y):
        return [c for c in self.constraints if set(c.scope) == {x, y} and len(c.scope) == 2]


def anti_affinity(x, y):
    return Binary(x, y, lambda a, b: a != b)


def zone_affinity(x, y):
    return Binary(x, y, lambda a, b: ZONE[a] == ZONE[b])


def zone_different(x, y):
    return Binary(x, y, lambda a, b: ZONE[a] != ZONE[b])


def figure3_csp():
    cons = [
        Unary("ml-train", lambda n: n == "N1"),          # C1 needs GPU (only N1)
        Unary("api-2", lambda n: ZONE[n] == "B"),        # C2 zone B only
        zone_affinity("cache", "ml-train"),              # C3
        zone_different("api-1", "api-2"),                # C4
    ]
    return FakeCSP(["ml-train", "cache", "api-1", "api-2"], cons)


# --------------------------------- tests --------------------------------------
def test_figure3_node_consistency():
    csp = figure3_csp()
    ok, d, stats = node_consistency(csp, csp.copy_domains())
    assert ok
    assert d["ml-train"] == {"N1"}
    assert d["api-2"] == {"N3", "N4"}
    assert d["cache"] == set(NODES) and d["api-1"] == set(NODES)
    assert stats["values_pruned"] == 3 + 2


def test_figure3_ac3_matches_paper():
    csp = figure3_csp()
    ok, d, stats = prune(csp)
    assert ok
    assert d == {
        "ml-train": {"N1"},
        "cache": {"N1", "N2"},
        "api-1": {"N1", "N2"},
        "api-2": {"N3", "N4"},
    }
    size = 1
    for v in d.values():
        size *= len(v)
    assert size == 8                      # 256 -> 8 candidates, as in Figure 3
    assert stats["failed_at"] is None


def test_revise_hand_cases():
    csp = FakeCSP(["a", "b"], [anti_affinity("a", "b")])
    # case 1: D(b)={N1}  ->  N1 must go from D(a)
    d = {"a": {"N1", "N2"}, "b": {"N1"}}
    assert revise(csp, d, "a", "b") is True and d["a"] == {"N2"}
    # case 2: D(b) has two values -> every value in D(a) still has a partner
    d = {"a": {"N1", "N2"}, "b": {"N1", "N2"}}
    assert revise(csp, d, "a", "b") is False and d["a"] == {"N1", "N2"}
    # case 3: zone affinity, D(b) only zone B -> D(a) loses zone A nodes
    csp = FakeCSP(["a", "b"], [zone_affinity("a", "b")])
    d = {"a": set(NODES), "b": {"N3"}}
    assert revise(csp, d, "a", "b") is True and d["a"] == {"N3", "N4"}


def test_revise_is_directional():
    csp = FakeCSP(["a", "b"], [anti_affinity("a", "b")])
    d = {"a": {"N1"}, "b": {"N1", "N2"}}
    assert revise(csp, d, "b", "a") is True and d["b"] == {"N2"}
    assert d["a"] == {"N1"}               # D(a) untouched


def test_detects_infeasible_wipeout():
    # three jobs pairwise anti-affine but only 2 nodes allowed each
    cons = [anti_affinity("a", "b"), anti_affinity("b", "c"), anti_affinity("a", "c")]
    csp = FakeCSP(["a", "b", "c"], cons)
    csp.domains = {v: {"N1", "N2"} for v in "abc"}
    # AC-3 is only pairwise: this is arc consistent (documented limitation!)
    ok, _, _ = ac3(csp, csp.copy_domains())
    assert ok is True
    # but forcing a={N1}, b={N1} is a wipe-out
    d = {"a": {"N1"}, "b": {"N1"}, "c": {"N1", "N2"}}
    ok, d2, _ = ac3(csp, d)
    assert ok is False and any(not v for v in d2.values())


def test_node_consistency_empty_domain():
    csp = FakeCSP(["a"], [Unary("a", lambda n: False)])
    ok, d, stats = node_consistency(csp, csp.copy_domains())
    assert ok is False and d["a"] == set()


def test_prune_reports_where_it_failed():
    csp = FakeCSP(["a"], [Unary("a", lambda n: False)])
    ok, _, s = prune(csp)
    assert not ok and s["failed_at"] == "node_consistency"


def test_inputs_never_mutated():
    csp = figure3_csp()
    d0 = csp.copy_domains()
    snapshot = copy.deepcopy(d0)
    node_consistency(csp, d0)
    ac3(csp, d0)
    prune(csp, d0)
    assert d0 == snapshot


def test_domains_never_grow():
    csp = figure3_csp()
    d0 = csp.copy_domains()
    ok, d1, _ = ac3(csp, d0)
    for v in d0:
        assert d1[v] <= d0[v]


def test_idempotent():
    csp = figure3_csp()
    _, d1, _ = prune(csp)
    ok, d2, stats = ac3(csp, d1)
    assert ok and d2 == d1 and stats["values_pruned"] == 0 and stats["revisions"] == 0


def test_stats_keys_and_counts():
    csp = figure3_csp()
    _, _, stats = prune(csp)
    for k in ("arcs_processed", "revisions", "values_pruned", "time"):
        assert k in stats
    assert stats["arcs_processed"] >= 2 * 2   # at least the initial arcs
    assert stats["revisions"] >= 1
    assert stats["time"] >= 0


def test_custom_queue_starts_only_from_given_arcs():
    csp = FakeCSP(["a", "b"], [anti_affinity("a", "b")])
    d = {"a": {"N1", "N2"}, "b": {"N1"}}
    ok, out, stats = ac3(csp, d, queue=[("a", "b")])
    assert ok and out["a"] == {"N2"} and stats["arcs_processed"] >= 1


def test_ac3_never_removes_a_solution_value():
    """Soundness: any value used in some full solution must survive AC-3."""
    csp = figure3_csp()
    ok, pruned, _ = prune(csp)
    solutions = []
    for combo in product(NODES, repeat=len(csp.variables)):
        asg = dict(zip(csp.variables, combo))
        if all(c.is_satisfied(asg) for c in csp.constraints):
            solutions.append(asg)
    assert solutions
    for asg in solutions:
        for v, n in asg.items():
            assert n in pruned[v]


# ------------------------- with the REAL CSP (P1) and the example files -------------------------
from pathlib import Path

from cluster_scheduler.instance import load_instance

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_figure1_instance_matches_figure3_pruning():
    inst = load_instance(EXAMPLES / "easy.json")
    ok, d, stats = prune(inst.csp)
    assert ok
    assert d["ml-train"] == {"N1"}                     # C1: needs the GPU node
    assert d["api-2"] == {"N3", "N4"}                  # C2: zone B only
    assert d["cache"] == {"N1", "N2"}                  # C3: same zone as ml-train
    assert d["api-1"] == {"N1", "N2"}                  # C4: other zone than api-2
    assert d["db"] == {"N1", "N2"}                     # sits with api-1


def test_ac3_alone_proves_unsat_ac3_example():
    inst = load_instance(EXAMPLES / "unsat_ac3.json")
    ok, d, stats = prune(inst.csp)
    assert not ok and stats["failed_at"] == "ac3"      # node consistency alone is not enough
    ok_nc, _, _ = node_consistency(inst.csp, inst.csp.copy_domains())
    assert ok_nc


def test_ac3_cannot_prove_pigeonhole_unsat():
    """AC-3 is only pairwise: it must NOT claim infeasibility here (search does)."""
    inst = load_instance(EXAMPLES / "unsat.json")
    ok, d, stats = prune(inst.csp)
    assert ok


def test_real_csp_inputs_not_mutated():
    inst = load_instance(EXAMPLES / "easy.json")
    before = inst.csp.copy_domains()
    ac3(inst.csp, before)
    assert before == inst.csp.copy_domains()


def test_arcs_processed_within_complexity_bound():
    inst = load_instance(EXAMPLES / "easy.json")
    ok, d, stats = prune(inst.csp)
    e = sum(len(inst.csp.neighbours(v)) for v in inst.csp.variables)    # directed arcs
    dmax = max(len(v) for v in inst.csp.domains.values())
    assert stats["arcs_processed"] <= e * (dmax + 1)     # each arc re-enters at most d times


# ------------------------- tests added after mutation checks -------------------------
def test_pruning_propagates_through_a_chain_of_arcs():
    """x - y - z with z restricted to zone B: z prunes y, and y's change must be
    pushed back onto the arc (x, y), otherwise x keeps its zone-A values."""
    cons = [zone_affinity("x", "y"), zone_affinity("y", "z"),
            Unary("z", lambda n: ZONE[n] == "B")]
    csp = FakeCSP(["x", "y", "z"], cons)
    ok, d, stats = prune(csp)
    assert ok
    assert d == {"x": {"N3", "N4"}, "y": {"N3", "N4"}, "z": {"N3", "N4"}}
    assert stats["revisions"] >= 2


def test_ac3_and_node_consistency_do_not_mutate_input_even_when_they_prune():
    csp = FakeCSP(["a", "b"], [anti_affinity("a", "b"), Unary("b", lambda n: n == "N1")])
    d0 = {"a": {"N1", "N2"}, "b": {"N1", "N2"}}
    snapshot = copy.deepcopy(d0)
    ok, d1, _ = node_consistency(csp, d0)
    assert d0 == snapshot and d1["b"] == {"N1"}
    ok, d2, _ = ac3(csp, d1)
    assert d1 == {"a": {"N1", "N2"}, "b": {"N1"}}      # input of ac3 untouched ...
    assert d2["a"] == {"N2"}                           # ... while the result IS pruned
    assert d0 == snapshot
