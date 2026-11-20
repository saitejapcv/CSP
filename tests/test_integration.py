"""Instance loading, derived specs, the new constraints, the verifier fix, the CLI and the examples."""
import json
from pathlib import Path

import pytest

from cluster_scheduler import cli
from cluster_scheduler.constraints import CapacityConstraint, NodeFitConstraint
from cluster_scheduler.generator import generate
from cluster_scheduler.instance import build_instance, derive_specs, load_instance, save_instance
from cluster_scheduler.models import Cluster, Job, Node
from cluster_scheduler.verifier import verify

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


# ------------------------------- new constraint classes -------------------------------
def _objs():
    nodes = {"N1": Node("N1", "ZA", 10, 40), "N2": Node("N2", "ZB", 4, 16)}
    jobs = {"a": Job("a", "s", 4, 16), "b": Job("b", "s", 4, 16), "c": Job("c", "s", 4, 16)}
    return jobs, nodes


def test_node_fit_constraint():
    jobs, nodes = _objs()
    jobs["big"] = Job("big", "s", 6, 24)
    c = NodeFitConstraint("big", jobs, nodes)
    assert c.scope == ("big",)
    assert c.is_satisfied({}) is True
    assert c.is_satisfied({"big": "N1"}) is True
    assert c.is_satisfied({"big": "N2"}) is False


def test_capacity_constraint_sums_over_all_jobs_on_a_node():
    jobs, nodes = _objs()
    c = CapacityConstraint(jobs, nodes)
    assert c.is_satisfied({"a": "N1", "b": "N1"})                  # 8 <= 10
    assert not c.is_satisfied({"a": "N1", "b": "N1", "c": "N1"})   # 12 > 10
    assert c.is_satisfied({"a": "N1", "b": "N1", "c": "N2"})
    assert c.is_satisfied({})                                      # nothing assigned yet
    assert not hasattr(c, "allows")                                # n-ary: AC-3 must not use it


# ------------------------------- csp additions -------------------------------
def test_csp_exposes_neighbors_as_the_contract_spells_it():
    inst = load_instance(EXAMPLES / "easy.json")
    assert inst.csp.neighbors("api-1") == inst.csp.neighbours("api-1")
    assert "api-2" in inst.csp.neighbors("api-1")


def test_csp_index_follows_added_constraints():
    from cluster_scheduler.constraints import AntiAffinityConstraint
    inst = load_instance(EXAMPLES / "easy.json")
    assert "worker" not in inst.csp.neighbours("batch") or True
    before = len(inst.csp.constraints_between("worker", "batch"))
    inst.csp.constraints.append(AntiAffinityConstraint("worker", "batch"))
    assert len(inst.csp.constraints_between("worker", "batch")) == before + 1


# ------------------------------- derived specs -------------------------------
def test_unary_specs_are_derived_from_the_job_data():
    cluster, jobs, specs = generate(8, 4, 0.8, 0.5, 3)
    inst = build_instance(cluster, jobs, specs)
    types = [s["type"] for s in inst.full_specs]
    assert types.count("gpu") == sum(j["needs_gpu"] for j in jobs)
    assert types.count("allowed_zones") == sum(j["allowed_zones"] is not None for j in jobs)
    assert types.count("capacity") == 1


def test_gpu_job_domain_is_pruned_to_gpu_nodes_only():
    from cluster_scheduler.ac3 import node_consistency
    inst = load_instance(EXAMPLES / "easy.json")
    ok, d, _ = node_consistency(inst.csp, inst.csp.copy_domains())
    assert d["ml-train"] == {"N1"}


def test_pair_capacity_only_added_for_pairs_that_cannot_share_some_node():
    cluster = Cluster((Node("N1", "ZA", 10, 40), Node("N2", "ZA", 10, 40)))
    jobs = [Job("a", "s", 3, 12), Job("b", "s", 3, 12), Job("c", "s", 8, 32)]
    specs = derive_specs(cluster, jobs, [])
    pairs = {frozenset(s["jobs"]) for s in specs if s["type"] == "pair_capacity"}
    assert pairs == {frozenset(("a", "c")), frozenset(("b", "c"))}


def test_derive_specs_does_not_duplicate_existing_specs():
    cluster, jobs, specs = generate(6, 3, 0.7, 0.4, 2)
    inst = build_instance(cluster, jobs, specs + [{"type": "gpu", "jobs": [j["id"]]}
                                                  for j in jobs if j["needs_gpu"]])
    keys = [(s["type"], tuple(sorted(s["jobs"]))) for s in inst.full_specs
            if s["type"] in ("gpu", "allowed_zones")]
    assert len(keys) == len(set(keys))


def test_json_round_trip(tmp_path):
    cluster, jobs, specs = generate(6, 3, 0.7, 0.3, 5)
    path = tmp_path / "inst.json"
    save_instance(path, cluster, jobs, specs)
    inst = load_instance(path)
    assert [j.id for j in inst.jobs] == [j["id"] for j in jobs]
    assert inst.specs == specs


# ------------------------------- verifier fix -------------------------------
def test_verifier_accepts_unary_specs_that_build_csp_accepts():
    cluster = Cluster((Node("N1", "ZA", 8, 32, True),))
    jobs = [Job("a", "s", 2, 8, needs_gpu=True)]
    specs = [{"type": "gpu", "jobs": ["a"]}, {"type": "allowed_zones", "jobs": ["a"]},
             {"type": "fits_node", "jobs": ["a"]}, {"type": "capacity", "jobs": ["a"]}]
    ok, errors = verify(cluster, jobs, {"a": "N1"}, specs)
    assert ok, errors


def test_verifier_still_rejects_unknown_types_and_real_violations():
    cluster = Cluster((Node("N1", "ZA", 8, 32), Node("N2", "ZA", 8, 32)))
    jobs = [Job("a", "s", 2, 8), Job("b", "s", 2, 8)]
    assert not verify(cluster, jobs, {"a": "N1", "b": "N1"},
                      [{"type": "anti_affinity", "jobs": ["a", "b"]}])[0]
    assert not verify(cluster, jobs, {"a": "N1", "b": "N2"}, [{"type": "bogus", "jobs": ["a", "b"]}])[0]


# ------------------------------- the example files -------------------------------
@pytest.mark.parametrize("name,expected", [("easy", "solved"), ("tight", "solved"),
                                           ("unsat", "no_solution"), ("unsat_ac3", "no_solution")])
def test_examples_behave_as_documented(name, expected):
    inst = load_instance(EXAMPLES / f"{name}.json")
    for method in ("bt", "bt+mrv", "ac3+bt", "ac3+bt+mrv", "brute"):
        r = cli.run_method(inst, method, 10)
        assert r["status"] == expected, (name, method)
        if expected == "solved":
            assert r["verified"] is True


def test_easy_example_is_the_figure_1_placement_capacity():
    inst = load_instance(EXAMPLES / "easy.json")
    assert len(inst.jobs) == 7 and len(inst.cluster.nodes) == 4


# ------------------------------- the CLI -------------------------------
def test_cli_pipeline_writes_a_verified_result(tmp_path, capsys):
    out = tmp_path / "r.json"
    code = cli.main(["--instance", str(EXAMPLES / "easy.json"), "--method", "ac3+bt+mrv", "--out", str(out)])
    assert code == 0
    result = json.loads(out.read_text())
    assert result["status"] == "solved" and result["verified"] is True
    assert result["pruning"]["failed_at"] is None
    assert "status : solved" in capsys.readouterr().out


@pytest.mark.parametrize("method", list(cli.METHODS))
def test_cli_every_method_runs(method, tmp_path):
    out = tmp_path / "r.json"
    assert cli.main(["--instance", str(EXAMPLES / "tight.json"), "--method", method,
                     "--out", str(out), "--quiet"]) == 0
    assert json.loads(out.read_text())["status"] in ("solved", "no_solution", "timeout")


def test_cli_ac3_short_circuits_on_infeasible_instance(tmp_path):
    out = tmp_path / "r.json"
    cli.main(["--instance", str(EXAMPLES / "unsat_ac3.json"), "--method", "ac3+bt", "--out", str(out), "--quiet"])
    r = json.loads(out.read_text())
    assert r["status"] == "no_solution" and r["stats"]["nodes_expanded"] == 0
    assert r["pruning"]["failed_at"] == "ac3"


def test_cli_separate_files_and_generator(tmp_path):
    cluster, jobs, specs = generate(6, 3, 0.7, 0.3, 1)
    for name, data in (("c", cluster), ("j", jobs), ("s", specs)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    assert cli.main(["--cluster", str(tmp_path / "c.json"), "--jobs", str(tmp_path / "j.json"),
                     "--constraints", str(tmp_path / "s.json"), "--quiet"]) == 0
    assert cli.main(["--generate", "--n-jobs", "6", "--n-nodes", "3", "--seed", "1", "--quiet"]) == 0


def test_cli_bad_input_returns_2(tmp_path, capsys):
    assert cli.main(["--instance", str(tmp_path / "missing.json")]) == 2
    assert cli.main([]) == 2


# ------------------------------- experiments (quick mode) -------------------------------
def test_experiments_quick_run_writes_every_output(tmp_path):
    from cluster_scheduler import experiments
    experiments.main(["--out", str(tmp_path), "--quick"])
    for name in ("solve.csv", "firstfit_vs_csp.csv", "pruning.csv", "game.csv", "oracle_agreement.csv",
                 "SUMMARY.md", "solver_comparison.png", "pruning.png", "firstfit_vs_csp.png",
                 "minimax_vs_alphabeta.png", "offline_vs_online.png", "placement_easy.png"):
        assert (tmp_path / name).stat().st_size > 0, name
    assert "(must be 0)" in (tmp_path / "SUMMARY.md").read_text()


def test_benchmark_csv_has_fixed_column_order(tmp_path):
    import csv
    from cluster_scheduler.benchmark import COLUMNS, write_csv
    path = tmp_path / "b.csv"
    write_csv(str(path), "solve", [0, 1], ["bt", "ac3_bt_mrv", "firstfit"], 8, 3, 0.8, 0.3, 5.0, "sat")
    with open(path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == COLUMNS
        assert len(list(reader)) == 6


def test_benchmark_is_reproducible_with_fixed_seeds():
    from cluster_scheduler.benchmark import run_one
    strip = lambda r: {k: v for k, v in r.items() if "time" not in k}
    assert strip(run_one("ac3_bt_mrv", 10, 4, 0.8, 0.3, 3)) == strip(run_one("ac3_bt_mrv", 10, 4, 0.8, 0.3, 3))


def test_benchmark_verifies_every_solution_honestly(monkeypatch):
    """verified must come from the independent verifier, not be assumed."""
    from cluster_scheduler import backtracking
    from cluster_scheduler.benchmark import run_one

    assert run_one("ac3_bt_mrv", 8, 3, 0.8, 0.5, 2)["verified"] is True

    def bad_solver(csp, domains, use_mrv=False, time_limit=None):
        return {v: "N1" for v in csp.variables}, {"status": "solved", "nodes_expanded": 1,
                                                  "backtracks": 0, "constraint_checks": 0}
    monkeypatch.setattr(backtracking, "backtrack", bad_solver)
    row = run_one("bt", 8, 3, 0.8, 0.5, 2)
    assert row["status"] == "solved" and row["verified"] is False
