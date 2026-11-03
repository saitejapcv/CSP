from cluster_scheduler.csp import CSP, build_csp
from cluster_scheduler.constraints import (
    AntiAffinityConstraint,
    ZoneAffinityConstraint,
    GPUConstraint,
)
import pytest

from cluster_scheduler.models import Node, Job, Cluster

def test_csp_initialization():
    variables = ["J1", "J2"]
    domains = {
        "J1": {"N1", "N2"},
        "J2": {"N1", "N2"},
    }

    constraints = []

    csp = CSP(variables, domains, constraints)

    assert csp.variables == variables
    assert csp.domains == domains




def test_neighbours():
    constraints = [
        AntiAffinityConstraint("J1", "J2"),
        AntiAffinityConstraint("J2", "J3"),
    ]

    csp = CSP(
        variables=["J1", "J2", "J3"],
        domains={},
        constraints=constraints,
    )

    assert csp.neighbours("J1") == {"J2"}
    assert csp.neighbours("J2") == {"J1", "J3"}
    assert csp.neighbours("J3") == {"J2"}


def test_neighbours_without_constraints():
    csp = CSP(
        variables=["J1", "J2"],
        domains={},
        constraints=[],
    )

    assert csp.neighbours("J1") == set()


def test_constraints_between():
    constraints = [
        AntiAffinityConstraint("J1", "J2"),
        AntiAffinityConstraint("J2", "J3"),
    ]

    csp = CSP(
        variables=["J1", "J2", "J3"],
        domains={},
        constraints=constraints,
    )

    result = csp.constraints_between("J1", "J2")

    assert len(result) == 1
    assert isinstance(result[0], AntiAffinityConstraint)


def test_constraints_between_reverse_order():
    constraint = AntiAffinityConstraint("J1", "J2")

    csp = CSP(
        variables=["J1", "J2"],
        domains={},
        constraints=[constraint],
    )

    assert csp.constraints_between("J2", "J1") == [constraint]


def test_constraints_between_no_match():
    constraint = AntiAffinityConstraint("J1", "J2")

    csp = CSP(
        variables=["J1", "J2", "J3"],
        domains={},
        constraints=[constraint],
    )

    assert csp.constraints_between("J1", "J3") == []


def test_is_consistent_valid_assignment():
    constraint = AntiAffinityConstraint("J1", "J2")

    csp = CSP(
        variables=["J1", "J2"],
        domains={},
        constraints=[constraint],
    )

    assignment = {"J1": "N1"}

    assert csp.is_consistent("J2", "N2", assignment) is True


def test_is_consistent_invalid_assignment():
    constraint = AntiAffinityConstraint("J1", "J2")

    csp = CSP(
        variables=["J1", "J2"],
        domains={},
        constraints=[constraint],
    )

    assignment = {"J1": "N1"}

    assert csp.is_consistent("J2", "N1", assignment) is False


def test_is_consistent_does_not_modify_assignment():
    constraint = AntiAffinityConstraint("J1", "J2")

    csp = CSP(
        variables=["J1", "J2"],
        domains={},
        constraints=[constraint],
    )

    assignment = {"J1": "N1"}

    csp.is_consistent("J2", "N2", assignment)

    assert assignment == {"J1": "N1"}


def test_copy_domains():
    domains = {
        "J1": {"N1", "N2"},
        "J2": {"N2", "N3"},
    }

    csp = CSP(
        variables=["J1", "J2"],
        domains=domains,
        constraints=[],
    )

    copied = csp.copy_domains()

    assert copied == domains
    assert copied is not domains
    assert copied["J1"] is not domains["J1"]
    assert copied["J2"] is not domains["J2"]


def test_copy_domains_independence():
    domains = {
        "J1": {"N1", "N2"},
        "J2": {"N2", "N3"},
    }

    csp = CSP(["J1", "J2"], domains, [])

    copied = csp.copy_domains()
    copied["J1"].remove("N1")

    assert "N1" in domains["J1"]

def test_build_csp_data():
    cluster = Cluster(
        nodes=(
            Node("N1", "A", 16, 32),
            Node("N2", "B", 16, 32),
        )
    )

    jobs = [
        Job("J1", "web", 4, 8),
        Job("J2", "web", 4, 8),
    ]

    constraint_specs = [
        {
            "type": "anti_affinity",
            "jobs": ["J1", "J2"],
        }
    ]

    assert len(cluster.nodes) == 2
    assert len(jobs) == 2
    assert constraint_specs[0]["type"] == "anti_affinity"

def test_build_csp_variables_and_domains():
    cluster = Cluster(
        nodes=(
            Node("N1", "A", 16, 32),
            Node("N2", "B", 16, 32),
        )
    )

    jobs = [
        Job("J1", "web", 4, 8),
        Job("J2", "ml", 8, 16),
    ]

    csp = build_csp(cluster, jobs, [])

    assert csp.variables == ["J1", "J2"]

    assert csp.domains == {
        "J1": {"N1", "N2"},
        "J2": {"N1", "N2"},
    }

    # Each job must have an independent domain set.
    assert csp.domains["J1"] is not csp.domains["J2"]


def test_build_csp_creates_anti_affinity():
    cluster = Cluster(
        nodes=(
            Node("N1", "A", 16, 32),
            Node("N2", "B", 16, 32),
        )
    )

    jobs = [
        Job("J1", "web", 4, 8),
        Job("J2", "web", 4, 8),
    ]

    specs = [
        {"type": "anti_affinity", "jobs": ["J1", "J2"]}
    ]

    csp = build_csp(cluster, jobs, specs)

    assert len(csp.constraints) == 1
    assert isinstance(csp.constraints[0], AntiAffinityConstraint)


def test_build_csp_creates_gpu_constraint():
    cluster = Cluster(
        nodes=(Node("N1", "A", 16, 32, gpu=True),)
    )

    jobs = [Job("J1", "ml", 4, 8, needs_gpu=True)]

    specs = [{"type": "gpu", "jobs": ["J1"]}]

    csp = build_csp(cluster, jobs, specs)

    assert isinstance(csp.constraints[0], GPUConstraint)


def test_build_csp_unknown_constraint():
    cluster = Cluster(
        nodes=(Node("N1", "A", 16, 32),)
    )

    jobs = [Job("J1", "web", 4, 8)]

    specs = [{"type": "unknown", "jobs": ["J1"]}]

    with pytest.raises(ValueError):
        build_csp(cluster, jobs, specs)
