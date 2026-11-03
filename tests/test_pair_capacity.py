
import pytest

from cluster_scheduler.models import Node, Job
from cluster_scheduler.constraints import PairCapacityConstraint


@pytest.fixture
def setup_capacity():
    jobs = {
        "J1": Job("J1", "web", 4, 8),
        "J2": Job("J2", "web", 4, 8),
        "J3": Job("J3", "ml", 8, 20),
    }

    nodes = {
        "N1": Node("N1", "A", 10, 20),
        "N2": Node("N2", "B", 10, 20),
    }

    return jobs, nodes


def test_jobs_on_different_nodes(setup_capacity):
    jobs, nodes = setup_capacity
    constraint = PairCapacityConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({"J1": "N1", "J2": "N2"}) is True


def test_jobs_fit_on_same_node(setup_capacity):
    jobs, nodes = setup_capacity
    constraint = PairCapacityConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({"J1": "N1", "J2": "N1"}) is True


def test_cpu_exceeds_capacity(setup_capacity):
    jobs, nodes = setup_capacity
    constraint = PairCapacityConstraint("J1", "J3", jobs, nodes)

    assert constraint.is_satisfied({"J1": "N1", "J3": "N1"}) is False


def test_ram_exceeds_capacity(setup_capacity):
    jobs, nodes = setup_capacity

    # Each job fits individually, but their combined RAM exceeds N1.
    jobs["J2"] = Job("J2", "web", 2, 16)

    constraint = PairCapacityConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({"J1": "N1", "J2": "N1"}) is False


def test_first_job_unassigned(setup_capacity):
    jobs, nodes = setup_capacity
    constraint = PairCapacityConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({"J2": "N1"}) is True


def test_second_job_unassigned(setup_capacity):
    jobs, nodes = setup_capacity
    constraint = PairCapacityConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({"J1": "N1"}) is True


def test_both_jobs_unassigned(setup_capacity):
    jobs, nodes = setup_capacity
    constraint = PairCapacityConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({}) is True


def test_allows_directly(setup_capacity):
    jobs, nodes = setup_capacity
    constraint = PairCapacityConstraint("J1", "J2", jobs, nodes)

    assert constraint.allows("J1", "N1", "J2", "N1") is True
    assert constraint.allows("J1", "N1", "J2", "N2") is True
    assert constraint.allows("J1", "N1", "J3", "N1") is False
