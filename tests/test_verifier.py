
from cluster_scheduler.models import Node, Job, Cluster
from cluster_scheduler.verifier import verify


# Shared test data
def create_cluster():
    nodes = (
        Node("N1", "east", 8, 16, gpu=True),
        Node("N2", "west", 8, 16, gpu=False),
        Node("N3", "east", 4, 8, gpu=False),
    )
    return Cluster(nodes=nodes)


def create_jobs():
    return [
        Job("J1", "training", 4, 8, needs_gpu=True),
        Job("J2", "database", 4, 6, allowed_zones=("east", "west")),
        Job("J3", "backend", 2, 4, allowed_zones=("west",)),
    ]


def test_valid_assignment():
    cluster = create_cluster()
    jobs = create_jobs()

    assignment = {
        "J1": "N1",
        "J2": "N3",
        "J3": "N2",
    }

    specs = [
        {"type": "anti_affinity", "jobs": ["J1", "J2"]},
        {"type": "zone_spread", "jobs": ["J2", "J3"]},
    ]

    valid, errors = verify(cluster, jobs, assignment, specs)

    assert valid is True
    assert errors == []


def test_gpu_violation():
    cluster = create_cluster()
    jobs = create_jobs()

    assignment = {
        "J1": "N2",
        "J2": "N3",
        "J3": "N2",
    }

    valid, errors = verify(cluster, jobs, assignment)

    assert valid is False
    assert any("GPU" in error for error in errors)


def test_allowed_zone_violation():
    cluster = create_cluster()
    jobs = create_jobs()

    assignment = {
        "J1": "N1",
        "J2": "N3",
        "J3": "N1",
    }

    valid, errors = verify(cluster, jobs, assignment)

    assert valid is False
    assert any("zone" in error for error in errors)


def test_anti_affinity_violation():
    cluster = create_cluster()
    jobs = create_jobs()

    assignment = {
        "J1": "N1",
        "J2": "N1",
        "J3": "N2",
    }

    specs = [
        {"type": "anti_affinity", "jobs": ["J1", "J2"]}
    ]

    valid, errors = verify(cluster, jobs, assignment, specs)

    assert valid is False
    assert any("different nodes" in error for error in errors)


def test_zone_affinity_violation():
    cluster = create_cluster()
    jobs = create_jobs()

    assignment = {
        "J1": "N1",
        "J2": "N2",
        "J3": "N2",
    }

    specs = [
        {"type": "zone_affinity", "jobs": ["J1", "J2"]}
    ]

    valid, errors = verify(cluster, jobs, assignment, specs)

    assert valid is False
    assert any("same zone" in error for error in errors)


def test_zone_spread_violation():
    cluster = create_cluster()
    jobs = create_jobs()

    assignment = {
        "J1": "N1",
        "J2": "N3",
        "J3": "N3",
    }

    specs = [
        {"type": "zone_spread", "jobs": ["J2", "J3"]}
    ]

    valid, errors = verify(cluster, jobs, assignment, specs)

    assert valid is False
    assert any("different zones" in error for error in errors)


def test_cpu_capacity_violation():
    cluster = create_cluster()
    jobs = [
        Job("J1", "service1", 6, 4),
        Job("J2", "service2", 6, 4),
    ]

    assignment = {
        "J1": "N1",
        "J2": "N1",
    }

    valid, errors = verify(cluster, jobs, assignment)

    assert valid is False
    assert any("CPU capacity exceeded" in error for error in errors)


def test_ram_capacity_violation():
    cluster = create_cluster()
    jobs = [
        Job("J1", "service1", 2, 12),
        Job("J2", "service2", 2, 8),
    ]

    assignment = {
        "J1": "N1",
        "J2": "N1",
    }

    valid, errors = verify(cluster, jobs, assignment)

    assert valid is False
    assert any("RAM capacity exceeded" in error for error in errors)


def test_missing_job_assignment():
    cluster = create_cluster()
    jobs = create_jobs()

    assignment = {
        "J1": "N1",
        "J2": "N3",
    }

    valid, errors = verify(cluster, jobs, assignment)

    assert valid is False
    assert any("has not been assigned" in error for error in errors)


def test_unknown_node_assignment():
    cluster = create_cluster()
    jobs = create_jobs()

    assignment = {
        "J1": "N99",
        "J2": "N3",
        "J3": "N2",
    }

    valid, errors = verify(cluster, jobs, assignment)

    assert valid is False
    assert any("unknown node" in error for error in errors)
