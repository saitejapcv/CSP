
from cluster_scheduler.models import Node, Job
from cluster_scheduler.constraints import GPUConstraint, AllowedZonesConstraint, AntiAffinityConstraint, \
    ZoneAffinityConstraint, ZoneSpreadConstraint


def create_gpu_constraint():
    jobs = {
        "J1": Job("J1", "ml", 8, 16, True),
        "J2": Job("J2", "web", 4, 8, False)
    }

    nodes = {
        "N1": Node("N1", "A", 16, 32, True),
        "N2": Node("N2", "B", 16, 32, False)
    }

    return GPUConstraint("J1", jobs, nodes)


def test_gpu_constraint_valid():
    constraint = create_gpu_constraint()

    assignment = {"J1": "N1"}

    assert constraint.is_satisfied(assignment) is True


def test_gpu_constraint_invalid():
    constraint = create_gpu_constraint()

    assignment = {"J1": "N2"}

    assert constraint.is_satisfied(assignment) is False


def test_gpu_constraint_unassigned():
    constraint = create_gpu_constraint()

    assignment = {}

    assert constraint.is_satisfied(assignment) is True


def test_gpu_not_required():
    jobs = {
        "J2": Job("J2", "web", 4, 8, False)
    }

    nodes = {
        "N2": Node("N2", "B", 16, 32, False)
    }

    constraint = GPUConstraint("J2", jobs, nodes)

    assignment = {"J2": "N2"}

    assert constraint.is_satisfied(assignment) is True


def create_zone_constraint(allowed_zones):
    jobs = {
        "J1": Job(
            "J1", "web", 4, 8,
            allowed_zones=allowed_zones
        )
    }

    nodes = {
        "N1": Node("N1", "A", 16, 32),
        "N2": Node("N2", "B", 16, 32),
        "N3": Node("N3", "C", 16, 32)
    }

    return AllowedZonesConstraint("J1", jobs, nodes)


def test_allowed_zone():
    constraint = create_zone_constraint(("A", "B"))

    assert constraint.is_satisfied({"J1": "N1"}) is True


def test_disallowed_zone():
    constraint = create_zone_constraint(("A", "B"))

    assert constraint.is_satisfied({"J1": "N3"}) is False


def test_no_zone_restriction():
    constraint = create_zone_constraint(None)

    assert constraint.is_satisfied({"J1": "N3"}) is True


def test_unassigned_job():
    constraint = create_zone_constraint(("A", "B"))

    assert constraint.is_satisfied({}) is True

def test_anti_affinity1():
    constraint = AntiAffinityConstraint("J1", "J2")

    assert constraint.is_satisfied({"J1": "N1", "J2": "N2"}) is True

def test_anti_affinity2():
    constraint = AntiAffinityConstraint("J1", "J2")

    assert constraint.is_satisfied({"J1": "N1", "J2": "N1"}) is False

def test_anti_affinity3():
    constraint = AntiAffinityConstraint("J1", "J2")

    assert constraint.is_satisfied({"J1": "N2"}) is True

def test_anti_affinity4():
    constraint = AntiAffinityConstraint("J1", "J2")

    assert constraint.is_satisfied({}) is True


def test_zone_affinity1():
    jobs = {
        "J1": Job("J1", "web", 4, 8, False, allowed_zones=("A", "B")),
        "J2": Job("J2", "web", 4, 8, False, allowed_zones=("A", "B")),
    }

    nodes = {
        "N1": Node("N1", "A", 16, 32),
        "N2": Node("N2", "A", 16, 32),
    }
    constraint = ZoneAffinityConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({}) is True

def test_zone_affinity2():
    jobs = {
        "J1": Job("J1", "web", 4, 8, False, allowed_zones=("A", "B")),
        "J2": Job("J2", "web", 4, 8, False, allowed_zones=("A", "B"))
    }

    nodes = {
        "N1": Node("N1", "A", 16, 32),
        "N2": Node("N2", "A", 16, 32),
    }
    constraint = ZoneAffinityConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({"J1": "N1", "J2": "N2"}) is True

def test_zone_affinity3():
    jobs = {
        "J1": Job("J1", "web", 4, 8, False, allowed_zones=("A", "B")),
        "J2": Job("J2", "web", 4, 8, False, allowed_zones=("A", "B"))
    }

    nodes = {
        "N1": Node("N1", "A", 16, 32),
        "N2": Node("N2", "B", 16, 32),
    }

    constraint = ZoneAffinityConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({"J1": "N1", "J2": "N2"}) is False

def test_zone_affinity4():
    jobs = {
        "J1": Job("J1", "web", 4, 8),
        "J2": Job("J2", "web", 4, 8)
    }

    nodes = {
        "N1": Node("N1", "A", 16, 32),
        "N2": Node("N2", "B", 16, 32)
    }

    constraint = ZoneAffinityConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({"J1": "N1"}) is True

def test_zone_spread1():
    jobs = {
        "J1": Job("J1", "web", 4, 8),
        "J2": Job("J2", "web", 4, 8)
    }

    nodes = {
        "N1": Node("N1", "A", 16, 32),
        "N2": Node("N2", "B", 16, 32)
    }

    constraint = ZoneSpreadConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({}) is True

def test_zone_spread2():
    jobs = {
        "J1": Job("J1", "web", 4, 8),
        "J2": Job("J2", "web", 4, 8)
    }

    nodes = {
        "N1": Node("N1", "A", 16, 32),
        "N2": Node("N2", "B", 16, 32)
    }

    constraint = ZoneSpreadConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({"J1": "N1"}) is True

def test_zone_spread3():
    jobs = {
        "J1": Job("J1", "web", 4, 8),
        "J2": Job("J2", "web", 4, 8)
    }

    nodes = {
        "N1": Node("N1", "A", 16, 32),
        "N2": Node("N2", "B", 16, 32)
    }

    constraint = ZoneSpreadConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({"J1": "N1", "J2": "N2"}) is True

def test_zone_spread4():
    jobs = {
        "J1": Job("J1", "web", 4, 8),
        "J2": Job("J2", "web", 4, 8)
    }

    nodes = {
        "N1": Node("N1", "A", 16, 32),
        "N2": Node("N2", "A", 16, 32)
    }

    constraint = ZoneSpreadConstraint("J1", "J2", jobs, nodes)

    assert constraint.is_satisfied({"J1": "N1", "J2": "N2"}) is False

