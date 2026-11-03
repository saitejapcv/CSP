
import pytest

from cluster_scheduler.models import Node, Job, Cluster


# -------------------------
# Node tests
# -------------------------

def test_node_creation():
    node = Node("N1", "A", 16, 32, True)

    assert node.id == "N1"
    assert node.zone == "A"
    assert node.cpu == 16
    assert node.ram == 32
    assert node.gpu is True


def test_node_rejects_invalid_cpu():
    with pytest.raises(ValueError):
        Node("N1", "A", 0, 32)


def test_node_rejects_invalid_ram():
    with pytest.raises(ValueError):
        Node("N1", "A", 16, 0)


def test_node_rejects_empty_id():
    with pytest.raises(ValueError):
        Node("", "A", 16, 32)


# -------------------------
# Job tests
# -------------------------

def test_job_creation():
    job = Job("ml-train", "ml", 8, 16, True)

    assert job.id == "ml-train"
    assert job.service == "ml"
    assert job.cpu == 8
    assert job.ram == 16
    assert job.needs_gpu is True


def test_job_rejects_invalid_cpu():
    with pytest.raises(ValueError):
        Job("J1", "ml", 0, 16)


def test_job_rejects_invalid_ram():
    with pytest.raises(ValueError):
        Job("J1", "ml", 8, 0)


def test_job_rejects_empty_id():
    with pytest.raises(ValueError):
        Job("", "ml", 8, 16)


# -------------------------
# Cluster tests
# -------------------------

def test_cluster_creation():
    n1 = Node("N1", "A", 16, 32)
    n2 = Node("N2", "B", 8, 16)

    cluster = Cluster((n1, n2))

    assert len(cluster.nodes) == 2
    assert cluster.nodes[0] == n1
    assert cluster.nodes[1] == n2
