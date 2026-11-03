
from abc import ABC, abstractmethod


class Constraint(ABC):
    """Base class for all CSP constraints."""

    def __init__(self, scope: tuple[str, ...]):
        self.scope = scope

    @abstractmethod
    def is_satisfied(self, assignment: dict[str, str]) -> bool:
        """Check whether the constraint is satisfied by an assignment."""
        pass


# --------------------------------------------------
# 1. GPU Constraint (Unary)
# --------------------------------------------------

class GPUConstraint(Constraint):
    def __init__(self, job_id, jobs, nodes):
        super().__init__((job_id,))
        self.jobs = jobs
        self.nodes = nodes

    def is_satisfied(self, assignment: dict[str, str]) -> bool:
        job_id = self.scope[0]
        node_id = assignment.get(job_id)

        if node_id is None:
            return True

        job = self.jobs[job_id]
        node = self.nodes[node_id]

        if job.needs_gpu and not node.gpu:
            return False

        return True


# --------------------------------------------------
# 2. Allowed Zones Constraint (Unary)
# --------------------------------------------------

class AllowedZonesConstraint(Constraint):
    def __init__(self, job_id, jobs, nodes):
        super().__init__((job_id,))
        self.jobs = jobs
        self.nodes = nodes

    def is_satisfied(self, assignment: dict[str, str]) -> bool:
        job_id = self.scope[0]
        node_id = assignment.get(job_id)

        if node_id is None:
            return True

        job = self.jobs[job_id]
        node = self.nodes[node_id]

        if job.allowed_zones is None:
            return True

        return node.zone in job.allowed_zones


# --------------------------------------------------
# 3. Anti-Affinity Constraint (Binary)
# --------------------------------------------------

class AntiAffinityConstraint(Constraint):
    def __init__(self, job1_id, job2_id):
        super().__init__((job1_id, job2_id))

    def allows(self, x, node_x, y, node_y):
        """Jobs must be assigned to different nodes."""
        return node_x != node_y

    def is_satisfied(self, assignment: dict[str, str]) -> bool:
        job1_id, job2_id = self.scope

        node1_id = assignment.get(job1_id)
        node2_id = assignment.get(job2_id)

        if node1_id is None or node2_id is None:
            return True

        return self.allows(job1_id, node1_id, job2_id, node2_id)


# --------------------------------------------------
# 4. Zone Affinity Constraint (Binary)
# --------------------------------------------------

class ZoneAffinityConstraint(Constraint):
    def __init__(self, job1_id, job2_id, jobs, nodes):
        super().__init__((job1_id, job2_id))
        self.jobs = jobs
        self.nodes = nodes

    def allows(self, x, node_x, y, node_y):
        """Jobs must be assigned to nodes in the same zone."""
        zone_x = self.nodes[node_x].zone
        zone_y = self.nodes[node_y].zone

        return zone_x == zone_y

    def is_satisfied(self, assignment: dict[str, str]) -> bool:
        job1_id, job2_id = self.scope

        node1_id = assignment.get(job1_id)
        node2_id = assignment.get(job2_id)

        if node1_id is None or node2_id is None:
            return True

        return self.allows(job1_id, node1_id, job2_id, node2_id)


# --------------------------------------------------
# 5. Zone Spread Constraint (Binary)
# --------------------------------------------------

class ZoneSpreadConstraint(Constraint):
    def __init__(self, job1_id, job2_id, jobs, nodes):
        super().__init__((job1_id, job2_id))
        self.jobs = jobs
        self.nodes = nodes

    def allows(self, x, node_x, y, node_y):
        """Jobs must be assigned to nodes in different zones."""
        zone_x = self.nodes[node_x].zone
        zone_y = self.nodes[node_y].zone

        return zone_x != zone_y

    def is_satisfied(self, assignment: dict[str, str]) -> bool:
        job1_id, job2_id = self.scope

        node1_id = assignment.get(job1_id)
        node2_id = assignment.get(job2_id)

        if node1_id is None or node2_id is None:
            return True

        return self.allows(job1_id, node1_id, job2_id, node2_id)


# --------------------------------------------------
# 6. Pair Capacity Constraint (Binary)
# --------------------------------------------------

class PairCapacityConstraint(Constraint):
    def __init__(self, job1_id, job2_id, jobs, nodes):
        super().__init__((job1_id, job2_id))
        self.jobs = jobs
        self.nodes = nodes

    def allows(self, x, node_x, y, node_y):
        """
        Check whether two jobs can fit together on a node.
        Jobs on different nodes are automatically compatible.
        """

        if node_x != node_y:
            return True

        job1 = self.jobs[x]
        job2 = self.jobs[y]
        node = self.nodes[node_x]

        total_cpu = job1.cpu + job2.cpu
        total_ram = job1.ram + job2.ram

        return total_cpu <= node.cpu and total_ram <= node.ram

    def is_satisfied(self, assignment: dict[str, str]) -> bool:
        job1_id, job2_id = self.scope

        node1_id = assignment.get(job1_id)
        node2_id = assignment.get(job2_id)

        if node1_id is None or node2_id is None:
            return True

        return self.allows(job1_id, node1_id, job2_id, node2_id)
