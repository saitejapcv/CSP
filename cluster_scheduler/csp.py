from cluster_scheduler.models import Cluster, Job
from cluster_scheduler.constraints import (
    GPUConstraint,
    AllowedZonesConstraint,
    AntiAffinityConstraint,
    ZoneAffinityConstraint,
    ZoneSpreadConstraint,
    PairCapacityConstraint,
    NodeFitConstraint,
    CapacityConstraint,
)


class CSP:
    def __init__(self, variables, domains, constraints, jobs=None, nodes=None):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        # Raw data, needed by the solver for the exact capacity check.
        self.jobs = jobs if jobs is not None else {}
        self.nodes = nodes if nodes is not None else {}
        self._index_size = -1
        self._neighbours = {}
        self._between = {}

    # AC-3 calls neighbours() and constraints_between() very often, so the
    # binary-constraint index is built once (and rebuilt if constraints change).
    def _build_index(self):
        if self._index_size == len(self.constraints):
            return

        neighbours = {}
        between = {}

        for constraint in self.constraints:
            if len(constraint.scope) == 2:
                x, y = constraint.scope
                neighbours.setdefault(x, set()).add(y)
                neighbours.setdefault(y, set()).add(x)
                between.setdefault(frozenset((x, y)), []).append(constraint)

        self._neighbours = neighbours
        self._between = between
        self._index_size = len(self.constraints)

    def neighbours(self, var):
        self._build_index()
        return set(self._neighbours.get(var, ()))

    # The interface contract (project plan, section 9) spells it "neighbors".
    def neighbors(self, var):
        return self.neighbours(var)

    def constraints_between(self, x, y):
        self._build_index()
        return list(self._between.get(frozenset((x, y)), ()))

    def is_consistent(self, var, node, assignment):
        # Create a temporary assignment.
        temp_assignment = assignment.copy()
        temp_assignment[var] = node

        # Check every constraint involving this variable.
        for constraint in self.constraints:
            if var in constraint.scope:
                if not constraint.is_satisfied(temp_assignment):
                    return False

        return True

    def copy_domains(self):
        return {
            var: set(domain)
            for var, domain in self.domains.items()
        }


def build_csp(cluster: Cluster, jobs: list[Job], constraint_specs: list[dict]) -> CSP:
    jobs_by_id = {job.id: job for job in jobs}
    nodes_by_id = {node.id: node for node in cluster.nodes}

    variables = [job.id for job in jobs]

    node_ids = set(nodes_by_id)

    domains = {
        job.id : node_ids.copy() for job in jobs
    }

    constraints = []

    for spec in constraint_specs:
        constraint_type = spec["type"]
        job_ids = spec["jobs"]

        if constraint_type == "gpu":
            constraint = GPUConstraint(
                job_ids[0], jobs_by_id, nodes_by_id,
            )
        elif constraint_type == "allowed_zones":
            constraint = AllowedZonesConstraint(
                job_ids[0], jobs_by_id, nodes_by_id,
            )
        elif constraint_type == "fits_node":
            constraint = NodeFitConstraint(
                job_ids[0], jobs_by_id, nodes_by_id,
            )
        elif constraint_type == "capacity":
            constraint = CapacityConstraint(jobs_by_id, nodes_by_id)
        elif constraint_type == "anti_affinity":
            constraint = AntiAffinityConstraint(
                job_ids[0], job_ids[1]
            )
        elif constraint_type == "zone_affinity":
            constraint = ZoneAffinityConstraint(
                job_ids[0], job_ids[1], jobs_by_id, nodes_by_id,
            )
        elif constraint_type == "zone_spread":
            constraint = ZoneSpreadConstraint(
                job_ids[0], job_ids[1], jobs_by_id, nodes_by_id,
            )
        elif constraint_type == "pair_capacity":
            constraint = PairCapacityConstraint(
                job_ids[0], job_ids[1], jobs_by_id, nodes_by_id,
            )
        else:
            raise ValueError("Unknown constraint type", constraint_type)

        constraints.append(constraint)


    return CSP(variables, domains, constraints, jobs_by_id, nodes_by_id)
