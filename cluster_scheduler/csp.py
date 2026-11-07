from cluster_scheduler.models import Cluster, Job
from cluster_scheduler.constraints import (
    GPUConstraint,
    AllowedZonesConstraint,
    AntiAffinityConstraint,
    ZoneAffinityConstraint,
    ZoneSpreadConstraint,
    PairCapacityConstraint,
)

class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints

    def neighbours(self, var):
        result = set()

        for constraint in self.constraints:
            if len(constraint.scope) == 2:
                x, y = constraint.scope

                if x == var:
                    result.add(y)
                elif y == var:
                    result.add(x)

        return result

    def constraints_between(self, x, y):
        return [
            constraint
            for constraint in self.constraints
            if len(constraint.scope) == 2
               and set(constraint.scope) == {x, y}
        ]

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


    return CSP(variables, domains, constraints)
