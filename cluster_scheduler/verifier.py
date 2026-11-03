
def verify(cluster, jobs, assignment, constraint_specs=()):
    """
    Independently verify a job-to-node assignment.

    Args:
        cluster: Cluster containing available nodes.
        jobs: List of Job objects.
        assignment: Dictionary mapping job IDs to node IDs.
        constraint_specs: Pairwise placement constraint specifications.

    Returns:
        valid: bool
        errors: list[str]
    """

    errors = []

    # Create lookup dictionaries
    nodes_by_id = {node.id: node for node in cluster.nodes}
    jobs_by_id = {job.id: job for job in jobs}

    # --------------------------------------------------
    # 1. Check assignment completeness and valid IDs
    # --------------------------------------------------

    for job_id in jobs_by_id:
        if job_id not in assignment:
            errors.append(f"Job {job_id} has not been assigned.")

    for job_id, node_id in assignment.items():
        if job_id not in jobs_by_id:
            errors.append(f"Unknown job ID: {job_id}")
        elif node_id not in nodes_by_id:
            errors.append(
                f"Job {job_id} is assigned to unknown node {node_id}."
            )

    # Continue only with valid job-node assignments
    valid_assignments = {
        job_id: node_id
        for job_id, node_id in assignment.items()
        if job_id in jobs_by_id and node_id in nodes_by_id
    }

    # --------------------------------------------------
    # 2. Check unary constraints
    # --------------------------------------------------

    for job_id, node_id in valid_assignments.items():
        job = jobs_by_id[job_id]
        node = nodes_by_id[node_id]

        # GPU requirement
        if job.needs_gpu and not node.gpu:
            errors.append(
                f"Job {job_id} requires GPU, but node {node_id} has no GPU."
            )

        # Allowed zones
        if (
            job.allowed_zones is not None
            and node.zone not in job.allowed_zones
        ):
            errors.append(
                f"Job {job_id} cannot run in zone {node.zone}."
            )

    # --------------------------------------------------
    # 3. Check pairwise constraints
    # --------------------------------------------------

    for spec in constraint_specs:
        constraint_type = spec["type"]
        job_ids = spec["jobs"]

        if len(job_ids) != 2:
            errors.append(
                f"{constraint_type} requires exactly two job IDs."
            )
            continue

        job1_id, job2_id = job_ids

        # Skip invalid or unassigned jobs
        if (
            job1_id not in valid_assignments
            or job2_id not in valid_assignments
        ):
            continue

        node1_id = valid_assignments[job1_id]
        node2_id = valid_assignments[job2_id]

        node1 = nodes_by_id[node1_id]
        node2 = nodes_by_id[node2_id]

        if constraint_type == "anti_affinity":
            if node1_id == node2_id:
                errors.append(
                    f"Jobs {job1_id} and {job2_id} must be on different nodes."
                )

        elif constraint_type == "zone_affinity":
            if node1.zone != node2.zone:
                errors.append(
                    f"Jobs {job1_id} and {job2_id} must be in the same zone."
                )

        elif constraint_type == "zone_spread":
            if node1.zone == node2.zone:
                errors.append(
                    f"Jobs {job1_id} and {job2_id} must be in different zones."
                )

        elif constraint_type == "pair_capacity":
            # Exact aggregate capacity is checked below.
            pass

        else:
            errors.append(
                f"Unknown constraint type: {constraint_type}"
            )

    # --------------------------------------------------
    # 4. Check aggregate CPU and RAM capacity
    # --------------------------------------------------

    for node_id, node in nodes_by_id.items():
        assigned_jobs = [
            jobs_by_id[job_id]
            for job_id, assigned_node in valid_assignments.items()
            if assigned_node == node_id
        ]

        total_cpu = sum(job.cpu for job in assigned_jobs)
        total_ram = sum(job.ram for job in assigned_jobs)

        if total_cpu > node.cpu:
            errors.append(
                f"Node {node_id} CPU capacity exceeded: "
                f"{total_cpu}/{node.cpu}."
            )

        if total_ram > node.ram:
            errors.append(
                f"Node {node_id} RAM capacity exceeded: "
                f"{total_ram}/{node.ram}."
            )

    # --------------------------------------------------
    # 5. Return verification result
    # --------------------------------------------------

    return len(errors) == 0, errors
