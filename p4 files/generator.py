import random
from itertools import combinations
 
 
def _make_cluster(rng, n_nodes, n_zones, gpu_fraction):
    zones = [f"Z{chr(65 + i)}" for i in range(n_zones)]
    n_gpu = max(1, round(n_nodes * gpu_fraction)) if gpu_fraction > 0 else 0
    gpu_ids = rng.sample(range(n_nodes), n_gpu)
    nodes = []
    for i in range(n_nodes):
        cpu = rng.choice([8, 16, 16, 32])
        nodes.append({"id": f"N{i + 1}", "zone": zones[i % n_zones],
                      "cpu": cpu, "ram": cpu * 4, "gpu": i in gpu_ids})
    return {"zones": zones, "nodes": nodes}
 
 
def _make_jobs_and_placement(rng, cluster, n_jobs, utilisation, gpu_fraction):
    nodes = cluster["nodes"]
    target = utilisation * sum(n["cpu"] for n in nodes)
    n_services = max(1, n_jobs // 2)
    for _ in range(200):
        weights = [rng.uniform(0.5, 2.0) for _ in range(n_jobs)]
        scale = target / sum(weights)
        jobs = []
        for i, w in enumerate(weights):
            cpu = max(1, round(w * scale))
            jobs.append({"id": f"job-{i + 1}",
                         "service": f"svc-{rng.randint(1, n_services)}",
                         "cpu": cpu, "ram": cpu * rng.choice([2, 3, 4]),
                         "needs_gpu": gpu_fraction > 0 and rng.random() < gpu_fraction,
                         "allowed_zones": None})
        free = {n["id"]: [n["cpu"], n["ram"]] for n in nodes}
        placement = {}
        for job in sorted(jobs, key=lambda j: -j["cpu"]):
            candidates = [n for n in nodes
                          if free[n["id"]][0] >= job["cpu"]
                          and free[n["id"]][1] >= job["ram"]
                          and (n["gpu"] or not job["needs_gpu"])]
            if not candidates:
                break
            n = rng.choice(candidates)
            free[n["id"]][0] -= job["cpu"]
            free[n["id"]][1] -= job["ram"]
            placement[job["id"]] = n["id"]
        if len(placement) == n_jobs:
            return jobs, placement
    raise ValueError("could not build a valid placement; lower utilisation")
 
 
def _derive_constraints(rng, cluster, jobs, placement, density):
    zone_of = {n["id"]: n["zone"] for n in cluster["nodes"]}
    specs = []
    for a, b in combinations([j["id"] for j in jobs], 2):
        if rng.random() >= density:
            continue
        options = []
        if placement[a] != placement[b]:
            options.append("anti_affinity")
        if zone_of[placement[a]] == zone_of[placement[b]]:
            options.append("zone_affinity")
        else:
            options.append("zone_spread")
        specs.append({"type": rng.choice(options), "jobs": [a, b]})
    for job in jobs:
        if rng.random() < density:
            home = zone_of[placement[job["id"]]]
            job["allowed_zones"] = sorted({home, rng.choice(cluster["zones"])})
    return specs
 
 
def _make_unsat(cluster, jobs, specs, kind):
    nodes = cluster["nodes"]
    if kind == "pigeonhole":
        k = len(nodes) + 1
        if len(jobs) < k:
            raise ValueError("pigeonhole needs n_jobs > n_nodes")
        for a, b in combinations([j["id"] for j in jobs[:k]], 2):
            specs.append({"type": "anti_affinity", "jobs": [a, b]})
    elif kind == "gpu":
        biggest = max(n["cpu"] for n in nodes if n["gpu"])
        jobs.append({"id": f"job-{len(jobs) + 1}", "service": "svc-huge",
                     "cpu": biggest + 1, "ram": 1,
                     "needs_gpu": True, "allowed_zones": None})
    else:
        raise ValueError(f"unknown unsat kind: {kind}")
 
 
def generate(n_jobs, n_nodes, utilisation, density, seed,
             n_zones=2, gpu_fraction=0.2, satisfiable=True, unsat_kind="pigeonhole"):
    rng = random.Random(seed)
    cluster = _make_cluster(rng, n_nodes, n_zones, gpu_fraction)
    jobs, placement = _make_jobs_and_placement(rng, cluster, n_jobs, utilisation, gpu_fraction)
    specs = _derive_constraints(rng, cluster, jobs, placement, density)
    if not satisfiable:
        _make_unsat(cluster, jobs, specs, unsat_kind)
    return cluster, jobs, specs
 
