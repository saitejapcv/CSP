"""instance.py - glue between JSON/dict data, P1's dataclasses and the CSP.

Why this file exists
--------------------
* P4's generator returns plain dicts, but P1's build_csp() needs Node/Job/Cluster objects.
* Job fields such as needs_gpu / allowed_zones only become CSP constraints if a
  "gpu" / "allowed_zones" spec is passed to build_csp(). The generator emits only
  binary specs, so without this step node consistency would prune nothing and
  solutions could place a GPU job on a CPU node.
* The pairwise capacity rule (layer 1 of the two-layer capacity design) has to be
  added as pair_capacity specs for AC-3 to use it.

JSON instance format (same shape generator.generate() returns):
    {
      "cluster": {"zones": ["ZA", "ZB"],
                  "nodes": [{"id": "N1", "zone": "ZA", "cpu": 16, "ram": 64, "gpu": true}, ...]},
      "jobs":    [{"id": "job-1", "service": "svc-1", "cpu": 4, "ram": 8,
                   "needs_gpu": false, "allowed_zones": null}, ...],
      "constraints": [{"type": "anti_affinity", "jobs": ["job-1", "job-2"]}, ...]
    }
"""

import json
from dataclasses import dataclass
from itertools import combinations

from cluster_scheduler.csp import CSP, build_csp
from cluster_scheduler.models import Cluster, Job, Node


@dataclass
class Instance:
    cluster: Cluster          # P1 objects
    jobs: list                # list[Job]
    specs: list               # the user-level specs (what the verifier is given)
    full_specs: list          # specs + derived unary / pairwise / capacity specs
    csp: CSP
    raw: dict                 # dict form {"cluster":..., "jobs":..., "constraints":...}


# ----------------------------------------------------------------------------
# dict <-> objects
# ----------------------------------------------------------------------------
def cluster_from_dict(data):
    nodes = data["nodes"] if isinstance(data, dict) else data
    return Cluster(nodes=tuple(
        Node(n["id"], n["zone"], int(n["cpu"]), int(n["ram"]), bool(n.get("gpu", False)))
        for n in nodes
    ))


def jobs_from_dicts(jobs):
    out = []
    for j in jobs:
        zones = j.get("allowed_zones")
        out.append(Job(
            j["id"], j.get("service", j["id"]), int(j["cpu"]), int(j["ram"]),
            bool(j.get("needs_gpu", False)),
            tuple(zones) if zones is not None else None,
        ))
    return out


# ----------------------------------------------------------------------------
# derived specs
# ----------------------------------------------------------------------------
def _key(spec):
    jobs = spec["jobs"]
    if spec["type"] in ("anti_affinity", "zone_affinity", "zone_spread", "pair_capacity"):
        return (spec["type"], frozenset(jobs))
    if spec["type"] == "capacity":
        return ("capacity",)
    return (spec["type"], jobs[0])


def derive_specs(cluster, jobs, specs, pairwise_capacity=True):
    """Return `specs` plus every rule implied by the raw data (no duplicates)."""
    full = list(specs)
    seen = {_key(s) for s in full}

    def add(spec):
        k = _key(spec)
        if k not in seen:
            seen.add(k)
            full.append(spec)

    nodes = cluster.nodes

    for job in jobs:
        if job.needs_gpu:
            add({"type": "gpu", "jobs": [job.id]})
        if job.allowed_zones is not None:
            add({"type": "allowed_zones", "jobs": [job.id]})
        if any(job.cpu > n.cpu or job.ram > n.ram for n in nodes):
            add({"type": "fits_node", "jobs": [job.id]})

    if pairwise_capacity:
        # Layer 1 of the capacity design: only pairs that cannot share SOME node
        # carry information, so only those get a pair_capacity constraint.
        for a, b in combinations(jobs, 2):
            if any(a.cpu + b.cpu > n.cpu or a.ram + b.ram > n.ram for n in nodes):
                add({"type": "pair_capacity", "jobs": [a.id, b.id]})

    # Layer 2: the exact n-ary rule (used by Backtracking and csp.is_consistent).
    add({"type": "capacity", "jobs": [j.id for j in jobs]})
    return full


# ----------------------------------------------------------------------------
# building and loading
# ----------------------------------------------------------------------------
def build_instance(cluster, jobs, specs=(), derive=True):
    """cluster/jobs/specs as returned by generator.generate() (dict form)."""
    specs = [dict(s, jobs=list(s["jobs"])) for s in specs]
    cluster_obj = cluster_from_dict(cluster)
    jobs_obj = jobs_from_dicts(jobs)
    full = derive_specs(cluster_obj, jobs_obj, specs) if derive else list(specs)
    csp = build_csp(cluster_obj, jobs_obj, full)
    cluster_dict = cluster if isinstance(cluster, dict) else {"nodes": list(cluster)}
    raw = {"cluster": cluster_dict, "jobs": list(jobs), "constraints": specs}
    return Instance(cluster_obj, jobs_obj, specs, full, csp, raw)


def load_instance(path):
    with open(path) as f:
        data = json.load(f)
    return build_instance(data["cluster"], data["jobs"], data.get("constraints", []))


def load_parts(cluster_path, jobs_path, constraints_path=None):
    with open(cluster_path) as f:
        cluster = json.load(f)
    with open(jobs_path) as f:
        jobs = json.load(f)
    specs = []
    if constraints_path:
        with open(constraints_path) as f:
            specs = json.load(f)
    return build_instance(cluster, jobs, specs)


def save_instance(path, cluster, jobs, specs):
    with open(path, "w") as f:
        json.dump({"cluster": cluster, "jobs": jobs, "constraints": list(specs)}, f, indent=2)
