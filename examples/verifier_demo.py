
from cluster_scheduler.models import Node, Job, Cluster
from cluster_scheduler.verifier import verify


# ----------------------------------------
# 1. Create cluster nodes
# ----------------------------------------

nodes = (
    Node("N1", "east", cpu=8, ram=16, gpu=True),
    Node("N2", "west", cpu=8, ram=16, gpu=False),
    Node("N3", "east", cpu=4, ram=8, gpu=False),
)

cluster = Cluster(nodes=nodes)


# ----------------------------------------
# 2. Create jobs
# ----------------------------------------

jobs = [
    Job(
        "J1",
        "training",
        cpu=4,
        ram=8,
        needs_gpu=True
    ),

    Job(
        "J2",
        "database",
        cpu=4,
        ram=6,
        allowed_zones=("east", "west")
    ),

    Job(
        "J3",
        "backend",
        cpu=2,
        ram=4,
        allowed_zones=("west",)
    ),
]


# ----------------------------------------
# 3. Define placement constraints
# ----------------------------------------

constraint_specs = [
    {
        "type": "anti_affinity",
        "jobs": ["J1", "J2"]
    },

    {
        "type": "zone_spread",
        "jobs": ["J2", "J3"]
    }
]


# ----------------------------------------
# 4. Create an assignment
# ----------------------------------------

assignment = {
    "J1": "N1",
    "J2": "N3",
    "J3": "N2"
}


# ----------------------------------------
# 5. Verify assignment
# ----------------------------------------

valid, errors = verify(
    cluster,
    jobs,
    assignment,
    constraint_specs
)


# ----------------------------------------
# 6. Display results
# ----------------------------------------

if valid:
    print("Assignment is valid!")
else:
    print("Assignment is invalid:")

    for error in errors:
        print("-", error)
