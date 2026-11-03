import argparse
import csv
import time
from ac3 import prune


CONFIGS = {
    "bt" : {"ac3":False, "mrv":False},
    "bt_mrv": {"ac3":False, "mrv":True},
    "ac3_bt": {"ac3":True, "mrv":False},
    "ac3_bt_mrv": {"ac3":True, "mrv":True},
}

COLUMNS = [
    "config","seed","n_jobs","n_nodes","utilization","density",
    "status","nodes_expanded","backtracks","constraint_checks",
    "arcs_processed","values_pruned","ac3_time","search_time","total_time",
    "pruned_before_search_pct","infeasible_by_ac3","verified",
]

PRUNING_COLUMNS = [
    "seed","n_jobs","n_nodes","utilization","density",
    "domain_values_before","domain_values_after_nc","domain_values_after_ac3",
    "pct_removed_by_nc","pct_removed_total","arcs_processed","revisions",
    "failed_at","ac3_time"
]

def _size(domains):
    return sum(len(d) for d in domains.values())

def _build(n_jobs,n_nodes,utilization,density,seed):
    from generator import generate
    from csp import build_csp
    cluster,jobs,specs = generate(n_jobs,n_nodes,utilization,density,seed)
    return cluster,jobs,build_csp(cluster,jobs,specs)

def run_one(config,n_jobs,n_nodes,utilization,density,seed,time_limit = 10.0):
    from backtracking import backtrack
    from verifier import verify

    cfg = CONFIGS[config]
    cluster,jobs,csp = _build(n_jobs,n_nodes,utilization,density,seed)

    t_total = time.perf_counter()
    domains = csp.copy_domains()
    before = _size(domains)
    ac3_stats = {"arcs_processed":0,"values_pruned":0,"time":0.0}
    row = {
        "config":config,"seed":seed,"n_jobs":n_jobs,"n_nodes":n_nodes,
        "utilization":utilization,"density":density,
        "infeasible_by_ac3":False,"verified":"",
    }

    if cfg["ac3"]:
        ok,domains,ac3_stats = prune(csp,domains)
        #if proven infeasible no search is required
        if not ok:
            row.update(
                status = "no_solution",nodes_expanded = 0,backtracks = 0,constraint_checks = 0,
                arcs_processed = ac3_stats["arcs_processed"],
                values_pruned = ac3_stats["values_pruned"],
                ac3_time = ac3_stats["time"],search_time = 0.0,
                total_time = time.perf_counter()-t_total,
                pruned_before_search_pct = 100.0 * (before - _size(domains)) /max(before,1),
                infeasible_by_ac3 = True,
            )
            return row


    t_search = time.perf_counter()
    assignment,st = backtrack(csp,domains,use_mrv = cfg["mrv"],time_limit = time_limit)
    search_time = time.perf_counter() - t_search

    verified = ""
    if assignment is not None:
        verified = verify(cluster,jobs,assignment)[0]

    row.update(
        status = st.get("status","solved" if assignment else "no_solution"),
        nodes_expanded = st.get("nodes_expanded",0),
        backtracks = st.get("backtracks",0),
        constraint_checks = st.get("constraint_checks",0),
        arcs_processed = ac3_stats["arcs_processed"],
        values_pruned = ac3_stats["values_pruned"],
        ac3_time = ac3_stats["time"],search_time = search_time,
        total_time = time.perf_counter()-t_total,
        pruned_before_search_pct = 100.0 *(before - _size(domains)) /max(before,1),
        verified = verified,
    )
    return row

def pruning_one(n_jobs,n_nodes,utilization,density,seed):
    from ac3 import node_consistency,ac3
    _,_,csp = _build(n_jobs,n_nodes,utilization,density,seed)
    d0 = csp.copy_domains()
    b = _size(d0)

    ok1,d1,_ = node_consistency(csp,d0)
    a1 = _size(d1)
    failed = "" if ok1 else "node_consistency"
    revisions = arcs = 0
    t = 0.0
    d2 = d1
    if ok1:
        ok2,d2,st = ac3(csp,d1)
        revisions,arcs,t = st["revisions"],st["arcs_processed"],st["time"]
        if not ok2:
            failed = "ac3"
    a2 = _size(d2)

    return {
        "seed":seed,"n_jobs":n_jobs,"n_nodes":n_nodes,
        "utilization":utilization,"density":density,
        "domain_values_before":b,
        "domain_values_after_nc": a1,
        "domain_values_after_ac3": a2,
        "pct_removed_by_nc":round(100.0 * (b-a1)/max(b,1),2),
        "pct_removed_total": round(100.0 * (b-a2)/max(b,1),2),
        "arcs_processed":arcs,
        "revisions":revisions,
        "failed_at":failed,"ac3_time":t,
    }
def parse_seeds(text):
    if "-" in text:
        a,b = text.split("-")
        return list(range(int(a),int(b)+1))
    return [int(s) for s in text.split(",")]

def main():
    p = argparse.ArgumentParser(description = "P2 benchmark harness")
    p.add_argument("--out",default = "results.csv")
    p.add_argument("--mode",choices = ["solve","pruning"],default = "solve")
    p.add_argument("--configs",default = ",".join(CONFIGS))
    p.add_argument("--seeds",default = "0-29",help = "e.g. 0-29 or 1,5,9")
    p.add_argument("--jobs",type = int,default = 10)
    p.add_argument("--nodes",type = int,default = 4)
    p.add_argument("--utilization",type = float,default = 0.8)
    p.add_argument("--density",type = float,default = 0.3)
    p.add_argument("--time-limit",type = float,default = 10.0)
    a = p.parse_args()

    seeds = parse_seeds(a.seeds)
    with open(a.out,"w",newline="") as f:
        if a.mode == "pruning":
            w =  csv.DictWriter(f,fieldnames=PRUNING_COLUMNS)
            w.writeheader()
            for s in seeds:
                w.writerow(pruning_one(a.jobs,a.nodes,a.utilization,a.density,s))
        else:
            w = csv.DictWriter(f,fieldnames=COLUMNS)
            w.writeheader()
            for s in seeds:
                for c in a.configs.split(","):
                    w.writerow(run_one(c,a.jobs,a.nodes,a.utilization,a.density,s,a.time_limit))

    print(f"wrote {a.out} ({len(seeds)} seeds,mode = {a.mode})")

if __name__ == "__main__":
    main()