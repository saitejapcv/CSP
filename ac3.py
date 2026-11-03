from collections import deque
from time import perf_counter


def _new_stats():
    return {"arcs_processed":0,"revisions":0,"values_pruned":0,"time":0.0}

def _copy_domains(domains):
    return {v : set(d) for v,d in domains.items()}


def _unary_constraints(csp,var):
    if hasattr(csp,"unary_constraints"):
        return list(csp.unary_constraints(var))
    return [c for c in getattr(csp,"constraints",[]) if tuple(c.scope) == (var,)]

def _binary_constraints(csp,xi,xj):
    return [
        c for c in csp.constraints_between(xi,xj)
        if len(c.scope) == 2 and hasattr(c,"allows")
    ]


# node consistency

def node_consistency(csp,domains) :
    t0 = perf_counter()
    doms = _copy_domains(domains)
    stats = _new_stats()
    ok = True

    for var in csp.variables:
        cons = _unary_constraints(csp,var)
        if cons:
            keep = {v for v in doms[var] if all(c.is_satisfied({var:v}) for c in cons)}
            stats["values_pruned"] += len(doms[var]) - len(keep)
            doms[var] = keep
        if not doms[var]:
            ok = False
            break
    stats["time"] = perf_counter() - t0
    return ok,doms,stats


def revise(csp,domains,xi,xj):
    cons = _binary_constraints(csp, xi, xj)
    #no binary constraint on this arc, nothing to prune
    if not cons:
        return False

    removed = False
    for a in list(domains[xi]):
        supported = any(all(c.allows(xi,a,xj,b) for c in cons)
            for b in domains[xj]
        )
        if not supported:
            domains[xi].discard(a)
            removed = True
    return removed


#AC 3
def ac3(csp,domains,queue = None):
    t0 = perf_counter()
    doms = _copy_domains(domains)
    stats = _new_stats()

    #an already empty means instance is not feasible.

    if any(not d for d in doms.values()):
        stats["time"] = perf_counter() - t0
        return False,doms,stats


    if queue is None:
        arcs = deque(
            (xi,xj)
            for xi in csp.variables
            for xj in csp.neighbors(xi)
        )
    else:
        arcs = deque(queue)

    while arcs:
        xi,xj = arcs.popleft()
        stats["arcs_processed"] += 1
        before = len(doms[xi])

        if revise(csp,doms,xi,xj):
            stats["revisions"] += 1
            stats["values_pruned"] += before - len(doms[xi])

            #wipe out (proven infeaasible)
            if not doms[xi]:
                stats["time"] = perf_counter() - t0
                return False,doms,stats

            for xk in csp.neighbors(xi):
                if xk != xj:
                    arcs.append((xk,xi))

    stats["time"] = perf_counter() - t0
    return True,doms,stats


def prune(csp,domains = None):

    if domains is None:
        domains =csp.copy_domains()

    ok,doms,s1 = node_consistency(csp,domains)

    if not ok:
        s1["failed_at"] = "node_consistency"
        s1.setdefault("arcs_processed",0)
        return False,doms,s1

    ok,doms,s2 = ac3(csp,doms)

    stats = {
        "arcs_processed": s2["arcs_processed"],
        "revisions": s2["revisions"],
        "values_pruned": s1["values_pruned"] + s2["values_pruned"],
        "time" : s1["time"] + s2["time"],
        "failed_at": None if ok else "ac3",    }

    return ok,doms,stats
