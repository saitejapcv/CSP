"""game.py (P4) - the online scheduling game used by Minimax / Alpha-Beta.

The adversary (MIN) picks which waiting job arrives next; the scheduler (MAX)
picks a node for it. The game ends when no job is waiting or the arriving job
fits nowhere. Utility = number of jobs placed.

Legal scheduler moves come in two flavours:
  use_ac3=True  (default, what the plan asks for) the CSP built by P1 is
                restricted to the arriving job plus the jobs already placed,
                node consistency + AC-3 (P2) prune its domain, and exact
                remaining capacity is applied on top. EVERY constraint type is
                therefore respected (GPU, zones, anti-affinity, zone affinity,
                zone spread, pairwise capacity).
  use_ac3=False the original capacity-only filter (capacity, GPU, allowed zones,
                anti-affinity). It ignores zone affinity / zone spread, so it is
                kept only as the simple "capacity filter first" version.
"""

from cluster_scheduler.ac3 import prune
from cluster_scheduler.csp import CSP
from cluster_scheduler.instance import build_instance


class Game:
    def __init__(self, cluster, jobs, specs=(), use_ac3=True):
        self.nodes = {n["id"]: n for n in cluster["nodes"]}
        self.jobs = {j["id"]: j for j in jobs}
        self.anti = {frozenset(s["jobs"]) for s in specs if s["type"] == "anti_affinity"}
        self.use_ac3 = use_ac3
        self.ac3_calls = 0
        self.cache_hits = 0
        self._cache = {}

        if use_ac3:
            self.csp = build_instance(cluster, jobs, specs).csp
            self._unary = {v: [] for v in self.csp.variables}
            for c in self.csp.constraints:
                if len(c.scope) == 1:
                    self._unary[c.scope[0]].append(c)

    def start(self):
        free = {nid: (n["cpu"], n["ram"]) for nid, n in self.nodes.items()}
        return State(self, tuple(sorted(self.jobs)), {}, free, None)

    def csp_allowed(self, job_id, placement):
        """Nodes the CSP rules allow for job_id given the jobs already placed (AC-3)."""
        csp = self.csp
        placed = tuple(sorted(
            (y, placement[y]) for y in csp.neighbours(job_id) if y in placement
        ))
        key = (job_id, placed)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        self.ac3_calls += 1
        variables = [job_id] + [y for y, _ in placed]
        constraints = list(self._unary[job_id])
        for y, _ in placed:
            constraints.extend(csp.constraints_between(job_id, y))

        domains = {job_id: set(self.nodes)}
        for y, node in placed:
            domains[y] = {node}

        sub = CSP(variables, domains, constraints, csp.jobs, csp.nodes)
        ok, pruned, _ = prune(sub, domains)
        allowed = frozenset(pruned[job_id]) if ok else frozenset()
        self._cache[key] = allowed
        return allowed


class State:
    def __init__(self, game, waiting, placement, free, current):
        self.game = game
        self.waiting = waiting
        self.placement = placement
        self.free = free
        self.current = current
        self._legal = None

    def is_max(self):  ##for checking whose turn it is
        return self.current is not None

    def legal_nodes(self):
        if self._legal is None:
            self._legal = self._compute_legal()
        return list(self._legal)

    def _compute_legal(self):
        job = self.game.jobs[self.current]

        if self.game.use_ac3:
            allowed = self.game.csp_allowed(self.current, self.placement)
            return [nid for nid, (cpu, ram) in self.free.items()
                    if nid in allowed and cpu >= job["cpu"] and ram >= job["ram"]]

        out = []
        for nid, (cpu, ram) in self.free.items():
            node = self.game.nodes[nid]
            if cpu < job["cpu"] or ram < job["ram"]:
                continue
            if job["needs_gpu"] and not node["gpu"]:
                continue
            if job["allowed_zones"] and node["zone"] not in job["allowed_zones"]:
                continue
            if any(placed == nid and frozenset((self.current, other)) in self.game.anti
                   for other, placed in self.placement.items()):
                continue
            out.append(nid)
        return out

    def is_terminal(self):
        if self.current is None:
            return not self.waiting
        return not self.legal_nodes()

    def utility(self):
        return len(self.placement)

    def moves(self, ordered=False):
        if self.current is None:
            return list(self.waiting)
        nodes = self.legal_nodes()
        if ordered:
            nodes.sort(key=lambda n: self.free[n][0])
        return nodes

    def apply(self, move):
        if self.current is None:
            waiting = tuple(j for j in self.waiting if j != move)
            return State(self.game, waiting, self.placement, self.free, move)
        job = self.game.jobs[self.current]
        free = dict(self.free)
        cpu, ram = free[move]
        free[move] = (cpu - job["cpu"], ram - job["ram"])
        placement = dict(self.placement)
        placement[self.current] = move
        return State(self.game, self.waiting, placement, free, None)
