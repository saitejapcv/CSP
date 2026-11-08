class Game:
    def __init__(self, cluster, jobs, specs=()):
        self.nodes = {n["id"]: n for n in cluster["nodes"]}
        self.jobs = {j["id"]: j for j in jobs}
        self.anti = {frozenset(s["jobs"]) for s in specs if s["type"] == "anti_affinity"}
 
    def start(self):
        free = {nid: (n["cpu"], n["ram"]) for nid, n in self.nodes.items()}
        return State(self, tuple(sorted(self.jobs)), {}, free, None)
 
 
class State:
    def __init__(self, game, waiting, placement, free, current):
        self.game = game
        self.waiting = waiting
        self.placement = placement
        self.free = free
        self.current = current
 
    def is_max(self):  ##for checking whose turn it is 
        return self.current is not None
 
    def legal_nodes(self):
        job = self.game.jobs[self.current]
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
 
