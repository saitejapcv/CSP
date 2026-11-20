"""backtracking.py (P3) - recursive Backtracking with MRV and exact capacity.

    backtrack(csp, domains, use_mrv=False, time_limit=None) -> (assignment | None, stats)

* `domains` are the (usually AC-3-pruned) domains. They are never modified.
* Unary and binary constraints are checked against the jobs already placed.
* Capacity is the exact n-ary rule: per-node running CPU/RAM totals are added on
  assign and subtracted on undo, so every capacity check is O(1).
* MRV picks the unassigned job with the fewest values that are still legal given
  the current partial assignment; ties go to the job with the most unassigned
  neighbours (degree heuristic).
* stats: nodes_expanded (assignments tried), backtracks (dead ends: a call that
  returns failure), constraint_checks, time, status.
* status is exactly one of "solved", "no_solution", "timeout".
  A timeout means "gave up", NOT "proved infeasible"; the two are never mixed.
"""

from time import perf_counter


class _Timeout(Exception):
    pass


class Checker:
    """Partial assignment + running per-node totals + the consistency test.

    Shared by backtrack() and the first-fit baseline so both use the same rules.
    """

    def __init__(self, csp):
        self.csp = csp
        self.jobs = csp.jobs
        self.nodes = csp.nodes
        self.capacity_on = bool(csp.jobs) and bool(csp.nodes)

        self.assignment = {}
        self.used_cpu = {nid: 0 for nid in self.nodes}
        self.used_ram = {nid: 0 for nid in self.nodes}
        self.checks = 0

        self.unary = {v: [] for v in csp.variables}
        for c in csp.constraints:
            if len(c.scope) == 1 and c.scope[0] in self.unary:
                self.unary[c.scope[0]].append(c)

        # binary[v] = [(other_var, [constraints with allows]), ...]
        self.binary = {}
        for v in csp.variables:
            entries = []
            for other in csp.neighbours(v):
                cons = [c for c in csp.constraints_between(v, other) if hasattr(c, "allows")]
                if cons:
                    entries.append((other, cons))
            self.binary[v] = entries

    # ---- the consistency test -------------------------------------------------
    def ok(self, var, node):
        """Can `var` go on `node` given the jobs already placed?"""
        for c in self.unary[var]:
            self.checks += 1
            if not c.is_satisfied({var: node}):
                return False

        for other, cons in self.binary[var]:
            placed = self.assignment.get(other)
            if placed is None:
                continue
            for c in cons:
                self.checks += 1
                if not c.allows(var, node, other, placed):
                    return False

        if self.capacity_on:
            self.checks += 1
            job, n = self.jobs[var], self.nodes[node]
            if (self.used_cpu[node] + job.cpu > n.cpu
                    or self.used_ram[node] + job.ram > n.ram):
                return False

        return True

    # ---- assign / undo --------------------------------------------------------
    def assign(self, var, node):
        self.assignment[var] = node
        if self.capacity_on:
            job = self.jobs[var]
            self.used_cpu[node] += job.cpu
            self.used_ram[node] += job.ram

    def unassign(self, var):
        node = self.assignment.pop(var)
        if self.capacity_on:
            job = self.jobs[var]
            self.used_cpu[node] -= job.cpu
            self.used_ram[node] -= job.ram


class _Search(Checker):
    def __init__(self, csp, domains, use_mrv, time_limit):
        super().__init__(csp)
        self.use_mrv = use_mrv
        self.deadline = None if time_limit is None else perf_counter() + time_limit
        self.stats = {"nodes_expanded": 0, "backtracks": 0, "constraint_checks": 0}

        # Values are tried in cluster order so runs are deterministic.
        order = {nid: i for i, nid in enumerate(self.nodes)}
        self.values = {
            v: sorted(domains[v], key=lambda n: order.get(n, len(order)))
            for v in csp.variables
        }

    def _select(self):
        """Return (variable, legal_values_or_None)."""
        unassigned = [v for v in self.csp.variables if v not in self.assignment]
        if not self.use_mrv:
            return unassigned[0], None

        best_key, best_var, best_legal = None, None, None
        for v in unassigned:
            legal = [n for n in self.values[v] if self.ok(v, n)]
            degree = sum(1 for o, _ in self.binary[v] if o not in self.assignment)
            key = (len(legal), -degree)
            if best_key is None or key < best_key:
                best_key, best_var, best_legal = key, v, legal
            if not legal:        # a job with no legal node: dead end right now
                break
        return best_var, best_legal

    def _recurse(self):
        if len(self.assignment) == len(self.csp.variables):
            return True
        if self.deadline is not None and perf_counter() > self.deadline:
            raise _Timeout

        var, legal = self._select()
        candidates = legal if legal is not None else self.values[var]

        for node in candidates:
            if legal is None and not self.ok(var, node):
                continue
            self.assign(var, node)
            self.stats["nodes_expanded"] += 1
            if self._recurse():
                return True
            self.unassign(var)          # undo

        self.stats["backtracks"] += 1   # every value failed: dead end
        return False


def backtrack(csp, domains, use_mrv=False, time_limit=None):
    """Find a placement or prove there is none. See the module docstring."""
    start = perf_counter()
    search = _Search(csp, domains, use_mrv, time_limit)

    try:
        solved = search._recurse()
        status = "solved" if solved else "no_solution"
    except _Timeout:
        solved, status = False, "timeout"

    stats = dict(search.stats)
    stats["constraint_checks"] = search.checks
    stats["time"] = perf_counter() - start
    stats["status"] = status
    return (dict(search.assignment) if solved else None), stats
