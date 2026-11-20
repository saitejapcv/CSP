# Cloud Cluster Scheduler

Job placement for a simulated cloud, solved as a **Constraint Satisfaction Problem** with
node consistency, **AC-3**, **Backtracking + MRV**, plus a small **Minimax / Alpha-Beta**
online-scheduling add-on. Pure Python; the solvers use no CSP or search library.

Given a cluster (zones, nodes with CPU / RAM / GPU) and jobs with placement rules, the scheduler
either returns a placement that satisfies **every** rule (checked by an independent verifier) or
correctly reports that none exists.

## Install

Python 3.10 or newer (tested on 3.13). Only `matplotlib` (plots) and `pytest` are needed.

```bash
pip install matplotlib pytest          # or: uv sync
```

## Run it

```bash
# solve an instance (JSON in, verified result JSON out)
python -m cluster_scheduler.cli --instance examples/easy.json --method ac3+bt+mrv --out result.json

# the other methods: bt | bt+mrv | ac3+bt | ac3+bt+mrv | firstfit | brute
python -m cluster_scheduler.cli --instance examples/tight.json --method firstfit      # greedy fails here
python -m cluster_scheduler.cli --instance examples/unsat_ac3.json --method ac3+bt    # AC-3 alone proves it

# seeded random instance
python -m cluster_scheduler.cli --generate --n-jobs 12 --n-nodes 4 --utilisation 0.85 --seed 3

# tests (293)
python -m pytest

# reproduce EVERY table and figure with fixed seeds (about 10 seconds) -> results/
python -m cluster_scheduler.experiments --out results
```

Statuses are exactly `solved`, `no_solution`, `timeout`. A timeout means "gave up", never "proved infeasible".

## Layout

| File | Owner | What it does |
|---|---|---|
| `models.py`, `constraints.py`, `csp.py`, `verifier.py` | P1 | dataclasses, constraint classes, the CSP, the independent verifier |
| `ac3.py`, `benchmark.py` | P2 | node consistency, `revise`, AC-3; experiment runner writing CSV |
| `backtracking.py`, `baselines.py`, `cli.py` | P3 | Backtracking + MRV + exact capacity; first-fit and brute force; command line |
| `game.py`, `minimax.py`, `alphabeta.py`, `generator.py`, `visualize.py` | P4 | online game, Minimax, Alpha-Beta, seeded generator, plots |
| `instance.py`, `experiments.py` | glue | dict/JSON to objects to CSP; one command for all experiments |
| `examples/` | P1 | `easy` (Figure 1), `tight`, `unsat`, `unsat_ac3` |
| `tests/` | all | one file per module plus integration tests |

## How it works

1. **Read** the JSON and build the CSP: one variable per job, domain = the nodes it may use.
2. **Node consistency** removes values that break a unary rule (GPU, allowed zones, "job fits an empty node").
3. **AC-3** removes values with no compatible partner among neighbouring jobs (anti-affinity, zone
   affinity, zone spread, pairwise capacity). An empty domain proves infeasibility.
4. **Backtracking** on the pruned domains, MRV (fewest legal nodes, ties by degree), exact per-node
   CPU/RAM running totals added on assign and subtracted on undo.
5. **Verify** with `verifier.py`, which shares no code with the solver.

**Capacity is not binary.** AC-3 only sees pairs, so capacity is handled in two layers: (1) a
*pairwise* rule (`cpu_i + cpu_j <= node.cpu`) that is safe but not sufficient and is used for pruning;
(2) the *exact* n-ary rule enforced by Backtracking. AC-3 never solves the problem by itself: three jobs
that must be on different nodes with only two nodes (`examples/unsat.json`) pass AC-3 and are only
refuted by search.

### Instance JSON

```json
{"cluster": {"zones": ["ZA","ZB"], "nodes": [{"id":"N1","zone":"ZA","cpu":16,"ram":64,"gpu":true}]},
 "jobs": [{"id":"ml","service":"ml","cpu":8,"ram":32,"needs_gpu":true,"allowed_zones":null}],
 "constraints": [{"type":"anti_affinity","jobs":["api-1","api-2"]}]}
```
Constraint types: `anti_affinity`, `zone_affinity`, `zone_spread`, `pair_capacity` (binary);
`gpu`, `allowed_zones`, `fits_node` (unary) and `capacity` (n-ary) are derived from the job and node data
automatically (`instance.derive_specs`), so you only list the binary rules.

## Statistics

`nodes_expanded` (assignments tried), `backtracks` (dead ends), `constraint_checks`, `arcs_processed`,
`values_pruned`, `time`, `status`. AC-3 costs O(e * d^3) for e arcs and domain size d.

## Results

`python -m cluster_scheduler.experiments` writes `results/SUMMARY.md` with all tables and the figures
(`solver_comparison.png`, `firstfit_vs_csp.png`, `pruning.png`, `minimax_vs_alphabeta.png`,
`offline_vs_online.png`, `placement_easy.png`). Every table is regenerated from fixed seeds.
