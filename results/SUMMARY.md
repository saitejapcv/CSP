# Experiment summary (fixed seeds)

### Backtracking configurations (14 jobs, 4 nodes, utilisation 0.85, 50 seeds)

| config | instances | solved | timeout | mean nodes | mean backtracks | mean checks | mean ms | failed verifier |
|---|---|---|---|---|---|---|---|---|
| bt | 50 | 50 | 0 | 247.9 | 233.9 | 2419 | 0.85 | 0 |
| bt_mrv | 50 | 50 | 0 | 18.2 | 4.2 | 1440 | 0.52 | 0 |
| ac3_bt | 50 | 50 | 0 | 49.2 | 35.2 | 416 | 0.70 | 0 |
| ac3_bt_mrv | 50 | 50 | 0 | 17.5 | 3.5 | 1035 | 0.86 | 0 |
| firstfit | 50 | 2 | 0 | 6.6 | 0.0 | 45 | 0.11 | 0 |

### First-fit vs the CSP solver (every instance is satisfiable by construction)

| utilisation | instances | first-fit solved | AC-3+BT+MRV solved |
|---|---|---|---|
| 0.6 | 30 | 17% | 100% |
| 0.7 | 30 | 17% | 100% |
| 0.8 | 30 | 17% | 100% |
| 0.9 | 30 | 7% | 100% |

### How much AC-3 prunes, and when it proves infeasibility alone

| instance kind | instances | % removed by node consistency | % removed in total | infeasible found by node consistency | infeasible found by AC-3 | not detected |
|---|---|---|---|---|---|---|
| sat | 20 | 17.0 | 46.9 | 0 | 0 | 20 |
| unsat_pigeonhole | 20 | 14.0 | 41.9 | 0 | 7 | 13 |
| unsat_gpu | 20 | 21.5 | 21.5 | 20 | 0 | 0 |
| unsat_zone_conflict | 20 | 19.9 | 23.8 | 4 | 16 | 0 |
AC-3 / node consistency reported infeasible on 0 satisfiable instances (must be 0). 'not detected' on an unsat kind means only search can prove it.

### Online game (3 nodes, utilisation 0.9; moves filtered by AC-3)

| pool size | games | Minimax nodes | Alpha-Beta nodes | nodes saved | offline best case | online worst case | gap | Minimax = Alpha-Beta |
|---|---|---|---|---|---|---|---|---|
| 5 | 10 | 2483 | 1058 | 51% | 5.00 | 5.00 | 0.00 | True |
| 6 | 10 | 12337 | 3828 | 65% | 5.00 | 3.10 | 1.90 | True |
| 7 | 10 | 54510 | 11774 | 75% | 5.20 | 2.40 | 2.80 | True |

When everything fits (pool 5) the gap is 0: a scheduler that knows the pool can follow one valid placement whatever the arrival order. The gap appears once the pool is overloaded.

### Agreement with the brute-force oracle (5 jobs, 3 nodes)

60/60 random tiny instances agree (30 satisfiable, 30 unsatisfiable): every solver configuration returns the same solved / no_solution as brute force, and AC-3 never removes a value used by a real solution.

### Placement for examples/easy.json

verified = True (see placement_easy.png)
