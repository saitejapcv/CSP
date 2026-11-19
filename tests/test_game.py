"""Tests for generator.py, game.py, minimax.py, alphabeta.py (written from P3's point of view):
identical values, terminal handling, node counts, legal moves honour every rule."""
import pytest

from cluster_scheduler import alphabeta, minimax
from cluster_scheduler.game import Game
from cluster_scheduler.generator import generate
from cluster_scheduler.instance import build_instance
from cluster_scheduler.verifier import verify


def make_game(seed, **kw):
    cluster, jobs, specs = generate(5, 3, 0.7, 0.3, seed)
    return Game(cluster, jobs, specs, **kw)


def _node(i, zone, cpu, gpu=False):
    return {"id": i, "zone": zone, "cpu": cpu, "ram": cpu * 4, "gpu": gpu}


def _job(i, cpu, zones=None):
    return {"id": i, "service": "s", "cpu": cpu, "ram": cpu * 4, "needs_gpu": False, "allowed_zones": zones}


# ------------------------------------ generator ------------------------------------
def test_generator_is_deterministic():
    assert generate(6, 3, 0.7, 0.2, 1) == generate(6, 3, 0.7, 0.2, 1)
    assert generate(6, 3, 0.7, 0.2, 1) != generate(6, 3, 0.7, 0.2, 2)


@pytest.mark.parametrize("seed", range(25))
def test_generated_satisfiable_instances_have_a_valid_placement(seed):
    from cluster_scheduler.baselines import brute_force
    cluster, jobs, specs = generate(5, 3, 0.8, 0.4, seed)
    inst = build_instance(cluster, jobs, specs)
    assignment, _ = brute_force(inst.cluster, inst.jobs, inst.specs)
    assert assignment is not None


def test_generator_utilisation_knob():
    cluster, jobs, _ = generate(8, 4, 0.8, 0.0, 3)
    demand = sum(j["cpu"] for j in jobs)
    supply = sum(n["cpu"] for n in cluster["nodes"])
    assert 0.55 <= demand / supply <= 1.0


def test_generator_density_knob():
    low = len(generate(10, 4, 0.7, 0.05, 4)[2])
    high = len(generate(10, 4, 0.7, 0.6, 4)[2])
    assert high > low


@pytest.mark.parametrize("kind", ["pigeonhole", "gpu", "zone_conflict"])
def test_generator_unsat_kinds_really_are_unsatisfiable(kind):
    from cluster_scheduler.baselines import brute_force
    cluster, jobs, specs = generate(5, 3, 0.7, 0.2, 7, satisfiable=False, unsat_kind=kind)
    inst = build_instance(cluster, jobs, specs)
    if len(jobs) <= 8:
        assert brute_force(inst.cluster, inst.jobs, inst.specs)[0] is None


# ------------------------------------ the game ------------------------------------
@pytest.mark.parametrize("use_ac3", [True, False])
@pytest.mark.parametrize("seed", range(15))
def test_minimax_equals_alphabeta_and_never_expands_more(seed, use_ac3):
    state = make_game(seed, use_ac3=use_ac3).start()
    v1, _, s1 = minimax.solve(state)
    v2, _, s2 = alphabeta.solve(state)
    assert v1 == v2
    assert s2["nodes_expanded"] <= s1["nodes_expanded"]


def test_alpha_beta_saves_nodes_overall():
    total1 = total2 = 0
    for seed in range(10):
        state = make_game(seed).start()
        total1 += minimax.solve(state)[2]["nodes_expanded"]
        total2 += alphabeta.solve(state)[2]["nodes_expanded"]
    assert total2 < total1


def test_value_bounds():
    assert 0 <= minimax.solve(make_game(0).start())[0] <= 5


def test_terminal_states_and_utility():
    cluster = {"zones": ["ZA"], "nodes": [_node("N1", "ZA", 4)]}
    game = Game(cluster, [_job("a", 3), _job("b", 3)], [])
    root = game.start()
    assert not root.is_terminal() and not root.is_max()
    s = root.apply("a")                       # adversary sends a
    assert s.is_max() and s.legal_nodes() == ["N1"]
    s = s.apply("N1")                         # scheduler places it
    s = s.apply("b")                          # adversary sends b: N1 has only 1 CPU left
    assert s.is_terminal() and s.utility() == 1
    assert minimax.solve(root)[0] == 1        # only one of the two can ever be placed


def test_empty_waiting_list_is_terminal():
    cluster = {"zones": ["ZA"], "nodes": [_node("N1", "ZA", 4)]}
    game = Game(cluster, [], [])
    assert game.start().is_terminal() and game.start().utility() == 0


def test_moves_do_not_mutate_the_parent_state():
    state = make_game(0).start()
    child = state.apply(state.moves()[0])
    assert state.current is None and len(state.waiting) == 5 and state.placement == {}
    assert child.current is not None


# ----------------- legal moves are filtered through AC-3 (all rule types) -----------------
def _two_zone_game(spec, **kw):
    cluster = {"zones": ["ZA", "ZB"], "nodes": [_node("N1", "ZA", 16), _node("N2", "ZA", 16),
                                                 _node("N3", "ZB", 16)]}
    return Game(cluster, [_job("a", 2), _job("b", 2)], [spec], **kw)


def test_ac3_filter_respects_zone_affinity():
    game = _two_zone_game({"type": "zone_affinity", "jobs": ["a", "b"]})
    s = game.start().apply("a").apply("N3").apply("b")       # a is in zone B
    assert s.legal_nodes() == ["N3"]


def test_ac3_filter_respects_zone_spread():
    game = _two_zone_game({"type": "zone_spread", "jobs": ["a", "b"]})
    s = game.start().apply("a").apply("N1").apply("b")       # a is in zone A
    assert s.legal_nodes() == ["N3"]


def test_ac3_filter_respects_anti_affinity():
    game = _two_zone_game({"type": "anti_affinity", "jobs": ["a", "b"]})
    s = game.start().apply("a").apply("N1").apply("b")
    assert "N1" not in s.legal_nodes() and set(s.legal_nodes()) == {"N2", "N3"}


def test_capacity_only_filter_ignores_zone_rules_which_is_why_ac3_is_the_default():
    game = _two_zone_game({"type": "zone_affinity", "jobs": ["a", "b"]}, use_ac3=False)
    s = game.start().apply("a").apply("N3").apply("b")
    assert set(s.legal_nodes()) == {"N1", "N2", "N3"}        # wrong: b may leave zone B


def _play_random_games(seed):
    import random
    cluster, jobs, specs = generate(6, 3, 0.8, 0.5, seed)
    game = Game(cluster, jobs, specs)
    rng = random.Random(seed)
    state = game.start()
    while not state.is_terminal():
        state = state.apply(rng.choice(state.moves()))
    return cluster, jobs, specs, state


@pytest.mark.parametrize("seed", range(25))
def test_every_placement_made_in_the_game_passes_the_independent_verifier(seed):
    cluster, jobs, specs, state = _play_random_games(seed)
    inst = build_instance(cluster, jobs, specs)
    placed_jobs = [j for j in inst.jobs if j.id in state.placement]
    ok, errors = verify(inst.cluster, placed_jobs, state.placement, inst.specs)
    assert ok, errors


def test_best_case_is_at_least_the_online_value_and_equals_it_when_everything_fits():
    for seed in range(6):
        game = make_game(seed)
        online = minimax.solve(game.start())[0]
        offline, _ = minimax.best_case(game.start())
        assert offline >= online
        assert offline == online == 5          # pool fits: scheduler follows one fixed plan
