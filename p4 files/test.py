import pytest
from generator import generate
from game import Game
import minimax
import alphabeta
 
 
def make_game(seed):
    cluster, jobs, specs = generate(5, 3, 0.7, 0.3, seed)
    return Game(cluster, jobs, specs)
 
 
def test_generator_deterministic():
    assert generate(6, 3, 0.7, 0.2, 1) == generate(6, 3, 0.7, 0.2, 1)
 
 
@pytest.mark.parametrize("seed", range(15))
def test_minimax_equals_alphabeta(seed):
    state = make_game(seed).start()
    v1, _, s1 = minimax.solve(state)
    v2, _, s2 = alphabeta.solve(state)
    assert v1 == v2
    assert s2["nodes_expanded"] <= s1["nodes_expanded"]
 
 
def test_value_bounds():
    assert 0 <= minimax.solve(make_game(0).start())[0] <= 5