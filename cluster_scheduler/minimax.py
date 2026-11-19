def minimax(state, stats):
    stats["nodes_expanded"] += 1
    if state.is_terminal():
        return state.utility()
 
    if state.is_max():
        best = float('-inf')
        for move in state.moves():
            val = minimax(state.apply(move), stats)
            best = max(best, val)
        return best
 
    else:
        best = float('inf')
        for move in state.moves():
            val = minimax(state.apply(move), stats)
            best = min(best, val)
        return best
 
 
def solve(state):
    stats = {"nodes_expanded": 1}
    if state.is_terminal():
        return state.utility(), None, stats
    best_move = None
    best = float('-inf') if state.is_max() else float('inf')
    for move in state.moves():
        val = minimax(state.apply(move), stats)
        if (state.is_max() and val > best) or (not state.is_max() and val < best):
            best, best_move = val, move
    return best, best_move, stats
 


def best_case(state):
    """Offline (clairvoyant) value: the scheduler ALSO chooses the arrival order.

    Every level maximises, so this is the largest number of jobs any arrival order
    can place - the offline optimum the online worst case is compared with.
    Returns (value, stats).
    """
    stats = {"nodes_expanded": 0}
    total = len(state.game.jobs)

    def go(s):
        stats["nodes_expanded"] += 1
        if s.is_terminal():
            return s.utility()
        best = 0
        for move in s.moves():
            best = max(best, go(s.apply(move)))
            if best == total:      # cannot do better than placing everything
                break
        return best

    return go(state), stats
