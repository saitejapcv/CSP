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
 
