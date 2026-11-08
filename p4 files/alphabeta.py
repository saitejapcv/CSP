def alphabeta(state, alpha, beta, stats, ordered=True):
    stats["nodes_expanded"] += 1
    if state.is_terminal():
        return state.utility()
 
    if state.is_max():
        best = float('-inf')
        for move in state.moves(ordered):
            val = alphabeta(state.apply(move), alpha, beta, stats, ordered)
            best = max(best, val)
            alpha = max(alpha, best)
 
            if beta <= alpha:
                break
        return best
 
    else:
        best = float('inf')
        for move in state.moves(ordered):
            val = alphabeta(state.apply(move), alpha, beta, stats, ordered)
            best = min(best, val)
            beta = min(beta, best)
 
            if beta <= alpha:
                break
        return best
 
 
def solve(state, ordered=True):
    stats = {"nodes_expanded": 1}
    if state.is_terminal():
        return state.utility(), None, stats
    alpha, beta = float('-inf'), float('inf')
    best_move = None
    best = float('-inf') if state.is_max() else float('inf')
    for move in state.moves(ordered):
        val = alphabeta(state.apply(move), alpha, beta, stats, ordered)
        if state.is_max():
            if val > best:
                best, best_move = val, move
            alpha = max(alpha, best)
        else:
            if val < best:
                best, best_move = val, move
            beta = min(beta, best)
        if beta <= alpha:
            break
    return best, best_move, stats

