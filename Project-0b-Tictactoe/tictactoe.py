"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    X always goes first. If counts are equal -> X, else -> O.
    """
    x_count = sum(cell == X for row in board for cell in row)
    o_count = sum(cell == O for row in board for cell in row)
    return X if x_count == o_count else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    If the game is over, returns an empty set.
    """
    if terminal(board):
        return set()
    return {(i, j)
            for i in range(3)
            for j in range(3)
            if board[i][j] is EMPTY}


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    Does not mutate the original board.
    Raises ValueError for invalid actions.
    """
    if action is None or len(action) != 2:
        raise ValueError("Action must be a tuple (i, j).")
    i, j = action
    if not (0 <= i < 3 and 0 <= j < 3):
        raise ValueError("Action out of bounds.")
    if board[i][j] is not EMPTY:
        raise ValueError("Cell already occupied.")

    new_board = copy.deepcopy(board)
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game (X or O), if there is one; otherwise None.
    """
    lines = []

    # Rows and columns
    lines.extend(board)  # rows
    lines.extend([[board[r][c] for r in range(3)] for c in range(3)])  # cols

    # Diagonals
    lines.append([board[i][i] for i in range(3)])
    lines.append([board[i][2 - i] for i in range(3)])

    for line in lines:
        if line[0] is not None and line.count(line[0]) == 3:
            return line[0]
    return None


def terminal(board):
    """
    Returns True if game is over (win or draw), False otherwise.
    """
    if winner(board) is not None:
        return True
    # Any empty cell? If none, it's a draw -> terminal
    return all(cell is not EMPTY for row in board for cell in row)


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    w = winner(board)
    if w == X:
        return 1
    if w == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    If the game is terminal, returns None.
    Uses alpha-beta pruning for efficiency.
    """
    if terminal(board):
        return None

    turn = player(board)

    # Optional: short-circuit if a winning move is immediately available
    for a in actions(board):
        if winner(result(board, a)) == turn:
            return a

    def max_value(state, alpha, beta):
        if terminal(state):
            return utility(state), None
        v = -math.inf
        best_action = None
        for a in sorted(actions(state)):  # deterministic tie-breaking
            score, _ = min_value(result(state, a), alpha, beta)
            if score > v:
                v, best_action = score, a
            alpha = max(alpha, v)
            if alpha >= beta:
                break  # beta cut-off
        return v, best_action

    def min_value(state, alpha, beta):
        if terminal(state):
            return utility(state), None
        v = math.inf
        best_action = None
        for a in sorted(actions(state)):
            score, _ = max_value(result(state, a), alpha, beta)
            if score < v:
                v, best_action = score, a
            beta = min(beta, v)
            if alpha >= beta:
                break  # alpha cut-off
        return v, best_action

    if turn == X:
        _, act = max_value(board, -math.inf, math.inf)
    else:
        _, act = min_value(board, -math.inf, math.inf)
    return act
