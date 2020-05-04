"""
Tic Tac Toe Player
"""

import math

#Additional imports
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

def empty_counter(board):
    """
    Returns number of empty cells on the board.
    """
    empty_slots = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] is EMPTY:
                empty_slots += 1
    return empty_slots


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    empty_slots = empty_counter(board)

    if empty_slots == 0:
        return None
    elif empty_slots%2 != 0:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] is EMPTY:
                actions.add((i, j))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] is not EMPTY:
        raise Exception('Invalid Move!')

    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if (board[0][0] == board[1][0] == board[2][0] == X or board[0][0] == board[0][1] == board[0][2] == X or
        board[0][1] == board[1][1] == board[2][1] == X or board[1][0] == board[1][1] == board[1][2] == X or
        board[0][2] == board[1][2] == board[2][2] == X or board[2][0] == board[2][1] == board[2][2] == X or
        board[0][0] == board[1][1] == board[2][2] == X or board[0][2] == board[1][1] == board[2][0] == X):
        return X
    elif (board[0][0] == board[1][0] == board[2][0] == O or board[0][0] == board[0][1] == board[0][2] == O or
        board[0][1] == board[1][1] == board[2][1] == O or board[1][0] == board[1][1] == board[1][2] == O or
        board[0][2] == board[1][2] == board[2][2] == O or board[2][0] == board[2][1] == board[2][2] == O or
        board[0][0] == board[1][1] == board[2][2] == O or board[0][2] == board[1][1] == board[2][0] == O):
        return O
    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None or empty_counter(board) == 0:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def max_value(board):
    """
    Returns the 'score' of the board, w.r.t player X
    """
    if terminal(board):
        return utility(board)
    v = -69
    for action in actions(board):
        if v < min_value(result(board, action)):
            v = min_value(result(board, action))
    return v


def min_value(board):
    """
    Returns the 'score' of the board, w.r.t player O
    """
    if terminal(board):
        return utility(board)
    v = 69
    for action in actions(board):
        if v > max_value(result(board, action)):
            v = max_value(result(board, action))
    return v


def MAX(board):
    """
    Returns optimal action for player X
    """
    v = -69
    move = None
    for action in actions(board):
        if v < min_value(result(board, action)):
            move = action
            v = min_value(result(board, action))
    return move


def MIN(board):
    """
    Returns optimal action for player O
    """
    v = 69
    move = None
    for action in actions(board):
        if v > max_value(result(board, action)):
            move = action
            v = max_value(result(board, action))
    return move


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    if player(board) == X:
        return MAX(board)
    else:
        return MIN(board)
