"""Fast one-move terminal threats using the current Python rules engine."""

from time import perf_counter

import Team19_engine as engine
from Team19_search import SearchTimeout


DIRECTIONS = ((-1, 0), (0, -1), (1, 0), (0, 1))
EDGE = tuple(r in (0, 8) or c in (0, 8) for r in range(9) for c in range(9))


def ray(row, col, dr, dc):
    points = []
    row, col = row + dr, col + dc
    while 0 <= row < 9 and 0 <= col < 9:
        points.append((row, col))
        row, col = row + dr, col + dc
    return tuple(points)


RAYS = tuple(tuple(ray(r, c, dr, dc) for dr, dc in DIRECTIONS)
             for r in range(9) for c in range(9))


def poll(deadline):
    if perf_counter() >= deadline:
        raise SearchTimeout("Arena search deadline")


def _attacker_to(board, row, col):
    for direction in RAYS[9 * row + col]:
        for rr, cc in direction:
            square = board[rr][cc]
            if square == "A":
                return [[rr, cc], [row, col]]
            if square not in (".", "X"):
                break
    return None


def winning_move(state, attacker):
    """Find an immediate win on a playable position, with no history side effects."""
    if state.king is None:
        return None
    row, col = state.king
    board = state.board
    if not attacker:
        for direction in RAYS[9 * row + col]:
            for rr, cc in direction:
                square = board[rr][cc]
                if square not in (".", "X"):
                    break
                if square == "." and EDGE[9 * rr + cc]:
                    return [[row, col], [rr, cc]]
        if state.attackers <= 5:
            for move in engine.get_legal_moves(board, False):
                if engine._apply_legal_move(state, move).winner() == "D":
                    return move
        return None

    protected = abs(row - 4) + abs(col - 4) <= 1
    for dr, dc in DIRECTIONS:
        rr, cc = row + dr, col + dc
        br, bc = row - dr, col - dc
        if not (0 <= rr < 9 and 0 <= cc < 9 and 0 <= br < 9 and 0 <= bc < 9):
            continue
        if board[rr][cc] != ".":
            continue
        if protected:
            if any(board[row + kr][col + kc] not in ("A", "X")
                   for kr, kc in DIRECTIONS if (kr, kc) != (dr, dc)):
                continue
        elif board[br][bc] != "A":
            continue
        if move := _attacker_to(board, rr, cc):
            return move
    return None
