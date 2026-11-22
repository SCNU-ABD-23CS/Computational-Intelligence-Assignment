"""Fast, static Tablut evaluation for Group 19.

Scores always use the attackers' perspective.  Terminal adjudication belongs
to the rules engine and search module, so this module only describes ongoing
positions and deliberately remains independent of those modules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numba import njit as _njit


Board = list[list[str]]

FEATURE_NAMES = (
    "material_balance",
    "low_attacker_penalty",
    "king_escape_routes",
    "king_edge_proximity",
    "king_capture_pressure",
    "king_guard_cover",
    "perimeter_blockade",
)
"""Stable weight order shared with the offline parameter optimiser."""


DEFAULT_WEIGHTS: list[float] = [
    8_500.0,
    7_500.0,
    10_000.0,
    9_000.0,
    10_000.0,
    7_000.0,
    3_500.0,
]

_BOARD_EDGE = 8
_CENTER = (4, 4)
_CASTLE_ADJACENT = frozenset(((3, 4), (4, 3), (4, 5), (5, 4)))
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MAX_WEIGHT = 10_000.0


def _ray(row: int, column: int, dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    squares = []
    row, column = row + dr, column + dc
    while 0 <= row <= _BOARD_EDGE and 0 <= column <= _BOARD_EDGE:
        squares.append((row, column))
        row, column = row + dr, column + dc
    return tuple(squares)


_KING_RAYS = tuple(
    tuple(tuple(_ray(row, column, dr, dc) for dr, dc in _DIRECTIONS)
          for column in range(_BOARD_EDGE + 1))
    for row in range(_BOARD_EDGE + 1)
)


def evaluate_board(board: Board, weights: list[float] | None = None) -> float:
    """Return a bounded, finite heuristic score for ``board``.

    Positive scores favour Attackers and negative scores favour Defenders.
    All seven feature values lie in ``[-1.0, 1.0]``.  Explicit weights must
    match ``FEATURE_NAMES`` in length, be finite numbers, and have absolute
    value at most 10,000.  This makes every ordinary score strictly smaller
    than the +/-100,000 terminal scores assigned by ``Team19_search``.

    The function reads its inputs only.  A board without a King is tolerated
    to keep this hot path finite for diagnostic callers, although the rules
    engine should classify that state as an attacker win before evaluation.
    """
    active_weights = normalize_weights(DEFAULT_WEIGHTS if weights is None else weights)
    features = _features(board)
    return _weighted_score(active_weights, features)


def _weighted_score(weights: Sequence[float], features: Sequence[float]) -> float:
    return sum(weight * feature for weight, feature in zip(weights, features))


def normalize_weights(weights: Sequence[object]) -> list[float]:
    """Return a validated copy of a seven-feature weight vector.

    Player uses this at construction time before it gives an offline-tuned
    vector to search.  It is also the single validation path for direct
    evaluator calls, so a training export cannot use a broader contract than
    the one enforced during search.
    """
    if len(weights) != len(FEATURE_NAMES):
        raise ValueError(f"weights must contain exactly seven values in {FEATURE_NAMES!r}")

    validated: list[float] = []
    for index, weight in enumerate(weights):
        if isinstance(weight, bool):
            raise ValueError(f"weight {index} must be a finite number")
        if isinstance(weight, (str, bytes, bytearray)):
            raise ValueError(f"weight {index} must be numeric, not text")
        try:
            number = float(weight)
        except OverflowError as error:
            raise ValueError(f"weight {index} must be between -10,000 and 10,000") from error
        except (TypeError, ValueError) as error:
            raise ValueError(f"weight {index} must be a finite number") from error
        if not math.isfinite(number):
            raise ValueError(f"weight {index} must be finite")
        if abs(number) > _MAX_WEIGHT:
            raise ValueError(f"weight {index} must be between -10,000 and 10,000")
        validated.append(number)
    return validated


def _features(board: Board) -> tuple[float, float, float, float, float, float, float]:
    attackers = 0
    defenders = 0
    king: tuple[int, int] | None = None

    for row, cells in enumerate(board):
        for column, piece in enumerate(cells):
            if piece == "A":
                attackers += 1
            elif piece == "D":
                defenders += 1
            elif piece == "K" and king is None:
                king = (row, column)

    return _features_from_counts(board, attackers, defenders, king)


def _features_from_counts(
    board: Board, attackers: int, defenders: int, king: tuple[int, int] | None,
) -> tuple[float, float, float, float, float, float, float]:
    attackers = min(16, attackers)
    defenders = min(8, defenders)
    material_balance = attackers / 16.0 - defenders / 8.0
    low_attacker_penalty = -max(0, 5 - attackers) / 5.0
    escape_routes, guard_cover, perimeter_blockade = _king_lines(board, king)
    edge_proximity = -_edge_proximity(king)
    capture_pressure = _capture_pressure(board, king)
    return (
        material_balance,
        low_attacker_penalty,
        escape_routes,
        edge_proximity,
        capture_pressure,
        guard_cover,
        perimeter_blockade,
    )


def _king_lines(
    board: Board, king: tuple[int, int] | None,
) -> tuple[float, float, float]:
    """Score escape rays, reachable landings, useful guards and attacker blockades."""
    if king is None:
        return 0.0, 0.0, 0.0
    row, column = king
    if not (0 < row < _BOARD_EDGE and 0 < column < _BOARD_EDGE):
        return 0.0, 0.0, 0.0

    routes = 0
    landings = 0
    shields = 0
    blockade = 0.0
    for (dr, dc), ray in zip(_DIRECTIONS, _KING_RAYS[row][column]):
        if board[row + dr][column + dc] == "D" and board[row - dr][column - dc] == "A":
            shields += 1
        blocked = False
        for distance, (rr, cc) in enumerate(ray, 1):
            square = board[rr][cc]
            if square == ".":
                if not blocked:
                    landings += 1
            elif square == "A":
                # A reserve blocker behind a guard matters only if the guard moves.
                blockade += (9 - distance) * (0.25 if blocked else 1.0)
                blocked = True
                break
            elif square != "X":
                blocked = True
        if not blocked:
            routes += 1
    guard_cover = -0.55 * landings / 16
    if shields:
        # A guard negates capture pressure along an ordinary sandwich axis.
        # Keep its credit below the default gain from an attacker closing in.
        shield_weight = 0.45 if king == _CENTER or king in _CASTLE_ADJACENT else 0.18
        guard_cover -= shield_weight * shields / 4
    return -routes / 4.0, guard_cover, blockade / 32


def _edge_proximity(king: tuple[int, int] | None) -> float:
    if king is None:
        return 0.0
    row, column = king
    distance = min(row, column, _BOARD_EDGE - row, _BOARD_EDGE - column)
    return 1.0 - min(4, max(0, distance)) / 4.0


def _capture_pressure(board: Board, king: tuple[int, int] | None) -> float:
    if king is None:
        return 0.0
    row, column = king
    if row in (0, _BOARD_EDGE) or column in (0, _BOARD_EDGE):
        return 0.0
    if king == _CENTER:
        return _adjacent_count(board, king, "A") / 4.0
    if king in _CASTLE_ADJACENT:
        surrounding_attackers = 0
        for row_step, column_step in _DIRECTIONS:
            neighbour = (row + row_step, column + column_step)
            if neighbour != _CENTER and _cell(board, *neighbour) == "A":
                surrounding_attackers += 1
        return surrounding_attackers / 3.0

    vertical_pressure = _opposing_pair_count(board, row, column, (-1, 0), (1, 0))
    horizontal_pressure = _opposing_pair_count(board, row, column, (0, -1), (0, 1))
    return max(vertical_pressure, horizontal_pressure) / 2.0


def _opposing_pair_count(
    board: Board,
    row: int,
    column: int,
    first: tuple[int, int],
    second: tuple[int, int],
) -> int:
    first_square = _cell(board, row + first[0], column + first[1])
    second_square = _cell(board, row + second[0], column + second[1])
    if first_square not in ("A", ".") or second_square not in ("A", "."):
        return 0
    return int(first_square == "A") + int(second_square == "A")


def _adjacent_count(board: Board, king: tuple[int, int] | None, piece: str) -> int:
    if king is None:
        return 0
    row, column = king
    return sum(
        _cell(board, row + row_step, column + column_step) == piece
        for row_step, column_step in _DIRECTIONS
    )


def _cell(board: Board, row: int, column: int) -> str | None:
    if row < 0 or column < 0 or row >= len(board) or column >= len(board[row]):
        return None
    return board[row][column]


# Native search uses the same encoding as the rules kernel. Geometry is passed
# by the caller so the public evaluator can still be used independently.
_NATIVE_A, _NATIVE_D, _NATIVE_X = 1, 2, 4
_NATIVE_EDGE_DISTANCE = np.array(
    [min(r, c, 8 - r, 8 - c) for r in range(9) for c in range(9)], dtype=np.int8,
)
_NATIVE_DISTANCE = np.array(
    [[abs(a // 9 - b // 9) + abs(a % 9 - b % 9) for b in range(81)]
     for a in range(81)], dtype=np.int8,
)

(
    _ATTACKER_OPEN, _ATTACKER_LATE, _DEFENDER_MATERIAL, _LOW_ATTACKERS,
    _ESCAPE_BASE, _ESCAPE_ONE, _ESCAPE_TWO, _ESCAPE_THREE, _ESCAPE_FOUR, _ESCAPE_LONG,
    _KING_MOBILITY, _KING_EDGE, _DIRECT_ESCAPE_DEFENCE, _DIRECT_ESCAPE_ATTACK,
    _PST_INTERIOR, _PST_EDGE, _ATTACKER_NEAR, _ATTACKER_ADJACENT, _GUARD_NEAR,
    _GUARD_ACTIVE, _BLOCKED_DIRECTIONS, _RAY_PRESSURE,
) = range(22)


@_njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
def _native_escape(board, king, queue, seen, near, edge):
    seen.fill(False)
    seen[king] = True
    queue[0] = king
    length = 1
    head = 0
    distance = 1
    mobility = 0
    while head < length:
        end = length
        route = False
        while head < end:
            square = queue[head]
            head += 1
            for direction in range(4):
                target = near[square, direction]
                while target >= 0:
                    cell = board[target]
                    if cell == 0:
                        mobility += distance == 1
                        if edge[target]:
                            route = True
                            if distance > 1:
                                return distance, mobility
                        elif not seen[target]:
                            seen[target] = True
                            queue[length] = target
                            length += 1
                    elif cell != _NATIVE_X and target != king:
                        break
                    target = near[target, direction]
        if route:
            return distance, mobility
        distance += 1
    return 0, mobility


def _native_piece_bonus(weights):
    """Precompute the positional terms that depend only on two squares."""
    near_king = np.maximum(0, 4 - _NATIVE_DISTANCE)
    attacking = (
        weights[_PST_INTERIOR] * _NATIVE_EDGE_DISTANCE
        + weights[_PST_EDGE] * (_NATIVE_EDGE_DISTANCE == 0)
        + weights[_ATTACKER_NEAR] * near_king
        + weights[_ATTACKER_ADJACENT] * (_NATIVE_DISTANCE == 1)
    )
    return np.stack((attacking, weights[_GUARD_NEAR] * near_king))


@_njit(fastmath={"contract"}, no_cfunc_wrapper=True)
def _native_evaluate(board, attacker, attackers, defenders, king, weights, queue, seen,
                     near, edge, piece_bonus):
    """Score an ongoing flat position from the attackers' perspective."""
    distance, mobility = _native_escape(board, king, queue, seen, near, edge)
    material = weights[_ATTACKER_OPEN if defenders >= 5 else _ATTACKER_LATE] * attackers
    score = material + weights[_DEFENDER_MATERIAL] * defenders
    score += weights[_LOW_ATTACKERS] * max(0, 6 - attackers)
    score += weights[_ESCAPE_BASE + min(5, distance)]
    score += weights[_KING_MOBILITY] * mobility
    score += weights[_KING_EDGE] * (4 - _NATIVE_EDGE_DISTANCE[king])
    if distance == 1:
        score += weights[_DIRECT_ESCAPE_ATTACK if attacker else _DIRECT_ESCAPE_DEFENCE]

    for square in range(81):
        cell = board[square]
        if cell == _NATIVE_A:
            score += piece_bonus[0, king, square]
        elif cell == _NATIVE_D:
            score += piece_bonus[1, king, square]
            for direction in range(4):
                target = near[square, direction]
                if target >= 0 and board[target] == 0:
                    score += weights[_GUARD_ACTIVE]
                    break

    if weights[_BLOCKED_DIRECTIONS] != 0 or weights[_RAY_PRESSURE] != 0:
        closed, pressure = 0, 0
        for direction in range(4):
            square = near[king, direction]
            while square >= 0:
                cell = board[square]
                if cell == _NATIVE_A:
                    closed += 1
                    pressure += 9 - _NATIVE_DISTANCE[king, square]
                    break
                if cell != 0 and cell != _NATIVE_X:
                    break
                square = near[square, direction]
        score += closed * weights[_BLOCKED_DIRECTIONS] + pressure * weights[_RAY_PRESSURE]

    # Keep evolved heuristics distinct from the search's integer mate scores.
    return int(round(min(80_000.0, max(-80_000.0, score))))
