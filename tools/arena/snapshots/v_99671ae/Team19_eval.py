"""Fast, static Tablut evaluation for Group 19.

Scores always use the attackers' perspective.  Terminal adjudication belongs
to the rules engine and search module, so this module only describes ongoing
positions and deliberately remains independent of those modules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


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
    return float(sum(weight * feature for weight, feature in zip(active_weights, features)))


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
    perimeter_attackers = 0
    king: tuple[int, int] | None = None

    for row, cells in enumerate(board):
        for column, piece in enumerate(cells):
            if piece == "A":
                attackers = min(16, attackers + 1)
                if row in (0, _BOARD_EDGE) or column in (0, _BOARD_EDGE):
                    perimeter_attackers = min(16, perimeter_attackers + 1)
            elif piece == "D":
                defenders = min(8, defenders + 1)
            elif piece == "K" and king is None:
                king = (row, column)

    material_balance = attackers / 16.0 - defenders / 8.0
    low_attacker_penalty = -max(0, 5 - attackers) / 5.0
    escape_routes = -_escape_route_count(board, king) / 4.0
    edge_proximity = -_edge_proximity(king)
    capture_pressure = _capture_pressure(board, king)
    guard_cover = -_adjacent_count(board, king, "D") / 4.0
    perimeter_blockade = perimeter_attackers / 16.0
    return (
        material_balance,
        low_attacker_penalty,
        escape_routes,
        edge_proximity,
        capture_pressure,
        guard_cover,
        perimeter_blockade,
    )


def _escape_route_count(board: Board, king: tuple[int, int] | None) -> int:
    if king is None:
        return 0
    row, column = king
    if not (0 < row < _BOARD_EDGE and 0 < column < _BOARD_EDGE):
        return 0

    routes = 0
    for row_step, column_step in _DIRECTIONS:
        next_row = row + row_step
        next_column = column + column_step
        while 0 <= next_row <= _BOARD_EDGE and 0 <= next_column <= _BOARD_EDGE:
            square = _cell(board, next_row, next_column)
            if square not in (".", "X"):
                break
            if (
                next_row in (0, _BOARD_EDGE) or next_column in (0, _BOARD_EDGE)
            ) and square == ".":
                routes += 1
                break
            next_row += row_step
            next_column += column_step
    return routes


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
