"""Static, fail-closed opening-book lookup for Group 19.

``Team19_book_data.BOOK_LINES`` contains complete paths from the official
initial position. Every path alternates moves and only its last move is a
recommendation. This module validates and expands that immutable corpus once
at import time. Lookups only probe the resulting read-only index.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Sequence
from types import MappingProxyType

from Team19_engine import apply_move, check_winner, compute_hash, get_legal_moves


_INITIAL_BOARD = (
    (".", ".", ".", "A", "A", "A", ".", ".", "."),
    (".", ".", ".", ".", "A", ".", ".", ".", "."),
    (".", ".", ".", ".", "D", ".", ".", ".", "."),
    ("A", ".", ".", ".", "D", ".", ".", ".", "A"),
    ("A", "A", "D", "D", "K", "D", "D", "A", "A"),
    ("A", ".", ".", ".", "D", ".", ".", ".", "A"),
    (".", ".", ".", ".", "D", ".", ".", ".", "."),
    (".", ".", ".", ".", "A", ".", ".", ".", "."),
    (".", ".", ".", "A", "A", "A", ".", ".", "."),
)
_BOARD_EDGE = 8

try:
    from Team19_book_data import BOOK_LINES as _DATA_LINES
except Exception:
    _DATA_LINES = ()

BOOK_LINES = _DATA_LINES

Move = tuple[int, int, int, int]
MovePair = tuple[tuple[int, int], tuple[int, int]]
BoardKey = tuple[tuple[str, ...], ...]
HistoryKey = frozenset[int]
IndexKey = tuple[BoardKey, bool, HistoryKey]


def choose_book_move(
    board: list[list[str]],
    is_attacker: bool,
    game_history: set[int],
    legal_moves: Iterable[Sequence[Sequence[int]]],
) -> list[list[int]] | None:
    """Return a verified book move, or ``None`` without changing state.

    The key contains the complete past-position set, so a matching board
    reached through another history cannot use a book entry. Empty, stale,
    malformed, ambiguous, and repeated-successor cases fail closed.
    """
    try:
        if len(game_history) > MAX_HISTORY_SIZE:
            return None
        history = frozenset(game_history)
        key = (_freeze_board(board), is_attacker, history)
    except (TypeError, ValueError):
        return None
    recommendation = BOOK_INDEX.get(key)
    if recommendation is None:
        return None

    pair = _as_pair(recommendation)
    if pair is None or not _is_legal(pair, legal_moves):
        return None
    try:
        successor = apply_move(board, _as_lists(recommendation), is_attacker)
        if compute_hash(successor, not is_attacker) in history:
            return None
    except Exception:
        return None
    return _as_lists(recommendation)


def build_book_index(lines: object) -> tuple[MappingProxyType, int]:
    """Build the immutable D4-expanded index used by production and tests."""
    try:
        source_lines = tuple(lines)
    except TypeError:
        return MappingProxyType({}), 0

    entries: dict[IndexKey, Move | None] = {}
    maximum_history_size = 0
    for raw_line in source_lines:
        line = _as_line(raw_line)
        if line is None:
            continue
        # Collapse stabilizer duplicates from this one source path first.  The
        # first transform is its stated orientation, which deliberately keeps
        # the registered root orientation on a symmetric initial board.
        source_entries: dict[IndexKey, Move] = {}
        for transformed in _d4_lines(line):
            state = _state_before_recommendation(transformed)
            if state is None:
                continue
            expected_board, side, history = state
            candidate = transformed[-1]
            if not _final_move_is_legal(expected_board, side, candidate):
                continue
            key = (expected_board, side, history)
            if key not in source_entries:
                source_entries[key] = candidate

        # Different source paths that claim different moves for one exact
        # board/side/history are unsafe, even if one happens to be identity
        # orientation.  Make the shared state unavailable rather than allow
        # transform priority or input order to mask the disagreement.
        for key, candidate in source_entries.items():
            maximum_history_size = max(maximum_history_size, len(key[2]))
            if key not in entries:
                entries[key] = candidate
            elif entries[key] != candidate:
                entries[key] = None

    index = {key: value for key, value in entries.items() if value is not None}
    return MappingProxyType(index), maximum_history_size


def _state_before_recommendation(
    line: tuple[Move, ...],
) -> tuple[BoardKey, bool, HistoryKey] | None:
    board = [list(row) for row in _INITIAL_BOARD]
    is_attacker = True
    history = {compute_hash(board, is_attacker)}
    for move in line[:-1]:
        try:
            if _as_lists(move) not in get_legal_moves(board, is_attacker):
                return None
            board = apply_move(board, _as_lists(move), is_attacker)
            is_attacker = not is_attacker
            state_hash = compute_hash(board, is_attacker)
            next_moves = get_legal_moves(board, is_attacker)
            if check_winner(board, is_attacker, bool(next_moves)) is not None:
                return None
        except Exception:
            return None
        if state_hash in history:
            return None
        history.add(state_hash)
    return _freeze_board(board), is_attacker, frozenset(history)


def _final_move_is_legal(board: BoardKey, is_attacker: bool, move: Move) -> bool:
    try:
        return _as_lists(move) in get_legal_moves(board, is_attacker)
    except Exception:
        return False


def _d4_lines(line: tuple[Move, ...]) -> tuple[tuple[Move, ...], ...]:
    return tuple(
        tuple(_transform_move(move, rotations, reflected) for move in line)
        for rotations in range(4)
        for reflected in (False, True)
    )


def _transform_move(move: Move, rotations: int, reflected: bool) -> Move:
    start_row, start_column = _transform_square(move[0], move[1], rotations, reflected)
    end_row, end_column = _transform_square(move[2], move[3], rotations, reflected)
    return (start_row, start_column, end_row, end_column)


def _transform_square(row: int, column: int, rotations: int, reflected: bool) -> tuple[int, int]:
    for _ in range(rotations):
        row, column = column, _BOARD_EDGE - row
    if reflected:
        column = _BOARD_EDGE - column
    return row, column


def _as_line(raw_line: object) -> tuple[Move, ...] | None:
    if not isinstance(raw_line, (tuple, list)) or not raw_line:
        return None
    moves: list[Move] = []
    for raw_move in raw_line:
        try:
            raw_values = tuple(raw_move)
        except (TypeError, ValueError):
            return None
        if len(raw_values) != 4 or any(isinstance(value, bool) for value in raw_values):
            return None
        try:
            move = tuple(operator.index(value) for value in raw_values)
        except (TypeError, ValueError):
            return None
        moves.append(move)
    return tuple(moves)


def _as_pair(move: object) -> MovePair | None:
    try:
        if len(move) == 4:
            raw_values = tuple(move)
            if any(isinstance(value, bool) for value in raw_values):
                return None
            values = tuple(operator.index(value) for value in raw_values)
            return ((values[0], values[1]), (values[2], values[3]))
        (row, column), (target_row, target_column) = move
        raw_values = (row, column, target_row, target_column)
        if any(isinstance(value, bool) for value in raw_values):
            return None
        values = tuple(operator.index(value) for value in raw_values)
    except (TypeError, ValueError):
        return None
    return ((values[0], values[1]), (values[2], values[3]))


def _is_legal(pair: MovePair, legal_moves: Iterable[Sequence[Sequence[int]]]) -> bool:
    try:
        if pair in legal_moves:
            return True
    except TypeError:
        pass
    try:
        return any(_as_pair(move) == pair for move in legal_moves)
    except TypeError:
        return False


def _as_lists(move: Move) -> list[list[int]]:
    return [[move[0], move[1]], [move[2], move[3]]]


def _freeze_board(board: Sequence[Sequence[str]]) -> BoardKey:
    return tuple(tuple(row) for row in board)


BOOK_INDEX, MAX_HISTORY_SIZE = build_book_index(BOOK_LINES)
