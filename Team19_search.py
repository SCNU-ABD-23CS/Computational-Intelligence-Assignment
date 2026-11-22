"""Alpha-Beta search for Group 19, owned by member 2.

Scores always use the attacker's perspective. The core accepts rule and
evaluation functions so it can be tested before teammate modules are available.
The Player adapter uses iterative deepening; experiments can fix the depth.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import ctypes
from dataclasses import dataclass, field
import math
import sys
import time
from typing import NamedTuple, TYPE_CHECKING

import numpy as np
from numba import njit as _njit

if TYPE_CHECKING:
    from Team19_engine import _SearchPosition


Board = list[list[str]]
Move = list[list[int]]
Weights = list[float] | None

WIN_SCORE = 100_000.0
_MAX_ITERATIVE_DEPTH = 64
_MAX_RETURN_RESERVE = 0.005
_MAX_MOVE_HINTS = 16_384
_TERMINAL_SCORES = {"A": WIN_SCORE, "D": -WIN_SCORE, "DRAW": 0.0}


class SearchTimeout(TimeoutError):
    """A depth was interrupted; its partial result must be discarded."""

    nodes_visited = 0
    cutoffs = 0


@dataclass(frozen=True)
class SearchFunctions:
    """Functions supplied by members 1 and 3, or by a test fixture.

    All callbacks must leave their arguments unchanged. apply_move returns an
    independent board. check_winner accepts the actual availability of moves
    as its third argument. compute_hash, when supplied, includes the side to
    move and enables repetition checks against both history and the path.
    """

    get_legal_moves: Callable[[Board, bool], list[Move]]
    apply_move: Callable[[Board, Move, bool], Board]
    check_winner: Callable[[Board, bool, bool], str | None]
    evaluate_board: Callable[[Board, Weights], float]
    compute_hash: Callable[[Board, bool], int] | None = None


@dataclass(frozen=True)
class SearchResult:
    """A completed fixed-depth result; no move is chosen at depth zero or a terminal.

    nodes_visited includes the root, leaves and repeated positions. cutoffs
    counts bounds that actually skip at least one remaining sibling.
    """

    move: Move | None
    score: float
    nodes_visited: int
    cutoffs: int
    terminal_plies: int | None = None


@dataclass(frozen=True)
class _FastSearch:
    root: _SearchPosition
    apply_move: Callable[[_SearchPosition, Move], _SearchPosition]
    has_legal_move: Callable[[Board, bool], bool]
    evaluate: Callable[[_SearchPosition], float]


def _move_key(move: Move) -> tuple[int, int, int, int]:
    (row, col), (end_row, end_col) = move
    return row, col, end_row, end_col


@dataclass
class _MoveOrdering:
    """Per-turn hints only. A board hash alone is unsafe for caching a score."""

    hints: dict[int, Move] = field(default_factory=dict)
    killers: dict[int, tuple[Move, Move | None]] = field(default_factory=dict)
    attack_history: dict[tuple[int, int, int, int], int] = field(default_factory=dict)
    defence_history: dict[tuple[int, int, int, int], int] = field(default_factory=dict)

    def history(self, is_attacker: bool) -> dict[tuple[int, int, int, int], int]:
        return self.attack_history if is_attacker else self.defence_history

    def record_cutoff(self, move: Move, is_attacker: bool, ply: int, depth: int) -> None:
        first, _ = self.killers.get(ply, (None, None))
        if move != first:
            self.killers[ply] = (move, first)
        history = self.history(is_attacker)
        key = _move_key(move)
        history[key] = history.get(key, 0) + depth * depth

    def record_hint(self, position_hash: int, move: Move) -> None:
        if len(self.hints) < _MAX_MOVE_HINTS or position_hash in self.hints:
            self.hints[position_hash] = move


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise SearchTimeout("Search reached its deadline")


def _capture_targets(
    board: Board, is_attacker: bool, deadline: float | None,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    last = len(board) - 1
    enemies = ("D", "K") if is_attacker else ("A",)
    # Only landings adjacent to an enemy with an occupied square beyond it
    # can capture. The engine decides castle exceptions and multiple captures.
    targets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, row in enumerate(board):
        _check_deadline(deadline)
        for c, piece in enumerate(row):
            if piece not in enemies:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if (
                    0 <= rr <= last and 0 <= cc <= last
                    and 0 <= r - dr <= last and 0 <= c - dc <= last
                    and board[rr][cc] == "." and board[r - dr][c - dc] != "."
                ):
                    targets.setdefault((rr, cc), []).append((r, c))
    return targets


def _king_advance(
    board: Board, r: int, c: int, rr: int, cc: int,
) -> tuple[bool, int]:
    if board[r][c] != "K":
        return False, 0
    last = len(board) - 1
    distance = min(rr, cc, last - rr, last - cc)
    return distance == 0, max(0, min(r, c, last - r, last - c) - distance)


def _tactical_order(
    board: Board, moves: list[Move], is_attacker: bool,
    functions: SearchFunctions, deadline: float | None,
) -> list[Move]:
    """Order legal Tablut moves stably, using the engine to confirm captures."""
    targets = _capture_targets(board, is_attacker, deadline)

    def priority(move: Move) -> tuple[bool, int, int]:
        _check_deadline(deadline)
        (r, c), (rr, cc) = move
        escape, advance = _king_advance(board, r, c, rr, cc)
        captures = 0
        wins = escape
        if neighbours := targets.get((rr, cc)):
            child = functions.apply_move(board, move, is_attacker)
            for nr, nc in neighbours:
                if child[nr][nc] == ".":
                    captures += 1
                    wins = wins or board[nr][nc] == "K"
        return wins, captures, advance

    return sorted(moves, key=priority, reverse=True)


def _order_legal_moves(
    position: _SearchPosition, moves: list[Move], is_attacker: bool,
    fast: _FastSearch, deadline: float | None,
    ordering: _MoveOrdering | None = None, ply: int = 0,
) -> tuple[list[Move], dict[int, _SearchPosition]]:
    """Keep the children already made while resolving possible captures."""
    targets = _capture_targets(position.board, is_attacker, deadline)
    children: dict[int, _SearchPosition] = {}
    winning_side = "A" if is_attacker else "D"
    killers = ordering.killers.get(ply, ()) if ordering is not None else ()
    history = ordering.history(is_attacker) if ordering is not None else None

    def priority(move: Move) -> tuple[bool, int, int, bool, int]:
        _check_deadline(deadline)
        (r, c), (rr, cc) = move
        wins, advance = _king_advance(position.board, r, c, rr, cc)
        captures = 0
        if (rr, cc) in targets:
            child = fast.apply_move(position, move)
            children[id(move)] = child
            captures = (position.attackers + position.defenders
                        - child.attackers - child.defenders + (child.king is None))
            wins = wins or child.winner() == winning_side
        return wins, captures, advance, move in killers, history.get(_move_key(move), 0) if history else 0

    return sorted(moves, key=priority, reverse=True), children


def _ordered_children(
    state: _SearchPosition, moves: list[Move], is_attacker: bool, position_hash: int,
    fast: _FastSearch, deadline: float | None, ordering: _MoveOrdering | None, ply: int,
) -> Iterator[tuple[Move, _SearchPosition | None]]:
    if ordering is not None:
        hint = ordering.hints.get(position_hash)
        if hint is not None and hint in moves:
            # A good hint can cut off before we have to order the other moves.
            yield hint, None
            moves.remove(hint)
    ranked, children = _order_legal_moves(state, moves, is_attacker, fast, deadline, ordering, ply)
    for move in ranked:
        yield move, children.get(id(move))


def search_fixed_depth(
    board: Board,
    is_attacker: bool,
    depth: int,
    *,
    functions: SearchFunctions,
    weights: Weights = None,
    deadline: float | None = None,
    game_history: set[int] | None = None,
    tactical_ordering: bool = False,
) -> SearchResult:
    """Search at most depth individual moves with an initially unbounded window.

    Attackers maximize; defenders minimize. Wins score +/- WIN_SCORE and draws
    score zero. Other leaves use evaluate_board(board, weights); its finite
    heuristic scores should lie strictly between the two win scores.

    Terminal and no-move checks precede horizon evaluation. Repetition checks
    also precede evaluation when a hasher is supplied. The current root may
    already occur in game_history without being treated as a new repetition.

    No board, weight list or real-game history is modified by the search.
    A deadline is an absolute time.perf_counter() timestamp. An interruption
    raises SearchTimeout rather than returning a partially searched root.
    Callbacks are synchronous, so their running time cannot be preempted.
    Engine order is the baseline. Set tactical_ordering=True for Tablut ordering.
    """
    if type(depth) is not int or depth < 0:
        raise ValueError("depth must be a non-negative integer")
    result, _ = _search_depth(
        board, is_attacker, depth, functions, weights, deadline, game_history,
        tactical_ordering=tactical_ordering,
    )
    return result


def _search_depth(
    board: Board,
    is_attacker: bool,
    depth: int,
    functions: SearchFunctions,
    weights: Weights,
    deadline: float | None,
    game_history: set[int] | None,
    *,
    playable_root_moves: list[Move] | None = None,
    tactical_ordering: bool = False,
    fast: _FastSearch | None = None,
    root_children: dict[int, _SearchPosition] | None = None,
    ordering: _MoveOrdering | None = None,
) -> tuple[SearchResult, bool]:
    """Return the depth result and whether any scored leaf used the heuristic."""
    if deadline is not None and math.isnan(deadline):
        raise ValueError("deadline must not be NaN")
    if game_history and functions.compute_hash is None:
        raise ValueError("compute_hash is required when game_history is non-empty")

    history = game_history if game_history is not None else set()
    search_path: set[int] = set()
    nodes_visited = 0
    cutoffs = 0
    evaluated_horizon = False

    def visit(
        position: Board,
        attackers_turn: bool,
        remaining: int,
        alpha: float,
        beta: float,
        state: _SearchPosition | None = None,
    ) -> tuple[float, Move | None]:
        nonlocal nodes_visited, cutoffs, evaluated_horizon
        _check_deadline(deadline)
        nodes_visited += 1
        ply = depth - remaining
        is_root = ply == 0
        position_hash = None
        moves = None
        if is_root and playable_root_moves is not None:
            moves = playable_root_moves
        elif state is not None:
            winner = state.winner()
            if winner == "A":
                return WIN_SCORE - ply, None
            if winner == "D":
                return -WIN_SCORE + ply, None
            if remaining:
                moves = functions.get_legal_moves(position, attackers_turn)
                if not moves:
                    return 0.0, None
            elif not fast.has_legal_move(position, attackers_turn):
                return 0.0, None
        else:
            moves = functions.get_legal_moves(position, attackers_turn)
            _check_deadline(deadline)
            winner = functions.check_winner(position, attackers_turn, bool(moves))
            _check_deadline(deadline)
            if winner is not None:
                if winner not in _TERMINAL_SCORES:
                    raise ValueError(f"Unexpected winner marker: {winner!r}")
                return _TERMINAL_SCORES[winner], None
            if not moves:
                return 0.0, None

        if state is not None:
            position_hash = state.attacker_hash if attackers_turn else state.defender_hash
        elif functions.compute_hash is not None:
            position_hash = functions.compute_hash(position, attackers_turn)
            _check_deadline(deadline)
        if position_hash is not None:
            if not is_root and (
                position_hash in history or position_hash in search_path
            ):
                return 0.0, None
            search_path.add(position_hash)

        try:
            if remaining == 0:
                if state is not None:
                    score = fast.evaluate(state)
                else:
                    _check_deadline(deadline)
                    score = float(functions.evaluate_board(position, weights))
                    if not math.isfinite(score):
                        raise ValueError("evaluate_board must return a finite score")
                _check_deadline(deadline)
                evaluated_horizon = True
                return score, None

            best_score = -math.inf if attackers_turn else math.inf
            best_move = None
            children = root_children if is_root else None
            should_order = tactical_ordering and not (is_root and playable_root_moves is not None)
            move_count = len(moves)
            if state is not None and should_order:
                candidates = _ordered_children(
                    state, moves, attackers_turn, position_hash, fast, deadline, ordering, ply,
                )
            else:
                if should_order:
                    moves = _tactical_order(position, moves, attackers_turn, functions, deadline)
                candidates = ((move, children.get(id(move)) if children else None) for move in moves)
            best_possible = (WIN_SCORE - ply - 1) * (1 if attackers_turn else -1)
            for index, (move, child_state) in enumerate(candidates):
                _check_deadline(deadline)
                if state is None:
                    child = functions.apply_move(position, move, attackers_turn)
                else:
                    if child_state is None:
                        child_state = fast.apply_move(state, move)
                    child = child_state.board
                score, _ = visit(child, not attackers_turn, remaining - 1, alpha, beta, child_state)
                # Keep the first tie: a pruned child may return only a bound.
                if best_move is None or (
                    score > best_score if attackers_turn else score < best_score
                ):
                    best_score = score
                    best_move = move
                if attackers_turn:
                    alpha = max(alpha, best_score)
                else:
                    beta = min(beta, best_score)
                if alpha >= beta or (state is not None and best_score == best_possible):
                    if index + 1 < move_count:
                        cutoffs += 1
                        if (ordering is not None and score != best_possible
                                and child_state.attackers == state.attackers
                                and child_state.defenders == state.defenders
                                and child_state.king is not None):
                            ordering.record_cutoff(move, attackers_turn, ply, remaining)
                    break
            if ordering is not None:
                ordering.record_hint(position_hash, best_move)
            return best_score, best_move
        finally:
            if position_hash is not None:
                search_path.remove(position_hash)

    try:
        score, move = visit(board, is_attacker, depth, -math.inf, math.inf,
                            fast.root if fast is not None else None)
        _check_deadline(deadline)
    except SearchTimeout as error:
        error.nodes_visited = nodes_visited
        error.cutoffs = cutoffs
        raise
    terminal_plies = None
    if fast is not None and abs(score) >= WIN_SCORE - depth:
        terminal_plies = int(WIN_SCORE - abs(score))
        score = math.copysign(WIN_SCORE, score)
    return SearchResult(move, score, nodes_visited, cutoffs, terminal_plies), evaluated_horizon


def _load_functions() -> SearchFunctions:
    """Import real teammate modules only when the Player adapter is called."""
    from Team19_engine import apply_move, check_winner, compute_hash, get_legal_moves
    from Team19_eval import evaluate_board

    return SearchFunctions(
        get_legal_moves=get_legal_moves,
        apply_move=apply_move,
        check_winner=check_winner,
        evaluate_board=evaluate_board,
        compute_hash=compute_hash,
    )


def _prepare_fast_search(
    board: Board, functions: SearchFunctions, weights: Weights,
) -> _FastSearch | None:
    # Synthetic SearchFunctions can use any board, hash or scoring convention.
    import Team19_engine as engine
    import Team19_eval as evaluation

    if not hasattr(engine, "_search_position") or not hasattr(evaluation, "_features_from_counts"):
        return None
    if (functions.get_legal_moves is not engine.get_legal_moves
            or functions.apply_move is not engine.apply_move
            or functions.check_winner is not engine.check_winner
            or functions.compute_hash is not engine.compute_hash
            or functions.evaluate_board is not evaluation.evaluate_board):
        return None
    active = tuple(evaluation.normalize_weights(
        evaluation.DEFAULT_WEIGHTS if weights is None else weights,
    ))
    extract = evaluation._features_from_counts
    weighted_score = evaluation._weighted_score

    def evaluate(state: _SearchPosition) -> float:
        return weighted_score(active, extract(state.board, state.attackers, state.defenders, state.king))

    return _FastSearch(engine._search_position(board), engine._apply_legal_move,
                       engine._has_legal_move, evaluate)


def choose_best_move(
    board: Board,
    is_attacker: bool,
    deadline: float,
    game_history: set[int],
    weights: Weights = None,
) -> Move:
    """Return the move from the deepest completed Alpha-Beta iteration.

    Requires the real Team19_engine and Team19_eval modules and a nonterminal
    board with legal moves. The same absolute deadline applies to every depth.
    If depth one cannot finish, return the first legal move. Search also stops
    when it proves a win for the side to move, when no heuristic leaves were
    needed, or at the recursion guard.
    """
    # Leave time to unwind a partial iteration and hand the move back to Player.
    available = max(0.0, deadline - time.perf_counter())
    deadline -= min(_MAX_RETURN_RESERVE, available * 0.10)
    functions = _load_functions()
    legal_moves = functions.get_legal_moves(board, is_attacker)
    winner = functions.check_winner(board, is_attacker, bool(legal_moves))
    if winner is not None or not legal_moves:
        raise ValueError("choose_best_move requires a nonterminal board with legal moves")
    fast = _prepare_fast_search(board, functions, weights)
    ordering = _MoveOrdering() if fast is not None else None
    best_move = legal_moves[0]
    root_children = None
    try:
        if fast is None:
            ordered_moves = _tactical_order(board, legal_moves, is_attacker, functions, deadline)
        else:
            ordered_moves, root_children = _order_legal_moves(
                fast.root, legal_moves, is_attacker, fast, deadline,
            )
    except SearchTimeout:
        return [coordinate[:] for coordinate in best_move]
    root_order = ordered_moves
    for depth in range(1, _MAX_ITERATIVE_DEPTH + 1):
        try:
            result, evaluated_horizon = _search_depth(
                board, is_attacker, depth, functions, weights, deadline, game_history,
                playable_root_moves=root_order, tactical_ordering=True,
                fast=fast, root_children=root_children, ordering=ordering,
            )
        except SearchTimeout:
            break
        if result.move is None:
            raise RuntimeError("A completed search on a playable root did not choose a move")
        best_move = result.move
        won = result.terminal_plies is not None and result.score == (
            WIN_SCORE if is_attacker else -WIN_SCORE
        )
        if won or not evaluated_horizon:
            break
        if best_move != root_order[0]:
            root_order = [best_move] + [move for move in ordered_moves if move != best_move]
    return [coordinate[:] for coordinate in best_move]


# The public functions above keep their original callback and depth contracts.
# The Player adapter owns a separate compiled workspace for real matches.
try:
    from Team19_engine import (
        _A, _AREA, _BEYOND, _CASTLE, _D, _EDGE, _K, _KING_CASTLE, _MOVE_CAPACITY, _NEAR, _X,
        _flat, _has_legal, _immediate_win, _legal, _move_list, _pop, _push, _search_position, _winner,
    )
    from Team19_eval import (
        DEFAULT_WEIGHTS as _PYTHON_DEFAULT_WEIGHTS,
        _NATIVE_DISTANCE, _NATIVE_EDGE_DISTANCE, _native_evaluate, _native_piece_bonus,
    )
except ImportError:
    _NATIVE_AVAILABLE = False
else:
    _NATIVE_AVAILABLE = True


_MATE = int(WIN_SCORE)
_Q_LIMIT = 4
_NATIVE_PLY_SLOTS = _MAX_ITERATIVE_DEPTH + _Q_LIMIT + 2
_NATIVE_TT_SIZE = 1 << 18
_NODE_COUNT, _CUTOFFS, _REDUCTIONS, _RESEARCHES, _STOPPED, _EVALUATIONS = range(6)


# Read the OS monotonic counter from compiled search, without objmode or a
# per-node transition back to Python. Deadlines are transferred as durations.
if sys.platform == "win32":
    _kernel = ctypes.WinDLL("kernel32")
    _query_counter = _kernel.QueryPerformanceCounter
    _query_counter.argtypes = (ctypes.POINTER(ctypes.c_int64),)
    _query_counter.restype = ctypes.c_int
    _frequency = ctypes.c_int64()
    if not _kernel.QueryPerformanceFrequency(ctypes.byref(_frequency)):
        raise OSError("Cannot read the performance counter frequency")
    _COUNTER_SCALE = 1.0 / _frequency.value
    _CLOCK_DTYPE = np.int64

    def _read_native_clock(buffer):
        _query_counter(buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
        return buffer[0] * _COUNTER_SCALE

    @_njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
    def _native_now(buffer):
        _query_counter(buffer.ctypes)
        return buffer[0] * _COUNTER_SCALE
else:
    _libc = ctypes.CDLL(None)
    _clock_gettime = _libc.clock_gettime
    _clock_gettime.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_long))
    _clock_gettime.restype = ctypes.c_int
    _CLOCK_MONOTONIC = time.CLOCK_MONOTONIC
    _CLOCK_DTYPE = np.dtype(ctypes.c_long)

    def _read_native_clock(buffer):
        _clock_gettime(_CLOCK_MONOTONIC, buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_long)))
        return float(buffer[0]) + float(buffer[1]) * 1e-9

    @_njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
    def _native_now(buffer):
        _clock_gettime(_CLOCK_MONOTONIC, buffer.ctypes)
        return buffer[0] + buffer[1] * 1e-9


class _NativeWork(NamedTuple):
    moves: np.ndarray
    priorities: np.ndarray
    captured_at: np.ndarray
    captured_pieces: np.ndarray
    path: np.ndarray
    tt_keys: np.ndarray
    tt_moves: np.ndarray
    tt_generations: np.ndarray
    generation: np.ndarray
    killers: np.ndarray
    history: np.ndarray
    queue: np.ndarray
    seen: np.ndarray
    scratch: np.ndarray
    clock: np.ndarray
    counts: np.ndarray
    weights: np.ndarray
    piece_bonus: np.ndarray


def _native_work(weights) -> _NativeWork:
    return _NativeWork(
        np.empty((_NATIVE_PLY_SLOTS, _MOVE_CAPACITY), dtype=np.int16),
        np.empty((_NATIVE_PLY_SLOTS, _MOVE_CAPACITY), dtype=np.int32),
        np.empty((_NATIVE_PLY_SLOTS, 4), dtype=np.int16),
        np.empty((_NATIVE_PLY_SLOTS, 4), dtype=np.int8),
        np.empty(_NATIVE_PLY_SLOTS, dtype=np.uint64),
        np.zeros(_NATIVE_TT_SIZE, dtype=np.uint64),
        np.full(_NATIVE_TT_SIZE, -1, dtype=np.int16),
        np.zeros(_NATIVE_TT_SIZE, dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.full((_NATIVE_PLY_SLOTS, 2), -1, dtype=np.int16),
        np.zeros((2, _AREA * _AREA), dtype=np.int32),
        np.empty(_AREA, dtype=np.int16),
        np.empty(_AREA, dtype=np.bool_),
        np.empty(_MOVE_CAPACITY, dtype=np.int16),
        np.empty(2, dtype=_CLOCK_DTYPE),
        np.zeros(6, dtype=np.int64),
        weights,
        _native_piece_bonus(weights),
    )


@_njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
def _native_capture_priority(board, source, target, attacker):
    priority = 0
    for direction in range(4):
        far = _BEYOND[target, direction]
        if far < 0:
            continue
        victim = board[_NEAR[target, direction]]
        ally = board[far] if far != source else _X if source == _CASTLE else 0
        if attacker:
            if victim == _K:
                if ally == _A or _KING_CASTLE[_NEAR[target, direction]]:
                    priority += 12000
            elif victim == _D and (ally == _A or ally == _X or ally == _K and far == _CASTLE):
                priority += 6500
        elif victim == _A and (ally == _D or ally == _K or ally == _X):
            priority += 6000
    return priority


@_njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
def _native_seen(played, path, stamp, ply):
    low, high = 0, len(played)
    while low < high:
        mid = (low + high) // 2
        if played[mid] < stamp:
            low = mid + 1
        else:
            high = mid
    if low < len(played) and played[low] == stamp:
        return True
    for earlier in range(ply):
        if path[earlier] == stamp:
            return True
    return False


@_njit(inline="always")
def _native_tt_hint(stamp, work):
    slot = int(stamp & np.uint64(_NATIVE_TT_SIZE - 1))
    if work.tt_generations[slot] >= work.generation[1] and work.tt_keys[slot] == stamp:
        return int(work.tt_moves[slot])
    return -1


@_njit(inline="always")
def _native_tt_store(stamp, move, work):
    slot = int(stamp & np.uint64(_NATIVE_TT_SIZE - 1))
    work.tt_keys[slot] = stamp
    work.tt_moves[slot] = move
    work.tt_generations[slot] = work.generation[0]


@_njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
def _native_order(board, attacker, king, ply, normal, danger, hint, work):
    moves, priorities = work.moves[ply], work.priorities[ply]
    count = _legal(board, attacker, moves)
    side = 0 if attacker else 1
    useful = 0
    for index in range(count):
        move = moves[index]
        source, target = divmod(move, _AREA)
        tactical = _native_capture_priority(board, source, target, attacker)
        king_contact = attacker and _NATIVE_DISTANCE[king, target] == 1
        if not normal and not danger and tactical == 0 and not king_contact:
            continue
        priority = tactical
        if board[source] == _K:
            priority += 220 * (_NATIVE_EDGE_DISTANCE[source] - _NATIVE_EDGE_DISTANCE[target])
        if normal:
            if move == hint:
                priority += 2_000_000
            elif move == work.killers[ply, 0]:
                priority += 5000
            elif move == work.killers[ply, 1]:
                priority += 4000
            priority += min(3500, work.history[side, move])
        moves[useful], priorities[useful] = move, priority
        useful += 1
    return useful


@_njit(inline="always")
def _native_score(board, attacker, attackers, defenders, king, work):
    work.counts[_EVALUATIONS] += 1
    value = _native_evaluate(board, attacker, attackers, defenders, king, work.weights,
                             work.queue, work.seen, _NEAR, _EDGE, work.piece_bonus)
    return value if attacker else -value


# The caller owns every array, including the recursive frames. The kernel only
# returns scalars, so no Numba-managed allocation is needed during recursion.
@_njit(no_cfunc_wrapper=True, _nrt=False)
def _native_visit(board, attacker, hash_a, hash_d, attackers, defenders, king,
                  depth, qleft, ply, alpha, beta, deadline, played, work, lmr):
    counts = work.counts
    counts[_NODE_COUNT] += 1
    if counts[_NODE_COUNT] & 511 == 0 and _native_now(work.clock) >= deadline:
        counts[_STOPPED] = 1
    if counts[_STOPPED]:
        return 0, -1

    winner = _winner(attackers, king)
    if winner:
        value = _MATE - ply
        return (value if (winner == _A) == attacker else -value), -1
    stamp = hash_a if attacker else hash_d
    if ply and _native_seen(played, work.path, stamp, ply):
        return 0, -1

    normal = depth > 0
    if normal or qleft or ply == 1:
        winning_move = _immediate_win(board, attacker, attackers, king, work.scratch)
        if winning_move >= 0:
            return _MATE - ply - 1, int(winning_move)
    if not normal:
        if not _has_legal(board, attacker):
            return 0, -1
        if qleft == 0:
            return _native_score(board, attacker, attackers, defenders, king, work), -1

    best, best_move = -2 * _MATE, -1
    can_reduce = lmr and depth >= 3 and ply > 0 and beta == alpha + 1
    danger = False
    if not normal or can_reduce:
        danger = _immediate_win(board, not attacker, attackers, king, work.scratch) >= 0
    if not normal and not danger:
        best = _native_score(board, attacker, attackers, defenders, king, work)
        if best >= beta:
            counts[_CUTOFFS] += 1
            return best, -1
        alpha = max(alpha, best)

    work.path[ply] = stamp
    hint = _native_tt_hint(stamp, work) if normal else -1
    useful = _native_order(board, attacker, king, ply, normal, danger, hint, work)
    if normal and useful == 0:
        return 0, -1
    moves, priorities = work.moves[ply], work.priorities[ply]
    captured_at, captured_pieces = work.captured_at[ply], work.captured_pieces[ply]
    child_depth = max(0, depth - 1)
    child_qleft = qleft if normal else qleft - 1
    first = True
    for index in range(useful):
        rank = index
        for sibling in range(index + 1, useful):
            if priorities[sibling] > priorities[rank]:
                rank = sibling
        moves[index], moves[rank] = moves[rank], moves[index]
        priorities[index], priorities[rank] = priorities[rank], priorities[index]
        move = np.int64(moves[index])
        source, target = divmod(move, _AREA)
        ha, hd, a, d, k, captured = _push(board, move, hash_a, hash_d, attackers, defenders,
                                           king, captured_at, captured_pieces)
        if not normal and not danger and captured == 0:
            if _immediate_win(board, attacker, a, k, work.scratch) < 0:
                _pop(board, move, captured_at, captured_pieces, captured)
                continue

        trial_depth = child_depth
        if (can_reduce and index >= 4 and not danger and captured == 0 and k == king
                and move != hint and move != work.killers[ply, 0] and move != work.killers[ply, 1]
                and _NATIVE_DISTANCE[king, source] > 2 and _NATIVE_DISTANCE[king, target] > 2
                and abs(alpha) < _MATE - _NATIVE_PLY_SLOTS
                and _immediate_win(board, attacker, a, k, work.scratch) < 0):
            counts[_REDUCTIONS] += 1
            trial_depth -= 1

        narrow = not first
        while True:
            child, _ = _native_visit(
                board, not attacker, ha, hd, a, d, k, trial_depth, child_qleft, ply + 1,
                -alpha - 1 if narrow else -beta, -alpha, deadline, played, work, lmr,
            )
            if counts[_STOPPED]:
                break
            if trial_depth < child_depth and -child > alpha:
                counts[_RESEARCHES] += 1
                trial_depth = child_depth
            elif narrow and alpha < -child < beta:
                narrow = False
            else:
                break
        first = False
        _pop(board, move, captured_at, captured_pieces, captured)
        if counts[_STOPPED]:
            return 0, -1
        value = -child
        if value > best:
            best, best_move = value, move
        alpha = max(alpha, best)
        if alpha >= beta or best == _MATE - ply - 1:
            counts[_CUTOFFS] += index + 1 < useful
            if normal and captured == 0 and best != _MATE - ply - 1:
                if move != work.killers[ply, 0]:
                    work.killers[ply, 1] = work.killers[ply, 0]
                    work.killers[ply, 0] = move
                side = 0 if attacker else 1
                work.history[side, move] += depth * depth
            break
    if normal and best_move >= 0:
        _native_tt_store(stamp, best_move, work)
    return best, best_move


def _native_position(board):
    # The Python path already has a compact root scan. Use it on the input
    # board so no separate standalone JIT dispatch is needed here each turn.
    root = _search_position(board)
    king = -1 if root.king is None else root.king[0] * 9 + root.king[1]
    return _flat(board), (np.uint64(root.attacker_hash), np.uint64(root.defender_hash),
                          root.attackers, root.defenders, king)


@dataclass(frozen=True)
class _NativeTelemetry:
    depth: int
    nodes: int
    score: float | None
    fallback: bool
    reductions: int = 0
    reduced_researches: int = 0


class _NativeSearch:
    """A single player's per-game hints and independently timed search."""

    def __init__(self, initial_board, weights=None):
        if not _NATIVE_AVAILABLE:
            raise ImportError("The compiled rules or evaluator are unavailable")
        from Team19_weights import native_weights

        self.work = _native_work(native_weights(weights))
        self.legacy_weights = tuple(_PYTHON_DEFAULT_WEIGHTS)
        self.last_search = None
        self.iterations = []
        self.interrupted_depth = self.interrupted_nodes = None
        started = time.perf_counter()
        self.warmup(initial_board)
        self.warmup_seconds = time.perf_counter() - started
        self.reset_game()

    def warmup(self, board):
        flat, state = _native_position(board)
        # The large recursive kernel has one signature for both sides, every
        # depth, and all search modes. Use it without a timed cutoff here.
        self.work.generation[0] += 1
        self._iteration(flat, True, state, 1, 0, -2 * _MATE, 2 * _MATE,
                        math.inf, np.empty(0, dtype=np.uint64), False)

    def reset_game(self):
        self.work.generation[1] = self.work.generation[0] + 1
        self.reset_trace()

    def reset_trace(self):
        self.last_search = None
        self.iterations.clear()
        self.interrupted_depth = self.interrupted_nodes = None

    def accepts(self, weights):
        return weights is None or tuple(weights) == self.legacy_weights

    def _iteration(self, flat, side, state, depth, qleft, alpha, beta, deadline, played, lmr):
        return _native_visit(flat, side, *state, depth, qleft, 0, alpha, beta,
                             deadline, played, self.work, lmr)

    def choose(self, board: Board, is_attacker: bool, deadline: float,
               game_history: set[int], weights: Weights = None) -> Move:
        self.reset_trace()
        if not self.accepts(weights):
            return choose_best_move(board, is_attacker, deadline, game_history, weights)
        work = self.work
        flat, state = _native_position(board)
        played = np.asarray(sorted(game_history), dtype=np.uint64)
        work.generation[0] += 1
        work.killers.fill(-1)
        work.history.fill(0)
        work.counts.fill(0)
        now = time.perf_counter()
        available = max(0.0, deadline - now)
        search_time = available - min(0.004, available * 0.10)
        cutoff = _read_native_clock(work.clock) + search_time
        soft_deadline = now + search_time

        best, score = -1, 0
        for depth in range(1, _MAX_ITERATIVE_DEPTH + 1):
            if depth > 1 and time.perf_counter() >= soft_deadline:
                break
            started = time.perf_counter()
            nodes, cutoffs = int(work.counts[_NODE_COUNT]), int(work.counts[_CUTOFFS])
            evaluated = int(work.counts[_EVALUATIONS])
            reduced = int(work.counts[_REDUCTIONS])
            # The first complete pass is bounded to one full move and exact
            # immediate threats. It runs even if the deeper-search time is zero.
            first = depth == 1
            qleft = 0 if first else _Q_LIMIT
            stop_at = math.inf if first else cutoff
            alpha, beta = (-2 * _MATE, 2 * _MATE) if first else (score - 850, score + 850)
            value, move = self._iteration(
                flat, is_attacker, state, depth, qleft, alpha, beta, stop_at, played, True,
            )
            if not first and not work.counts[_STOPPED] and (value <= alpha or value >= beta):
                value, move = self._iteration(
                    flat, is_attacker, state, depth, qleft, -2 * _MATE, 2 * _MATE,
                    stop_at, played, True,
                )
            # Selectivity may hide a later defence. Certify mate scores with
            # the normal depth in every branch before the early-exit decision.
            if (not work.counts[_STOPPED] and abs(value) >= _MATE - _NATIVE_PLY_SLOTS
                    and work.counts[_REDUCTIONS] > reduced):
                value, move = self._iteration(
                    flat, is_attacker, state, depth, qleft, -2 * _MATE, 2 * _MATE,
                    stop_at, played, False,
                )
            if work.counts[_STOPPED]:
                self.interrupted_depth = depth
                self.interrupted_nodes = int(work.counts[_NODE_COUNT]) - nodes
                break
            if move < 0:
                if first:
                    raise ValueError("Native search requires a nonterminal board with legal moves")
                raise RuntimeError("A completed native depth did not produce a move")

            score, best = value, move
            terminal = _MATE - abs(value) if abs(value) >= _MATE - _NATIVE_PLY_SLOTS else None
            display = math.copysign(WIN_SCORE, value) if terminal is not None else float(value)
            display = display if is_attacker else -display
            result = SearchResult(_move_list(move), display,
                                  int(work.counts[_NODE_COUNT]) - nodes,
                                  int(work.counts[_CUTOFFS]) - cutoffs, terminal)
            self.iterations.append((depth, result, time.perf_counter() - started,
                                    bool(work.counts[_EVALUATIONS] > evaluated)))
            if terminal is not None:
                break

        depth, result, _, _ = self.iterations[-1]
        self.last_search = _NativeTelemetry(
            depth, int(work.counts[_NODE_COUNT]), result.score, False,
            int(work.counts[_REDUCTIONS]), int(work.counts[_RESEARCHES]),
        )
        return _move_list(best)
