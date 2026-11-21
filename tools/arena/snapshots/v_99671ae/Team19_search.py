"""Alpha-Beta search for Group 19, owned by member 2.

Scores always use the attacker's perspective. The core accepts rule and
evaluation functions so it can be tested before teammate modules are available.
The Player adapter uses iterative deepening; experiments can fix the depth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import time


Board = list[list[str]]
Move = list[list[int]]
Weights = list[float] | None

WIN_SCORE = 100_000.0
_MAX_ITERATIVE_DEPTH = 64
_MAX_RETURN_RESERVE = 0.005
_TERMINAL_SCORES = {"A": WIN_SCORE, "D": -WIN_SCORE, "DRAW": 0.0}


class SearchTimeout(TimeoutError):
    """A depth was interrupted; its partial result must be discarded."""


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


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() >= deadline:
        raise SearchTimeout("Search reached its deadline")


def _tactical_order(
    board: Board, moves: list[Move], is_attacker: bool,
    functions: SearchFunctions, deadline: float | None,
) -> list[Move]:
    """Order legal Tablut moves stably, using the engine to confirm captures."""
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

    def priority(move: Move) -> tuple[bool, int, int]:
        _check_deadline(deadline)
        (r, c), (rr, cc) = move
        escape = False
        advance = 0
        if board[r][c] == "K":
            distance = min(rr, cc, last - rr, last - cc)
            escape = distance == 0
            advance = max(0, min(r, c, last - r, last - c) - distance)
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
        *,
        is_root: bool = False,
    ) -> tuple[float, Move | None]:
        nonlocal nodes_visited, cutoffs, evaluated_horizon
        _check_deadline(deadline)
        nodes_visited += 1
        if is_root and playable_root_moves is not None:
            moves = playable_root_moves
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

        position_hash = None
        if functions.compute_hash is not None:
            position_hash = functions.compute_hash(position, attackers_turn)
            _check_deadline(deadline)
            if not is_root and (
                position_hash in history or position_hash in search_path
            ):
                return 0.0, None
            search_path.add(position_hash)

        try:
            if remaining == 0:
                _check_deadline(deadline)
                score = float(functions.evaluate_board(position, weights))
                _check_deadline(deadline)
                if not math.isfinite(score):
                    raise ValueError("evaluate_board must return a finite score")
                evaluated_horizon = True
                return score, None

            best_score = -math.inf if attackers_turn else math.inf
            best_move = None
            if tactical_ordering and not (is_root and playable_root_moves is not None):
                moves = _tactical_order(position, moves, attackers_turn, functions, deadline)
            for index, move in enumerate(moves):
                _check_deadline(deadline)
                child = functions.apply_move(position, move, attackers_turn)
                score, _ = visit(child, not attackers_turn, remaining - 1, alpha, beta)
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
                if alpha >= beta and index + 1 < len(moves):
                    cutoffs += 1
                    break
            return best_score, best_move
        finally:
            if position_hash is not None:
                search_path.remove(position_hash)

    score, move = visit(board, is_attacker, depth, -math.inf, math.inf, is_root=True)
    _check_deadline(deadline)
    return SearchResult(move, score, nodes_visited, cutoffs), evaluated_horizon


def _load_functions() -> SearchFunctions:
    """Import real teammate modules only when the Player adapter is called."""
    from .Team19_engine import apply_move, check_winner, compute_hash, get_legal_moves
    from .Team19_eval import evaluate_board

    return SearchFunctions(
        get_legal_moves=get_legal_moves,
        apply_move=apply_move,
        check_winner=check_winner,
        evaluate_board=evaluate_board,
        compute_hash=compute_hash,
    )


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
    when the result is proven without heuristic leaves or at the recursion guard.
    """
    # Leave time to unwind a partial iteration and hand the move back to Player.
    available = max(0.0, deadline - time.perf_counter())
    deadline -= min(_MAX_RETURN_RESERVE, available * 0.10)
    functions = _load_functions()
    legal_moves = functions.get_legal_moves(board, is_attacker)
    winner = functions.check_winner(board, is_attacker, bool(legal_moves))
    if winner is not None or not legal_moves:
        raise ValueError("choose_best_move requires a nonterminal board with legal moves")
    best_move = legal_moves[0]
    try:
        ordered_moves = _tactical_order(board, legal_moves, is_attacker, functions, deadline)
    except SearchTimeout:
        return [coordinate[:] for coordinate in best_move]
    root_order = ordered_moves
    for depth in range(1, _MAX_ITERATIVE_DEPTH + 1):
        try:
            result, evaluated_horizon = _search_depth(
                board, is_attacker, depth, functions, weights, deadline, game_history,
                playable_root_moves=root_order, tactical_ordering=True,
            )
        except SearchTimeout:
            break
        if result.move is None:
            raise RuntimeError("A completed search on a playable root did not choose a move")
        best_move = result.move
        if not evaluated_horizon:
            break
        if best_move != root_order[0]:
            root_order = [best_move] + [move for move in ordered_moves if move != best_move]
    return [coordinate[:] for coordinate in best_move]
