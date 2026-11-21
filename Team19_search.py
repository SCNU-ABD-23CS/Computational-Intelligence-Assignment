"""Alpha-Beta search for Group 19, owned by member 2.

Scores always use the attacker's perspective. The core accepts rule and
evaluation functions so it can be tested before teammate modules are available.
The Player adapter uses iterative deepening; experiments can fix the depth.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import math
import time
from typing import TYPE_CHECKING

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
