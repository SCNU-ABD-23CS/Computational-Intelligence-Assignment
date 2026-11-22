"""Player adapter for Group 19, owned by member 4.

The referee in Tablut.py only ever calls play_attacker and play_defender. This
module owns everything around the search module: the time budget of a whole
game, the real game history, and the guarantee that a legal move is returned
even when a teammate module is missing or raises an exception.

Two facts from Tablut.py drive the design:

* the budget is per game, not per move, so every call must ask for a share of
  what is still left instead of a fixed thinking time;
* past_positions is seeded with the hash of the initial position and then
  extended once per move, always keyed by the side to move next. The same
  positions are recorded here so that repetition draws line up with the
  referee.

Missing modules are tolerated on purpose: the first week integration can run
end to end before the engine, evaluator and search are all present, and a
late integration mistake costs weaker moves rather than a forfeit.
"""

from __future__ import annotations

import math
import operator
import time

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

# The referee may shorten TIME_BUDGET when it tests all the bots.
_DEFAULT_TIME_BUDGET = 60.0
# Never handed to the search; covers building and returning the move.
_RETURN_RESERVE = 0.05
# Skip search very close to the bookkeeping reserve. Smaller allocated shares
# are still useful when the game clock can afford them.
_MIN_AVAILABLE_TIME = 0.02
_SHORT_SEARCH_TIME = 0.008
_MIN_MOVES_LEFT = 12
_MOVES_PER_PIECE = 1.5
_OPENING_MOVE_ALLOWANCE = 13

Square = tuple[int, int]
MovePair = tuple[Square, Square]


def _import_member(module_name: str, attribute: str):
    """Return one callable from a teammate module, or None if it is absent."""
    try:
        module = __import__(module_name, fromlist=[attribute])
    except Exception:
        return None
    member = getattr(module, attribute, None)
    return member if callable(member) else None


def _read_time_budget() -> float:
    """Read the referee's per-game budget instead of assuming sixty seconds."""
    try:
        budget = float(getattr(__import__("Tablut"), "TIME_BUDGET", _DEFAULT_TIME_BUDGET))
    except Exception:
        return _DEFAULT_TIME_BUDGET
    return budget if math.isfinite(budget) and budget > 0.0 else _DEFAULT_TIME_BUDGET


def _load_weights() -> list[float] | None:
    """Tuned weights from member 5 if they were exported, else the default.

    Returning None is meaningful: the evaluator then applies its own defaults,
    so this file works before member 5 has produced any tuned vector.
    """
    try:
        evaluator = __import__("Team19_eval", fromlist=["DEFAULT_WEIGHTS", "normalize_weights"])
        normalize_weights = evaluator.normalize_weights
        default_weights = evaluator.DEFAULT_WEIGHTS
    except Exception:
        return None

    try:
        tuned_module = __import__("Team19_weights", fromlist=["WEIGHTS"])
        tuned_weights = tuned_module.WEIGHTS
    except Exception:
        tuned_weights = None
    if tuned_weights is not None:
        try:
            return normalize_weights(tuned_weights)
        except Exception:
            pass

    try:
        return normalize_weights(default_weights)
    except Exception:
        return None


class Player:
    """Referee-facing player: one instance may be reused across games."""

    def __init__(self):
        # Loaded once here, never inside the timed methods.
        self._choose_best_move = _import_member("Team19_search", "choose_best_move")
        self._apply_move = _import_member("Team19_engine", "apply_move")
        self._check_winner = _import_member("Team19_engine", "check_winner")
        self._compute_hash = _import_member("Team19_engine", "compute_hash")

        self.weights = _load_weights()
        self._native = None
        self._native_choice = None
        self.native_init_error = None
        self.warmup_seconds = 0.0
        native_search = _import_member("Team19_search", "_NativeSearch")
        if native_search is not None:
            try:
                self._native = native_search(_INITIAL_BOARD)
            except Exception as error:
                self.native_init_error = error
            else:
                self.warmup_seconds = self._native.warmup_seconds
                self._native_choice = self._native.choose
                self._choose_best_move = self._native_choice
        self.reset_game()

    def reset_game(self) -> None:
        """Start a fresh game when a harness explicitly reuses this player."""
        self.total_budget = _read_time_budget()
        self.remaining = self.total_budget
        self.moves_played = 0
        self.history: set[int] = set()
        self._last_board: list[list[str]] | None = None
        self._last_is_attacker: bool | None = None
        self._opening_replies: set[tuple[tuple[str, ...], ...]] = set()
        # Last exception that forced a fallback move, kept for debugging.
        self.last_error: Exception | None = None
        self.last_search = None
        if self._native is not None:
            self._native.reset_game()

    # =================================================
    # Referee entry points
    def play_attacker(self, board: list[list[str]]) -> list[list[int]]:
        return self._play(board, True)

    def play_defender(self, board: list[list[str]]) -> list[list[int]]:
        return self._play(board, False)

    def _play(self, board: list[list[str]], is_attacker: bool) -> list[list[int]]:
        """Return a move for a playable position and charge the actual time."""
        started = time.perf_counter()
        self.last_search = None
        position = [list(row) for row in board]
        try:
            return self._select_move(position, is_attacker, started)
        except Exception as error:
            # Swallowing the error is what keeps the referee alive, but it must
            # stay observable: a silent fallback hides real bugs, so the last
            # failure is kept here for tests and debugging.
            self.last_error = error
            legal = self._reference_legal_moves(position, is_attacker)
            if not legal:
                raise
            move = [list(square) for square in min(legal)]
            self.moves_played += 1
            self._last_board = self._slide_without_captures(position, move)
            self._last_is_attacker = is_attacker
            return move
        finally:
            self.remaining = max(0.0, self.remaining - (time.perf_counter() - started))

    # =================================================
    # One decision
    def _select_move(
        self, position: list[list[str]], is_attacker: bool, started: float,
    ) -> list[list[int]]:
        seed_initial_history = self._start_new_game_if_needed(position, is_attacker)

        legal = self._reference_legal_moves(position, is_attacker)
        if not legal:
            raise ValueError("The referee must supply a board with a legal move")
        fallback = min(legal)
        step_budget = self._step_budget(position)
        deadline = started + step_budget
        search = self._choose_best_move
        native = (
            self._native is not None
            and (search is self._native_choice or getattr(search, "__wrapped__", None) is self._native_choice)
            and self._native.accepts(self.weights)
        )
        if native:
            bookkeeping_deadline = math.inf
        else:
            bookkeeping_deadline = started + self.remaining - _RETURN_RESERVE if step_budget else started
        if seed_initial_history and time.perf_counter() < bookkeeping_deadline:
            self._record_opponent_position([list(row) for row in _INITIAL_BOARD], True)
        if time.perf_counter() < bookkeeping_deadline:
            self._record_opponent_position(position, is_attacker)

        chosen = None
        if search is not None and (native or time.perf_counter() < deadline):
            try:
                result = search(
                    [row[:] for row in position], is_attacker, deadline, set(self.history),
                    self.weights,
                )
            except Exception as error:
                self.last_error = error
            else:
                candidate = self._as_pair(result)
                if candidate in legal:
                    chosen = candidate
                    if native:
                        self.last_search = self._native.last_search

        if chosen is None and time.perf_counter() < deadline:
            chosen = self._winning_move(position, is_attacker, legal, deadline)
        move = [list(square) for square in (chosen or fallback)]

        after_move = self._record_own_position(position, move, is_attacker, bookkeeping_deadline)
        self.moves_played += 1
        self._last_board = (
            after_move if after_move is not None
            else self._slide_without_captures(position, move)
        )
        self._last_is_attacker = is_attacker
        return move

    # =================================================
    # Time budget
    def _step_budget(self, board: list[list[str]]) -> float:
        """Share of the remaining game budget this move may spend.

        The share uses both the remaining time and the expected moves left.
        Actual time is charged afterwards, so unused allocation carries over.
        """
        available = self.remaining - _RETURN_RESERVE
        if available <= _MIN_AVAILABLE_TIME:
            return 0.0
        opening_allowance = max(0, _OPENING_MOVE_ALLOWANCE - self.moves_played)
        moves_left = max(
            _MIN_MOVES_LEFT, int(_MOVES_PER_PIECE * _piece_count(board)) + opening_allowance,
        )
        fair_share = available / moves_left
        short_search = min(_SHORT_SEARCH_TIME, available / _MIN_MOVES_LEFT)
        return max(fair_share, short_search)

    # =================================================
    # Game history
    def _start_new_game_if_needed(self, board: list[list[str]], is_attacker: bool) -> bool:
        """Reset on known game boundaries; return whether to seed the initial hash."""
        last = self._last_board
        opening_reply = None
        if not is_attacker and self._is_attack_slide(_INITIAL_BOARD, board):
            opening_reply = tuple(tuple(row) for row in board)

        new_game = last is None or is_attacker != self._last_is_attacker
        if last is not None and not new_game:
            new_game = _piece_count(board) > _piece_count(last)
            if is_attacker:
                new_game = new_game or self._is_initial_board(board)
            elif opening_reply is not None:
                new_game = (
                    new_game or opening_reply in self._opening_replies
                    or not self._is_attack_slide(last, board)
                )
        if new_game:
            self.reset_game()
        if opening_reply is not None:
            self._opening_replies.add(opening_reply)
        return new_game and opening_reply is not None

    @staticmethod
    def _is_attack_slide(
        before: tuple[tuple[str, ...], ...] | list[list[str]], after: list[list[str]],
    ) -> bool:
        """Check if the second board could follow a single attacker slide."""
        changed = [
            (row, col) for row, line in enumerate(before)
            for col, piece in enumerate(line) if piece != after[row][col]
        ]
        if len(changed) != 2:
            return False
        start, end = changed
        if before[start[0]][start[1]] != "A":
            start, end = end, start
        start_row, start_col = start
        end_row, end_col = end
        if (before[start_row][start_col], after[start_row][start_col],
                before[end_row][end_col], after[end_row][end_col]) != ("A", ".", ".", "A"):
            return False
        if start_row == end_row:
            return all(before[start_row][col] in (".", "X")
                       for col in range(min(start_col, end_col) + 1, max(start_col, end_col)))
        if start_col == end_col:
            return all(before[row][start_col] in (".", "X")
                       for row in range(min(start_row, end_row) + 1, max(start_row, end_row)))
        return False

    @staticmethod
    def _slide_without_captures(board: list[list[str]], move: list[list[int]]) -> list[list[str]]:
        position = [row[:] for row in board]
        (start_row, start_col), (end_row, end_col) = move
        position[end_row][end_col] = position[start_row][start_col]
        position[start_row][start_col] = "X" if (start_row, start_col) == (4, 4) else "."
        return position

    def _record_opponent_position(self, board: list[list[str]], is_attacker: bool) -> None:
        position_hash = self._position_hash(board, is_attacker)
        if position_hash is not None:
            self.history.add(position_hash)

    def _record_own_position(
        self, board: list[list[str]], move: list[list[int]], is_attacker: bool, deadline: float,
    ) -> list[list[str]] | None:
        """Record the position our move leads to and return it, if possible."""
        if self._apply_move is None or time.perf_counter() >= deadline:
            return None
        try:
            new_board = self._apply_move(
                [row[:] for row in board], [square[:] for square in move], is_attacker,
            )
        except Exception:
            return None
        if time.perf_counter() < deadline:
            position_hash = self._position_hash(new_board, not is_attacker)
            if position_hash is not None:
                self.history.add(position_hash)
        return [list(row) for row in new_board]

    def _position_hash(self, board: list[list[str]], is_attacker: bool) -> int | None:
        if self._compute_hash is None:
            return None
        try:
            return int(self._compute_hash(board, is_attacker))
        except Exception:
            return None

    # =================================================
    # Legal moves and fallbacks
    @staticmethod
    def _reference_legal_moves(board: list[list[str]], is_attacker: bool) -> set[MovePair]:
        """Movement rules of the referee, independent of the engine module."""
        own = ("A",) if is_attacker else ("D", "K")
        rows = len(board)
        cols = len(board[0]) if rows else 0
        moves = set()
        for row in range(rows):
            for col in range(cols):
                if board[row][col] not in own:
                    continue
                for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    target_row, target_col = row + delta_row, col + delta_col
                    while (0 <= target_row < rows and 0 <= target_col < cols
                           and board[target_row][target_col] in (".", "X")):
                        if board[target_row][target_col] == ".":
                            moves.add(((row, col), (target_row, target_col)))
                        target_row += delta_row
                        target_col += delta_col
        return moves

    @staticmethod
    def _as_pair(move) -> MovePair | None:
        try:
            (start_row, start_col), (end_row, end_col) = move
            return ((operator.index(start_row), operator.index(start_col)),
                    (operator.index(end_row), operator.index(end_col)))
        except (TypeError, ValueError):
            return None

    def _winning_move(
        self, board: list[list[str]], is_attacker: bool, legal: set[MovePair], deadline: float,
    ) -> MovePair | None:
        """Look for an immediate win within the time left for this move."""
        if self._apply_move is None or self._check_winner is None:
            return None
        target = "A" if is_attacker else "D"
        for move in sorted(legal):
            if time.perf_counter() >= deadline:
                break
            try:
                child = self._apply_move(
                    [row[:] for row in board], [list(square) for square in move], is_attacker,
                )
                if time.perf_counter() >= deadline:
                    break
                winner = self._check_winner(child, not is_attacker, True)
            except Exception:
                return None
            if winner == target and time.perf_counter() < deadline:
                return move
        return None

    @staticmethod
    def _is_initial_board(board: list[list[str]]) -> bool:
        if len(board) != len(_INITIAL_BOARD):
            return False
        return all(tuple(row) == _INITIAL_BOARD[index] for index, row in enumerate(board))


def _count(board: list[list[str]], piece: str) -> int:
    return sum(row.count(piece) for row in board)


def _piece_count(board: list[list[str]]) -> int:
    return sum(_count(board, piece) for piece in ("A", "D", "K"))


# ==== End of file
