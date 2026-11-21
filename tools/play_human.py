"""Play the current Team19 bot in a terminal using the course referee."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import copy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Tablut as rules
from Team19 import Player
import Team19_eval as evaluation
import Team19_search as search


Board = search.Board
Square = tuple[int, int]
Move = tuple[Square, Square]
SAVED_POSITIONS = Path(__file__).resolve().parent / "saved_positions"
BOARD_SIZE = 9
CANDIDATE_BUDGET = 0.5
SNAPSHOT_VERSION = 1
SNAPSHOT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
SIDE_NAMES = {True: "Attackers (A)", False: "Defenders (D)"}
SQUARE_PATTERN = r"(-?\d+)(?:\s*,\s*|\s+)(-?\d+)"
MOVE_PATTERN = re.compile(rf"{SQUARE_PATTERN}(?:\s*->\s*|\s+){SQUARE_PATTERN}")
HELP = "Move as 4,3 -> 4,1   |   ? r,c / moves r,c   |   undo   |   save [name]   |   quit"


def side_code(attacker: bool) -> str:
    return "A" if attacker else "D"


def in_board(square: Square) -> bool:
    return all(0 <= value < BOARD_SIZE for value in square)


def move_pair(value: object) -> Move | None:
    """Accept the referee's two integer coordinate pairs."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    for square in value:
        if (not isinstance(square, (tuple, list)) or len(square) != 2
                or any(type(index) is not int for index in square)):
            return None
    return tuple(value[0]), tuple(value[1])


def move_text(move: Move) -> str:
    (row, col), (target_row, target_col) = move
    return f"{row},{col} -> {target_row},{target_col}"


class RecordedError(RuntimeError):
    """A prior Player error read from a JSON snapshot."""


def error_text(error: Exception) -> str:
    if isinstance(error, RecordedError):
        return str(error)
    return f"{type(error).__name__}: {' '.join(str(error).split())}"


@dataclass(frozen=True)
class DepthRecord:
    depth: int
    result: search.SearchResult
    elapsed: float
    evaluated_horizon: bool


@dataclass
class SearchTrace:
    depths: list[DepthRecord] = field(default_factory=list)
    interrupted_depth: int | None = None
    interrupted_nodes: int | None = None
    allocation: float | None = None
    player_start: float | None = None
    history: set[int] | None = None
    returned_move: Move | None = None
    called: bool = False
    error: str | None = None
    is_attacker: bool | None = None

    @contextmanager
    def observe(self, player: Player):
        """Record calls made by the real Player and search, one turn at a time."""
        allocate = player._step_budget
        choose = player._choose_best_move
        search_depth = search._search_depth

        def record_allocation(board):
            self.player_start = player.remaining
            self.allocation = allocate(board)
            return self.allocation

        def record_choice(board, is_attacker, deadline, game_history, weights=None):
            self.called = True
            self.is_attacker = is_attacker
            self.history = game_history
            move = choose(board, is_attacker, deadline, game_history, weights)
            self.returned_move = move_pair(move)
            return move

        def record_depth(board, is_attacker, depth, *args, **kwargs):
            started = time.perf_counter()
            try:
                result, evaluated = search_depth(board, is_attacker, depth, *args, **kwargs)
            except search.SearchTimeout as error:
                self.interrupted_depth = depth
                self.interrupted_nodes = error.nodes_visited
                raise
            self.depths.append(DepthRecord(depth, result, time.perf_counter() - started, evaluated))
            return result, evaluated

        with ExitStack() as stack:
            stack.enter_context(patch.object(player, "_step_budget", record_allocation))
            if choose is not None:
                stack.enter_context(patch.object(player, "_choose_best_move", record_choice))
                stack.enter_context(patch.object(search, "_search_depth", record_depth))
            yield

    def stop_reason(self) -> str:
        if self.interrupted_depth is not None:
            return f"Search deadline at D{self.interrupted_depth}"
        if self.error:
            return "Player recorded an exception"
        if not self.called:
            return "Search skipped"
        if not self.depths:
            return "Stopped before D1 completed"
        last = self.depths[-1]
        winning_score = search.WIN_SCORE if self.is_attacker else -search.WIN_SCORE
        if last.result.terminal_plies is not None and last.result.score == winning_score:
            return f"Proven win in {last.result.terminal_plies} plies"
        if not last.evaluated_horizon:
            return "Proven score (no heuristic leaves)"
        if last.depth == search._MAX_ITERATIVE_DEPTH:
            return f"Depth cap reached at D{last.depth}"
        return f"Search returned after D{last.depth}"


@dataclass(frozen=True)
class TurnStats:
    elapsed: float
    depth: int
    nodes: int
    fallback: bool
    error: bool


@dataclass(frozen=True)
class MoveEvent:
    move: Move
    piece: str
    captures: tuple[tuple[int, int, str], ...]


@dataclass(frozen=True)
class Checkpoint:
    referee: rules.Tablut
    player: Player
    attackers_turn: bool
    remaining: float
    last_move: MoveEvent | None
    move_count: int
    ai_turn_count: int


def terminal_reason(game: rules.Tablut, message: str) -> str:
    """Describe an outcome already determined by the official referee."""
    if game.winner == "A":
        return "king captured"
    if game.winner == "D":
        if any(
            piece == "K" and (0 in (row, col) or BOARD_SIZE - 1 in (row, col))
            for row, squares in enumerate(game.board) for col, piece in enumerate(squares)
        ):
            return "king escaped to the edge"
        return "fewer than two attackers remain"
    if message.strip() == "Repeated position found":
        return "repeated position"
    if message.strip() == "Player cannot move":
        return "no legal move"
    return "referee declared a draw"


class Game:
    def __init__(self, human_attacker: bool):
        self.referee = rules.Tablut()
        self.referee.past_positions = [self.referee.compute_zobrist_hash(True)]
        self.player = Player()
        self.human_attacker = human_attacker
        self.attackers_turn = True
        self.budget = rules.TIME_BUDGET
        self.remaining = self.budget
        self.moves: list[Move] = []
        self.ai_turns: list[TurnStats] = []
        self.checkpoints: list[Checkpoint] = []
        self.last_move: MoveEvent | None = None
        self.outcome: str | None = None
        self.forfeit: str | None = None

    def apply(self, move: Move) -> bool:
        before = self.referee.board
        if not self.referee.make_move(self.attackers_turn, move):
            return False
        start, _ = move
        captures = tuple(
            (row, col, piece)
            for row, squares in enumerate(before) for col, piece in enumerate(squares)
            if (row, col) != start and piece in "ADK" and self.referee.board[row][col] == "."
        )
        self.last_move = MoveEvent(move, before[start[0]][start[1]], captures)
        self.moves.append(move)
        self.attackers_turn = not self.attackers_turn
        with redirect_stdout(StringIO()) as message:
            finished = self.referee.check_if_game_over(self.attackers_turn)
        if finished:
            self.outcome = terminal_reason(self.referee, message.getvalue())
        return True

    def play_human(self, move: Move) -> bool:
        referee = copy(self.referee)
        referee.board = [row[:] for row in referee.board]
        referee.past_positions = referee.past_positions[:]
        player = copy(self.player)
        player.history = player.history.copy()
        player._opening_replies = player._opening_replies.copy()
        if player._last_board is not None:
            player._last_board = [row[:] for row in player._last_board]
        state = Checkpoint(referee, player, self.attackers_turn, self.remaining,
                           self.last_move, len(self.moves), len(self.ai_turns))
        if not self.apply(move):
            self.referee.rounds_played = referee.rounds_played
            return False
        self.checkpoints.append(state)
        return True

    def undo(self) -> bool:
        if not self.checkpoints:
            return False
        state = self.checkpoints.pop()
        self.referee = state.referee
        self.player = state.player
        self.attackers_turn = state.attackers_turn
        self.remaining = state.remaining
        self.last_move = state.last_move
        del self.moves[state.move_count:]
        del self.ai_turns[state.ai_turn_count:]
        self.outcome = None
        self.forfeit = None
        return True

    def play_ai(self) -> tuple[SearchTrace, TurnStats]:
        trace = SearchTrace()
        previous_error = self.player.last_error
        position = [row[:] for row in self.referee.board]
        play = self.player.play_attacker if self.attackers_turn else self.player.play_defender
        returned = None
        failed = None
        with trace.observe(self.player):
            # Tablut.play measures the callback with time.time(), after copying the board.
            started = time.time()
            try:
                returned = play(position)
            except Exception as error:
                elapsed = time.time() - started
                failed = error
            else:
                elapsed = time.time() - started
                self.remaining -= elapsed

        move = move_pair(returned)
        new_error = self.player.last_error
        if new_error is not None and new_error is not previous_error:
            trace.error = error_text(new_error)
        if failed is not None:
            trace.error = error_text(failed)
            self.forfeit, self.outcome = "EXCEPTION", "Team19 raised an exception"
        elif self.remaining < 0:
            self.forfeit, self.outcome = "TIMEOUT", "Team19 exceeded the game clock"
        elif move is None or not self.apply(move):
            self.forfeit, self.outcome = "ILLEGAL", "Team19 returned an illegal move"
        if self.forfeit:
            self.referee.winner = side_code(self.human_attacker)

        depth = trace.depths[-1].depth if trace.depths else 0
        fallback = move is not None and self.forfeit != "ILLEGAL" and (
            not trace.depths or move != trace.returned_move or trace.error is not None
        )
        stats = TurnStats(elapsed, depth, sum(record.result.nodes_visited for record in trace.depths),
                          fallback, trace.error is not None)
        self.ai_turns.append(stats)
        return trace, stats

    def referee_state(self) -> dict:
        return {
            "board": self.referee.board,
            "side_to_move": side_code(self.attackers_turn),
            "past_positions": self.referee.past_positions,
            "rounds_played": self.referee.rounds_played,
            "winner": self.referee.winner,
        }


def player_state(player: Player) -> dict:
    return {
        "remaining": player.remaining,
        "moves_played": player.moves_played,
        "history": sorted(player.history),
        "last_board": player._last_board,
        "last_is_attacker": player._last_is_attacker,
        "opening_replies": sorted(player._opening_replies),
        "last_error": error_text(player.last_error) if player.last_error is not None else None,
    }


def save_snapshot(game: Game, name: str, *, web_reports: list[dict] | None = None) -> Path:
    name = name or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if not SNAPSHOT_NAME.fullmatch(name):
        raise ValueError("Use a name containing only letters, digits, hyphens and underscores.")
    data = {
        "version": SNAPSHOT_VERSION,
        "human": side_code(game.human_attacker),
        "budget": game.budget,
        "remaining": game.remaining,
        "referee": game.referee_state(),
        "moves": game.moves,
        "player": player_state(game.player),
        "ai_turns": [asdict(turn) for turn in game.ai_turns],
    }
    if web_reports is not None:
        data["web_reports"] = web_reports
    contents = json.dumps(data, indent=2, allow_nan=False)
    SAVED_POSITIONS.mkdir(exist_ok=True)
    destination = SAVED_POSITIONS / f"{name}.json"
    with destination.open("x", encoding="utf-8") as file:
        file.write(contents + "\n")
    return destination


def valid_board(board: object) -> bool:
    return isinstance(board, list) and len(board) == BOARD_SIZE and all(
        isinstance(row, list) and len(row) == BOARD_SIZE
        and all(isinstance(cell, str) and cell in {".", "A", "D", "K", "X"} for cell in row)
        for row in board
    )


def seconds(value: object) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite, nonnegative time in seconds.")
    return float(value)


def restore_snapshot(game: Game, data: dict, *, on_move: Callable[[Game], None] | None = None) -> None:
    """Replay with the official referee, then restore the bot's timed state."""
    for value in data["moves"]:
        move = move_pair(value)
        if (game.outcome or move is None or not all(map(in_board, move))
                or not game.apply(move)):
            raise ValueError("The snapshot's move list contains an illegal continuation.")
        if on_move is not None:
            on_move(game)
    if game.outcome or game.referee_state() != data["referee"]:
        raise ValueError("The saved referee state does not match the move list.")

    game.remaining = seconds(data["remaining"])
    if game.remaining > game.budget:
        raise ValueError("The saved referee clock exceeds the budget.")
    for entry in data["ai_turns"]:
        turn = TurnStats(**entry)
        seconds(turn.elapsed)
        if (type(turn.nodes) is not int or turn.nodes < 0
                or type(turn.depth) is not int or turn.depth < 0
                or type(turn.fallback) is not bool or type(turn.error) is not bool):
            raise ValueError("Invalid saved AI turn statistics.")
        game.ai_turns.append(turn)
    expected = (len(game.moves) + int(not game.human_attacker)) // 2
    total_elapsed = sum(turn.elapsed for turn in game.ai_turns)
    if len(game.ai_turns) != expected or not math.isclose(
        game.remaining + total_elapsed, game.budget, abs_tol=1e-8
    ):
        raise ValueError("The saved AI turn statistics do not match the game clock.")

    saved = data["player"]
    player = game.player
    remaining = seconds(saved["remaining"])
    moves_played = saved["moves_played"]
    history = saved["history"]
    last_board = saved["last_board"]
    last_side = saved["last_is_attacker"]
    openings = saved["opening_replies"]
    last_error = saved["last_error"]
    if (remaining > game.budget or type(moves_played) is not int
            or not 0 <= moves_played <= expected
            or any(type(key) is not int or not 0 <= key < 2**64 for key in history)
            or (last_board is not None and not valid_board(last_board))
            or (last_side is not None and (type(last_side) is not bool
                                           or last_side == game.human_attacker))
            or not all(valid_board(board) for board in openings)
            or (last_error is not None and not isinstance(last_error, str))):
        raise ValueError("Invalid saved Team19 state.")
    player.remaining = remaining
    player.moves_played = moves_played
    player.history = set(history)
    player._last_board = last_board
    player._last_is_attacker = last_side
    player._opening_replies = {tuple(map(tuple, board)) for board in openings}
    if last_error is not None:
        player.last_error = RecordedError(last_error)


def use_color(disabled: bool) -> bool:
    if disabled or "NO_COLOR" in os.environ or not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return os.environ.get("TERM") != "dumb"

    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel.GetStdHandle.restype = wintypes.HANDLE
    kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    handle = kernel.GetStdHandle(wintypes.DWORD(-11).value)
    mode = wintypes.DWORD()
    return bool(kernel.GetConsoleMode(handle, ctypes.byref(mode))
                and kernel.SetConsoleMode(handle, mode.value | 0x0004))


def show_board(game: Game, color: bool) -> None:
    event = game.last_move
    start, end = event.move if event else (None, None)
    palette = {"A": 94, "D": 33, "K": 91, "X": 96}
    print("\n     " + "   ".join(map(str, range(BOARD_SIZE))))
    for row, squares in enumerate(game.referee.board):
        cells = []
        for col, piece in enumerate(squares):
            mark = (row, col)
            cell = f"({piece})" if mark == start else f"[{piece}]" if mark == end else f" {piece} "
            if color and piece in palette:
                cell = f"\033[{palette[piece]}m{cell}\033[0m"
            cells.append(cell)
        print(f"  {row} " + " ".join(cells))
    if event and event.captures:
        captures = ", ".join(f"{piece}@({row},{col})" for row, col, piece in event.captures)
        print(f"Captured {captures}")


def feature_breakdown(before: Board, after: Board, weights: search.Weights) -> list[dict]:
    active = evaluation.normalize_weights(evaluation.DEFAULT_WEIGHTS if weights is None else weights)
    old = [weight * value for weight, value in zip(active, evaluation._features(before))]
    new = [weight * value for weight, value in zip(active, evaluation._features(after))]
    return [
        {"name": name, "before": old_score, "after": new_score, "delta": new_score - old_score}
        for name, old_score, new_score in zip(evaluation.FEATURE_NAMES, old, new)
    ]


def show_features(before: Board, after: Board, weights: search.Weights) -> None:
    features = feature_breakdown(before, after, weights)
    print(f"  Static features (+A / -D)   total {sum(f['before'] for f in features):+.0f}"
          f" -> {sum(f['after'] for f in features):+.0f}")
    cells = [
        f"{feature['name']:<20} {feature['after']:+7.0f} ({feature['delta']:+.0f})"
        for feature in features
    ]
    for index in range(0, len(cells), 2):
        print("    " + "  ".join(f"{cell:<37}" for cell in cells[index:index + 2]).rstrip())


def analyze_candidates(
    board: Board, attacker: bool, actual: Move, trace: SearchTrace,
    official_history: set[int], weights: search.Weights,
) -> dict:
    deadline = time.perf_counter() + CANDIDATE_BUDGET
    functions = search._load_functions()
    history = official_history if trace.history is None else trace.history
    moves = [move_pair(move) for move in functions.get_legal_moves(board, attacker)]
    # Evaluate the played move first so it gets a score even in a partial scan.
    if actual in moves:
        moves.remove(actual)
        moves.insert(0, actual)

    def fill_scores(candidates, depth, scores):
        for candidate in candidates:
            if candidate in scores:
                continue
            only_move = [list(square) for square in candidate]

            def one_root_move(position, side):
                # Keep the original root in the search path. A child that
                # repeats a previous state must count as a draw even at D1.
                return [only_move] if position is board else functions.get_legal_moves(position, side)

            try:
                result = search.search_fixed_depth(
                    board, attacker, depth, functions=replace(functions, get_legal_moves=one_root_move),
                    weights=weights, deadline=deadline, game_history=history,
                    tactical_ordering=depth > 1,
                )
            except search.SearchTimeout:
                break
            scores[candidate] = result.score

    def ranked(scores):
        return sorted(scores, key=scores.__getitem__, reverse=attacker)

    static_scores: dict[Move, float] = {}
    fill_scores(moves, 1, static_scores)
    exact_scores: dict[Move, float] = {}
    shortlist = []
    deep = None
    depth = trace.depths[-1].depth if trace.depths else 0
    if depth > 1:
        previous = [move_pair(record.result.move) for record in reversed(trace.depths)]
        shortlist = [move for move in dict.fromkeys([actual, *previous, *ranked(static_scores)[:6]])
                     if move in moves]
        best = move_pair(trace.depths[-1].result.move)
        if best in shortlist:
            exact_scores[best] = trace.depths[-1].result.score
        fill_scores(shortlist, depth, exact_scores)
        deep = {"depth": depth, "scored": len(exact_scores), "total": len(shortlist)}
    if exact_scores and len(exact_scores) >= min(3, len(shortlist)):
        scores, display_depth, total = exact_scores, depth, len(shortlist)
    else:
        scores, display_depth, total = static_scores, 1, len(moves)

    best_moves = ranked(scores)[:3]
    return {
        "depth": display_depth, "scored": len(scores), "total": total, "deep": deep,
        "entries": [
            {"move": move, "score": scores[move], "played": move == actual,
             "gap": abs(scores[best_moves[0]] - scores[move])}
            for move in best_moves
        ],
    }


def show_candidates(
    board: Board, attacker: bool, actual: Move, trace: SearchTrace,
    official_history: set[int], weights: search.Weights,
) -> None:
    comparison = analyze_candidates(board, attacker, actual, trace, official_history, weights)
    depth, deep = comparison["depth"], comparison["deep"]
    if deep:
        print(f"  D{deep['depth']} shortlist   {deep['scored']}/{deep['total']} scored")
    if depth == 1:
        print(f"  D1 candidates   {comparison['scored']}/{comparison['total']} scored")
    for rank, entry in enumerate(comparison["entries"], 1):
        selected = " * played" if entry["played"] else ""
        print(f"    {rank}. {move_text(entry['move']):<12} {entry['score']:+10.1f}"
              f"  gap {entry['gap']:8.1f} [D{depth}]{selected}")
    if not comparison["entries"]:
        print("    No candidate scores available.")


def show_ai_turn(
    game: Game, attacker: bool, before: Board, history: set[int],
    trace: SearchTrace, stats: TurnStats, top3: bool,
) -> None:
    tags = []
    if game.forfeit:
        tags.append(f"[{game.forfeit}]")
    if stats.fallback:
        tags.append("[FALLBACK]")
    if stats.error and game.forfeit != "EXCEPTION":
        tags.append("[ERROR]")
    title = f"Team19 {side_code(attacker)}"
    if not game.forfeit and game.last_move:
        event = game.last_move
        title += f"   {event.piece} {move_text(event.move)}   captures {len(event.captures)}"
    print(" ".join([title, *tags]))
    allocated = f"{trace.allocation:.3f}s" if trace.allocation is not None else "n/a"
    nps = stats.nodes / stats.elapsed if stats.elapsed > 0 else 0
    print(f"  Allocated {allocated}   elapsed {stats.elapsed:.3f}s   "
          f"referee left {game.remaining:.3f}/{game.budget:g}s   NPS {nps:,.0f}")
    if trace.player_start is not None:
        used = trace.player_start - game.player.remaining
        print(f"  Player clock charged {used:.3f}s   left {game.player.remaining:.3f}s")
    for record in trace.depths:
        result = record.result
        move = move_pair(result.move)
        chosen = move_text(move) if move else "-"
        rate = 100 * result.cutoffs / result.nodes_visited if result.nodes_visited else 0
        heuristic = "" if record.evaluated_horizon else "  proven"
        print(f"  D{record.depth:<2} {chosen:<12} {result.score:+10.1f}"
              f"   {result.nodes_visited:>6,} nodes   {result.cutoffs:>5,} cuts"
              f" ({rate:.1f}/100 nodes)   {record.elapsed:.3f}s{heuristic}")
    print(f"  {trace.stop_reason()}")
    if trace.error:
        print(f"  {trace.error}")
    if not game.forfeit and game.last_move:
        if top3:
            show_candidates(before, attacker, game.last_move.move, trace, history, game.player.weights)
        show_features(before, game.referee.board, game.player.weights)


def illegal_reason(board: Board, attacker: bool, move: Move) -> str:
    (row, col), (target_row, target_col) = move
    own_pieces = "A" if attacker else "DK"
    if board[row][col] not in own_pieces:
        return f"Select one of your own pieces ({own_pieces})."
    if move[0] == move[1]:
        return "Choose a different destination."
    if board[target_row][target_col] != ".":
        return "The destination must be empty and cannot be the castle."
    if row != target_row and col != target_col:
        return "Pieces slide on a row or column."
    return "The referee rejected the move. Check for a piece blocking the path."


def legal_destinations(game: Game, source: Square) -> list[Square]:
    row, col = source
    trial = copy(game.referee)
    targets = [(row, i) for i in range(BOARD_SIZE) if i != col]
    targets.extend((i, col) for i in range(BOARD_SIZE) if i != row)
    legal = []
    for target in targets:
        trial.board = game.referee.board
        if trial.make_move(game.attackers_turn, (source, target)):
            legal.append(target)
    return legal


def show_destinations(game: Game, text: str) -> None:
    match = re.fullmatch(SQUARE_PATTERN, text.strip())
    if not match:
        print("Enter ? r,c or moves r,c, for example ? 0,3.")
        return
    try:
        source = tuple(map(int, match.groups()))
    except ValueError:
        print("Coordinates must be between 0 and 8.")
        return
    if not in_board(source):
        print("Coordinates must be between 0 and 8.")
        return
    legal = [f"{row},{col}" for row, col in legal_destinations(game, source)]
    print(f"  {source[0]},{source[1]} can move to " + ("  ".join(legal) or "no legal destination"))


def human_turn(game: Game, color: bool) -> bool:
    while True:
        line = input(f"{side_code(game.attackers_turn)} move> ").strip()
        if not line:
            continue
        command, *argument = line.split(maxsplit=1)
        text = argument[0] if argument else ""
        command = command.lower()
        if command in ("quit", "q"):
            return False
        if command == "help":
            print(HELP)
        elif line.startswith("?") or command == "moves":
            show_destinations(game, line[1:] if line.startswith("?") else text)
        elif command == "undo":
            if game.undo():
                print("Restored the position and clocks before your last move.")
                show_board(game, color)
            else:
                print("There is no earlier human turn in this session.")
        elif command == "save":
            try:
                print(f"Saved {save_snapshot(game, text)}")
            except (OSError, ValueError) as error:
                print(f"Save failed. {error}")
        else:
            match = MOVE_PATTERN.fullmatch(line)
            if not match:
                print(f"Enter four coordinates. {HELP}")
                continue
            try:
                row, col, target_row, target_col = map(int, match.groups())
            except ValueError:
                print("Coordinates must be between 0 and 8.")
                continue
            move = ((row, col), (target_row, target_col))
            if not all(map(in_board, move)):
                print("Coordinates must be between 0 and 8.")
            elif game.play_human(move):
                return True
            else:
                print(illegal_reason(game.referee.board, game.attackers_turn, move))


def show_summary(game: Game) -> None:
    print("\nGame summary")
    if game.outcome is None:
        print("  Session ended early.")
    elif game.referee.winner is None:
        print(f"  Draw ({game.outcome}). Course points   Team19 1   Human 1")
    else:
        winner = SIDE_NAMES[game.referee.winner == "A"]
        human_points = 2 if game.referee.winner == side_code(game.human_attacker) else 0
        print(f"  {winner} win ({game.outcome}). Course points   Team19 {2 - human_points}"
              f"   Human {human_points}")
    spent = game.budget - game.remaining
    plies = len(game.moves)
    rounds = (plies + bool(game.forfeit) + 1) // 2
    count = len(game.ai_turns)
    print(f"  {rounds} rounds   {plies} legal plies   {count} AI turns")
    print(f"  Team19 charged {spent:.3f}/{game.budget:g}s ({100 * spent / game.budget:.1f}%)")
    if not count:
        return
    depths = Counter(turn.depth for turn in game.ai_turns)
    average_depth = sum(depth * frequency for depth, frequency in depths.items()) / count
    distribution = "   ".join(f"D{depth} {depths[depth]}" for depth in sorted(depths))
    duration = sum(turn.elapsed for turn in game.ai_turns)
    nps = sum(turn.nodes for turn in game.ai_turns) / duration if duration > 0 else 0
    print(f"  Completed depths   {distribution}   average {average_depth:.2f}")
    print(f"  NPS (completed depths) {nps:,.0f}   "
          f"slowest turn {max(turn.elapsed for turn in game.ai_turns):.3f}s   "
          f"fallbacks {sum(turn.fallback for turn in game.ai_turns)}   "
          f"errors {sum(turn.error for turn in game.ai_turns)}")


def play_session(game: Game, color: bool, top3: bool) -> None:
    print(f"Human {SIDE_NAMES[game.human_attacker]}   Team19 {SIDE_NAMES[not game.human_attacker]}"
          f"   AI game clock {game.budget:g}s")
    print(HELP)
    show_board(game, color)
    try:
        while game.outcome is None:
            if game.attackers_turn == game.human_attacker:
                if not human_turn(game, color):
                    break
                show_board(game, color)
            else:
                board = game.referee.board
                history = set(game.referee.past_positions)
                attacker = game.attackers_turn
                trace, stats = game.play_ai()
                if not game.forfeit:
                    show_board(game, color)
                show_ai_turn(game, attacker, board, history, trace, stats, top3)
    except (EOFError, KeyboardInterrupt):
        print()
    show_summary(game)


def parse_budget(value: str) -> float:
    try:
        budget = seconds(float(value))
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError("Enter a positive, finite game budget in seconds.") from error
    if budget == 0:
        raise argparse.ArgumentTypeError("The game budget must be positive.")
    return budget


def interactive_options(default_budget: float) -> tuple[str, float]:
    while True:
        human = input("Human side (A/D) [A]> ").strip().upper() or "A"
        if human in ("A", "D"):
            break
        print("Enter A for Attackers or D for Defenders.")
    while True:
        value = input(f"Team19 game budget in seconds [{default_budget:g}]> ").strip()
        try:
            return human, parse_budget(value) if value else default_budget
        except argparse.ArgumentTypeError as error:
            print(error)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=str.upper, choices=("A", "D"), help="human side (default A)")
    parser.add_argument("--budget", type=parse_budget, help="AI time per game, in seconds (default 60)")
    parser.add_argument("--load", type=Path, help="resume a JSON snapshot with its saved side and clocks")
    parser.add_argument("--no-top3", action="store_true", help="skip candidate analysis")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI board colors")
    args = parser.parse_args(argv)
    snapshot = None
    default_budget = rules.TIME_BUDGET
    side = args.side or "A"
    budget = args.budget if args.budget is not None else default_budget
    if args.load:
        try:
            snapshot = json.loads(args.load.read_text(encoding="utf-8"))
            if not isinstance(snapshot, dict) or snapshot.get("version") != SNAPSHOT_VERSION:
                raise ValueError("Unsupported snapshot format.")
            side = snapshot["human"]
            budget = seconds(snapshot["budget"])
            if side not in ("A", "D") or budget == 0:
                raise ValueError("Invalid saved side or budget.")
            if (args.side is not None and args.side != side
                    or args.budget is not None and args.budget != budget):
                raise ValueError("The options disagree with the saved side or budget.")
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
            parser.error(f"{args.load} | {error}")
    elif args.side is None and args.budget is None and sys.stdin.isatty():
        try:
            side, budget = interactive_options(default_budget)
        except (EOFError, KeyboardInterrupt):
            print()
            return

    with patch.object(rules, "TIME_BUDGET", budget):
        game = Game(side == "A")
        if snapshot is not None:
            try:
                restore_snapshot(game, snapshot)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                parser.error(f"{args.load} | {error}")
            print(f"Loaded {args.load}   {len(game.moves)} plies played")
            print("Undo is available after your first move in this session.")
        play_session(game, use_color(args.no_color), top3=not args.no_top3)


if __name__ == "__main__":
    main()
