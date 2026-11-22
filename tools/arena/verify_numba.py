"""Validate the native engine against the archived rules suite and official referee."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import Tablut

from . import HERE, openings
from .bots import grandmaster_numba as gm
from .bots import numba_rules as native


RULES_TEST_COMMIT = "50fc259d39bb1c54a2a615bb708998b52a56d245"


def existing_tests(repo: Path, implementation=native) -> None:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in ("test_engine", "test_interface_contract"):
        source = subprocess.check_output(
            ("git", "show", f"{RULES_TEST_COMMIT}:tests/{name}.py"), cwd=repo,
        ).decode("utf-8")
        module = ModuleType(name)
        with patch.dict("sys.modules", {"Team19_engine": implementation}):
            exec(compile(source, f"{RULES_TEST_COMMIT}/tests/{name}.py", "exec"), module.__dict__)
        suite.addTests(loader.loadTestsFromModule(module))
    if suite.countTestCases() != 40:
        raise RuntimeError("The historical rules suite should contain 40 tests")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("Historical rules tests did not all pass")


def _native_state(engine, flat):
    ha, hd, attackers, defenders, king = engine._describe(flat)
    return np.uint64(ha), np.uint64(hd), attackers, defenders, king


def team19_rules():
    """Expose the new compiled primitives to the archived Python rule tests."""
    import Team19_engine as engine

    def compute_hash(board: list[list[str]], is_attacker: bool) -> int:
        return int(engine._describe(engine._flat(board))[0 if is_attacker else 1])

    def get_legal_moves(board: list[list[str]], is_attacker: bool) -> list[list[list[int]]]:
        moves = np.empty(engine._MOVE_CAPACITY, dtype=np.int16)
        length = engine._legal(engine._flat(board), is_attacker, moves)
        return [engine._move_list(int(code)) for code in moves[:length]]

    def apply_move(board: list[list[str]], move: list[list[int]], is_attacker: bool) -> list[list[str]]:
        if move not in get_legal_moves(board, is_attacker):
            raise ValueError("Illegal move")
        (r, c), (rr, cc) = move
        code = (r * 9 + c) * 81 + rr * 9 + cc
        flat = engine._flat(board)
        engine._push(flat, code, *_native_state(engine, flat),
                     np.empty(4, dtype=np.int16), np.empty(4, dtype=np.int8))
        return engine._nested(flat)

    def check_winner(board: list[list[str]], is_attacker_next: bool,
                     has_legal_moves: bool = True) -> str | None:
        _, _, attackers, _, king = engine._describe(engine._flat(board))
        winner = engine._winner(attackers, king)
        return engine._CELLS[winner] if winner else None if has_legal_moves else engine.DRAW

    constants = {name: getattr(engine, name) for name in (
        "ATTACKER", "DEFENDER", "KING", "EMPTY", "CASTLE", "SIZE", "CASTLE_ROW", "CASTLE_COL",
        "WIN_ATTACKER", "WIN_DEFENDER", "DRAW",
    )}
    return SimpleNamespace(**constants, compute_hash=compute_hash,
                           get_legal_moves=get_legal_moves, apply_move=apply_move,
                           check_winner=check_winner)


def official_moves(game, attacker):
    # Enumerate orthogonal endpoints and ask the official checker about each.
    board = game.board
    own = ("A",) if attacker else ("D", "K")
    result = set()
    for row, cells in enumerate(board):
        for col, cell in enumerate(cells):
            if cell not in own:
                continue
            for dest in range(9):
                for rr, cc in ((row, dest), (dest, col)):
                    if (rr, cc) == (row, col) or board[rr][cc] != ".":
                        continue
                    if game.make_move(attacker, [[row, col], [rr, cc]]):
                        result.add((row * 9 + col) * 81 + rr * 9 + cc)
                    game.board = board
    return result


def _official_immediate_moves(game, moves, attacker, attackers, king):
    """Find official one-move wins without examining irrelevant landing squares."""
    board = game.board
    king_row, king_col = divmod(king, 9)
    wins = set()
    for code in moves:
        source, target = divmod(code, 81)
        row, col = divmod(target, 9)
        if attacker and abs(row - king_row) + abs(col - king_col) != 1:
            continue
        if not attacker and attackers > 5 and not (
            source == king and (row in (0, 8) or col in (0, 8))
        ):
            continue
        if not game.make_move(attacker, native._move_list(code)):
            raise AssertionError("Official legal move was rejected on replay")
        if attacker:
            won = game.board[king_row][king_col] != "K"
        else:
            won = (source == king and (row in (0, 8) or col in (0, 8))
                   or sum(line.count("A") for line in game.board) < 2)
        game.board = board
        if won:
            wins.add(code)
    return wins


def differential(steps: int = 5000, *, engine=native, verify_immediate: bool = False) -> dict:
    rng = random.Random(20261121)
    referee = Tablut.Tablut()
    referee.past_positions = [referee.compute_zobrist_hash(True)]
    flat = engine._flat(referee.board)
    state = _native_state(engine, flat)
    attacker = True
    endings = Counter()
    scratch = Tablut.Tablut()
    move_buffer = np.empty(engine._MOVE_CAPACITY, dtype=np.int16)
    immediate_buffer = np.empty_like(move_buffer)
    taken = np.empty(4, dtype=np.int16)
    pieces = np.empty(4, dtype=np.int8)
    started = perf_counter()
    for number in range(1, steps + 1):
        scratch.board = referee.board
        judge_moves = official_moves(scratch, attacker)
        length = engine._legal(flat, attacker, move_buffer)
        legal = {int(move) for move in move_buffer[:length]}
        if legal != judge_moves or len(legal) != length:
            raise AssertionError(f"Legal set mismatch at ply {number}")
        if verify_immediate:
            fast_win = engine._immediate_win(flat, attacker, state[2], state[4], immediate_buffer)
            judge_wins = _official_immediate_moves(scratch, legal, attacker, state[2], state[4])
            if (fast_win >= 0 and fast_win not in judge_wins) or (fast_win < 0 and judge_wins):
                raise AssertionError(f"Immediate win mismatch at ply {number}")
        move = rng.choice(sorted(legal))
        original = flat.copy()
        *updated, captured = engine._push(flat, move, *state, taken, pieces)
        updated[0] = np.uint64(updated[0])
        updated[1] = np.uint64(updated[1])
        child = flat.copy()
        engine._pop(flat, move, taken, pieces, captured)
        if not np.array_equal(flat, original):
            raise AssertionError(f"Undo mismatch at ply {number}")
        flat = child
        state = tuple(updated)
        if not referee.make_move(attacker, engine._move_list(move)) or engine._nested(flat) != referee.board:
            raise AssertionError(f"Board or captures mismatch at ply {number}")
        if state[0] != referee.compute_zobrist_hash(True) or state[1] != referee.compute_zobrist_hash(False):
            raise AssertionError(f"Dual incremental hash mismatch at ply {number}")
        if state != _native_state(engine, flat):
            raise AssertionError(f"Material or king metadata mismatch at ply {number}")
        attacker = not attacker
        winner = engine._winner(state[2], state[4])
        hash_next = state[0 if attacker else 1]
        own_verdict = (engine._CELLS[winner] if winner else
                       "DRAW" if not engine._has_legal(flat, attacker)
                       or int(hash_next) in referee.past_positions else None)
        with redirect_stdout(StringIO()) as output:
            game_over = referee.check_if_game_over(attacker)
        judge_verdict = (referee.winner or "DRAW") if game_over else None
        if own_verdict != judge_verdict:
            raise AssertionError(f"Terminal or repetition mismatch at ply {number}")
        if game_over:
            endings[output.getvalue().strip() or judge_verdict] += 1
            referee = Tablut.Tablut()
            referee.past_positions = [referee.compute_zobrist_hash(True)]
            flat = engine._flat(referee.board)
            state = _native_state(engine, flat)
            attacker = True
    return {"plies": steps, "differences": 0, "immediateChecked": verify_immediate,
            "terminalGames": dict(endings),
            "elapsed": perf_counter() - started}


def timing() -> dict:
    samples = []
    suite = openings.load_suite()
    with patch.object(Tablut, "TIME_BUDGET", 60.0):
        player = gm.Player()
        kernels = (gm._visit, native._push, native._legal, native._describe)
        signatures = [tuple(fn.signatures) for fn in kernels]
        for opening in (openings.standard(), suite[0], suite[5], suite[11]):
            for attacker in (True, False):
                game = openings.start(opening)
                if not attacker:
                    next_move = native.get_legal_moves(game.referee.board, True)[0]
                    assert game.referee.make_move(True, next_move)
                    assert not game.referee.check_if_game_over(False)
                player.reset_game()
                player.seed_history(game.referee.past_positions)
                started = perf_counter()
                move = (player.play_attacker(game.referee.board) if attacker else
                        player.play_defender(game.referee.board))
                elapsed = perf_counter() - started
                if not game.referee.make_move(attacker, move):
                    raise AssertionError("Illegal move in timing test")
                trace = player.last_search
                samples.append({"opening": opening.name, "side": "A" if attacker else "D",
                                "seconds": elapsed, "nodes": trace.nodes,
                                "nps": trace.nodes / elapsed, "depth": trace.depth,
                                "fallback": trace.fallback})
        if [tuple(fn.signatures) for fn in kernels] != signatures:
            raise AssertionError("A timed callback triggered a new JIT specialization")
    return {"initWarmupSeconds": player.warmup_seconds, "newSignaturesDuringPlay": 0,
            "medianNps": median(row["nps"] for row in samples),
            "medianDepth": median(row["depth"] for row in samples), "samples": samples}


def _v2_timing(player, cold_init: float) -> dict:
    import Team19_engine as engine
    import Team19_eval as evaluation
    import Team19_search as search

    if player.native_init_error is not None or player._native is None:
        raise RuntimeError("The native Player failed to initialize") from player.native_init_error
    kernels = (search._native_visit, engine._legal, engine._push,
               engine._immediate_win, evaluation._native_evaluate)
    signatures = [tuple(kernel.signatures) for kernel in kernels]
    if any(not specializations for specializations in signatures):
        raise AssertionError("The warmup did not compile every search kernel")

    suite = openings.load_suite()
    samples = []
    with patch.object(Tablut, "TIME_BUDGET", 60.0):
        for opening in (openings.standard(), suite[0], suite[5], suite[11]):
            for attacker in (True, False):
                started_game = openings.start(opening)
                game = started_game.referee
                if not attacker:
                    move = engine.get_legal_moves(game.board, True)[0]
                    if not game.make_move(True, move) or game.check_if_game_over(False):
                        raise AssertionError("The defender timing position was not playable")
                player.reset_game()
                if opening.moves:
                    player._last_board = started_game.last_board[attacker]
                    player._last_is_attacker = attacker
                    if not attacker:
                        player._opening_replies.add(
                            tuple(tuple(row) for row in started_game.positions[1]["board"])
                        )
                player.history.update(game.past_positions)
                started = perf_counter()
                move = (player.play_attacker(game.board) if attacker
                        else player.play_defender(game.board))
                elapsed = perf_counter() - started
                trace = player.last_search
                if not game.make_move(attacker, move) or trace is None or trace.fallback:
                    raise AssertionError("Native timing test did not return a searched legal move")
                samples.append({
                    "opening": opening.name, "side": "A" if attacker else "D",
                    "seconds": elapsed, "nodes": trace.nodes, "nps": trace.nodes / elapsed,
                    "depth": trace.depth, "fallback": trace.fallback,
                })
    if [tuple(kernel.signatures) for kernel in kernels] != signatures:
        raise AssertionError("A timed callback triggered a new JIT specialization")
    return {"coldPlayerSeconds": cold_init, "initWarmupSeconds": player.warmup_seconds,
            "newSignaturesDuringPlay": 0, "medianNps": median(row["nps"] for row in samples),
            "medianDepth": median(row["depth"] for row in samples), "samples": samples}


def _cold_v2_timing() -> dict:
    # The verification modules import Numba. Start a separate process so even
    # the Player's first dependency imports are included in its cold timing.
    program = """
from json import dumps
from time import perf_counter

from Team19 import Player

started = perf_counter()
player = Player()
elapsed = perf_counter() - started

from tools.arena.verify_numba import _v2_timing

print(dumps(_v2_timing(player, elapsed)))
"""
    output = subprocess.check_output(
        (sys.executable, "-B", "-c", program), cwd=HERE.parents[1],
        text=True, encoding="utf-8",
    )
    return json.loads(output)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("rules", "timing"))
    parser.add_argument("--repo", type=Path, default=Path(Tablut.__file__).resolve().parent)
    parser.add_argument("--team19-v2", action="store_true", help="verify the submission's native kernels")
    args = parser.parse_args()
    if args.mode == "rules":
        if args.team19_v2:
            import Team19_engine as engine

            existing_tests(args.repo, team19_rules())
            results = differential(engine=engine, verify_immediate=True)
        else:
            existing_tests(args.repo)
            results = differential()
        result = {"rulesTests": 40, "reference": RULES_TEST_COMMIT, **results}
    else:
        result = _cold_v2_timing() if args.team19_v2 else timing()
    dest = HERE / "results"
    dest.mkdir(exist_ok=True)
    prefix = "v2-numba" if args.team19_v2 else "numba"
    (dest / f"{prefix}-{args.mode}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
