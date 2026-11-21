"""Validate the native engine against the archived rules suite and official referee."""

from argparse import ArgumentParser
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import random
from statistics import median
import subprocess
from time import perf_counter
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

import Tablut

from . import HERE, openings
from .bots import grandmaster_numba as gm
from .bots import numba_rules as native


RULES_TEST_COMMIT = "50fc259d39bb1c54a2a615bb708998b52a56d245"


def existing_tests(repo: Path) -> None:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in ("test_engine", "test_interface_contract"):
        source = subprocess.check_output(
            ("git", "show", f"{RULES_TEST_COMMIT}:tests/{name}.py"), cwd=repo,
        ).decode("utf-8")
        source = source.replace("from Team19_engine import (", "from tools.arena.bots.numba_rules import (")
        source = source.replace("import Team19_engine\n",
                                "from tools.arena.bots import numba_rules as Team19_engine\n")
        module = ModuleType(name)
        exec(compile(source, f"{RULES_TEST_COMMIT}/tests/{name}.py", "exec"), module.__dict__)
        suite.addTests(loader.loadTestsFromModule(module))
    if suite.countTestCases() != 40:
        raise RuntimeError("The historical rules suite should contain 40 tests")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("Historical rules tests did not all pass")


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


def differential(steps: int = 5000) -> dict:
    rng = random.Random(20261121)
    referee = Tablut.Tablut()
    referee.past_positions = [referee.compute_zobrist_hash(True)]
    flat = native._flat(referee.board)
    state = gm._root_state(flat)
    attacker = True
    endings = Counter()
    scratch = Tablut.Tablut()
    move_buffer = np.empty(native._MOVE_CAPACITY, dtype=np.int16)
    taken = np.empty(4, dtype=np.int16)
    pieces = np.empty(4, dtype=np.int8)
    started = perf_counter()
    for number in range(1, steps + 1):
        scratch.board = referee.board
        judge_moves = official_moves(scratch, attacker)
        length = native._legal(flat, attacker, move_buffer)
        legal = {int(move) for move in move_buffer[:length]}
        if legal != judge_moves or len(legal) != length:
            raise AssertionError(f"Legal set mismatch at ply {number}")
        move = rng.choice(sorted(legal))
        original = flat.copy()
        *updated, captured = native._push(flat, move, *state, taken, pieces)
        updated[0] = np.uint64(updated[0])
        updated[1] = np.uint64(updated[1])
        child = flat.copy()
        native._pop(flat, move, taken, pieces, captured)
        if not np.array_equal(flat, original):
            raise AssertionError(f"Undo mismatch at ply {number}")
        flat = child
        state = tuple(updated)
        if not referee.make_move(attacker, native._move_list(move)) or native._nested(flat) != referee.board:
            raise AssertionError(f"Board or captures mismatch at ply {number}")
        if state[0] != referee.compute_zobrist_hash(True) or state[1] != referee.compute_zobrist_hash(False):
            raise AssertionError(f"Dual incremental hash mismatch at ply {number}")
        if state != gm._root_state(flat):
            raise AssertionError(f"Material or king metadata mismatch at ply {number}")
        attacker = not attacker
        winner = native._winner(state[2], state[4])
        hash_next = state[0 if attacker else 1]
        own_verdict = (native._CELLS[winner] if winner else
                       "DRAW" if not native._has_legal(flat, attacker)
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
            flat = native._flat(referee.board)
            state = gm._root_state(flat)
            attacker = True
    return {"plies": steps, "differences": 0, "terminalGames": dict(endings),
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


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("rules", "timing"))
    parser.add_argument("--repo", type=Path, default=Path(Tablut.__file__).resolve().parent)
    args = parser.parse_args()
    if args.mode == "rules":
        existing_tests(args.repo)
        result = {"rulesTests": 40, "reference": RULES_TEST_COMMIT, **differential()}
    else:
        result = timing()
    dest = HERE / "results"
    dest.mkdir(exist_ok=True)
    (dest / f"numba-{args.mode}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
