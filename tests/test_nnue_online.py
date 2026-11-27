"""Hard pre-release checks for the opt-in Numba NNUE search path."""

from __future__ import annotations

import math
import os
from pathlib import Path
from statistics import median
import time
import unittest
from unittest.mock import patch

import numpy as np

import Tablut
import Team19
import Team19_engine as engine
import Team19_nnue as nnue
import Team19_search as search


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "member5-c750-k6000-l0p5-tanh-precalib5-center1e3-listwise20-seed20261127"
DATA = ROOT / "tools/nnue/results/volume1200"
WEIGHTS = DATA / RUN_ID / "best_weights.npz"
SCRIPTED_OPENING = (
    [[0, 3], [0, 2]],
    [[2, 4], [2, 3]],
    [[0, 5], [0, 6]],
    [[3, 4], [2, 4]],
    [[1, 4], [1, 3]],
    [[4, 4], [3, 4]],
)


def opening_positions():
    board = Tablut.Tablut().board
    positions = [([row[:] for row in board], True)]
    for index, move in enumerate(SCRIPTED_OPENING):
        board = engine.apply_move(board, move, index % 2 == 0)
        positions.append(([row[:] for row in board], index % 2 == 1))
    return positions


def fixed_iteration(bot, board, attacker, depth=3):
    flat, state = search._native_position(board)
    work = bot.work
    work.counts.fill(0)
    work.killers.fill(-1)
    work.history.fill(0)
    from Team19_tt import advance_generation

    advance_generation(work.tt)
    repetition = search._ATTACKER_REPETITION_SCORE if attacker else 0
    siege = attacker and state[3] <= 6 and state[2] >= state[3] + 4
    value, move = bot._iteration(
        flat, attacker, state, depth, 0, -2 * search._MATE, 2 * search._MATE,
        math.inf, np.empty(0, dtype=np.uint64), True, repetition, siege,
    )
    return value, move, work.counts.copy()


def zero_head_search(board):
    work = nnue.make_nnue_work(
        True,
        head_weight=np.zeros((2, nnue.HIDDEN), dtype=np.float32),
        head_bias=np.zeros(2, dtype=np.float32),
    )
    return search._NativeSearch(board, nnue_work=work)


def incremental_audit(steps=240):
    """Play legal games and compare stack updates with full recomputation."""
    work = nnue.make_nnue_work(True)
    board = engine._flat(Tablut.Tablut().board)
    attacker = True
    rng = np.random.default_rng(20261127)
    moves = np.empty(engine._MOVE_CAPACITY, dtype=np.int16)
    captured_at = np.empty(4, dtype=np.int16)
    captured_pieces = np.empty(4, dtype=np.int8)
    reference = np.empty(nnue.HIDDEN, dtype=np.float32)
    max_error = 0.0
    capture_steps = king_steps = castle_steps = games = 0
    completed = 0
    while completed < steps:
        if games == 0 and completed == 0:
            games = 1
        state = engine._describe(board)
        winner = engine._winner(state[2], state[4])
        count = engine._legal(board, attacker, moves)
        if winner or not count:
            board = engine._flat(Tablut.Tablut().board)
            attacker = True
            games += 1
            continue
        if completed < len(SCRIPTED_OPENING):
            start, end = SCRIPTED_OPENING[completed]
            move = (start[0] * 9 + start[1]) * 81 + end[0] * 9 + end[1]
            if move not in moves[:count]:
                raise AssertionError("Scripted opening move is not legal")
        else:
            # Stable random choice; scripted opening already forces king/castle
            # coverage and real games provide many subsequent capture positions.
            move = int(moves[int(rng.integers(count))])
        source = move // 81
        moved = board[source]
        ply = completed % 8
        nnue._nnue_full(board, work.hidden_weight, work.hidden_bias, work.acc_stack[ply])
        ha, hd, a, d, k = engine._describe(board)
        ha, hd = np.uint64(ha), np.uint64(hd)
        ha, hd, a, d, k, captured = engine._push(
            board, move, ha, hd, a, d, k, captured_at, captured_pieces,
        )
        nnue._nnue_push(
            board, move, ply, captured_at, captured_pieces, captured, work,
        )
        nnue._nnue_full(board, work.hidden_weight, work.hidden_bias, reference)
        error = float(np.max(np.abs(work.acc_stack[ply + 1] - reference)))
        max_error = max(max_error, error)
        if error >= 1e-4:
            raise AssertionError(f"Incremental/full mismatch at step {completed}: {error}")
        capture_steps += captured > 0
        king_steps += moved == engine._K
        castle_steps += moved == engine._K and source == engine._CASTLE
        attacker = not attacker
        completed += 1
    if completed < 200 or not capture_steps or not king_steps or not castle_steps:
        raise AssertionError("The real-game audit did not cover every required move class")
    return {
        "steps": completed, "games": games, "capture_steps": int(capture_steps),
        "king_steps": int(king_steps), "castle_departures": int(castle_steps),
        "max_abs_accumulator_error": max_error,
    }


def validation_boards():
    path = DATA / "team19_nnue_val.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return np.array(data["boards"], copy=True), "volume1200_validation"
    boards = [engine._flat(board) for board, _ in opening_positions()]
    return np.stack(boards), "scripted_regression_positions"


def offline_weights():
    """Use the sealed NPZ when present, otherwise verify the deployed arrays."""
    from tools.nnue.model import load_weights

    if WEIGHTS.is_file():
        return load_weights(WEIGHTS), "sealed_npz"
    import Team19_nnue_weights as static

    return {
        "hidden_weight": static.HIDDEN_WEIGHT_BY_FEATURE.T,
        "hidden_bias": static.HIDDEN_BIAS,
        "head_weight": static.HEAD_WEIGHT,
        "head_bias": static.HEAD_BIAS,
        "residual_clip": static.RESIDUAL_CLIP,
        "eval_scale": static.EVAL_SCALE,
        "output_activation": np.array("tanh"),
    }, "read_only_static_arrays"


def measure_nps(bot, positions, attacker, seconds, depth=3):
    selected = [board for board, side in positions if side == attacker]
    for board in selected:
        fixed_iteration(bot, board, attacker, depth)
    nodes = 0
    iterations = 0
    started = time.perf_counter()
    while True:
        for board in selected:
            _, _, counts = fixed_iteration(bot, board, attacker, depth)
            nodes += int(counts[search._NODE_COUNT])
            iterations += 1
        elapsed = time.perf_counter() - started
        if elapsed >= seconds:
            break
    return {"nodes": nodes, "seconds": elapsed, "nps": nodes / elapsed,
            "fixed_depth": depth, "iterations": iterations}


def nps_audit(seconds, rounds=3):
    """Interleave paired measurements and use medians to reduce OS scheduling bias."""
    if rounds < 1:
        raise ValueError("NPS audit needs at least one round")
    positions = opening_positions()
    baseline = search._NativeSearch(Team19._INITIAL_BOARD)
    candidate = search._NativeSearch(Team19._INITIAL_BOARD, use_nnue=True)
    result = {}
    for attacker, label in ((True, "attacker"), (False, "defender")):
        samples = []
        for round_index in range(rounds):
            # Reverse order every round so thermal throttling or background load
            # does not systematically penalise whichever engine runs second.
            pair = ((baseline, "v4"), (candidate, "nnue"))
            if round_index % 2:
                pair = pair[::-1]
            sample = {}
            for bot, engine_name in pair:
                sample[engine_name] = measure_nps(bot, positions, attacker, seconds)
            sample["ratio"] = sample["nnue"]["nps"] / sample["v4"]["nps"]
            samples.append(sample)
        base = dict(samples[0]["v4"])
        online = dict(samples[0]["nnue"])
        base["nps"] = median(sample["v4"]["nps"] for sample in samples)
        online["nps"] = median(sample["nnue"]["nps"] for sample in samples)
        for engine_name, aggregate in (("v4", base), ("nnue", online)):
            for field in ("nodes", "seconds", "iterations"):
                aggregate[field] = sum(sample[engine_name][field] for sample in samples)
        result[label] = {
            "v4": base,
            "nnue": online,
            "ratio": median(sample["ratio"] for sample in samples),
            "round_ratios": [sample["ratio"] for sample in samples],
        }
    total_nodes = sum(result[side]["v4"]["nodes"] + result[side]["nnue"]["nodes"]
                      for side in result)
    result["geometric_mean_ratio"] = math.sqrt(
        result["attacker"]["ratio"] * result["defender"]["ratio"]
    )
    result["measured_nodes"] = total_nodes
    result["rounds_per_side"] = rounds
    result["numba_threads"] = os.environ.get("NUMBA_NUM_THREADS", "1")
    return result


class NNUEOnlineTests(unittest.TestCase):
    def test_zero_heads_reproduce_frozen_v4_fixed_depth_node_for_node(self):
        positions = opening_positions()
        baseline = search._NativeSearch(Team19._INITIAL_BOARD)
        zero = zero_head_search(Team19._INITIAL_BOARD)
        for board, attacker in positions:
            for depth in (1, 2, 3):
                with self.subTest(attacker=attacker, depth=depth):
                    left = fixed_iteration(baseline, board, attacker, depth)
                    right = fixed_iteration(zero, board, attacker, depth)
                    self.assertEqual(left[0], right[0])
                    self.assertEqual(left[1], right[1])
                    np.testing.assert_array_equal(left[2], right[2])

    def test_incremental_stack_matches_full_recalculation_over_240_real_plies(self):
        result = incremental_audit(240)
        self.assertGreaterEqual(result["steps"], 200)
        self.assertLess(result["max_abs_accumulator_error"], 1e-4)
        print("INCREMENTAL_AUDIT", result, flush=True)

    def test_numba_matches_offline_numpy_on_validation_positions(self):
        from tools.nnue.model import numpy_residual

        boards, source = validation_boards()
        weights, weight_source = offline_weights()
        work = nnue.make_nnue_work(True)
        maximum = 0.0
        for side in (0, 1):
            expected = numpy_residual(
                weights, boards, np.full(len(boards), side, dtype=np.int64),
            )
            actual = np.asarray([
                nnue.residual_flat(board, bool(side), work) for board in boards
            ], dtype=np.float32)
            maximum = max(maximum, float(np.max(np.abs(expected - actual))))
        self.assertLess(maximum, 5e-4)
        print("OFFLINE_ALIGNMENT", {"source": source, "weights": weight_source,
                                  "positions_per_head": len(boards),
                                  "max_abs_residual_error": maximum}, flush=True)

    def test_single_thread_attacker_and_defender_nps_hard_floor(self):
        seconds = float(os.environ.get("TEAM19_NNUE_NPS_SECONDS", "0.12"))
        rounds = int(os.environ.get("TEAM19_NNUE_NPS_ROUNDS", "3"))
        result = nps_audit(seconds, rounds)
        # Always expose both sides and every paired round, including on failure.
        print("NPS_AUDIT", result, flush=True)
        for side in ("attacker", "defender"):
            self.assertGreaterEqual(result[side]["ratio"], 0.65, side)


class NNUEIntegrationTests(unittest.TestCase):
    def test_default_player_is_off_and_book_nnue_combination_is_independent(self):
        default = Team19.Player()
        self.assertFalse(default._book_enabled)
        self.assertFalse(default._nnue_enabled)
        self.assertFalse(default._native.work.nnue.enabled)
        combined = Team19.Player(use_book=True, use_nnue=True)
        self.assertTrue(combined._book_enabled)
        self.assertTrue(combined._native.work.nnue.enabled)
        with patch.object(Tablut, "TIME_BUDGET", 10.0):
            move = combined.play_attacker(Tablut.Tablut().board)
        self.assertTrue(combined.last_book_hit)
        self.assertEqual(move, [[0, 5], [1, 5]])

    def test_nnue_without_book_executes_real_native_search(self):
        player = Team19.Player(use_nnue=True)
        referee = Tablut.Tablut()
        with patch.object(Tablut, "TIME_BUDGET", 1.0):
            move = player.play_attacker(referee.board)
        self.assertIsNone(player.last_error)
        self.assertIsNotNone(player.last_search)
        self.assertFalse(player.last_book_hit)
        self.assertIn(move, engine.get_legal_moves(referee.board, True))

    def test_arena_exposes_both_blind_review_entrants(self):
        from tools.arena import runner

        self.assertFalse(runner.TEAMS["Team19-v4"].use_nnue)
        self.assertTrue(runner.TEAMS["Team19-NNUE"].use_nnue)
        self.assertFalse(runner.TEAMS["Team19-NNUE"].use_book)
        self.assertTrue(runner.TEAMS["Team19-Book-NNUE"].use_book)
        self.assertTrue(runner.TEAMS["Team19-Book-NNUE"].use_nnue)

    def test_static_arrays_are_read_only_and_runtime_module_has_no_forbidden_imports(self):
        import Team19_nnue_weights as weights

        for array in (weights.HIDDEN_WEIGHT_BY_FEATURE, weights.HIDDEN_BIAS,
                      weights.HEAD_WEIGHT, weights.HEAD_BIAS):
            self.assertFalse(array.flags.writeable)
            self.assertTrue(array.flags.c_contiguous)
        source = (ROOT / "Team19_nnue.py").read_text(encoding="utf-8")
        for forbidden in ("import torch", "import tools", "subprocess", "cache=True"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
