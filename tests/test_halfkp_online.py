"""Coverage for the opt-in HalfKP search and its portable weights."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import Tablut
import Team19
import Team19_engine as engine
import Team19_halfkp as halfkp
import Team19_search as search
from Team19_halfkp_weights import load_weights
from tools.arena import runner
from tools.arena.openings import standard, start
from tools.nnue.verify_halfkp_online import (
    ALIGNMENT_LIMIT, RUN, VALIDATION, alignment_audit, incremental_audit, zero_head_audit,
)


class HalfKPOnlineTests(unittest.TestCase):
    def test_zero_head_reproduces_v4(self):
        result = zero_head_audit()
        self.assertGreaterEqual(result["positions"], 7)
        self.assertEqual(result["matching_searches"], result["positions"] * 3)
        self.assertGreater(result["nodes_per_version"], 0)

    def test_incremental_matches_rebuild_and_pop_on_referee_moves(self):
        result = incremental_audit()
        self.assertGreaterEqual(result["steps"], 200)
        self.assertLess(result["max_abs_accumulator_error"], 1e-4)
        for field in ("ordinary", "single_captures", "multiple_captures", "king_moves",
                      "throne_departures"):
            self.assertGreater(result[field], 0, field)

    def test_halfkp_incremental_rare_moves(self):
        empty = "........."
        # Coordinates are (row, column). Each step specifies its exact victims.
        cases = (
            ("double capture between attackers",
             ("...A.....", "...D.....", ".AD......", empty, "....K....",
              "...A.....", empty, empty, empty),
             True, (((5, 3), (2, 3), {(1, 3): "D", (2, 2): "D"}),)),
            ("triple capture with the empty throne as an anchor",
             (empty, empty, "..A...K..", "..D......", "A..DX....",
              "..D......", "..A......", empty, empty),
             True, (((4, 0), (4, 2), {(3, 2): "D", (4, 3): "D", (5, 2): "D"}),)),
            ("king leaves the throne and captures an attacker",
             (empty, ".A.......", empty, empty, "....K.AD.",
              empty, empty, ".......A.", empty),
             False, (((4, 4), (4, 5), {(4, 6): "A"}),)),
            ("off-throne king captures one attacker against the empty throne",
             (empty, ".A.......", "..K......", empty, "...AX....",
              empty, empty, ".......A.", empty),
             False, (((2, 2), (4, 2), {(4, 3): "A"}),)),
            ("off-throne king captures two attackers",
             (empty, ".A....D..", "......A..", "....DA...", "....X....",
              empty, "......K..", ".......A.", empty),
             False, (((6, 6), (3, 6), {(2, 6): "A", (3, 5): "A"}),)),
            ("king captured on an ordinary square",
             ("....A....", empty, "..AK.....", empty, "....X....",
              empty, empty, empty, empty),
             True, (((0, 4), (2, 4), {(2, 3): "K"}),)),
            ("king captured next to the throne by three attackers and X",
             (empty, empty, "....A....", "...AK....", "....X....",
              ".....A...", empty, empty, empty),
             True, (((5, 5), (3, 5), {(3, 4): "K"}),)),
            ("king captured on the throne by four attackers",
             (empty, empty, empty, "....A....", "...AK..A.",
              "....A....", empty, empty, empty),
             True, (((4, 7), (4, 5), {(4, 4): "K"}),)),
            ("three attackers around the throne let the king anchor a guard capture",
             (empty, empty, empty, "....A....", "...AKD..A",
              "....A....", empty, empty, empty),
             True, (((4, 8), (4, 6), {(4, 5): "D"}),)),
            ("nested triple capture, king slide, and capture",
             (empty, empty, "..A...K..", "..D......", "A..DX....",
              "..D.D....", "..A......", empty, empty),
             True, (
                 ((4, 0), (4, 2), {(3, 2): "D", (4, 3): "D", (5, 2): "D"}),
                 ((2, 6), (2, 5), {}),
                 ((6, 2), (6, 4), {(5, 4): "D"}),
             )),
        )
        max_acc_error = max_head_error = 0.0
        checked_positions = checked_moves = 0
        for name, rows, attacker, steps in cases:
            with self.subTest(case=name):
                acc_error, head_error = self._check_rare_move_line(rows, attacker, steps)
                max_acc_error = max(max_acc_error, acc_error)
                max_head_error = max(max_head_error, head_error)
                checked_positions += 1
                checked_moves += len(steps)
        print(f"HALFKP_RARE_MOVES positions={checked_positions} moves={checked_moves} "
              f"max_abs_accumulator_error={max_acc_error:.9g} "
              f"max_abs_head_error={max_head_error:.9g}", flush=True)

    def _check_rare_move_line(self, rows, attacker, steps):
        referee = Tablut.Tablut()
        referee.board = [list(row) for row in rows]
        referee.past_positions = [referee.compute_zobrist_hash(attacker)]
        board = engine._flat(referee.board)
        state = engine._describe(board)
        work = halfkp.make_halfkp_work(len(steps) + 1)
        rebuilt = halfkp.make_halfkp_work(1)
        work.acc_stack.fill(np.nan)
        work.ksq_stack.fill(-2)
        halfkp._halfkp_root(board, state[-1], work)
        squares = np.empty((len(steps), 4), dtype=np.int16)
        pieces = np.empty((len(steps), 4), dtype=np.int8)
        undo = []
        max_acc_error = max_head_error = 0.0
        for ply, (origin, destination, victims) in enumerate(steps):
            context = f"ply {ply + 1}: {origin} -> {destination}"
            before = board.copy()
            parent_acc = work.acc_stack[:ply + 1].copy()
            parent_king = work.ksq_stack[:ply + 1].copy()
            self.assertTrue(referee.make_move(attacker, [list(origin), list(destination)]),
                            context)
            expected_board = engine._flat(referee.board)
            expected_state = engine._describe(expected_board)
            king_captured = "K" in victims.values()
            self.assertEqual(referee.check_if_game_over(not attacker), king_captured, context)
            self.assertEqual(referee.winner, "A" if king_captured else None, context)

            source = origin[0] * 9 + origin[1]
            target = destination[0] * 9 + destination[1]
            move = source * engine._AREA + target
            ha, hd, a, d, king = state
            *child_state, captures = engine._push(
                board, move, np.uint64(ha), np.uint64(hd), a, d, king, squares[ply], pieces[ply],
            )
            state = tuple(child_state)
            self.assertEqual(captures, len(victims), context)
            self.assertEqual({
                divmod(int(square), 9): engine._CELLS[piece]
                for square, piece in zip(squares[ply, :captures], pieces[ply, :captures])
            }, victims, context)
            np.testing.assert_array_equal(board, expected_board, err_msg=context)
            self.assertEqual(state, expected_state, context)

            halfkp._halfkp_push(board, source, target, state[-1], ply,
                                squares[ply], pieces[ply], captures, work)
            halfkp._halfkp_root(expected_board, expected_state[-1], rebuilt)
            self.assertEqual(work.ksq_stack[ply + 1], rebuilt.ksq_stack[0], context)
            acc_error = float(np.max(np.abs(work.acc_stack[ply + 1] - rebuilt.acc_stack[0])))
            self.assertLess(acc_error, 1e-4, context)
            max_acc_error = max(max_acc_error, acc_error)
            for side in (False, True):
                delta = halfkp._halfkp_residual(side, ply + 1, work)
                reference = halfkp._halfkp_residual(side, 0, rebuilt)
                head_error = abs(delta - reference)
                self.assertLess(head_error, ALIGNMENT_LIMIT, f"{context}, {'A' if side else 'D'} head")
                max_head_error = max(max_head_error, head_error)
            np.testing.assert_array_equal(work.acc_stack[:ply + 1], parent_acc, err_msg=context)
            np.testing.assert_array_equal(work.ksq_stack[:ply + 1], parent_king, err_msg=context)
            undo.append((move, captures, before, parent_acc, parent_king))
            attacker = not attacker

        # Unwind in LIFO order using each frame's original capture buffers.
        for ply in range(len(undo) - 1, -1, -1):
            move, captures, before, parent_acc, parent_king = undo[ply]
            engine._pop(board, move, squares[ply], pieces[ply], captures)
            context = f"undo ply {ply + 1}"
            np.testing.assert_array_equal(board, before, err_msg=context)
            np.testing.assert_array_equal(work.acc_stack[:ply + 1], parent_acc, err_msg=context)
            np.testing.assert_array_equal(work.ksq_stack[:ply + 1], parent_king, err_msg=context)
        return max_acc_error, max_head_error

    def test_exported_heads_match_frozen_npz(self):
        weights = RUN / "stage3.npz"
        if not weights.is_file() or not VALIDATION.is_file():
            self.skipTest("Local sealed weights or validation NPZ is missing")
        result = alignment_audit(weights, VALIDATION)
        self.assertEqual(result["total_predictions"], 6332)
        self.assertEqual(result["heads"]["defender"]["index"], 0)
        self.assertEqual(result["heads"]["attacker"]["index"], 1)
        self.assertLess(result["max_abs_error"], ALIGNMENT_LIMIT)

    def test_read_only_weights_are_shared_and_workspaces_are_independent(self):
        arrays = load_weights()
        self.assertEqual(arrays[0].shape, (13122, 64))
        self.assertEqual(arrays[1].shape, (2210,))
        for array in arrays:
            self.assertEqual(array.dtype, np.float32)
            self.assertTrue(array.flags.c_contiguous)
            self.assertFalse(array.flags.writeable)
        left = halfkp.make_halfkp_work(search._NATIVE_PLY_SLOTS)
        right = halfkp.make_halfkp_work(search._NATIVE_PLY_SLOTS)
        self.assertIs(left.embedding, right.embedding)
        self.assertFalse(np.shares_memory(left.acc_stack, right.acc_stack))
        self.assertFalse(np.shares_memory(left.ksq_stack, right.ksq_stack))

    def test_default_and_legacy_modes_keep_their_own_search(self):
        default = Team19.Player()
        self.assertFalse(default._halfkp_enabled)
        self.assertIsNone(default._native.work.halfkp)
        old = Team19.Player(use_nnue=True, use_book=True)
        self.assertTrue(old._native.work.nnue.enabled)
        self.assertIsNone(old._native.work.halfkp)
        with self.assertRaisesRegex(ValueError, "one residual"):
            Team19.Player(use_nnue=True, use_halfkp=True)

    def test_arena_loads_both_halfkp_variants(self):
        game = start(standard())
        board = game.referee.board
        with patch.object(Tablut, "TIME_BUDGET", 1.0):
            for alias, use_book in (("halfkp", False), ("bookhalfkp", True)):
                name = runner.TEAM_NAMES[alias]
                entrant = runner.TEAMS[name]
                self.assertTrue(entrant.use_halfkp)
                self.assertFalse(entrant.use_nnue)
                self.assertEqual(entrant.use_book, use_book)
                player = runner._player(name, 0, game, True)
                self.assertIsNone(player.native_init_error)
                self.assertIsNotNone(player._native.work.halfkp)
                move = player.play_attacker(board)
                self.assertIn(move, engine.get_legal_moves(board, True))
                self.assertIsNone(player.last_error)
                self.assertEqual(player.last_book_hit, use_book)
                if use_book:
                    self.assertEqual(move, [[0, 5], [1, 5]])
                else:
                    self.assertIsNotNone(player.last_search)
                    self.assertFalse(player.last_search.fallback)


if __name__ == "__main__":
    unittest.main()
