"""The submission's native rules, weights and search regressions."""

from functools import wraps
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import Tablut
import Team19
import Team19_engine as engine
import Team19_eval as evaluation
import Team19_search as search
from Team19_tt import lookup
from Team19_weights import DEFAULT_WEIGHTS, FEATURE_NAMES, native_weights

from . import openings
from .bots import grandmaster_numba as gm


class NativeEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.buffers = SimpleNamespace(queue=np.empty(81, dtype=np.int16),
                                       seen=np.empty(81, dtype=np.bool_))

    def score(self, flat, side, weights):
        _, _, attackers, defenders, king = engine._describe(flat)
        return evaluation._native_evaluate(
            flat, side, attackers, defenders, king, weights, self.buffers.queue,
            self.buffers.seen, engine._NEAR, engine._EDGE, evaluation._native_piece_bonus(weights),
        )

    def test_warm_start_matches_grandmaster_on_all_openings(self):
        weights = native_weights((
            520.0, 625.0, -1050.0, -400.0, 2750.0, -7300.0, -4300.0, -2100.0, -900.0,
            160.0, -160.0, -450.0, -1400.0, -200.0, 35.0, -15.0, 100.0, 330.0, -75.0,
            -105.0, 0.0, 0.0,
        ))
        for opening in [openings.standard(), *openings.load_suite()]:
            flat = engine._flat(openings.start(opening).referee.board)
            _, _, attackers, defenders, king = engine._describe(flat)
            for side in (True, False):
                with self.subTest(opening=opening.name, attacker=side):
                    expected = gm._evaluate.py_func(flat, side, attackers, defenders, king,
                                                     self.buffers)
                    self.assertEqual(self.score(flat, side, weights), expected)

    def test_new_weights_measure_blocked_king_rays(self):
        board = [["."] * 9 for _ in range(9)]
        board[4][4] = "K"
        board[4][7] = board[8][8] = "A"
        weights = native_weights(np.zeros(len(FEATURE_NAMES)))
        weights[FEATURE_NAMES.index("blocked_king_directions")] = 50
        weights[FEATURE_NAMES.index("king_ray_pressure")] = 10
        self.assertEqual(self.score(engine._flat(board), True, weights), 110)
        board[4][7], board[4][6] = ".", "A"
        self.assertEqual(self.score(engine._flat(board), True, weights), 120)

    def test_native_weights_are_finite_independent_float64_vectors(self):
        self.assertEqual(len(FEATURE_NAMES), len(DEFAULT_WEIGHTS))
        source = np.asarray(DEFAULT_WEIGHTS, dtype=np.float32)
        result = native_weights(source)
        self.assertEqual(result.dtype, np.dtype("float64"))
        self.assertFalse(np.shares_memory(source, result))
        for invalid in ([0], [np.inf] * len(source), ["0"] * len(source),
                        [10_001] * len(source)):
            with self.assertRaises(ValueError):
                native_weights(invalid)


class NativeRulesTests(unittest.TestCase):
    def test_defender_can_win_by_taking_the_penultimate_attacker(self):
        game = Tablut.Tablut()
        board = [["."] * 9 for _ in range(9)]
        board[4][4] = "K"
        for row, col in ((2, 4), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4)):
            board[row][col] = "D"
        board[2][3] = board[8][8] = "A"
        game.board = board
        flat = engine._flat(board)
        scratch = np.empty(engine._MOVE_CAPACITY, dtype=np.int16)
        code = engine._immediate_win(flat, False, 2, 40, scratch)
        self.assertGreaterEqual(code, 0)
        self.assertTrue(game.make_move(False, engine._move_list(code)))
        self.assertTrue(game.check_if_game_over(True))
        self.assertEqual(game.winner, "D")


def _escape_threat():
    board = [["."] * 9 for _ in range(9)]
    board[4][4] = "X"
    board[1][4] = "K"
    board[1][3] = board[1][5] = board[2][4] = "D"
    board[0][1] = board[7][7] = "A"
    return board


class NativeSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = search._NativeSearch(Team19._INITIAL_BOARD)

    def setUp(self):
        self.bot.reset_game()

    def test_player_completes_depth_one_when_the_soft_budget_is_zero(self):
        with patch.object(Tablut, "TIME_BUDGET", 10.0):
            player = Team19.Player()
        board = _escape_threat()
        player._last_board = [row[:] for row in board]
        player._last_is_attacker = True
        player.moves_played = 100
        player.remaining = 0.06
        self.assertEqual(player._step_budget(board, native=True), 0.0)
        self.assertEqual(player.play_attacker(board), [[0, 1], [0, 4]])
        self.assertIsNone(player.last_error)
        self.assertEqual(player.last_search.depth, 1)
        self.assertFalse(player.last_search.fallback)
        self.assertIn(engine.compute_hash(board, True), player.history)
        after = engine.apply_move(board, [[0, 1], [0, 4]], True)
        self.assertIn(engine.compute_hash(after, False), player.history)

    def test_long_game_reserves_referee_time_with_callback_overhead(self):
        for budget in (10.0, 60.0):
            with self.subTest(budget=budget), patch.object(Tablut, "TIME_BUDGET", budget):
                player = Team19.Player()
                clock = SimpleNamespace(now=0.0)
                board = _escape_threat()
                referee = Tablut.Tablut()
                referee_remaining = budget

                @wraps(player._native_choice)
                def play_depth_one(position, side, deadline, history, weights):
                    # Model using the reserved search share and at least the
                    # cost of the mandatory native pass on every callback.
                    clock.now += max(0.0004, deadline - clock.now)
                    return player._native_choice(position, side, 0.0, history, weights)

                player._choose_best_move = play_depth_one
                with patch.object(Team19, "time", SimpleNamespace(perf_counter=lambda: clock.now)):
                    for _ in range(520):
                        if player.remaining < 0.5:
                            self.assertLessEqual(player._step_budget(board, native=True), 0.008)
                        started = clock.now
                        reply = player.play_attacker(board)
                        referee_remaining -= clock.now - started + 0.00015
                        self.assertGreater(referee_remaining, 0.05)
                        self.assertLess(player.remaining, referee_remaining)
                        self.assertIsNone(player.last_error)
                        self.assertEqual(player.last_search.depth, 1)
                        self.assertFalse(player.last_search.fallback)
                        referee.board = [row[:] for row in board]
                        self.assertTrue(referee.make_move(True, reply))
                self.assertEqual(player.moves_played, 520)
                self.assertEqual(player.remaining, 0.0)

    def test_quiescence_searches_the_quiet_escape_block(self):
        flat, state = search._native_position(_escape_threat())
        self.bot.work.counts.fill(0)
        value, move = self.bot._iteration(
            flat, True, state, 0, 1, -2 * search._MATE, 2 * search._MATE,
            float("inf"), np.empty(0, dtype=np.uint64), True,
        )
        self.assertEqual(engine._move_list(move), [[0, 1], [0, 4]])
        self.assertGreater(value, -search._MATE + 2)

    def test_transposition_move_survives_turns_but_cannot_override_repetition(self):
        board = Tablut.Tablut().board
        result = self.bot.choose(board, True, 0.0, set())
        work = self.bot.work
        flat, state = search._native_position(board)
        stamp = state[0]
        table = work.tt
        slot = lookup(table, stamp)
        self.assertGreaterEqual(slot, 0)
        hint = table.moves[slot]
        self.assertEqual(engine._move_list(hint), result)
        table.generation[0] += 1
        self.assertEqual(lookup(table, stamp), slot)
        work.counts.fill(0)
        played = np.array([stamp], dtype=np.uint64)
        score, move = search._native_visit(
            flat, True, *state, 2, 4, 1, -2 * search._MATE, 2 * search._MATE,
            float("inf"), played, work, True, 0, False,
        )
        self.assertEqual((score, move, work.counts[search._NODE_COUNT]), (0, -1, 1))
        self.bot.reset_game()
        self.assertEqual(lookup(table, stamp), -1)
        self.assertEqual(table.moves[slot], hint)

    def test_native_search_prefers_a_proven_escape_to_repetition(self):
        board = [["."] * 9 for _ in range(9)]
        board[4][4], board[7][2] = "X", "K"
        board[5][6] = board[3][6] = "A"
        board[6][2] = board[7][1] = board[7][3] = "D"
        winning = engine.apply_move(board, [[7, 2], [8, 2]], False)
        move = self.bot.choose(board, False, 0.0, {engine.compute_hash(winning, True)})
        self.assertEqual(move, [[7, 2], [8, 2]])
        self.assertEqual(engine.check_winner(engine.apply_move(board, move, False), True), "D")
        self.assertEqual(self.bot.last_search.score, -search.WIN_SCORE)

    def test_attacker_avoids_a_cheap_repetition_but_draws_to_prevent_a_loss(self):
        board = Tablut.Tablut().board
        first = self.bot.choose(board, True, 0.0, set())
        child = engine.apply_move(board, first, True)
        self.assertNotEqual(self.bot.choose(board, True, 0.0, {engine.compute_hash(child, False)}),
                            first)
        self.assertGreater(self.bot.last_search.score, search._ATTACKER_REPETITION_SCORE)

        board = _escape_threat()
        block = [[0, 1], [0, 4]]
        child = engine.apply_move(board, block, True)
        self.assertEqual(self.bot.choose(board, True, 0.0, {engine.compute_hash(child, False)}),
                         block)
        self.assertEqual(self.bot.last_search.score, search._ATTACKER_REPETITION_SCORE)

        # The experiment API continues to value the same official draw at zero.
        result = search.search_fixed_depth(
            board, True, 2, functions=search._load_functions(),
            game_history={engine.compute_hash(child, False)},
        )
        self.assertEqual((result.move, result.score), (block, 0.0))

    def test_siege_score_rewards_reaching_a_guard_flank(self):
        board = [list(row) for row in (
            ".........", ".........", "......A..", "....AADD.",
            "....XAD..", "...AAAKDA", "....AAAA.", ".........", ".........",
        )]
        staged = engine.apply_move(board, [[5, 8], [4, 8]], True)

        def score(position, siege):
            flat, state = search._native_position(position)
            self.bot.work.counts.fill(0)
            return -self.bot._iteration(
                flat, False, state, 0, 0, -2 * search._MATE, 2 * search._MATE,
                float("inf"), np.empty(0, dtype=np.uint64), True, siege=siege,
            )[0]

        self.assertGreater(score(staged, True) - score(staged, False),
                           score(board, True) - score(board, False))
