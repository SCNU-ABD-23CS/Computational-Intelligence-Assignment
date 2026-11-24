"""Replay regressions for quiet escapes, blocker captures and critical time use."""

from functools import wraps
from math import inf
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import Tablut
import Team19
import Team19_search as search
from Team19_engine import _move_list

from .snapshots.v_be23011 import Team19_search as v3
from .test_team19_v2 import _escape_threat


# Official 60-second v3 vs Grandmaster-Plus logs. Each token encodes r1,c1,r2,c2.
_O12 = (
    "4181 3436 4737 2423 0535 2325 3034 4575 3767 4373 0343 4252 4042 2524 3834 7525 "
    "1424 4445 4363 4535 4838 3536 0406 3616"
)
_O09 = (
    "7473 4353 7372 4626 0343 3433 8363 4442 1412 4243 8565 3332 1213 4535 8464 2656 "
    "0402 3252 0222 5676 1323 2425 5854 7673 4745 7376 4846 3536 6566 7675 5455 5373 "
    "5060 7565 6434 6564 3454 7374 5453 6463 3033 6364 7273 5272 5553 6463"
)
_O11 = "1410 4535 4757 3537 0545 3431 5053 4252 8353 6463 4142 3141 3034 4333 3436 4442 1012"


def _replay(line, plies):
    game = Tablut.Tablut()
    game.past_positions = [game.compute_zobrist_hash(True)]
    for index, code in enumerate(line.split()[:plies]):
        move = [list(map(int, code[:2])), list(map(int, code[2:]))]
        assert game.make_move(index % 2 == 0, move)
        assert not game.check_if_game_over(index % 2 == 1)
    return game


def _depth(kernel, bot, game, attacker, depth, qleft=4):
    flat, state = kernel._native_position(game.board)
    bot.reset_game()
    kernel.advance_generation(bot.work.tt)
    bot.work.killers.fill(-1)
    bot.work.history.fill(0)
    bot.work.counts.fill(0)
    value, move = bot._iteration(
        flat, attacker, state, depth, qleft, -2 * search._MATE, 2 * search._MATE,
        inf, np.array(sorted(game.past_positions), dtype=np.uint64), True,
        search._ATTACKER_REPETITION_SCORE if attacker else 0,
    )
    return value, _move_list(move) if move >= 0 else None, int(bot.work.counts[0])


class TacticalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        board = Tablut.Tablut().board
        cls.current = search._NativeSearch(board)
        cls.baseline = v3._NativeSearch(board)

    def test_o11_quiet_escape_threat_enters_the_static_search(self):
        game = _replay(_O11, 17)
        old_score, old_move, _ = _depth(v3, self.baseline, game, False, 0, qleft=2)
        score, move, nodes = _depth(search, self.current, game, False, 0, qleft=2)
        self.assertIsNone(old_move)
        self.assertGreater(score, old_score)
        self.assertLess(nodes, 1000)
        self.assertTrue(game.make_move(False, move))
        r, c = next((r, row.index("K")) for r, row in enumerate(game.board) if "K" in row)
        self.assertTrue(any(
            game.make_move(False, [[r, c], target])
            for target in ([0, c], [8, c], [r, 0], [r, 8])
        ))
        self.assertTrue(game.check_if_game_over(True))
        self.assertEqual(game.winner, "D")

    def test_o12_keeps_a_reply_to_the_double_lane_turn(self):
        game = _replay(_O12, 18)
        _, old_move, old_nodes = _depth(v3, self.baseline, game, True, 3)
        _, move, nodes = _depth(search, self.current, game, True, 3)
        self.assertLess(max(nodes, old_nodes), 55_000)
        self.assertEqual(old_move, [[4, 3], [6, 3]])
        self.assertNotEqual(move, old_move)
        self.assertTrue(game.make_move(True, old_move))
        self.assertTrue(game.make_move(False, [[4, 5], [7, 5]]))
        self.assertEqual(game.board[7][5], "K")
        self.assertTrue(all(game.board[7][c] == "." for c in range(5)))
        self.assertTrue(all(game.board[7][c] == "." for c in range(6, 9)))
        game = _replay(_O12, 18)
        self.assertTrue(game.make_move(True, move))
        if game.make_move(False, [[4, 5], [7, 5]]):
            # When the turn is legal, the attackers may answer by capturing the king.
            self.assertTrue(game.make_move(True, [[6, 7], [6, 5]]))
            self.assertTrue(game.check_if_game_over(False))
            self.assertEqual(game.winner, "A")

    def test_o12_reports_two_one_and_zero_immediate_replies(self):
        for plies, remaining in ((20, 2), (22, 1), (24, 0)):
            with self.subTest(plies=plies):
                game = _replay(_O12, plies)
                self.current.reset_game()
                self.current.choose(game.board, True, 0.0, set(game.past_positions))
                self.assertEqual(self.current.last_search.live_replies, remaining)

        game = _replay(_O12, 20)
        old_score, _, old_nodes = _depth(v3, self.baseline, game, True, 2)
        score, _, nodes = _depth(search, self.current, game, True, 2)
        self.assertGreater(old_score, -90_000)
        self.assertEqual(score, -search._MATE + 6)
        self.assertLess(max(nodes, old_nodes), 1000)

    def test_o09_reconsiders_a_recapturable_block_at_the_same_depth(self):
        game = _replay(_O09, 44)
        _, old_move, old_nodes = _depth(v3, self.baseline, game, True, 6)
        score, move, nodes = _depth(search, self.current, game, True, 6)
        self.assertLess(max(nodes, old_nodes), 125_000)
        self.assertLess(score, 0)
        self.assertEqual(old_move, [[5, 5], [5, 3]])
        self.assertEqual(move, [[6, 0], [6, 3]])
        self.assertTrue(game.make_move(True, old_move))
        self.assertTrue(game.make_move(False, [[6, 4], [6, 3]]))
        self.assertEqual(game.board[5][3], ".")
        game = _replay(_O09, 44)
        self.assertTrue(game.make_move(True, move))
        self.assertFalse(game.make_move(False, [[6, 4], [6, 3]]))
        self.assertTrue(game.make_move(False, [[7, 2], [6, 2]]))
        self.assertEqual(game.board[6][3], ".")

    def test_an_open_lane_cannot_override_a_repetition_draw(self):
        game = _replay(_O12, 24)
        move = [[6, 3], [6, 2]]
        self.assertTrue(game.make_move(True, move))
        repeated = game.compute_zobrist_hash(False)
        game = _replay(_O12, 24)
        self.current.reset_game()
        chosen = self.current.choose(game.board, True, 0.0, {*game.past_positions, repeated})
        self.assertEqual(chosen, move)
        self.assertEqual(self.current.last_search.score, search._ATTACKER_REPETITION_SCORE)

    def test_risk_uses_hard_limit_while_a_quiet_or_unique_reply_stops(self):
        cases = (
            (Tablut.Tablut(), (2, False)),
            (_replay(_O09, 44), (2, True)),
            (_replay(_O12, 22), (1, False)),
        )
        for game, (depth, extended) in cases:
            self.current.reset_game()
            self.current.time_window = (0.020, 0.050)
            clock = SimpleNamespace(now=0.0)
            real_iteration = self.current._iteration
            deadlines = []

            def capped(flat, side, state, layer, qleft, alpha, beta, end, *args):
                deadlines.append(end)
                if layer > 2:
                    self.current.work.counts[search._STOPPED] = 1
                    return 0, -1
                result = real_iteration(flat, side, state, layer, qleft, alpha, beta, inf, *args)
                if layer == 2:
                    clock.now = 0.030
                return result

            with (patch.object(search, "time", SimpleNamespace(perf_counter=lambda: clock.now)),
                  patch.object(search, "_read_native_clock", return_value=100.0),
                  patch.object(self.current, "_iteration", side_effect=capped)):
                self.current.choose(game.board, True, 0.020, set(game.past_positions))
            self.assertEqual(self.current.last_search.depth, depth)
            self.assertEqual(self.current.last_search.used_hard_limit, extended)
            if depth == 2:
                self.assertAlmostEqual(deadlines[1], 100.046 if extended else 100.018)


class ClockTests(unittest.TestCase):
    def test_520_calls_even_if_each_move_uses_its_hard_limit(self):
        board = _escape_threat()
        for budget in (10.0, 60.0):
            with self.subTest(budget=budget), patch.object(Tablut, "TIME_BUDGET", budget):
                player = Team19.Player()
                referee = Tablut.Tablut()
                referee_remaining = budget
                clock = SimpleNamespace(now=0.0)

                @wraps(player._native_choice)
                def hard_limit(position, side, deadline, history, weights):
                    soft, hard = player._native.time_window
                    self.assertEqual(soft, deadline)
                    if player.remaining < 0.5 or budget == 10.0:
                        self.assertEqual(hard, soft)
                    clock.now += max(0.0004, hard - clock.now)
                    return player._native_choice(position, side, 0.0, history, weights)

                player._choose_best_move = hard_limit
                with patch.object(Team19, "time", SimpleNamespace(perf_counter=lambda: clock.now)):
                    for _ in range(520):
                        started = clock.now
                        move = player.play_attacker(board)
                        referee_remaining -= clock.now - started + 0.00015
                        self.assertGreater(referee_remaining, 0.05)
                        self.assertLess(player.remaining, referee_remaining)
                        self.assertIsNone(player.last_error)
                        self.assertFalse(player.last_search.fallback)
                        self.assertEqual(player.last_search.depth, 1)
                        referee.board = [row[:] for row in board]
                        self.assertTrue(referee.make_move(True, move))
                self.assertEqual(player.moves_played, 520)
