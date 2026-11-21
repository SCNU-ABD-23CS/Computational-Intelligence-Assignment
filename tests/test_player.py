"""Player regressions exercised against the course referee."""

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import Tablut
import Team19


def referee_move(board, move, is_attacker):
    referee = Tablut.Tablut()
    referee.board = [row[:] for row in board]
    if not referee.make_move(is_attacker, move):
        raise ValueError("Illegal test move")
    return referee.board


def referee_hash(board, is_attacker):
    referee = Tablut.Tablut()
    referee.board = board
    return referee.compute_zobrist_hash(is_attacker)


class PlayerTests(unittest.TestCase):
    def setUp(self):
        self.referee = Tablut.Tablut()
        self.referee.past_positions = [self.referee.compute_zobrist_hash(True)]
        time_budget_patch = patch.object(Tablut, "TIME_BUDGET", 10.0)
        time_budget_patch.start()
        self.addCleanup(time_budget_patch.stop)
        self.clock = SimpleNamespace(now=0.0)
        clock_patch = patch.object(
            Team19, "time", SimpleNamespace(perf_counter=lambda: self.clock.now),
        )
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.player = Team19.Player()
        self.player._apply_move = referee_move
        self.player._compute_hash = referee_hash
        self.player._check_winner = None

    def apply_nonterminal(self, is_attacker, move):
        self.assertTrue(self.referee.make_move(is_attacker, move))
        self.assertFalse(self.referee.check_if_game_over(not is_attacker))

    def test_opening_like_defender_turn_preserves_budget_and_history(self):
        replies = iter(([[2, 4], [2, 3]], [[2, 3], [2, 4]], [[3, 4], [3, 3]]))
        histories = []

        def select(board, is_attacker, deadline, history, weights):
            self.assertFalse(is_attacker)
            histories.append(history)
            self.clock.now += 0.1
            self.assertLess(self.clock.now, deadline)
            return next(replies)

        self.player._choose_best_move = select
        for attacker_move in ([[0, 3], [0, 2]], [[0, 5], [0, 6]], [[0, 2], [0, 3]]):
            self.apply_nonterminal(True, attacker_move)
            reply = self.player.play_defender(self.referee.board)
            self.assertEqual(histories[-1], set(self.referee.past_positions))
            self.apply_nonterminal(False, reply)

        self.assertEqual(self.player.moves_played, 3)
        self.assertAlmostEqual(self.player.remaining, 9.7)

    def test_reused_defender_resets_for_a_new_standard_opening(self):
        for budget, attacker_move in ((10, [[0, 3], [0, 2]]), (60, [[0, 5], [0, 6]])):
            self.referee = Tablut.Tablut()
            self.referee.past_positions = [self.referee.compute_zobrist_hash(True)]
            self.apply_nonterminal(True, attacker_move)
            Tablut.TIME_BUDGET = budget

            def select(board, is_attacker, deadline, history, weights):
                self.assertEqual(history, set(self.referee.past_positions))
                self.clock.now += 0.1
                return [[2, 4], [2, 3]]

            self.player._choose_best_move = select
            reply = self.player.play_defender(self.referee.board)
            self.apply_nonterminal(False, reply)
            self.assertEqual(self.player.moves_played, 1)
            self.assertAlmostEqual(self.player.remaining, budget - 0.1)

    def test_explicit_reset_clears_state_and_reads_new_budget(self):
        self.player._choose_best_move = Mock(return_value=[[0, 3], [0, 2]])
        self.player.play_attacker(self.referee.board)
        self.assertTrue(self.player.history)
        Tablut.TIME_BUDGET = 60
        self.player.reset_game()
        self.assertEqual(self.player.moves_played, 0)
        self.assertEqual(self.player.remaining, 60)
        self.assertEqual(self.player.history, set())

    def test_invalid_engine_move_is_never_used_as_fallback(self):
        illegal = [[0, 0], [0, 1]]
        engine = SimpleNamespace(
            get_legal_moves=lambda *_: [illegal], apply_move=referee_move,
            compute_hash=referee_hash, check_winner=lambda *_: None,
        )
        search = SimpleNamespace(choose_best_move=lambda *_: illegal)
        with patch.dict("sys.modules", {"Team19_engine": engine, "Team19_search": search}):
            player = Team19.Player()
        move = player.play_attacker(self.referee.board)
        self.assertTrue(self.referee.make_move(True, move))

    def test_malformed_or_failing_search_uses_a_legal_fallback(self):
        cases = (
            Mock(return_value=[[0.25, 5], [0, 6]]),
            Mock(return_value=[[0, 5], [0, 6], [4, 4]]),
            Mock(side_effect=RuntimeError("failed search")),
        )
        for search in cases:
            with self.subTest(search=search):
                player = Team19.Player()
                player._choose_best_move = search
                referee = Tablut.Tablut()
                move = player.play_attacker(referee.board)
                self.assertEqual(move, [[0, 3], [0, 0]])
                self.assertTrue(referee.make_move(True, move))
                if isinstance(search.side_effect, RuntimeError):
                    self.assertIs(player.last_error, search.side_effect)

    def test_low_budget_skips_search_and_tactical_fallback(self):
        self.apply_nonterminal(True, [[0, 3], [0, 2]])
        self.apply_nonterminal(False, [[2, 4], [2, 3]])
        self.player._last_board = self.referee.board
        self.player._last_is_attacker = True
        self.player.moves_played = 30
        self.player.remaining = 0.06
        self.player._choose_best_move = Mock()

        def slow_apply(board, move, is_attacker):
            self.clock.now += 0.002
            return referee_move(board, move, is_attacker)

        self.player._apply_move = Mock(side_effect=slow_apply)
        self.player._check_winner = Mock(return_value=None)
        self.player._compute_hash = Mock(side_effect=referee_hash)
        move = self.player.play_attacker(self.referee.board)
        self.player._choose_best_move.assert_not_called()
        self.player._check_winner.assert_not_called()
        self.player._apply_move.assert_not_called()
        self.player._compute_hash.assert_not_called()
        self.assertLess(self.clock.now, 0.06)
        self.assertTrue(self.referee.make_move(True, move))

    def test_tactical_fallback_stops_at_the_search_deadline(self):
        self.player._choose_best_move = Mock(side_effect=RuntimeError("failed search"))

        def slow_apply(board, move, is_attacker):
            self.clock.now += 0.1
            return referee_move(board, move, is_attacker)

        self.player._apply_move = Mock(side_effect=slow_apply)
        self.player._check_winner = Mock(return_value=None)
        move = self.player.play_attacker(self.referee.board)
        self.assertEqual(self.player._check_winner.call_count, 1)
        self.assertEqual(self.player._apply_move.call_count, 3)
        self.assertTrue(self.referee.make_move(True, move))

    def test_small_but_usable_clock_still_calls_search(self):
        self.apply_nonterminal(True, [[0, 3], [0, 2]])
        self.apply_nonterminal(False, [[2, 4], [2, 3]])
        self.player._last_board = self.referee.board
        self.player._last_is_attacker = True
        self.player.remaining = 0.1398
        self.player.moves_played = 60
        move = [[0, 2], [1, 2]]

        def search_with_clock(board, is_attacker, deadline, history, weights):
            self.assertTrue(is_attacker)
            self.assertGreater(deadline, self.clock.now)
            self.assertLess(deadline, self.clock.now + 0.02)
            self.assertIn(referee_hash(board, is_attacker), history)
            return move

        self.player._choose_best_move = Mock(side_effect=search_with_clock)
        self.assertEqual(self.player.play_attacker(self.referee.board), move)
        self.player._choose_best_move.assert_called_once()
        self.assertTrue(self.referee.make_move(True, move))

    def test_long_endgame_keeps_a_share_for_future_turns(self):
        board = [["."] * 9 for _ in range(9)]
        board[4][4] = "K"
        board[0][4] = "A"
        board[8][4] = "A"
        self.player.remaining = 5.0
        for moves_played in (20, 60, 120):
            self.player.moves_played = moves_played
            self.assertAlmostEqual(self.player._step_budget(board), (5.0 - 0.05) / 12)

    def test_without_teammate_modules_finishes_a_game(self):
        output = StringIO()
        missing = {"Team19_engine": None, "Team19_eval": None, "Team19_search": None}
        with patch.dict("sys.modules", missing), redirect_stdout(output):
            self.referee.play(Team19.Player(), Team19.Player())
        result = output.getvalue()
        self.assertIn("Game over", result)
        for error in ("Illegal move", "exceeded time", "Exception caught"):
            self.assertNotIn(error, result)


class PlayerIntegrationTests(unittest.TestCase):
    def assert_real_search_move(self, player):
        referee = Tablut.Tablut()
        search_callback = Mock(wraps=player._choose_best_move)
        player._choose_best_move = search_callback
        with patch.object(Tablut, "TIME_BUDGET", 10.0):
            move = player.play_attacker(referee.board)
        search_callback.assert_called_once()
        self.assertIsNone(player.last_error)
        self.assertTrue(referee.make_move(True, move))
        self.assertTrue(player.history)

    def test_valid_tuned_weights_reach_real_search_unchanged(self):
        from Team19_eval import normalize_weights

        tuned = [1_000.0, 1_100.0, 1_200.0, 1_300.0, 1_400.0, 1_500.0, 1_600.0]
        with patch.object(Tablut, "TIME_BUDGET", 10.0), patch.dict(
            "sys.modules", {"Team19_weights": SimpleNamespace(WEIGHTS=tuned)},
        ):
            player = Team19.Player()
        self.assertEqual(player.weights, normalize_weights(tuned))
        self.assertIsNot(player.weights, tuned)
        self.assertIsNotNone(player._choose_best_move)
        self.assertIsNotNone(player._apply_move)
        self.assert_real_search_move(player)

    def test_invalid_tuned_weights_fall_back_to_valid_defaults_before_real_search(self):
        from Team19_eval import DEFAULT_WEIGHTS, normalize_weights

        invalid_vectors = (
            [1.0],
            [float("nan")] * 7,
            [10_001.0] * 7,
            ["1000"] * 7,
        )
        expected = normalize_weights(DEFAULT_WEIGHTS)
        for tuned in invalid_vectors:
            with self.subTest(tuned=tuned):
                with patch.object(Tablut, "TIME_BUDGET", 10.0), patch.dict(
                    "sys.modules", {"Team19_weights": SimpleNamespace(WEIGHTS=tuned)},
                ):
                    player = Team19.Player()
                self.assertEqual(player.weights, expected)
                self.assert_real_search_move(player)


if __name__ == "__main__":
    unittest.main()
