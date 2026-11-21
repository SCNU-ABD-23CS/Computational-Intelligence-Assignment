"""Arena-only acceptance checks. The coursework suite remains independent."""

from importlib import import_module
from io import BytesIO
import json
import unittest
from unittest.mock import patch

import Tablut
import Team19
import Team19_engine

from . import openings, report, runner


class HoldingPlayer:
    def __init__(self, move):
        self.move = move

    def play_attacker(self, board):
        board[4][4] = "X"
        return self.move

    def play_defender(self, board):
        return self.move


class BrokenPlayer(HoldingPlayer):
    def play_attacker(self, board):
        raise RuntimeError("test exception")


class ArenaTests(unittest.TestCase):
    def test_historical_modules_keep_distinct_policies_and_share_rule_hashes(self):
        versions = [runner.TEAMS[team] for team in runner.LEAGUE[:3]]
        board = Tablut.Tablut().board
        modules = [import_module(team.module, runner.__package__) for team in versions]
        players = [module.Player() for module in modules]
        self.assertIs(modules[0], Team19)
        self.assertEqual(len({type(bot).__module__ for bot in players}), 3)
        self.assertEqual(len({bot._choose_best_move.__module__ for bot in players}), 3)
        self.assertEqual(len({bot._apply_move.__module__ for bot in players}), 3)
        results = []
        for team, player in zip(versions, players):
            functions = import_module(team.search, runner.__package__)._load_functions()
            self.assertEqual(player._compute_hash(board, True), Team19_engine.compute_hash(board, True))
            results.append(functions.evaluate_board(board, player.weights))
        self.assertEqual(results, [656.25, -2493.75, -4375.0])
        self.assertEqual(import_module("Team19_engine").__name__, "Team19_engine")

    def test_first_timed_turn_sees_the_complete_prefix(self):
        opening = openings.load_suite()[0]
        expected = openings.start(opening).referee.past_positions
        captured = []
        original = runner.ObservedReferee.play

        def observe(instance, *players):
            captured.append(instance)
            return original(instance, *players)

        match = runner.Match("RandomBot", "GreedyBot", 10, opening)
        with (
            patch.object(runner, "_player", return_value=HoldingPlayer(None)),
            patch.object(runner.ObservedReferee, "play", observe),
        ):
            row = runner.run_match(match)
        self.assertEqual(row["reason"], "illegal move")
        self.assertEqual(captured[0].past_positions, expected)
        self.assertEqual(len(expected), 5)

    def test_official_referee_uses_deepcopy_and_calls_illegal_move(self):
        move = [[0, 3], [0, 2]]
        match = runner.Match("RandomBot", "GreedyBot", 10, openings.standard())
        with patch.object(runner, "_player", side_effect=(HoldingPlayer(move), HoldingPlayer(None))):
            row = runner.run_match(match)
        self.assertEqual((row["winner"], row["reason"]), ("A", "illegal move"))
        self.assertEqual(row["attempts"][0]["event"]["positionAfter"]["board"][4][4], "K")
        self.assertEqual(report.replay(row)["final"]["winner"], "A")

    def test_official_clock_decides_a_timeout_before_applying_the_move(self):
        match = runner.Match("RandomBot", "GreedyBot", 10, openings.standard())
        with (
            patch.object(runner, "_player", return_value=HoldingPlayer([[0, 3], [0, 2]])),
            patch.object(runner.time, "time", side_effect=(100.0, 111.0)),
        ):
            row = runner.run_match(match)
        self.assertEqual((row["winner"], row["reason"], row["remaining"]["A"]), ("D", "timeout", -1))
        self.assertNotIn("event", row["attempts"][0])

    def test_official_referee_adjudicates_callback_exception(self):
        match = runner.Match("RandomBot", "GreedyBot", 10, openings.standard())
        with patch.object(runner, "_player", return_value=BrokenPlayer(None)):
            row = runner.run_match(match)
        self.assertEqual((row["winner"], row["reason"]), ("D", "exception"))
        self.assertIn("test exception", row["attempts"][0]["exception"])

    def test_resume_rejects_duplicates_and_preserves_last_complete_line(self):
        prefix = json.dumps({"id": "first", "schedule": "hash"}).encode() + b"\n"
        recorded = BytesIO(prefix + b'{"id": "incom')
        self.assertEqual(runner._read_finished(recorded, "hash", {"first"}), {"first"})
        self.assertEqual(recorded.getvalue(), prefix)
        with self.assertRaises(ValueError):
            runner._read_finished(BytesIO(prefix * 2), "hash", {"first"})

    def test_fixed_opening_suite_is_reproducible(self):
        self.assertEqual(openings.load_suite(), openings.generate())


if __name__ == "__main__":
    unittest.main()
