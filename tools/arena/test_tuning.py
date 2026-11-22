"""Tests for member 5's reproducible offline weight experiments."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import Tablut
import Team19
import Team19_engine as engine
import Team19_eval as evaluation

from .tuning import (
    Opponent,
    Settings,
    aggregate_games,
    evaluate_weights,
    export_weights,
    load_weight_file,
    opponent_pool,
    play_weight_game,
    select_openings,
    weight_digest,
    write_evolution_svg,
    write_weight_file,
)
from .weight_player import FixedDepthWeightPlayer, IterativeWeightPlayer


class WeightFileTests(unittest.TestCase):
    def test_round_trip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            original = write_weight_file(path, evaluation.DEFAULT_WEIGHTS, {"test": True})
            loaded = load_weight_file(path)
            self.assertEqual(loaded["weights"], evaluation.DEFAULT_WEIGHTS)
            self.assertEqual(loaded["weightDigest"], original["weightDigest"])
            document = json.loads(path.read_text(encoding="utf-8"))
            document["weights"][0] += 1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_weight_file(path)

    def test_export_requires_matching_real_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = write_weight_file(root / "candidate.json", evaluation.DEFAULT_WEIGHTS)
            validation = {
                "format": "team19-weight-validation-v1",
                "candidateDigest": candidate["weightDigest"],
                "candidate": {"weightDigest": candidate["weightDigest"],
                              "pointRate": 0.75, "forfeits": 0},
                "default": {"pointRate": 0.5},
            }
            validation_path = root / "validation.json"
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            output = export_weights(root / "candidate.json", validation_path,
                                    root / "Team19_weights.py")
            namespace = {}
            exec(output.read_text(encoding="utf-8"), namespace)
            self.assertEqual(namespace["WEIGHTS"], evaluation.DEFAULT_WEIGHTS)

            validation["candidate"]["pointRate"] = 0.5
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Export refused"):
                export_weights(root / "candidate.json", validation_path,
                               root / "refused.py")

            validation["candidate"]["pointRate"] = 0.75
            validation["candidateDigest"] = "0" * 64
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different candidate"):
                export_weights(root / "candidate.json", validation_path,
                               root / "mismatch.py")


class ProtocolTests(unittest.TestCase):
    def test_settings_and_opponents_are_validated(self):
        with self.assertRaises(ValueError):
            Settings(depth=0)
        with self.assertRaises(ValueError):
            Settings(budget=0)
        with self.assertRaisesRegex(ValueError, "incumbent"):
            opponent_pool("incumbent")
        pool = opponent_pool("default,greedy,random")
        self.assertEqual([item.name for item in pool], ["default", "greedy", "random"])
        with self.assertRaises(ValueError):
            opponent_pool("greedy,greedy")

    def test_training_and_validation_openings_can_be_disjoint(self):
        training = select_openings(2, 0)
        validation = select_openings(2, 2)
        self.assertTrue({item.name for item in training}.isdisjoint(
            item.name for item in validation
        ))

    def test_aggregate_uses_course_points_and_forfeits(self):
        games = [
            {"points": 2, "candidateForfeit": False, "gameSeconds": 1.0},
            {"points": 1, "candidateForfeit": False, "gameSeconds": 3.0},
            {"points": 0, "candidateForfeit": True, "gameSeconds": 2.0},
        ]
        result = aggregate_games(games)
        self.assertEqual((result["points"], result["maxPoints"]), (3, 6))
        self.assertEqual((result["wins"], result["draws"], result["losses"]), (1, 1, 1))
        self.assertEqual(result["forfeits"], 1)
        self.assertEqual(result["meanGameSeconds"], 2.0)

    def test_evolution_svg_is_generated_without_plot_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_evolution_svg([
                {"generation": 0, "bestPointRate": 0.5, "meanPointRate": 0.4},
                {"generation": 1, "bestPointRate": 0.75, "meanPointRate": 0.6},
            ], Path(directory) / "evolution.svg")
            text = path.read_text(encoding="utf-8")
            self.assertIn("<svg", text)
            self.assertIn("GA training fitness", text)


class PlayerIntegrationTests(unittest.TestCase):
    def test_fixed_depth_player_returns_legal_move_without_mutating_board(self):
        board = [row[:] for row in Tablut.Tablut().board]
        before = [row[:] for row in board]
        player = FixedDepthWeightPlayer(evaluation.DEFAULT_WEIGHTS, depth=1)
        move = player.play_attacker(board)
        self.assertIn(move, engine.get_legal_moves(before, True))
        self.assertEqual(board, before)

    def test_iterative_player_uses_explicit_candidate_weights(self):
        weights = [0.0] * len(evaluation.FEATURE_NAMES)
        player = IterativeWeightPlayer(weights)
        self.assertEqual(player.weights, weights)
        self.assertIsInstance(player, Team19.Player)

    def test_one_official_game_is_recorded(self):
        result = play_weight_game(
            evaluation.DEFAULT_WEIGHTS,
            Opponent("random", "bot"),
            select_openings(1)[0],
            True,
            Settings(depth=1, budget=10.0),
            20261122,
        )
        self.assertIn(result["points"], (0, 1, 2))
        self.assertFalse(result["candidateForfeit"])
        self.assertEqual(result["candidateSide"], "A")


if __name__ == "__main__":
    unittest.main()
