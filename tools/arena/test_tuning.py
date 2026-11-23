"""Regression checks for resumable evolution and its independent benchmark."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import Team19_engine
from Team19_weights import DEFAULT_WEIGHTS

from . import openings, tune_ga, validate_v2
from .snapshots.v_91ffcc8 import Team19 as historic
from .snapshots.v_91ffcc8 import Team19_engine as historic_engine


class TuningTests(unittest.TestCase):
    def test_mutations_can_move_both_zero_weights(self):
        weights = np.asarray(DEFAULT_WEIGHTS)
        weights[-2:] = 0.0
        sigma = tune_ga.mutation_sigma(weights)
        self.assertTrue(np.all(sigma[-2:] > 0))
        rng = np.random.default_rng(123)
        population = [{"weights": weights.tolist()}]
        mutants = np.array([tune_ga.mutate(population, sigma, rng) for _ in range(20)])
        self.assertTrue(np.all(np.any(mutants[:, -2:] != 0, axis=0)))
        self.assertTrue(np.all(np.isfinite(mutants)))

    def test_resume_reproduces_the_uninterrupted_mutation_sequence(self):
        class Clock:
            now = 0

            def time(self):
                return self.now

        clock = Clock()

        class Trainer:
            def __init__(self, cases, depth, weights):
                pass

            def score(self, weights, deadline):
                clock.now += 1
                return float(weights[-1]), 0.0, []

        def args(output, seconds, resume=None):
            return SimpleNamespace(output=output, time_limit=seconds, depth=4, seed=123,
                                   resume=resume, init_weights=None)

        with TemporaryDirectory(dir=tune_ga.HERE) as folder, patch.multiple(
            tune_ga, perf_counter=clock.time, FixedDepthTrainer=Trainer,
        ):
            root = Path(folder)
            one, split = root / "one.json", root / "split.json"
            tune_ga.train(args(one, 6))
            tune_ga.train(args(split, 3))
            tune_ga.train(args(split, 3, split.with_suffix(".checkpoint.json")))
            def history(path):
                data = json.loads(path.read_text(encoding="utf-8"))
                return [entry["weights"] for entry in data["history"]]

            self.assertEqual(history(one), history(split))

    def test_validation_openings_exclude_training_symmetries(self):
        with TemporaryDirectory(dir=tune_ga.HERE) as folder:
            path = Path(folder) / "held-out.json"
            suite = validate_v2.held_out_openings(path)
            self.assertEqual(suite, validate_v2.held_out_openings(path))
        key = lambda opening: openings._symmetry_key(openings.start(opening).referee.board)
        self.assertEqual(len({key(item) for item in suite}), 12)
        self.assertTrue({key(item) for item in suite}.isdisjoint(map(key, openings.load_suite())))

    def test_new_archive_uses_its_own_engine_and_search(self):
        player = historic.Player()
        self.assertIsNot(historic_engine, Team19_engine)
        self.assertIs(player._apply_move, historic_engine.apply_move)
        self.assertEqual(player._choose_best_move.__module__, historic.__package__ + ".Team19_search")

    def test_preference_for_consistent_paired_wins(self):
        result = {"games": 24, "points": 26, "positivePairs": 4, "negativePairs": 5}
        self.assertFalse(validate_v2._eligible(result))
        result["positivePairs"] = 6
        self.assertTrue(validate_v2._eligible(result))
