"""Fast HalfKP regressions using synthetic data and unopened sealed questions."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .eval_halfkp import _ranking, final_report
from .halfkp_data import SPLIT_FILES, load_split, ranking_groups, split_audit
from .halfkp_features import FEATURES, KING_STRIDE, PERM, feature_indices, pack_boards
from .halfkp_model import HalfKP, export_weights, load_weights, numpy_residual, torch_indices
from .train_halfkp import augmented_batches, parse_args, train, validation_nll


def _fixture(count: int, *, game: str = "train", king: int = 49) -> dict:
    boards = np.zeros((count, 81), dtype=np.int8)
    boards[:, 40], boards[:, king] = 4, 3
    boards[np.arange(count), np.arange(count)] = 1
    boards[:, 62] = 2
    deep = np.linspace(-1150, 1550, count, dtype=np.float32)
    return {
        "boards": boards, "sides": np.zeros(count, dtype=np.int8),
        "base_evals": deep / 2, "deep_evals": deep,
        "outcomes": np.ones(count, dtype=np.float32),
        "game_ids": np.full(count, game), "depths": np.full(count, 5, dtype=np.int16),
    }


class FeatureTests(unittest.TestCase):
    def test_d4_transforms_king_and_pieces_in_lockstep(self):
        boards = _fixture(2)["boards"]
        kings, packed = pack_boards(boards)
        self.assertEqual(len({tuple(row) for row in PERM}), 8)
        for view, permutation in enumerate(PERM):
            transformed = np.empty_like(boards)
            transformed[:, permutation] = boards
            self.assertTrue(np.all((transformed == 3).sum(axis=1) == 1))
            expected = np.sort(feature_indices(*pack_boards(transformed)), axis=1)
            encoded = feature_indices(kings, packed, view)
            np.testing.assert_array_equal(np.sort(encoded, axis=1), expected)
            self.assertTrue(np.all((encoded > 0) & (encoded < FEATURES)))
            self.assertEqual(encoded[0, 0], permutation[49] * KING_STRIDE + permutation[0])
        without_throne = np.where(boards == 4, 0, boards)
        np.testing.assert_array_equal(feature_indices(*pack_boards(without_throne)),
                                      feature_indices(kings, packed))

    def test_torch_batch_indices_match_reference_for_padded_bags(self):
        boards = _fixture(3)["boards"]
        boards[1, 62] = 0
        boards[2, [2, 62]] = 0
        kings, pieces = pack_boards(boards)
        views = np.array([1, 3, 6])
        actual = torch_indices(torch.from_numpy(kings), torch.from_numpy(pieces),
                               torch.from_numpy(views), torch.tensor(PERM))
        np.testing.assert_array_equal(actual.numpy(), feature_indices(kings, pieces, views))
        self.assertEqual(actual[2].tolist(), [0, 0])

    def test_each_online_epoch_visits_all_eight_views_once(self):
        generator = torch.Generator().manual_seed(7)
        visited = [(int(row), int(view)) for rows, views in augmented_batches(5, 7, generator)
                   for row, view in zip(rows, views)]
        self.assertEqual(sorted(visited), [(row, view) for row in range(5) for view in range(8)])

    def test_numpy_and_torch_match_for_both_architectures(self):
        boards = _fixture(2)["boards"]
        sides = np.array([0, 1], dtype=np.int8)
        torch.manual_seed(7)
        for hidden2 in (0, 4):
            with self.subTest(hidden2=hidden2), TemporaryDirectory() as tmp:
                model = HalfKP(8, hidden2, 1000)
                with torch.no_grad():
                    model.side_calib[:] = torch.tensor([-3.0, 3.0])
                    model.heads.bias[:] = torch.tensor([-0.2, 0.2])
                    reference = model(torch.from_numpy(feature_indices(*pack_boards(boards))),
                                      torch.from_numpy(sides.astype(np.int64))).numpy()
                self.assertLess(reference[0], 0)
                self.assertGreater(reference[1], 0)
                path = Path(tmp) / "weights.npz"
                export_weights(path, model)
                weights = load_weights(path)
                self.assertNotIn("side_calib", weights)
                np.testing.assert_allclose(numpy_residual(weights, boards, sides),
                                           reference, atol=1e-4)


class DatasetTests(unittest.TestCase):
    def test_seven_field_train_filter_and_six_field_locked_validation(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.npz"
            fixture = _fixture(4)
            fixture["depths"][:] = [5, 32, 33, 64]
            np.savez(path, **fixture, additional_metadata=np.array("compatible"))
            train_rows, raw_rows = load_split(path, training=True)
            self.assertEqual(raw_rows, 4)
            np.testing.assert_array_equal(train_rows["depths"], [5, 32])
            self.assertEqual(len(load_split(path)[0]["boards"]), 4)
            del fixture["depths"]
            np.savez(path, **fixture)
            self.assertEqual(len(load_split(path)[0]["boards"]), 4)
            with self.assertRaisesRegex(ValueError, "depths"):
                load_split(path, training=True)

    def test_d4_overlap_checks_side_and_game_identity(self):
        fixture = _fixture(1)
        validation = _fixture(1, game="heldout")
        validation["boards"] = validation["boards"][:, PERM[3]]
        with self.assertRaisesRegex(ValueError, "D4-equivalent"):
            split_audit(fixture, validation)
        validation["sides"] = np.ones(1, dtype=np.int8)
        self.assertEqual(split_audit(fixture, validation)["shared_d4_positions"], 0)
        validation["game_ids"] = fixture["game_ids"]
        with self.assertRaisesRegex(ValueError, "1 game IDs"):
            split_audit(fixture, validation)

    def test_listwise_groups_are_disjoint_and_share_game_and_side(self):
        fixture = _fixture(14)
        fixture["game_ids"] = np.array(["first"] * 8 + ["second"] * 6)
        fixture["sides"][4:8] = 1
        fixture["deep_evals"][[0, 10]] = [100_000, -3000]
        groups = ranking_groups(fixture, seed=7)
        self.assertEqual(len(groups), 3)
        self.assertEqual(len(set(groups.ravel())), groups.size)
        self.assertFalse({0, 10}.intersection(groups.ravel()))
        for group in groups:
            self.assertEqual(len(set(fixture["game_ids"][group])), 1)
            self.assertEqual(len(set(fixture["sides"][group])), 1)

    def test_cpu_training_and_listwise_keep_seal_closed(self):
        with TemporaryDirectory() as tmp:
            data_dir, run_dir = Path(tmp), Path(tmp) / "run"
            fixture = _fixture(4)
            fixture["depths"][-1] = 64
            np.savez(data_dir / SPLIT_FILES["train"], **fixture)
            validation = _fixture(3, game="heldout", king=73)
            del validation["depths"]
            np.savez(data_dir / SPLIT_FILES["validation"], **validation)
            sealed = data_dir / "questions_400.json"
            sealed.write_text("unreadable by training", encoding="utf-8")
            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                if path == sealed:
                    raise AssertionError("Training tried to open the sealed question file")
                return original_open(path, *args, **kwargs)

            args = parse_args(["--data-dir", str(data_dir), "--run-dir", str(run_dir),
                               "--device", "cpu", "--hidden", "8", "--hidden2", "4",
                               "--calibration-epochs", "1", "--epochs", "1",
                               "--listwise-epochs", "1", "--batch-size", "64"])
            with (patch.object(Path, "open", guarded_open),
                  patch("tools.nnue.train_halfkp.validation_nll", wraps=validation_nll) as val_nll):
                with self.assertRaises(FileNotFoundError):
                    final_report(data_dir, run_dir)
                metrics = train(args)
            self.assertEqual(val_nll.call_count, 2)
            self.assertFalse(metrics["selection"]["sealed_questions_used"])
            self.assertEqual(metrics["data"]["train_rows"], 3)
            self.assertEqual(metrics["data"]["validation_rows"], 3)
            self.assertEqual(metrics["selection"]["stage3"]["training_game_groups"], 1)
            self.assertIsNone(metrics["epochs"][-1]["val_nll"])
            self.assertEqual(set(metrics["models"]), {"stage2", "stage3"})
            for files in metrics["models"].values():
                for artifact in files.values():
                    self.assertTrue((run_dir / artifact["file"]).is_file())

    def test_sealed_metrics_match_legacy_tie_policy_and_report_is_cached(self):
        questions = [{"ranking": [0, 1, 2], "tie_top2": False},
                     {"ranking": [1, 0, 2], "tie_top2": True}]
        static = np.array([[3, 2, 1], [1, 2, 3]])
        labels = np.array([[3, 2, 1], [3, 3, 1]])
        scores = np.array([[3, 3, 1], [1, 3, -1]])
        metrics = _ranking(questions, static, scores, labels)
        self.assertEqual(metrics["main"]["nnue"]["top1_correct"], 1)
        self.assertEqual(metrics["main"]["nnue"]["pairwise"]["correct_credit"], 2.5)
        self.assertEqual(metrics["tie_top2_information_only"]["nnue"]["pairwise"]["compared_pairs"], 2)
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "final_evaluation.json").write_text(json.dumps(metrics), encoding="utf-8")
            with patch("tools.nnue.eval_halfkp._sealed_questions") as sealed:
                self.assertEqual(final_report(run_dir, run_dir), metrics)
            sealed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
