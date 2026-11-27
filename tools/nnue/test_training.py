"""Fast regressions for member 5's training contract and exported inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .model import (
    augment_train, deep_regression_mask, encode_boards, load_weights, mixed_targets,
    numpy_residual, scale_tanh_weights,
)
from .train_nnue import (
    ResidualNNUE, evaluate, export_weights, pretrain_side_calibration, training_objective,
)
from .train_listwise import augment_listwise, split_by_game


class TrainingContractTests(unittest.TestCase):
    def test_three_planes_and_empty_castle(self):
        board = np.zeros((1, 81), dtype=np.int8)
        board[0, [0, 1, 2, 3, 4]] = [0, 1, 2, 3, 4]
        features = encode_boards(board)
        self.assertEqual(features.shape, (1, 243))
        self.assertTrue(np.all(features[0, :3] == 0))  # empty
        self.assertTrue(np.all(features[0, 12:15] == 0))  # empty castle
        self.assertEqual(features[0, 3:6].tolist(), [1, 0, 0])
        self.assertEqual(features[0, 6:9].tolist(), [0, 1, 0])
        self.assertEqual(features[0, 9:12].tolist(), [0, 0, 1])

    def test_d4_is_train_only_transform_major(self):
        arrays = {
            "boards": np.arange(162, dtype=np.int8).reshape(2, 81) % 5,
            "sides": np.array([0, 1], dtype=np.int8),
            "base_evals": np.array([10, 20], dtype=np.float32),
            "deep_evals": np.array([30, 40], dtype=np.float32),
            "outcomes": np.array([0, 1], dtype=np.float32),
            "game_ids": np.array(["a", "b"]),
        }
        augmented = augment_train(arrays)
        self.assertEqual(len(augmented["boards"]), 16)
        self.assertEqual(augmented["base_evals"].tolist(), [10, 20] * 8)
        self.assertEqual(augmented["sides"].tolist(), [0, 1] * 8)
        self.assertEqual(len(arrays["boards"]), 2)  # source arrays remain untouched
        for variant in range(8):
            block = augmented["boards"][variant * 2:(variant + 1) * 2]
            self.assertEqual(np.sum(block == 1, axis=1).tolist(),
                             np.sum(arrays["boards"] == 1, axis=1).tolist())

    def test_masked_rows_keep_outcome_only(self):
        deep = np.array([1000, 90_000, -3000, 3000], dtype=np.float32)
        outcome = np.array([1, 0, 1, 0], dtype=np.float32)
        target, mask = mixed_targets(deep, outcome, scale=6000, outcome_weight=0.5)
        self.assertEqual(mask.tolist(), [True, False, False, False])
        self.assertEqual(target[1:].tolist(), outcome[1:].tolist())
        expected = 0.5 + 0.5 / (1 + np.exp(-1000 / 6000))
        self.assertAlmostEqual(float(target[0]), expected, places=6)

    def test_head_initialization_is_small_nonzero_and_asymmetric(self):
        torch.manual_seed(20261126)
        model = ResidualNNUE(750, 6000)
        weights = model.heads.weight.detach().numpy()
        self.assertGreater(float(np.std(weights)), 0.0)
        self.assertLess(float(np.max(np.abs(weights))), 0.01)
        self.assertEqual(int(np.sum(weights == weights[0, 0])), 1)
        self.assertEqual(model.heads.bias.detach().numpy().tolist(), [0.0, 0.0])

    def test_side_calibration_is_side_selected_logit_bias(self):
        model = ResidualNNUE(750, 6000)
        with torch.no_grad():
            model.hidden.weight.zero_()
            model.hidden.bias.zero_()
            model.heads.weight.zero_()
            model.heads.bias.zero_()
            model.side_calib.copy_(torch.tensor([0.5, -0.25]))
        features = torch.zeros(2, 243)
        sides = torch.tensor([0, 1], dtype=torch.int64)
        base = torch.tensor([0.0, 12000.0])
        prediction, residual = model(features, sides, base)
        expected = torch.sigmoid(torch.tensor([0.5, 1.75]))
        torch.testing.assert_close(prediction, expected)
        torch.testing.assert_close(residual, torch.zeros(2))
        prediction.sum().backward()
        self.assertIsNotNone(model.side_calib.grad)
        self.assertTrue(bool(torch.all(model.side_calib.grad != 0)))

    def test_pre_calibration_updates_only_side_calib(self):
        torch.manual_seed(11)
        model = ResidualNNUE(750, 6000)
        arrays = {
            "boards": np.zeros((4, 81), dtype=np.int8),
            "sides": np.array([0, 0, 1, 1], dtype=np.int8),
            "base_evals": np.zeros(4, dtype=np.float32),
            "targets": np.array([0, 0, 1, 1], dtype=np.float32),
        }
        args = SimpleNamespace(
            calibration_epochs=3, calibration_learning_rate=0.05, batch_size=2,
        )
        hidden_before = [parameter.detach().clone() for parameter in model.hidden.parameters()]
        heads_before = [parameter.detach().clone() for parameter in model.heads.parameters()]
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "calibration_history.jsonl"
            records = pretrain_side_calibration(
                model, arrays, args, torch.Generator().manual_seed(11), history,
            )
            self.assertEqual(len(records), 3)
            self.assertEqual(len(history.read_text(encoding="utf-8").splitlines()), 3)
        for parameter, before in zip(model.hidden.parameters(), hidden_before):
            torch.testing.assert_close(parameter, before, rtol=0.0, atol=0.0)
        for parameter, before in zip(model.heads.parameters(), heads_before):
            torch.testing.assert_close(parameter, before, rtol=0.0, atol=0.0)
        self.assertLess(float(model.side_calib[0].detach()), 0.0)
        self.assertGreater(float(model.side_calib[1].detach()), 0.0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(records[-1]["optimized_parameters"], ["side_calib"])
        self.assertEqual(records[-1]["frozen_parameters"], ["hidden", "heads"])

    def test_center_penalty_uses_pre_tanh_selected_not_validation(self):
        model = ResidualNNUE(750, 6000)
        with torch.no_grad():
            model.hidden.weight.zero_()
            model.hidden.bias.zero_()
            model.heads.weight.zero_()
            model.heads.bias.copy_(torch.tensor([2.0, -3.0]))
            model.side_calib.zero_()
        features = torch.zeros(2, 243)
        sides = torch.tensor([0, 1], dtype=torch.int64)
        base = torch.zeros(2)
        target = torch.full((2,), 0.5)
        objective, smooth_l1, selected_mse = training_objective(
            model, features, sides, base, target, center_weight=1e-3,
        )
        torch.testing.assert_close(selected_mse, torch.tensor(6.5))
        torch.testing.assert_close(smooth_l1, F.smooth_l1_loss(
            model(features, sides, base)[0], target,
        ))
        torch.testing.assert_close(objective, smooth_l1 + 1e-3 * 6.5)
        arrays = {
            "boards": np.zeros((2, 81), dtype=np.int8),
            "sides": sides.numpy().astype(np.int8),
            "base_evals": np.zeros(2, dtype=np.float32),
            "targets": target.numpy(),
        }
        metrics = evaluate(model, arrays, batch_size=2)
        clamped = torch.clamp(model(features, sides, base)[0], 1e-7, 1.0 - 1e-7)
        expected_nll = F.binary_cross_entropy(clamped, target)
        self.assertAlmostEqual(metrics["nll"], float(expected_nll.detach()), places=7)

    def test_exported_numpy_matches_torch_and_selects_heads(self):
        torch.manual_seed(7)
        model = ResidualNNUE(750, 6000)
        with torch.no_grad():
            model.heads.weight.normal_(0, 0.1)
            model.heads.bias.copy_(torch.tensor([2.0, -3.0]))
            model.side_calib.copy_(torch.tensor([0.4, -0.2]))
        boards = (np.arange(162).reshape(2, 81) % 5).astype(np.int8)
        sides = np.array([1, 0], dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            export_weights(model, path)
            with np.load(path, allow_pickle=False) as saved:
                self.assertNotIn("side_calib", saved.files)
            weights = load_weights(path)
            np.testing.assert_allclose(
                weights["head_weight"], model.heads.weight.detach().numpy() * 750,
                rtol=1e-6, atol=1e-6,
            )
            np.testing.assert_allclose(
                weights["head_bias"], model.heads.bias.detach().numpy() * 750,
                rtol=1e-6, atol=1e-6,
            )
            actual = numpy_residual(weights, boards, sides)
            features = torch.from_numpy(encode_boards(boards))
            base = torch.zeros(2)
            with torch.no_grad():
                _, expected = model(features, torch.from_numpy(sides.astype(np.int64)), base)
            np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)

    def test_development_scale_multiplies_tanh_residual_exactly(self):
        torch.manual_seed(17)
        model = ResidualNNUE(750, 6000)
        boards = (np.arange(162).reshape(2, 81) % 5).astype(np.int8)
        sides = np.array([1, 0], dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            export_weights(model, path)
            weights = load_weights(path)
        scaled = scale_tanh_weights(weights, 0.25)
        np.testing.assert_allclose(
            numpy_residual(scaled, boards, sides),
            0.25 * numpy_residual(weights, boards, sides),
            rtol=1e-6, atol=1e-6,
        )
        self.assertAlmostEqual(float(scaled["residual_clip"]), 187.5, places=5)
        self.assertEqual(float(scaled["eval_scale"]), float(weights["eval_scale"]))
        with self.assertRaises(ValueError):
            scale_tanh_weights(weights, 0.0)
        with self.assertRaises(ValueError):
            scale_tanh_weights(weights, 1.25)

    def test_listwise_game_split_and_d4_keep_candidate_labels_aligned(self):
        questions = [
            {"game_id": f"game-{index // 2}", "id": str(index)} for index in range(16)
        ]
        train, dev = split_by_game(questions, seed=20261127, dev_fraction=0.25)
        train_games = {row["game_id"] for row in train}
        dev_games = {row["game_id"] for row in dev}
        self.assertTrue(train_games.isdisjoint(dev_games))
        self.assertEqual(len(train) + len(dev), 16)
        arrays = {
            "boards": np.arange(2 * 3 * 81, dtype=np.int16).reshape(2, 3, 81) % 5,
            "static": np.arange(6, dtype=np.float32).reshape(2, 3),
            "deep": np.arange(6, dtype=np.float32).reshape(2, 3) + 10,
            "game_ids": np.array(["a", "b"]),
        }
        arrays["boards"] = arrays["boards"].astype(np.int8)
        augmented = augment_listwise(arrays)
        self.assertEqual(augmented["boards"].shape, (16, 3, 81))
        np.testing.assert_array_equal(augmented["boards"][:2], arrays["boards"])
        np.testing.assert_array_equal(augmented["static"][:2], arrays["static"])
        np.testing.assert_array_equal(augmented["deep"][:2], arrays["deep"])
        for variant in range(8):
            block = slice(variant * 2, (variant + 1) * 2)
            np.testing.assert_array_equal(augmented["static"][block], arrays["static"])
            np.testing.assert_array_equal(augmented["deep"][block], arrays["deep"])
            np.testing.assert_array_equal(augmented["game_ids"][block], arrays["game_ids"])

    def test_mask_is_conservative_and_finite(self):
        deep = np.array([89_999, 90_000, -90_000, 2999.999, -3000], dtype=np.float32)
        self.assertEqual(deep_regression_mask(deep).tolist(), [True, False, False, True, False])


if __name__ == "__main__":
    unittest.main()
