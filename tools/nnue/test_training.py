"""Fast regressions for member 5's training contract and exported inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .model import (
    augment_train, deep_regression_mask, encode_boards, load_weights, mixed_targets,
    numpy_residual,
)
from .train_nnue import ResidualNNUE, export_weights


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

    def test_mask_is_conservative_and_finite(self):
        deep = np.array([89_999, 90_000, -90_000, 2999.999, -3000], dtype=np.float32)
        self.assertEqual(deep_regression_mask(deep).tolist(), [True, False, False, True, False])


if __name__ == "__main__":
    unittest.main()
