"""Behavioral tests for the Group 19 static board evaluator."""

from __future__ import annotations

from copy import deepcopy
import math
import unittest

import Team19_search as search


def board_with(pieces: dict[tuple[int, int], str]) -> list[list[str]]:
    board = [["."] * 9 for _ in range(9)]
    board[4][4] = "X"
    for (row, column), piece in pieces.items():
        board[row][column] = piece
    return board


class EvaluationFeatureTests(unittest.TestCase):
    @staticmethod
    def weight_for(feature_index: int) -> list[float]:
        weights = [0.0] * 7
        weights[feature_index] = 1_000.0
        return weights

    def test_escape_route_counts_through_empty_castle(self):
        from Team19_eval import evaluate_board

        board = board_with({(4, 2): "K"})
        self.assertEqual(evaluate_board(board, self.weight_for(2)), -1_000.0)

    def test_piece_before_edge_blocks_that_escape_route(self):
        from Team19_eval import evaluate_board

        board = board_with({(4, 4): "K", (2, 4): "A"})
        self.assertEqual(evaluate_board(board, self.weight_for(2)), -750.0)

    def test_two_escape_routes_score_between_zero_and_four_routes(self):
        from Team19_eval import evaluate_board

        two_routes = board_with({(4, 4): "K", (2, 4): "A", (6, 4): "A"})
        no_routes = board_with({
            (4, 4): "K", (2, 4): "A", (6, 4): "A", (4, 2): "A", (4, 6): "A",
        })
        self.assertEqual(evaluate_board(two_routes, self.weight_for(2)), -500.0)
        self.assertEqual(evaluate_board(no_routes, self.weight_for(2)), 0.0)

    def test_edge_distance_rewards_the_defender_as_the_king_nears_an_edge(self):
        from Team19_eval import evaluate_board

        center = board_with({(4, 4): "K"})
        near_edge = board_with({(1, 4): "K"})
        weights = self.weight_for(3)
        self.assertEqual(evaluate_board(center, weights), 0.0)
        self.assertEqual(evaluate_board(near_edge, weights), -750.0)

    def test_capture_pressure_recognises_center_adjacent_and_normal_king_shapes(self):
        from Team19_eval import evaluate_board

        center = board_with({
            (4, 4): "K", (3, 4): "A", (5, 4): "A", (4, 3): "A", (4, 5): "A",
        })
        beside_castle = board_with({
            (3, 4): "K", (2, 4): "A", (3, 3): "A", (3, 5): "A",
        })
        normal = board_with({(2, 2): "K", (1, 2): "A", (3, 2): "A"})
        weights = self.weight_for(4)
        self.assertEqual(evaluate_board(center, weights), 1_000.0)
        self.assertEqual(evaluate_board(beside_castle, weights), 1_000.0)
        self.assertEqual(evaluate_board(normal, weights), 1_000.0)

    def test_edge_king_has_no_static_capture_pressure(self):
        from Team19_eval import evaluate_board

        board = board_with({(0, 2): "K", (0, 1): "A", (0, 3): "A"})
        self.assertEqual(evaluate_board(board, self.weight_for(4)), 0.0)

    def test_defender_in_a_normal_capture_axis_removes_that_static_pressure(self):
        from Team19_eval import evaluate_board

        board = board_with({(2, 2): "K", (1, 2): "A", (3, 2): "D"})
        self.assertEqual(evaluate_board(board, self.weight_for(4)), 0.0)

    def test_material_and_low_attacker_features_have_their_declared_signs(self):
        from Team19_eval import evaluate_board

        material = {(4, 4): "K"}
        for column in range(8):
            material[(0, column)] = "A"
        material[(1, 0)] = material[(1, 1)] = "D"
        self.assertGreater(evaluate_board(board_with(material), self.weight_for(0)), 0.0)

        scarce = board_with({(4, 4): "K", (0, 0): "A", (0, 1): "A", (0, 2): "A"})
        self.assertLess(evaluate_board(scarce, self.weight_for(1)), 0.0)

    def test_blockade_rewards_controlling_a_king_route_instead_of_standing_on_the_edge(self):
        from Team19_eval import evaluate_board

        weights = self.weight_for(6)
        idle = board_with({(4, 4): "K", (0, 0): "A", (8, 8): "A"})
        blocker = board_with({(4, 4): "K", (4, 7): "A", (8, 8): "A"})
        backup = board_with({(4, 4): "K", (4, 7): "A", (4, 5): "D", (8, 8): "A"})
        self.assertEqual(evaluate_board(idle, weights), 0.0)
        self.assertGreater(evaluate_board(blocker, weights), evaluate_board(backup, weights))
        self.assertGreater(evaluate_board(backup, weights), 0.0)

    def test_guard_feature_rewards_releasing_a_king_lane_without_threats(self):
        from Team19_eval import evaluate_board

        weights = self.weight_for(5)
        blocked = board_with({(4, 4): "K", (4, 3): "D", (0, 0): "A", (8, 8): "A"})
        opened = board_with({(4, 4): "K", (3, 3): "D", (0, 0): "A", (8, 8): "A"})
        self.assertLess(evaluate_board(opened, weights), evaluate_board(blocked, weights))

    def test_guard_feature_rewards_a_guard_covering_an_attacked_axis(self):
        from Team19_eval import evaluate_board

        weights = self.weight_for(5)
        backing = board_with({(4, 4): "K", (4, 2): "D", (4, 5): "A", (8, 8): "A"})
        shielding = board_with({(4, 4): "K", (4, 3): "D", (4, 5): "A", (8, 8): "A"})
        self.assertLess(evaluate_board(shielding, weights), evaluate_board(backing, weights))

    def test_leaving_the_guarded_ray_in_the_opening_loses_attack_value(self):
        from Tablut import Tablut
        from Team19_eval import evaluate_board

        game = Tablut()
        hold = evaluate_board(game.board)
        self.assertTrue(game.make_move(True, [[1, 4], [1, 3]]))
        self.assertLess(evaluate_board(game.board), hold)

    def test_default_score_increases_as_a_reserve_blocker_approaches_a_guard(self):
        from Team19_eval import evaluate_board

        for king, guard, squares in (
            ((4, 4), (3, 4), ((1, 3), (0, 4), (1, 4), (2, 4))),
            ((3, 3), (3, 4), ((2, 6), (3, 7), (3, 6), (3, 5))),
            ((4, 5), (3, 5), ((1, 6), (0, 5), (1, 5), (2, 5))),
        ):
            with self.subTest(king=king):
                fixed = {king: "K", guard: "D", (0, 0): "A", (8, 8): "A"}
                off, d4, d3, d2 = [
                    evaluate_board(board_with({**fixed, square: "A"}))
                    for square in squares
                ]
                self.assertLess(off, d4)
                self.assertLess(d4, d3)
                self.assertLess(d3, d2)

    def test_default_score_rewards_closing_on_an_ordinary_guarded_axis(self):
        from Team19_eval import evaluate_board

        pieces = {(3, 3): "K", (3, 4): "D", (0, 0): "A", (8, 8): "A"}
        approaching = board_with({**pieces, (3, 1): "A"})
        adjacent = board_with({**pieces, (3, 2): "A"})
        self.assertGreater(evaluate_board(adjacent), evaluate_board(approaching))


class EvaluationContractTests(unittest.TestCase):
    def test_default_weights_are_used_when_no_explicit_vector_is_supplied(self):
        import Team19_eval as evaluation

        board = board_with({(4, 4): "K", (0, 0): "A"})
        expected = evaluation.evaluate_board(board, evaluation.DEFAULT_WEIGHTS)
        self.assertEqual(evaluation.evaluate_board(board), expected)
        original = evaluation.DEFAULT_WEIGHTS[:]
        try:
            evaluation.DEFAULT_WEIGHTS[0] = 0.0
            self.assertEqual(
                evaluation.evaluate_board(board),
                evaluation.evaluate_board(board, evaluation.DEFAULT_WEIGHTS),
            )
        finally:
            evaluation.DEFAULT_WEIGHTS[:] = original

    def test_explicit_weights_are_validated_without_mutating_inputs(self):
        from Team19_eval import evaluate_board

        board = board_with({(4, 4): "K", (0, 0): "A"})
        original_board = deepcopy(board)
        weights = [1_000.0] * 7
        original_weights = weights[:]
        score = evaluate_board(board, weights)
        self.assertTrue(math.isfinite(score))
        self.assertEqual(board, original_board)
        self.assertEqual(weights, original_weights)
        with self.assertRaisesRegex(ValueError, "seven"):
            evaluate_board(board, [1.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_board(board, [1.0, 1.0, 1.0, math.inf, 1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "10,000"):
            evaluate_board(board, [10_001.0] * 7)

    def test_weight_validation_rejects_numeric_strings_and_huge_integers(self):
        from Team19_eval import evaluate_board

        board = board_with({(4, 4): "K"})
        with self.assertRaisesRegex(ValueError, "numeric"):
            evaluate_board(board, ["1"] * 7)
        with self.assertRaisesRegex(ValueError, "10,000"):
            evaluate_board(board, [10 ** 10_000] * 7)

    def test_score_is_finite_and_strictly_inside_terminal_bounds_for_missing_king_and_weights(self):
        from Team19_eval import evaluate_board

        board = board_with({(0, 0): "A"})
        score = evaluate_board(board, [10_000.0] * 7)
        self.assertTrue(math.isfinite(score))
        self.assertGreater(score, -100_000.0)
        self.assertLess(score, 100_000.0)

    def test_real_search_callback_uses_evaluation_at_a_nonterminal_horizon(self):
        from Team19_eval import evaluate_board
        from Tablut import Tablut
        from tests.test_search import OfficialRules

        rules = OfficialRules()
        board = [row[:] for row in Tablut().board]
        weights = [1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0, 6_000.0, 7_000.0]
        functions = search.SearchFunctions(
            rules.get_legal_moves, rules.apply_move, rules.check_winner,
            evaluate_board, rules.compute_hash,
        )
        result = search.search_fixed_depth(board, True, 0, functions=functions, weights=weights)
        self.assertIsNone(result.move)
        self.assertEqual(result.score, evaluate_board(board, weights))
        self.assertGreater(result.nodes_visited, 0)

    def test_depth_one_uses_the_best_official_child_for_each_side(self):
        from Team19_eval import evaluate_board
        from Tablut import Tablut
        from tests.test_search import OfficialRules

        rules = OfficialRules()
        weights = [1_000.0, 1_100.0, 1_200.0, 1_300.0, 1_400.0, 1_500.0, 1_600.0]
        functions = search.SearchFunctions(
            rules.get_legal_moves, rules.apply_move, rules.check_winner,
            evaluate_board, rules.compute_hash,
        )
        initial = [row[:] for row in Tablut().board]
        attacker_result = search.search_fixed_depth(
            initial, True, 1, functions=functions, weights=weights,
        )
        attacker_scores = [
            evaluate_board(rules.apply_move(initial, move, True), weights)
            for move in rules.get_legal_moves(initial, True)
        ]
        self.assertEqual(attacker_result.score, max(attacker_scores))

        after_attack = rules.apply_move(initial, rules.get_legal_moves(initial, True)[0], True)
        defender_result = search.search_fixed_depth(
            after_attack, False, 1, functions=functions, weights=weights,
        )
        defender_scores = [
            evaluate_board(rules.apply_move(after_attack, move, False), weights)
            for move in rules.get_legal_moves(after_attack, False)
        ]
        self.assertEqual(defender_result.score, min(defender_scores))


if __name__ == "__main__":
    unittest.main()
