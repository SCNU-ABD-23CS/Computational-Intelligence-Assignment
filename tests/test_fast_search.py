"""Checks for the tournament search path against the official referee."""

from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import random
import unittest
from unittest.mock import patch

from Tablut import Tablut
import Team19_engine as engine
import Team19_eval as evaluation
import Team19_search as search


def board_from(*rows):
    return [list(row) for row in rows]


class FastSearchTests(unittest.TestCase):
    def assert_child_matches_referee(self, referee, state, side, move):
        referee.board = state.board
        referee.past_positions = []
        referee.winner = None
        self.assertTrue(referee.make_move(side, move))
        child = engine._apply_legal_move(state, move)
        self.assertEqual(child.board, referee.board)
        self.assertEqual(child.attacker_hash, referee.compute_zobrist_hash(True))
        self.assertEqual(child.defender_hash, referee.compute_zobrist_hash(False))
        pieces = Counter(piece for row in child.board for piece in row)
        self.assertEqual((child.attackers, child.defenders), (pieces["A"], pieces["D"]))
        self.assertEqual(
            evaluation._features_from_counts(child.board, child.attackers, child.defenders, child.king),
            evaluation._features(child.board),
        )
        has_move = engine._has_legal_move(child.board, not side)
        self.assertEqual(has_move, bool(engine.get_legal_moves(child.board, not side)))
        with redirect_stdout(StringIO()):
            finished = referee.check_if_game_over(not side)
        expected = (referee.winner or "DRAW") if finished else None
        self.assertEqual(child.winner() or (None if has_move else "DRAW"), expected)
        return child

    def test_both_hashes_and_captures_over_legal_play(self):
        rng = random.Random(19)
        referee = Tablut()
        state = engine._search_position(referee.board)
        for ply in range(80):
            side = ply % 2 == 0
            moves = engine.get_legal_moves(state.board, side)
            if state.winner() or not moves:
                state = engine._search_position(Tablut().board)
                moves = engine.get_legal_moves(state.board, side)
            before = [row[:] for row in state.board]
            sample = rng.sample(moves, min(8, len(moves)))
            for move in sample:
                next_state = self.assert_child_matches_referee(referee, state, side, move)
            self.assertEqual(state.board, before)
            state = next_state

    def test_multiple_captures_king_capture_and_vacated_throne(self):
        cases = (
            (board_from(".........", ".........", "....A....", "...AK....",
                        "....X....", ".....A...", ".........", ".........", "........."),
             True, [[5, 5], [3, 5]]),
            (board_from(".........", ".........", ".........", ".....A...",
                        "....K.A..", ".....D...", ".........", ".........", "........."),
             False, [[4, 4], [4, 5]]),
        )
        # Landing at (2,3) takes two defenders using the north and west anchors.
        multi = board_from("...A.....", "...DA....", ".AD......", ".........",
                           "....K....", "...A.....", ".........", ".........", ".........")
        state = engine._search_position(multi)
        child = self.assert_child_matches_referee(Tablut(), state, True, [[5, 3], [2, 3]])
        self.assertEqual(state.defenders - child.defenders, 2)
        for board, side, move in cases:
            with self.subTest(move=move):
                self.assert_child_matches_referee(Tablut(), engine._search_position(board), side, move)

    def test_move_existence_can_cross_empty_throne(self):
        board = board_from(".........", ".........", ".........", "...D.....",
                           "..DAX....", "...D.....", ".........", "...K.....", ".........")
        self.assertTrue(engine._has_legal_move(board, True))
        board[4][5] = "D"
        self.assertFalse(engine._has_legal_move(board, True))

    def test_cached_move_cannot_override_real_history_or_a_win(self):
        board = Tablut().board
        functions = search._load_functions()
        fast = search._prepare_fast_search(board, functions, None)
        move = functions.get_legal_moves(board, True)[0]
        child = fast.apply_move(fast.root, move)
        next_hash = child.defender_hash
        reply = functions.get_legal_moves(child.board, False)[0]
        ordering = search._MoveOrdering(hints={next_hash: reply})
        for history in (set(), {next_hash}, set()):
            result, _ = search._search_depth(
                board, True, 2, functions, None, None, history, fast=fast,
                playable_root_moves=[move], tactical_ordering=True, ordering=ordering,
            )
            if next_hash in history:
                self.assertEqual(result.score, 0)
            else:
                self.assertNotEqual(result.score, 0)

        mate = board_from(".........", ".........", ".........", ".........",
                          "....X....", ".........", ".........", "..K..A.A.", ".........")
        fast = search._prepare_fast_search(mate, functions, None)
        escape = [[7, 2], [8, 2]]
        after = fast.apply_move(fast.root, escape).attacker_hash
        result, _ = search._search_depth(
            mate, False, 1, functions, None, None, {after}, fast=fast,
            playable_root_moves=[escape], ordering=ordering,
        )
        self.assertEqual((result.terminal_plies, result.score), (1, -search.WIN_SCORE))

    def test_proved_three_ply_win_stops_deepening_even_with_heuristic_leaves(self):
        board = board_from("....A....", ".D..K.A.A", ".........", ".........",
                           "....X....", "........A", "....D....", ".........", "A........")
        depths = []
        run_depth = search._search_depth

        def record(*args, **kwargs):
            result, evaluated = run_depth(*args, **kwargs)
            depths.append((args[2], result, evaluated))
            return result, evaluated

        with patch.object(search, "_search_depth", side_effect=record):
            move = search.choose_best_move(board, False, float("inf"), set())
        self.assertEqual([depth for depth, _, _ in depths], [1, 2, 3])
        _, proved, evaluated = depths[-1]
        self.assertTrue(evaluated)
        self.assertEqual((proved.terminal_plies, proved.score), (3, -search.WIN_SCORE))
        functions = search._load_functions()

        def constrain_root(position, side):
            return [move] if position is board else functions.get_legal_moves(position, side)

        fixed = search.search_fixed_depth(
            board, False, 3, functions=replace(functions, get_legal_moves=constrain_root),
        )
        self.assertEqual(fixed.score, -search.WIN_SCORE)

    def test_forced_loss_prefers_the_line_that_survives_longer(self):
        board = board_from(".........", "D.D.A....", "...A.....", "...A.AA..",
                           "..A.XK.AA", "....A.A.A", "...A.A...", ".........", ".........")
        functions = search._load_functions()
        fast = search._prepare_fast_search(board, functions, None)
        result, _ = search._search_depth(
            board, False, 4, functions, None, None, set(), fast=fast,
            tactical_ordering=True, ordering=search._MoveOrdering(),
        )
        self.assertEqual((result.terminal_plies, result.score), (4, search.WIN_SCORE))
        self.assertNotEqual(board[result.move[0][0]][result.move[0][1]], "K")


if __name__ == "__main__":
    unittest.main()
