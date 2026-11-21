"""Targeted checks for the three specialist searchers."""

from math import inf
import random
from time import perf_counter
import unittest

import Tablut
import Team19_engine as engine
from Team19_search import SearchTimeout

from .bots.bfs_strategist import escape_distance
from .bots.guided_mcts import Player as MonteCarloPlayer
from .bots.qsearch_pvs import PrincipalSearch
from .bots.tactics import winning_move


def board_with(pieces):
    board = [["."] * 9 for _ in range(9)]
    board[4][4] = "X"
    for (row, col), piece in pieces.items():
        board[row][col] = piece
    return board


class SpecialistTests(unittest.TestCase):
    def test_bfs_finds_two_slide_escape_and_detects_trap(self):
        pieces = {(4, 4): "K", (0, 4): "A", (8, 4): "A", (4, 0): "A", (4, 8): "A"}
        board = board_with(pieces)
        self.assertEqual(escape_distance("".join(map("".join, board)), 40)[0], 2)
        for row, col in ((5, 4), (3, 4), (4, 5), (4, 3)):
            board[row][col] = "D"
        self.assertEqual(escape_distance("".join(map("".join, board)), 40), (None, 0))

    def test_quiescence_considers_a_quiet_defence_before_standing_pat(self):
        root = engine._search_position(board_with({
            (1, 4): "K", (1, 3): "D", (1, 5): "D", (2, 4): "D", (0, 1): "A", (7, 7): "A",
        }))
        search = PrincipalSearch(lambda state: 0, set(), inf)
        score, move = search.visit(root, True, 0, 2, -inf, inf)
        self.assertEqual((score, move), (0, [[0, 1], [0, 4]]))
        self.assertFalse(search.path)

    def test_quiescence_respects_expired_search_deadline(self):
        root = engine._search_position(Tablut.Tablut().board)
        search = PrincipalSearch(lambda state: 0, set(), 0)
        with self.assertRaises(SearchTimeout):
            search.visit(root, True, 0, 4, -inf, inf)

    def test_guided_mcts_prunes_moves_that_allow_immediate_escape(self):
        root = engine._search_position(board_with({
            (1, 4): "K", (1, 3): "D", (1, 5): "D", (2, 4): "D", (0, 1): "A", (7, 7): "A",
        }))
        bot = MonteCarloPlayer(seed=19)
        move = bot.choose(root, True, engine.get_legal_moves(root.board, True), perf_counter() + 0.06)
        self.assertEqual(move, [[0, 1], [0, 4]])
        self.assertGreater(bot.last_search.simulations, 0)

    def test_immediate_threats_agree_with_exhaustive_children(self):
        rng = random.Random(19)
        for _ in range(8):
            position = engine._search_position(Tablut.Tablut().board)
            for ply in range(22):
                for attacker in (True, False):
                    moves = engine.get_legal_moves(position.board, attacker)
                    side = "A" if attacker else "D"
                    wins = [move for move in moves
                            if engine._apply_legal_move(position, move).winner() == side]
                    reported = winning_move(position, attacker)
                    self.assertIn(reported, wins or [None])
                attacker = ply % 2 == 0
                moves = engine.get_legal_moves(position.board, attacker)
                if not moves:
                    break
                position = engine._apply_legal_move(position, rng.choice(moves))
                if position.winner():
                    break


if __name__ == "__main__":
    unittest.main()
