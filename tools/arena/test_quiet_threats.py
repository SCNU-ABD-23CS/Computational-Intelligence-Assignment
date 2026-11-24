"""Regressions for quiet threat selection from the v4 vs v3 match logs."""

from copy import copy
import unittest

from numba import njit

import Team19_search as search
from Team19_engine import _move_list
from Team19_weights import native_weights

from .test_team19_v4 import _replay
from .verify_numba import official_moves


# arena-b5d227d32353. Each token encodes a move as r1, c1, r2, c2.
_FAST_O10 = (
    "4777 6461 0506 4353 0343 3433 4143 4434 8363 5453 4043 3332 "
    "5855 3242 3033 4232 3835 3433 1413 3230 5552 3331 1311"
)
_LONG_O03 = (
    "0507 4272 1416 2425 1626 4575 0424 4445 2425 4373 0363 7262 "
    "3835 4636 4746 4543 2523 6222 2624 5434 0767 7570 8474"
)


@njit(no_cfunc_wrapper=True)
def _selected(board, king, work):
    count, _ = search._native_order(board, False, king, 0, False, False, -1, work, False, 2)
    return count


def _king_exits(game):
    board = game.board
    r, c = next((r, row.index("K")) for r, row in enumerate(board) if "K" in row)
    exits = set()
    for target in ([0, c], [8, c], [r, 0], [r, 8]):
        if game.make_move(False, [[r, c], target]):
            exits.add(tuple(target))
        game.board = board
    return exits


def _replies(game, move):
    game = copy(game)
    assert game.make_move(False, move)
    assert _king_exits(game)
    threatened = game.board
    blocks, captures = 0, 0
    for reply in official_moves(game, True):
        assert game.make_move(True, _move_list(reply))
        if all("K" not in row for row in game.board):
            captures += 1
        elif not _king_exits(game):
            blocks += 1
        game.board = threatened
    return blocks + captures, captures


class QuietThreatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = search._native_work(native_weights())

    def check_threat(self, line, ply, strong, weak, strong_replies, weak_replies):
        game = _replay(line, ply - 1)
        self.assertEqual(_replies(game, strong), strong_replies)
        self.assertEqual(_replies(game, weak), weak_replies)
        flat, state = search._native_position(game.board)
        original = flat.tolist()
        count = _selected(flat, state[4], self.work)
        self.assertEqual(flat.tolist(), original)
        attackers = sum(row.count("A") for row in game.board)
        quiet, covered = [], set()
        for code in self.work.moves[0, :count]:
            move = _move_list(code)
            forecast = copy(game)
            self.assertTrue(forecast.make_move(False, move))
            if sum(row.count("A") for row in forecast.board) == attackers:
                exits = _king_exits(forecast)
                self.assertTrue(exits - covered)
                covered.update(exits)
                quiet.append(move)
        self.assertIn(strong, quiet)
        self.assertNotIn(weak, quiet)
        return quiet

    def test_fast_o10_ply18_prefers_the_safe_king_turn(self):
        self.check_threat(_FAST_O10, 18, [[3, 4], [3, 3]], [[3, 4], [6, 4]], (2, 0), (6, 4))

    def test_fast_o10_ply24_prefers_the_guard_move_with_no_blocks(self):
        quiet = self.check_threat(
            _FAST_O10, 24, [[3, 0], [4, 0]], [[3, 0], [2, 0]], (0, 0), (1, 0),
        )
        self.assertEqual(quiet, [[[3, 0], [4, 0]], [[3, 1], [4, 1]]])

    def test_long_o03_ply24_prefers_the_king_turn_with_one_block(self):
        self.check_threat(_LONG_O03, 24, [[4, 3], [4, 5]], [[4, 3], [4, 2]], (1, 0), (7, 3))
