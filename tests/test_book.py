"""Regression tests for the fail-closed Group 19 opening book."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import Mock, patch
import unittest

import Tablut
import Team19


Move = tuple[int, int, int, int]


def as_lists(move: Move) -> list[list[int]]:
    return [[move[0], move[1]], [move[2], move[3]]]


def standard_state_before(line: tuple[Move, ...]):
    from Team19_engine import apply_move, compute_hash, get_legal_moves

    board = Tablut.Tablut().board
    history = {compute_hash(board, True)}
    for index, move in enumerate(line[:-1]):
        is_attacker = index % 2 == 0
        assert as_lists(move) in get_legal_moves(board, is_attacker)
        board = apply_move(board, as_lists(move), is_attacker)
        history.add(compute_hash(board, not is_attacker))
    is_attacker = (len(line) - 1) % 2 == 0
    return board, is_attacker, history


class OpeningBookTests(unittest.TestCase):
    OPENING = ((0, 5, 1, 5),)
    DEFENDER_REPLY = ((0, 5, 1, 5), (2, 4, 2, 3))

    def setUp(self):
        import Team19_book as book

        self.book = book
        self.previous_lines = book.BOOK_LINES
        self.previous_index = book.BOOK_INDEX
        self.previous_max_history_size = book.MAX_HISTORY_SIZE
        self.addCleanup(setattr, book, "BOOK_LINES", self.previous_lines)
        self.addCleanup(setattr, book, "BOOK_INDEX", self.previous_index)
        self.addCleanup(setattr, book, "MAX_HISTORY_SIZE", self.previous_max_history_size)

    def install(self, lines):
        self.book.BOOK_LINES = lines
        self.book.BOOK_INDEX, self.book.MAX_HISTORY_SIZE = self.book.build_book_index(lines)

    def lookup(self, line, *, history=None, legal_moves=None, board=None, is_attacker=None):
        from Team19_engine import get_legal_moves

        expected_board, expected_side, expected_history = standard_state_before(line)
        return self.book.choose_book_move(
            expected_board if board is None else board,
            expected_side if is_attacker is None else is_attacker,
            expected_history if history is None else history,
            get_legal_moves(expected_board, expected_side) if legal_moves is None else legal_moves,
        )

    def test_exact_standard_prefix_returns_only_its_registered_recommendation(self):
        self.install((self.OPENING, self.DEFENDER_REPLY))
        self.assertEqual(self.lookup(self.OPENING), as_lists(self.OPENING[-1]))
        self.assertEqual(self.lookup(self.DEFENDER_REPLY), as_lists(self.DEFENDER_REPLY[-1]))

    def test_production_corpus_prefers_the_approved_root_orientation(self):
        import Team19_book as production_book
        from Team19_engine import compute_hash, get_legal_moves

        board = Tablut.Tablut().board
        history = {compute_hash(board, True)}
        self.assertGreater(len(production_book.BOOK_INDEX), 0)
        self.assertEqual(
            production_book.choose_book_move(board, True, history, get_legal_moves(board, True)),
            [[0, 5], [1, 5]],
        )

    def test_d4_expansion_preserves_root_orientation_and_maps_nonroot_lines(self):
        line = ((0, 5, 1, 5), (2, 4, 2, 3))
        self.install(((line[0],), line))
        self.assertEqual(self.lookup((line[0],)), [[0, 5], [1, 5]])

        rotated = self.book._d4_lines(line)[2]
        self.assertEqual(self.lookup(rotated), as_lists(rotated[-1]))

    def test_history_side_board_or_legal_mismatch_fails_closed(self):
        self.install((self.DEFENDER_REPLY,))
        board, side, history = standard_state_before(self.DEFENDER_REPLY)
        self.assertEqual(self.lookup(self.DEFENDER_REPLY), as_lists(self.DEFENDER_REPLY[-1]))

        missing_history = set(history)
        missing_history.pop()
        self.assertIsNone(self.lookup(self.DEFENDER_REPLY, history=missing_history))
        self.assertIsNone(self.lookup(self.DEFENDER_REPLY, is_attacker=not side))
        self.assertIsNone(self.lookup(self.DEFENDER_REPLY, legal_moves=[]))
        stale_board = deepcopy(board)
        stale_board[0][0] = "A"
        self.assertIsNone(self.lookup(self.DEFENDER_REPLY, board=stale_board))

    def test_conflicting_lines_and_repeating_successor_are_rejected(self):
        self.install((((0, 5, 1, 5),), ((0, 3, 0, 2),)))
        board, side, history = standard_state_before(self.OPENING)
        from Team19_engine import get_legal_moves
        self.assertIsNone(self.book.choose_book_move(board, side, history, get_legal_moves(board, side)))

        shared_prefix = (0, 5, 1, 5)
        first_nonroot = (shared_prefix, (2, 4, 2, 3))
        second_nonroot = (shared_prefix, (2, 4, 2, 5))
        self.install((first_nonroot, second_nonroot))
        self.assertIsNone(self.lookup(first_nonroot))

        repetition = (
            (0, 3, 0, 2),
            (2, 4, 2, 3),
            (0, 2, 0, 3),
            (2, 3, 2, 4),
        )
        self.install((repetition,))
        self.assertIsNone(self.lookup(repetition))

    def test_differently_oriented_source_conflicts_are_not_priority_masked(self):
        first = ((0, 5, 1, 5), (2, 4, 2, 3))
        # The first move is a 90-degree rotation of ``first``'s prefix, but
        # unrotates to a different defender recommendation at that prefix.
        conflicting_orientation = ((5, 8, 5, 7), (4, 6, 5, 6))
        self.install((first, conflicting_orientation))
        self.assertIsNone(self.lookup(first))
        self.assertIsNone(self.lookup(conflicting_orientation))

    def test_illegal_final_source_move_does_not_mask_legal_line(self):
        stale = ((0, 0, 0, 1),)
        self.install((self.OPENING, stale))
        self.assertEqual(self.lookup(self.OPENING), as_lists(self.OPENING[-1]))

    def test_malformed_or_missing_data_produces_no_move(self):
        self.install(((), ((0, 5, 1, 5), ("bad", 0, 0, 0))))
        self.assertIsNone(self.lookup(self.OPENING))
        self.install(())
        self.assertIsNone(self.lookup(self.OPENING))

    def test_boolean_coordinates_are_rejected_before_integer_coercion(self):
        self.assertIsNone(self.book._as_line(((True, 5, 1, 5),)))
        self.assertIsNone(self.book._as_pair((True, 5, 1, 5)))
        self.assertIsNone(self.book._as_pair(((True, 5), (1, 5))))

    def test_a_miss_does_not_replay_the_book(self):
        self.install((self.OPENING,))
        board, side, history = standard_state_before(self.OPENING)
        history.add(-1)
        with patch.object(self.book, "_state_before_recommendation", side_effect=AssertionError):
            self.assertIsNone(self.book.choose_book_move(board, side, history, []))


class PlayerBookIntegrationTests(unittest.TestCase):
    def test_player_uses_a_verified_book_move_before_search(self):
        import Team19_book as book

        previous_lines = book.BOOK_LINES
        previous_index = book.BOOK_INDEX
        previous_max_history_size = book.MAX_HISTORY_SIZE
        self.addCleanup(setattr, book, "BOOK_LINES", previous_lines)
        self.addCleanup(setattr, book, "BOOK_INDEX", previous_index)
        self.addCleanup(setattr, book, "MAX_HISTORY_SIZE", previous_max_history_size)
        book.BOOK_LINES = (OpeningBookTests.OPENING,)
        book.BOOK_INDEX, book.MAX_HISTORY_SIZE = book.build_book_index(book.BOOK_LINES)
        player = Team19.Player(use_book=True)
        referee = Tablut.Tablut()
        with patch.object(Tablut, "TIME_BUDGET", 10.0):
            move = player.play_attacker(referee.board)
        self.assertEqual(move, as_lists(OpeningBookTests.OPENING[-1]))
        self.assertTrue(player.last_book_hit)
        self.assertIsNone(player.last_search)
        self.assertIsNone(player.last_error)
        self.assertTrue(referee.make_move(True, move))

    def test_book_can_be_disabled_or_is_bypassed_for_replaced_search(self):
        import Team19_book as book

        previous_lines = book.BOOK_LINES
        previous_index = book.BOOK_INDEX
        previous_max_history_size = book.MAX_HISTORY_SIZE
        self.addCleanup(setattr, book, "BOOK_LINES", previous_lines)
        self.addCleanup(setattr, book, "BOOK_INDEX", previous_index)
        self.addCleanup(setattr, book, "MAX_HISTORY_SIZE", previous_max_history_size)
        book.BOOK_LINES = (OpeningBookTests.OPENING,)
        book.BOOK_INDEX, book.MAX_HISTORY_SIZE = book.build_book_index(book.BOOK_LINES)
        referee = Tablut.Tablut()

        disabled = Team19.Player(use_book=False)
        disabled._choose_best_move = Mock(return_value=[[0, 3], [0, 1]])
        with patch.object(Tablut, "TIME_BUDGET", 10.0):
            self.assertEqual(disabled.play_attacker(referee.board), [[0, 3], [0, 1]])
        self.assertFalse(disabled.last_book_hit)
        disabled._choose_best_move.assert_called_once()

        hooked = Team19.Player(use_book=True)
        hooked._choose_best_move = Mock(return_value=[[0, 3], [0, 1]])
        with patch.object(Tablut, "TIME_BUDGET", 10.0):
            self.assertEqual(hooked.play_attacker(Tablut.Tablut().board), [[0, 3], [0, 1]])
        self.assertFalse(hooked.last_book_hit)
        hooked._choose_best_move.assert_called_once()


if __name__ == "__main__":
    unittest.main()
