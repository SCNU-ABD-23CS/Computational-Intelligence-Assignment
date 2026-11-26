"""Fast regressions for offline opening-book trial plumbing."""

from __future__ import annotations

import unittest

from . import book_trials
from .openings import start


class RootFinalistTrialTests(unittest.TestCase):
    def test_root_finalists_are_three_forced_attacker_openings(self):
        self.assertEqual(
            book_trials.ROOT_FINALISTS,
            ((0, 3, 1, 3), (1, 4, 1, 2), (0, 3, 2, 3)),
        )
        self.assertEqual(
            [(case.prefix, case.side, case.opponents)
             for case in book_trials.root_finalist_cases()],
            [
                ((), "A", ("Team19-v3", "QSearch-PVS")),
                ((), "A", ("Team19-v3", "QSearch-PVS")),
                ((), "A", ("Team19-v3", "QSearch-PVS")),
            ],
        )

    def test_one_ply_root_replay_seeds_missing_defender_predecessor(self):
        case = book_trials.root_finalist_cases()[0]
        opening = book_trials._opening(case, book_trials.ROOT_FINALISTS[0])
        started = start(opening)
        self.assertNotIn(False, started.last_board)

        book_trials._seed_missing_predecessor(started, opening)

        self.assertEqual(started.last_board[False], started.positions[0]["board"])


if __name__ == "__main__":
    unittest.main()
