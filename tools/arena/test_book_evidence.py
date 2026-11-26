import json
from pathlib import Path
import unittest

from Team19 import DEFAULT_USE_BOOK
from .book_acceptance import summarize

from .book_corpus import _fingerprint
from .build_book import key, transform


class BookEvidenceTests(unittest.TestCase):
    def test_reflections_and_rotations_use_the_same_full_prefix(self):
        path = ((0, 5, 1, 5), (2, 4, 2, 3), (0, 3, 1, 3))
        for symmetry in range(8):
            self.assertEqual(key(path), key(tuple(transform(move, symmetry) for move in path)))
        self.assertNotEqual(key(path), key(path[::-1]))

    def test_replays_deduplicate_across_files_without_merging_different_games(self):
        game = dict(id="a", budget=60, teams={"A": "v4", "D": "qs"}, winner="A",
                    initial={"board": ["test"]}, prefix=[], attempts=[{"move": [[0, 5], [1, 5]]}])
        self.assertEqual(_fingerprint(game), _fingerprint(dict(game, id="b", stage="archive")))
        self.assertNotEqual(_fingerprint(game), _fingerprint(dict(game, attempts=[])))
        self.assertNotEqual(_fingerprint(game), _fingerprint(dict(game, budget=10)))

    def test_release_switch_and_report_match_the_paired_games(self):
        directory = Path(__file__).resolve().parent
        actual = summarize(directory / "book_paired60.jsonl.gz")
        expected = json.loads((directory / "book_acceptance.json").read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)
        self.assertEqual(DEFAULT_USE_BOOK, actual["enableByDefault"])
