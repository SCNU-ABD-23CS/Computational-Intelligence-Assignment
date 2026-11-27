"""Invariants of the member 1 blind-evaluation infrastructure."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from tools.arena import runner
from tools.arena.openings import _symmetry_key, load_suite, start
from tools.member1 import branches as branchlib
from tools.member1 import configs


class SuiteTest(unittest.TestCase):
    def setUp(self):
        self.suite = branchlib.load()

    def test_size_and_shape(self):
        self.assertTrue(24 <= len(self.suite) <= 48)
        self.assertEqual(len({branch.name for branch in self.suite}), len(self.suite))
        self.assertEqual(len({branch.signature for branch in self.suite}), len(self.suite))
        self.assertTrue(all(branch.plies in (4, 6) for branch in self.suite))

    def test_no_duplicate_up_to_rotation_and_reflection(self):
        keys = [_symmetry_key(start(branch.opening()).referee.board) for branch in self.suite]
        self.assertEqual(len(set(keys)), len(keys))

    def test_never_reuses_the_referee_openings(self):
        suite = load_suite()
        hashes = {opening.signature for opening in suite}
        keys = {_symmetry_key(start(opening).referee.board) for opening in suite}
        self.assertFalse({branch.signature for branch in self.suite} & hashes)
        self.assertFalse({branch.symmetry for branch in self.suite} & keys)

    def test_balanced_over_first_move_classes(self):
        counts = Counter(branch.branch_class for branch in self.suite)
        self.assertEqual(len(counts), 8)
        self.assertEqual(len(set(counts.values())), 1)

    def test_frozen_file_is_reproducible(self):
        rebuilt = branchlib.build(branchlib.SEED, len(self.suite))
        self.assertEqual([branch.signature for branch in rebuilt],
                         [branch.signature for branch in self.suite])

    def test_frozen_metadata_matches(self):
        payload = json.loads(branchlib.SUITE.read_text(encoding="utf-8"))
        self.assertEqual(payload["seed"], branchlib.SEED)
        self.assertEqual(payload["target"], len(self.suite))
        self.assertEqual(payload["design"]["classes"], 8)


class ConfigTest(unittest.TestCase):
    def test_build_load_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen = Path(directory) / "frozen"
            built = configs.build("probe", configs=frozen)
            self.assertEqual(len(built.files), len(configs.CORE))
            self.assertTrue(all(digest for digest in built.sha256.values()))
            loaded = configs.load("probe", frozen)
            self.assertEqual(loaded.files, built.files)


if __name__ == "__main__":
    unittest.main()
