"""Synthetic regressions for opening-book evidence filtering."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from Team19 import DEFAULT_USE_BOOK

from .book_acceptance import summarize
from .book_corpus import _fingerprint
from .build_book import export, key, screened, transform


PATH = ((0, 5, 1, 5), (2, 4, 2, 3), (0, 3, 1, 3))
OTHER_PATH = ((1, 4, 1, 0), (3, 4, 3, 3), (3, 8, 3, 4))
NEXT = ((4, 3, 7, 3), (1, 4, 1, 3))


def _nested(path):
    return [[[r1, c1], [r2, c2]] for r1, c1, r2, c2 in path]


def _row(path, opponent, *, winner="D", errors=None, fallbacks=0, next_moves=()):
    return {
        "prefix": _nested(path),
        "opponent": opponent,
        "winner": winner,
        "errors": [] if errors is None else errors,
        "fallbacks": fallbacks,
        "reason": "no legal move",
        "nextMoves": _nested(next_moves),
    }


def _screen(rows, lines=None):
    with TemporaryDirectory() as temporary:
        journal = Path(temporary) / "book_reply_trials.jsonl"
        journal.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        result = {} if lines is None else lines
        screened(result, journal)
        return result


def _acceptance_game(opponent, side, enabled):
    other = "D" if side == "A" else "A"
    teams = {side: "Team19-v4", other: opponent}
    return {
        "id": f"{opponent}-{side}-{'book' if enabled else 'bare'}",
        "budget": 60,
        "teams": teams,
        "winner": None,
        "reason": "repeated position",
        "opening": "standard",
        "prefix": [],
        "bookEnabled": enabled,
        "attempts": [{"side": side, "elapsed": 0.5 if enabled else 0.75,
                      "bookHit": enabled, "fallback": False}],
    }


class BookEvidenceTests(unittest.TestCase):
    def test_reflections_and_rotations_use_the_same_full_prefix(self):
        for symmetry in range(8):
            self.assertEqual(key(PATH), key(tuple(transform(move, symmetry) for move in PATH)))
        self.assertNotEqual(key(PATH), key(PATH[::-1]))

    def test_default_release_switch_is_explicit(self):
        self.assertIs(DEFAULT_USE_BOOK, False)

    def test_replays_deduplicate_across_files_without_merging_different_games(self):
        game = dict(id="a", budget=60, teams={"A": "v4", "D": "qs"}, winner="A",
                    initial={"board": ["test"]}, prefix=[], attempts=[{"move": [[0, 5], [1, 5]]}])
        self.assertEqual(_fingerprint(game), _fingerprint(dict(game, id="b", stage="archive")))
        self.assertNotEqual(_fingerprint(game), _fingerprint(dict(game, attempts=[])))
        self.assertNotEqual(_fingerprint(game), _fingerprint(dict(game, budget=10)))

    def test_synthetic_sixteen_game_acceptance_keeps_default_disabled_on_a_tie(self):
        opponents = ("Team19-v3", "Grandmaster-Numba", "Grandmaster-Plus", "QSearch-PVS")
        games = [
            _acceptance_game(opponent, side, enabled)
            for opponent in opponents for side in ("A", "D") for enabled in (False, True)
        ]
        with TemporaryDirectory() as temporary:
            journal = Path(temporary) / "paired.jsonl"
            journal.write_text("\n".join(json.dumps(game) for game in games), encoding="utf-8")
            actual = summarize(journal)
        self.assertEqual(actual["games"], 16)
        self.assertEqual(actual["points"], {"bare": 8, "book": 8})
        self.assertTrue(actual["candidateHealthy"])
        self.assertFalse(actual["enableByDefault"])
        self.assertIs(DEFAULT_USE_BOOK, actual["enableByDefault"])

    def test_any_two_or_more_opponents_support_a_path_and_draws_are_allowed(self):
        lines = _screen([
            _row(PATH, "Alpha", winner="D"),
            _row(PATH, "Beta", winner=None),
            _row(PATH, "Additional-opponent", winner="D"),
        ])
        self.assertEqual(key(lines[key(PATH[:-1])]), key(PATH))

    def test_losses_and_faults_are_not_hidden_by_order_or_duplicate_opponent_rows(self):
        winning = _row(PATH, "Alpha")
        losing = _row(PATH, "Alpha", winner="A")
        second_support = _row(PATH, "Beta")
        self.assertNotIn(key(PATH[:-1]), _screen([winning, losing, second_support]))
        self.assertNotIn(key(PATH[:-1]), _screen([losing, winning, second_support]))
        self.assertNotIn(
            key(PATH[:-1]),
            _screen([winning, _row(PATH, "Alpha", errors=["timeout"]), second_support]),
        )

    def test_d4_equivalent_full_paths_share_evidence_and_promote_one_reply(self):
        symmetry = 1
        rotated_path = tuple(transform(move, symmetry) for move in PATH)
        rotated_next = tuple(transform(move, symmetry) for move in NEXT)
        lines = _screen([
            _row(PATH, "Alpha", next_moves=NEXT),
            _row(rotated_path, "Beta", next_moves=rotated_next),
        ])
        self.assertEqual(key(lines[key(PATH[:-1])]), key(PATH))
        extension = PATH + NEXT
        self.assertEqual(key(lines[key(extension[:-1])]), key(extension))

    def test_ply4_needs_two_supporters_and_rejects_conflicting_replies(self):
        only_one = _screen([
            _row(PATH, "Alpha", next_moves=NEXT),
            _row(PATH, "Beta"),
        ])
        self.assertIn(key(PATH[:-1]), only_one)
        self.assertNotIn(key((PATH + NEXT)[:-1]), only_one)

        alternate = ((4, 3, 7, 3), (1, 4, 1, 2))
        extension_prefix = key((PATH + NEXT)[:-1])
        stale = PATH + NEXT
        conflicts = _screen([
            _row(PATH, "Alpha", next_moves=NEXT),
            _row(PATH, "Beta", next_moves=alternate),
        ], {extension_prefix: stale})
        self.assertIn(key(PATH[:-1]), conflicts)
        self.assertNotIn(extension_prefix, conflicts)

    def test_draw_or_fault_reply_conflict_invalidates_a_stale_ply4_line(self):
        alternate = ((4, 3, 7, 3), (1, 4, 1, 2))
        extension_prefix = key((PATH + NEXT)[:-1])
        stale = PATH + NEXT
        for outcome in (None, "A"):
            with self.subTest(outcome=outcome):
                lines = _screen([
                    _row(PATH, "Alpha", next_moves=NEXT),
                    _row(PATH, "Beta", winner=outcome, next_moves=alternate),
                ], {extension_prefix: stale})
                self.assertNotIn(extension_prefix, lines)

    def test_missing_evidence_is_an_explicit_build_error(self):
        with TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.jsonl"
            with self.assertRaises(FileNotFoundError):
                screened({}, missing)

    def test_missing_decisions_fails_before_replacing_an_output(self):
        with TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            (evidence / "book_reply_trials.jsonl").write_text("", encoding="utf-8")
            output = Path(temporary) / "Team19_book_data.py"
            output.write_text("unchanged", encoding="utf-8")
            with patch("tools.arena.build_book.historical", return_value={()}):
                with self.assertRaises(FileNotFoundError):
                    export(Path(temporary), output, evidence)
            self.assertEqual(output.read_text(encoding="utf-8"), "unchanged")

    def test_conflicting_evidence_removes_a_stale_historical_recommendation(self):
        alternate = PATH[:-1] + ((0, 3, 1, 2),)
        prefix = key(PATH[:-1])
        lines = _screen([
            _row(PATH, "Alpha"),
            _row(PATH, "Beta"),
        ], {prefix: alternate})
        self.assertNotIn(prefix, lines)

    def test_unrelated_ply3_branches_promote_independently(self):
        lines = _screen([
            _row(PATH, "Alpha"),
            _row(PATH, "Beta"),
            _row(OTHER_PATH, "Alpha"),
            _row(OTHER_PATH, "Beta"),
        ])
        self.assertEqual(key(lines[key(PATH[:-1])]), key(PATH))
        self.assertEqual(key(lines[key(OTHER_PATH[:-1])]), key(OTHER_PATH))
