"""Fast regressions for the NNUE data pipeline; no long searches or live games."""

from __future__ import annotations

import hashlib
import random
import unittest
from datetime import datetime, timezone

import Tablut
import Team19_engine as engine
from tools.arena.openings import Opening, move_event, start
from Team19_search import _native_position, _native_score, _native_work
from Team19_weights import native_weights

from .common import (
    MIN_DEPTH, NativeEvaluator, Sample, board_tuple, d4_key, encode_board, eligible_game,
    game_samples,
)
from .collect import PREFIX_MAX, PREFIX_MIN, random_opening
from .dataset import split_families


def _event_position(game, attacker):
    return {
        "board": ["".join(row) for row in game.board],
        "sideToMove": "A" if attacker else "D",
        "hash": f"{game.compute_zobrist_hash(attacker):016x}",
    }


def synthetic_game(plies: int = 12, seed: int = 7) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    for _ in range(200):
        game = Tablut.Tablut()
        game.past_positions = [game.compute_zobrist_hash(True)]
        initial = _event_position(game, True)
        events = []
        positions = [initial]
        aborted = False
        for ply in range(1, plies + 1):
            attacker = ply % 2 == 1
            legal = engine.get_legal_moves(game.board, attacker)
            rng.shuffle(legal)
            moved = False
            for move in legal:
                before = game.board
                if game.make_move(attacker, move) and not game.check_if_game_over(not attacker):
                    events.append(move_event(game, before, attacker, move, ply))
                    positions.append(events[-1]["positionAfter"])
                    moved = True
                    break
            if not moved:
                aborted = True
                break
        if aborted or len(events) < plies:
            continue
        record = {
            "id": f"synthetic-{seed}",
            "budget": 10.0,
            "opening": "synthetic",
            "teams": {"A": "Team19-v4", "D": "Team19-v3"},
            "winner": "D",
            "reason": "king escaped to the edge",
            "started": datetime.now(timezone.utc).isoformat(),
            "initial": initial,
            "prefix": events[:4],
            "attempts": [
                {"team": "Team19-v4", "side": event["side"], "score": 100.0 + event["ply"],
                 "depth": 5, "elapsed": 0.1, "playerRemaining": 5.0, "event": event}
                for event in events[4:]
            ],
        }
        return record, positions
    raise RuntimeError("Unable to build a non-terminal synthetic game")


class PipelineRegressions(unittest.TestCase):
    def test_prefix_and_root_score_align_before_move(self):
        game, positions = synthetic_game()
        samples = list(game_samples(game, "collection:test", "post_frozen_v4"))
        self.assertEqual([sample.ply for sample in samples], [9, 10, 11, 12])
        for sample in samples:
            self.assertEqual(sample.board, board_tuple(positions[sample.ply - 1]["board"]))
            self.assertEqual(sample.deep_eval, 100.0 + sample.ply)

    def test_only_capture_move_immediately_excluded(self):
        game, _ = synthetic_game()
        game["attempts"][3]["event"]["captures"] = [[0, 0, "D"]]  # ply 8 captures
        samples = list(game_samples(game, "collection:test", "post_frozen_v4"))
        self.assertNotIn(9, [sample.ply for sample in samples])
        self.assertIn(10, [sample.ply for sample in samples])

    def test_frozen_team_and_depth_filters(self):
        game, _ = synthetic_game()
        game["attempts"][4]["team"] = "Team19-v3"  # ply 9
        game["attempts"][5]["depth"] = MIN_DEPTH - 1  # ply 10
        samples = list(game_samples(game, "collection:test", "post_frozen_v4"))
        self.assertNotIn(9, [sample.ply for sample in samples])
        self.assertNotIn(10, [sample.ply for sample in samples])
        self.assertEqual([sample.ply for sample in samples], [11, 12])

    def test_whitelist_and_cutoff(self):
        game, _ = synthetic_game()
        game["started"] = "2026-11-23T00:00:00+00:00"
        self.assertFalse(eligible_game(game, "arena-unlisted-10s.jsonl")[0])
        self.assertTrue(eligible_game(game, "arena-9451a7e4dcf4-10s.jsonl")[0])
        game["budget"] = 1.0
        self.assertFalse(eligible_game(game, "arena-9451a7e4dcf4-10s.jsonl")[0])

    def test_d4_key_is_symmetric_and_side_sensitive(self):
        game, _ = synthetic_game()
        board = game["prefix"][1]["positionAfter"]["board"]
        mirrored = ["".join(reversed(row)) for row in board]
        self.assertEqual(d4_key(board, "A"), d4_key(mirrored, "A"))
        self.assertNotEqual(d4_key(board, "A"), d4_key(board, "D"))

    def test_engine_encoding_is_reused_verbatim(self):
        game, _ = synthetic_game()
        board = game["prefix"][1]["positionAfter"]["board"]
        encoded = encode_board(board)
        self.assertEqual(encoded.dtype, engine._flat(board).dtype)
        self.assertEqual(encoded.tolist(), engine._flat(board).tolist())
        self.assertLessEqual(set(encoded.tolist()), {0, 1, 2, 3, 4})

    def test_native_base_is_pre_siege_and_attacker_positive(self):
        game, _ = synthetic_game()
        board = game["prefix"][-1]["positionAfter"]["board"]
        evaluator = NativeEvaluator()
        flat, state = _native_position(board)
        work = _native_work(native_weights(None))
        attacker = float(_native_score(flat, True, state[2], state[3], state[4], work, False))
        defender = float(_native_score(flat, False, state[2], state[3], state[4], work, False))
        self.assertAlmostEqual(evaluator.evaluate(board, "A"), attacker, places=4)
        self.assertAlmostEqual(evaluator.evaluate(board, "D"), -defender, places=4)

    def test_random_prefixes_are_short_and_unique(self):
        used = set()
        for index in range(10):
            opening = random_opening(index, used)
            self.assertGreaterEqual(len(opening.moves), PREFIX_MIN)
            self.assertLessEqual(len(opening.moves), PREFIX_MAX)
            start(opening)
        self.assertEqual(len(used), 10)

    def test_split_keeps_families_whole(self):
        samples = []
        for family in range(20):
            for index in range(family % 4 + 1):
                digest = hashlib.sha256(f"{family}-{index}".encode()).hexdigest()
                board = tuple(f"row-{digest[:6]}-{row}" for row in range(9))
                samples.append(Sample(
                    board=board, side="A", deep_eval=0.0, depth=5, budget=10.0,
                    outcome=1.0, game_id=f"g{family}-{index}", source="collection:test",
                    ply=12, started="2026-11-25T00:00:00+00:00", opening="r",
                    opening_family=f"f{family}", position_hash=digest,
                    history=(1,), elapsed=0.1, player_remaining=5.0,
                    team="Team19-v4", provenance="post_frozen_v4",
                    move=((0, 0), (0, 1)),
                ))
        families, report = split_families(samples, 0.15)
        self.assertTrue(families)
        self.assertLess(len(families), 20)
        self.assertEqual(report["samples_validation"] + report["samples_train"], len(samples))
        validation = [sample for sample in samples if sample.opening_family in families]
        training = [sample for sample in samples if sample.opening_family not in families]
        self.assertTrue(validation)
        self.assertTrue(training)
        self.assertEqual(len(validation), report["samples_validation"])
        self.assertFalse({sample.game_id for sample in validation}
                         & {sample.game_id for sample in training})


if __name__ == "__main__":
    unittest.main()
