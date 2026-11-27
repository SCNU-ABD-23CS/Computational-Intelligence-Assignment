"""Build and validate the frozen unseen-opening branch suite (member 1, task A).

The suite is deliberately *not* the twelve four-ply openings in
``tools/arena/openings.json``. Those are the openings every Team19 snapshot has
already been tuned against, so reusing them would make a "blind" evaluation
anything but blind. This module generates fresh quiet branches from the official
standard position, removes rotation/reflection duplicates, keeps a balanced
spread over the ten symmetry classes of the attacker's first move, and freezes
the result to JSON so that every later run and every report quotes the same data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import date
from io import StringIO
import json
from pathlib import Path
import random

import Tablut
import Team19_engine as engine

from tools.arena.openings import (
    Opening,
    _has_immediate_win,
    _hanging_position,
    _symmetry_key,
    load_suite,
    move_event,
    start,
)


SEED = 20261126
TARGET = 48
# The 80 legal first attacker moves collapse to eight D4 classes, so six
# branches per class is exactly the balanced maximum for a 48-branch suite.
MAX_PER_CLASS = 6
DEPTH_CHOICES = (4, 4, 6)
MAX_ATTEMPTS = 60_000
SUITE = Path(__file__).with_name("blind_branches.json")


@dataclass(frozen=True)
class Branch:
    name: str
    branch_class: str
    plies: int
    moves: list[list[list[int]]]
    signature: str
    symmetry: str

    def opening(self) -> Opening:
        return Opening(self.name, self.moves, self.signature)


def _class_of(moves: list[list[list[int]]]) -> str:
    """Symmetry class of the position after the attacker's first move."""
    game = Tablut.Tablut()
    with redirect_stdout(StringIO()):
        game.make_move(True, moves[0])
    return _symmetry_key(game.board)


def _quiet_sequence(rng: random.Random, plies: int):
    """Play a random sequence and reject it on the first tactical blemish."""
    game = Tablut.Tablut()
    game.past_positions = [game.compute_zobrist_hash(True)]
    moves = []
    for ply in range(plies):
        attacker = ply % 2 == 0
        legal = engine.get_legal_moves(game.board, attacker)
        if not legal:
            return None
        move = rng.choice(legal)
        before = game.board
        with redirect_stdout(StringIO()):
            if not game.make_move(attacker, move) or game.check_if_game_over(not attacker):
                return None
        event = move_event(game, before, attacker, move, ply + 1)
        if event["captures"] or _hanging_position(game.board, not attacker, move[1]):
            return None
        moves.append(move)
    if any(_has_immediate_win(game.board, side) for side in (True, False)):
        return None
    return moves, game


def build(seed: int = SEED, target: int = TARGET) -> list[Branch]:
    rng = random.Random(seed)
    existing = load_suite()
    blocked_symmetries = {_symmetry_key(start(opening).referee.board) for opening in existing}
    blocked_signatures = {opening.signature for opening in existing}

    accepted: list[Branch] = []
    per_class: Counter[str] = Counter()
    signatures: set[str] = set()
    symmetries: set[str] = set()

    for _ in range(MAX_ATTEMPTS):
        if len(accepted) >= target:
            break
        plies = rng.choice(DEPTH_CHOICES)
        produced = _quiet_sequence(rng, plies)
        if produced is None:
            continue
        moves, game = produced
        signature = f"{game.past_positions[-1]:016x}"
        symmetry = _symmetry_key(game.board)
        branch_class = _class_of(moves)
        if signature in signatures or signature in blocked_signatures:
            continue
        if symmetry in symmetries or symmetry in blocked_symmetries:
            continue
        if per_class[branch_class] >= MAX_PER_CLASS:
            continue
        signatures.add(signature)
        symmetries.add(symmetry)
        per_class[branch_class] += 1
        accepted.append(Branch(f"n{len(accepted) + 1:02d}", branch_class, plies,
                               moves, signature, symmetry))
    if len(accepted) < target:
        raise RuntimeError(f"only {len(accepted)} branches accepted for target {target}")
    return accepted


def freeze(branches: list[Branch], path: Path = SUITE, *, seed: int = SEED) -> None:
    classes = sorted({branch.branch_class for branch in branches})
    payload = {
        "seed": seed,
        "generated": date.today().isoformat(),
        "target": len(branches),
        "design": {
            "source": "official standard position of Tablut.py",
            "plies": {str(depth): sum(branch.plies == depth for branch in branches)
                      for depth in sorted({branch.plies for branch in branches})},
            "classes": len(classes),
            "perClassCap": MAX_PER_CLASS,
            "filters": [
                "no capture inside the branch",
                "the last moved piece is not capturable immediately",
                "neither side has a one-move win at the final position",
                "D4 rotation/reflection duplicates removed",
                "no overlap with tools/arena/openings.json o01-o12",
            ],
        },
        "branches": [asdict(branch) for branch in branches],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load(path: Path = SUITE) -> list[Branch]:
    data = json.loads(path.read_text(encoding="utf-8"))
    branches = [Branch(**entry) for entry in data["branches"]]
    if not 24 <= len(branches) <= 48:
        raise ValueError("the blind suite must hold between 24 and 48 branches")
    if len({branch.name for branch in branches}) != len(branches):
        raise ValueError("duplicate branch name")
    if len({branch.signature for branch in branches}) != len(branches):
        raise ValueError("duplicate branch hash")
    if len({branch.symmetry for branch in branches}) != len(branches):
        raise ValueError("a rotation/reflection duplicate slipped in")
    if any(branch.plies not in DEPTH_CHOICES or branch.plies % 2 for branch in branches):
        raise ValueError("branches must be an even number of plies from the frozen depths")
    referee = load_suite()
    referee_suite = {opening.signature for opening in referee} | {
        _symmetry_key(start(opening).referee.board) for opening in referee}
    if {branch.signature for branch in branches} & referee_suite:
        raise ValueError("branch overlaps the referee opening suite")
    for branch in branches:
        start(branch.opening())
    return branches


def coverage(branches: list[Branch]) -> dict[str, int]:
    return dict(Counter(branch.branch_class for branch in branches))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target", type=int, default=TARGET,
                        help="24 to 48 branches")
    parser.add_argument("--out", type=Path, default=SUITE)
    args = parser.parse_args()
    built = build(args.seed, args.target)
    freeze(built, args.out, seed=args.seed)
    print(f"{len(built)} branches -> {args.out}")
    print("plies:", Counter(branch.plies for branch in built))
    for branch_class, count in sorted(coverage(built).items()):
        print(f"  class {count} branches")
