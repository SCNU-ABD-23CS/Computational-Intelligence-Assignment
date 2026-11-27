"""Shared, side-effect-free primitives for the Team19 NNUE data pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

import Team19_engine as engine
import Team19_eval as evaluation
from Team19_weights import native_weights


ROOT = Path(__file__).resolve().parents[2]
ARENA_RESULTS = ROOT / "tools" / "arena" / "results"
NNUE_RESULTS = ROOT / "tools" / "nnue" / "results"
V4_COMMIT = "f36f3dc"
V4_CUTOFF = datetime(2026, 11, 24, 12, 42, 59, tzinfo=timezone.utc)
WHITELIST = frozenset({
    "arena-9451a7e4dcf4-10s.jsonl",
    "arena-9451a7e4dcf4-60s.jsonl",
})
FROZEN_TEAMS = frozenset({"Team19-v4", "Team19-New"})
MIN_BUDGET = 10.0
# Our own collection runs at 6s and is gated by measured filtered depth >= 4.5;
# the 10s floor only applies to reviewed historical arena logs in WHITELIST.
MIN_COLLECT_BUDGET = 5.0
MIN_DEPTH = 4
MIN_PLY = 9
MATE_THRESHOLD = 90_000.0
OUTCOMES = {"A": 1.0, "D": 0.0, None: 0.5}


def parse_started(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def board_rows(board: Iterable[str] | Iterable[Iterable[str]]) -> list[list[str]]:
    return [list(row) for row in board]


def board_tuple(board: Iterable[str] | Iterable[Iterable[str]]) -> tuple[str, ...]:
    return tuple("".join(row) for row in board)


def d4_variants(board: Iterable[str] | Iterable[Iterable[str]]) -> tuple[str, ...]:
    grid = tuple(tuple(row) for row in board)
    variants = []
    for _ in range(4):
        variants.append("/".join("".join(row) for row in grid))
        variants.append("/".join("".join(reversed(row)) for row in grid))
        grid = tuple(zip(*reversed(grid)))
    return tuple(variants)


def d4_key(board: Iterable[str] | Iterable[Iterable[str]], side: str | None = None) -> str:
    key = min(d4_variants(board))
    return key if side is None else f"{key}|{side}"


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class Sample:
    board: tuple[str, ...]
    side: str
    deep_eval: float
    depth: int
    budget: float
    outcome: float
    game_id: str
    source: str
    ply: int
    started: str
    opening: str
    opening_family: str
    position_hash: str
    history: tuple[int, ...]
    elapsed: float
    player_remaining: float | None
    team: str
    provenance: str
    move: tuple[tuple[int, int], tuple[int, int]]

    @property
    def exact_key(self) -> tuple[tuple[str, ...], str]:
        return self.board, self.side

    @property
    def symmetry_key(self) -> str:
        return d4_key(self.board, self.side)

    @property
    def sample_id(self) -> str:
        return stable_digest("\n".join((*self.board, self.side)))

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, row: dict) -> "Sample":
        row = dict(row)
        row["board"] = tuple(row["board"])
        row["history"] = tuple(row["history"])
        row["move"] = tuple(tuple(square) for square in row["move"])
        return cls(**row)


def position_hash(position: dict) -> int:
    value = position.get("hash")
    if value:
        return int(value, 16)
    return int(engine.compute_hash(board_rows(position["board"]), position["sideToMove"] == "A"))


def eligible_game(game: dict, source: str) -> tuple[bool, str]:
    floor = MIN_BUDGET if source in WHITELIST else MIN_COLLECT_BUDGET
    if float(game.get("budget", 0.0)) < floor:
        return False, "budget"
    if source in WHITELIST:
        return True, "reviewed_whitelist"
    try:
        started = parse_started(game["started"])
    except (KeyError, TypeError, ValueError):
        return False, "invalid_started"
    if started < V4_CUTOFF:
        return False, "pre_frozen_v4"
    return True, "post_frozen_v4"


def game_samples(game: dict, source: str, provenance: str) -> Iterator[Sample]:
    """Pair each root score with the board before that move, including prefixes."""
    winner = game.get("winner")
    if winner not in OUTCOMES:
        raise ValueError(f"Unknown winner {winner!r} in {game.get('id')}")
    outcome = OUTCOMES[winner]
    positions = [game["initial"]]
    prefix = game.get("prefix") or []
    for event in prefix:
        positions.append(event["positionAfter"])
    opening_family = d4_key(positions[-1]["board"])
    previous_capture = bool(prefix and prefix[-1].get("captures"))
    for attempt in game.get("attempts") or []:
        event = attempt.get("event")
        if event is None:
            break
        position = positions[-1]
        team = attempt.get("team")
        score = attempt.get("score")
        depth = attempt.get("depth")
        if (team in FROZEN_TEAMS and score is not None and depth is not None
                and int(depth) >= MIN_DEPTH and int(event["ply"]) >= MIN_PLY
                and not previous_capture):
            board = board_tuple(position["board"])
            side = position["sideToMove"]
            if side != attempt.get("side"):
                raise ValueError(f"Side mismatch at {game.get('id')} ply {event['ply']}")
            yield Sample(
                board=board,
                side=side,
                deep_eval=float(score),
                depth=int(depth),
                budget=float(game["budget"]),
                outcome=float(outcome),
                game_id=str(game["id"]),
                source=source,
                ply=int(event["ply"]),
                started=str(game["started"]),
                opening=str(game.get("opening", "standard")),
                opening_family=opening_family,
                position_hash=str(position.get("hash") or f"{position_hash(position):016x}"),
                history=tuple(position_hash(item) for item in positions),
                elapsed=float(attempt.get("elapsed") or 0.0),
                player_remaining=attempt.get("playerRemaining"),
                team=str(team),
                provenance=provenance,
                move=tuple(tuple(int(x) for x in square) for square in event["move"]),
            )
        positions.append(event["positionAfter"])
        previous_capture = bool(event.get("captures"))


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{number}") from error


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    temporary.replace(path)


class NativeEvaluator:
    """The exact 22-feature, attacker-positive base used by frozen v4."""

    def __init__(self):
        self.weights = native_weights(None)
        self.piece_bonus = evaluation._native_piece_bonus(self.weights)
        self.queue = np.empty(engine._AREA, dtype=np.int16)
        self.seen = np.empty(engine._AREA, dtype=np.bool_)

    def evaluate(self, board: Iterable[str] | Iterable[Iterable[str]], side: str) -> float:
        rows = board_rows(board)
        root = engine._search_position(rows)
        if root.king is None:
            raise ValueError("The native base evaluator requires an ongoing position with a king")
        king = root.king[0] * 9 + root.king[1]
        flat = engine._flat(rows)
        value = evaluation._native_evaluate(
            flat, side == "A", root.attackers, root.defenders, king,
            self.weights, self.queue, self.seen, engine._NEAR, engine._EDGE,
            self.piece_bonus,
        )
        return float(value)


def encode_board(board: Iterable[str] | Iterable[Iterable[str]]) -> np.ndarray:
    """Return the engine's own 0/1/2/3/4 encoding; do not duplicate its mapping."""
    return engine._flat(board_rows(board))
