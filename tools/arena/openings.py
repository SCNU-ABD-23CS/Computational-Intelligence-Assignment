"""Generate and validate the fixed four-ply opening suite."""

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
import json
import random

import Tablut
import Team19_engine as engine

from . import HERE


SEED = 20261121
SUITE = HERE / "openings.json"


@dataclass(frozen=True)
class Opening:
    name: str
    moves: list[list[list[int]]]
    signature: str


@dataclass
class StartedGame:
    referee: Tablut.Tablut
    positions: list[dict]
    turns: list[dict]
    last_board: dict[bool, list[list[str]]]


def position(game: Tablut.Tablut, attacker: bool) -> dict:
    return {
        "board": ["".join(row) for row in game.board],
        "sideToMove": "A" if attacker else "D",
        "hash": f"{game.compute_zobrist_hash(attacker):016x}",
    }


def move_event(game: Tablut.Tablut, before: list[list[str]], attacker: bool, move: list,
               ply: int) -> dict:
    r0, c0 = move[0]
    captures = [[r, c, piece] for r, row in enumerate(before) for c, piece in enumerate(row)
                if piece in "ADK" and (r, c) != (r0, c0) and game.board[r][c] == "."]
    return {"ply": ply, "side": "A" if attacker else "D", "move": move,
            "piece": before[r0][c0], "captures": captures, "positionAfter": position(game, not attacker)}


def start(opening: Opening, game: Tablut.Tablut | None = None) -> StartedGame:
    game = Tablut.Tablut() if game is None else game
    game.past_positions = [game.compute_zobrist_hash(True)]
    started = StartedGame(game, [position(game, True)], [], {})
    with redirect_stdout(StringIO()):
        for ply, move in enumerate(opening.moves, 1):
            attacker = ply % 2 == 1
            before = game.board
            if not game.make_move(attacker, move) or game.check_if_game_over(not attacker):
                raise ValueError(f"Opening {opening.name} stops at ply {ply}")
            event = move_event(game, before, attacker, move, ply)
            started.turns.append(event)
            started.positions.append(event["positionAfter"])
            started.last_board[attacker] = game.board
    if opening.signature and opening.signature != started.positions[-1]["hash"]:
        raise ValueError(f"Opening {opening.name} has a hash mismatch")
    return started


def standard() -> Opening:
    return Opening("standard", [], "")


def _symmetry_key(board: list[list[str]]) -> str:
    grid = tuple(tuple(row) for row in board)
    variants = []
    for _ in range(4):
        variants.append("".join("".join(row) for row in grid))
        variants.append("".join("".join(reversed(row)) for row in grid))
        grid = tuple(zip(*reversed(grid)))
    return min(variants)


def _hanging_position(board: list[list[str]], next_attacker: bool, last_square: list[int]) -> bool:
    row, col = last_square
    for reply in engine.get_legal_moves(board, next_attacker):
        child = engine.apply_move(board, reply, next_attacker)
        if child[row][col] == "." or engine.check_winner(child, not next_attacker) == (
            "A" if next_attacker else "D"
        ):
            return True
    return False


def _has_immediate_win(board: list[list[str]], attacker: bool) -> bool:
    winner = "A" if attacker else "D"
    return any(engine.check_winner(engine.apply_move(board, move, attacker), not attacker) == winner
               for move in engine.get_legal_moves(board, attacker))


def generate(seed: int = SEED) -> list[Opening]:
    rng = random.Random(seed)
    suite = []
    hashes = set()
    symmetries = set()
    for _ in range(10_000):
        game = Tablut.Tablut()
        game.past_positions = [game.compute_zobrist_hash(True)]
        moves = []
        for ply in range(4):
            attacker = ply % 2 == 0
            move = rng.choice(engine.get_legal_moves(game.board, attacker))
            before = game.board
            with redirect_stdout(StringIO()):
                if not game.make_move(attacker, move) or game.check_if_game_over(not attacker):
                    break
            event = move_event(game, before, attacker, move, ply + 1)
            if event["captures"] or _hanging_position(game.board, not attacker, move[1]):
                break
            moves.append(move)
        if len(moves) != 4 or any(_has_immediate_win(game.board, side) for side in (True, False)):
            continue
        signature = game.past_positions[-1]
        symmetry = _symmetry_key(game.board)
        if signature in hashes or symmetry in symmetries:
            continue
        hashes.add(signature)
        symmetries.add(symmetry)
        suite.append(Opening(f"o{len(suite) + 1:02d}", moves, f"{signature:016x}"))
        if len(suite) == 12:
            return suite
    raise RuntimeError("Unable to generate 12 distinct quiet openings")


def load_suite() -> list[Opening]:
    data = json.loads(SUITE.read_text(encoding="utf-8"))
    suite = [Opening(**entry) for entry in data["openings"]]
    if data["seed"] != SEED or len(suite) != 12 or any(len(entry.moves) != 4 for entry in suite):
        raise ValueError("Expected the fixed set of twelve four-ply openings")
    if len({entry.signature for entry in suite}) != 12 or len({entry.name for entry in suite}) != 12:
        raise ValueError("Duplicate opening")
    for entry in suite:
        start(entry)
    return suite


if __name__ == "__main__":
    from dataclasses import asdict

    SUITE.write_text(json.dumps({"seed": SEED, "openings": [asdict(x) for x in generate()]}, indent=2)
                     + "\n", encoding="utf-8")
