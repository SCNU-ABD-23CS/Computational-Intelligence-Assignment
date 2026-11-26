"""Screen the ten standard first-move families against five independent search policies."""

import argparse
from importlib import import_module
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import Tablut
import Team19_engine as engine
from . import runner
from .openings import Opening, _symmetry_key


def first_families():
    board = Tablut.Tablut().board
    groups = {}
    for move in engine.get_legal_moves(board, True):
        key = _symmetry_key(engine.apply_move(board, move, True))
        groups.setdefault(key, []).append(move)
    return list(groups.values())


def screen(budget):
    root = Tablut.Tablut().board
    root_hash = engine.compute_hash(root, True)
    names = ["v4", "v3", "gmp", "gm", "qs"]
    bots = {}
    for name in names:
        entrant = runner.TEAMS[runner.TEAM_NAMES[name]]
        module = import_module(entrant.module, runner.__package__)
        # Search directly with the frozen kernel, including when a book is installed.
        bots[name] = module.Player() if entrant.search else module.Player(seed=19)

    def choose(name, board, side, history):
        player = bots[name]
        player.reset_game()
        if name in ("v4", "v3"):
            native = player._native
            move = native.choose(board, side, perf_counter() + budget / 32, set(history))
            telemetry = native.last_search
        else:
            player.seed_history(history)
            move = (player.play_attacker if side else player.play_defender)(board)
            telemetry = player.last_search
        if move not in engine.get_legal_moves(board, side) or telemetry is None or telemetry.fallback:
            raise RuntimeError(f"{name} failed to complete a legal search")
        return dict(move=move, depth=int(telemetry.depth), score=int(telemetry.score))

    families = []
    for moves in first_families():
        first = min(moves)
        board = engine.apply_move(root, first, True)
        history = {root_hash, engine.compute_hash(board, False)}
        replies = {name: choose(name, board, False, history) for name in names}
        branches = {}
        for response in replies.values():
            second = response["move"]
            key = str(second)
            if key in branches:
                continue
            after = engine.apply_move(board, second, False)
            hist = history | {engine.compute_hash(after, True)}
            third = choose("v4", after, True, hist)
            after = engine.apply_move(after, third["move"], True)
            fourth = choose("v4", after, False, hist | {engine.compute_hash(after, False)})
            branches[key] = [second, third, fourth]
        families.append(dict(first=first, orbit=len(moves), replies=replies,
                             continuations=list(branches.values()),
                             worstV4Score=min(line[-1]["score"] for line in branches.values())))
    return sorted(families, key=lambda item: -item["worstV4Score"])


def validate_replies(families, budget, stream):
    """Test the frozen defender after each forced two-ply prefix against two attackers."""
    from functools import partial
    from Team19 import Player

    for entry in families:
        prefix = [entry["first"], entry["replies"]["v4"]["move"]]
        for opponent in ("Team19-v3", "Grandmaster-Plus"):
            match = runner.Match(opponent, "Team19-v4", budget, Opening("screen", prefix, ""))
            game = runner.run_match(match, factories={"Team19-v4": partial(Player, use_book=False)})
            turns = game["attempts"]
            result = dict(prefix=prefix, opponent=opponent, budget=budget, winner=game["winner"],
                          reason=game["reason"], plies=len(turns),
                          nextMoves=[t["move"] for t in turns[:2] if "event" in t],
                          fallbacks=sum(bool(t.get("fallback")) for t in turns),
                          errors=sum("error" in t or "exception" in t for t in turns))
            stream.write(json.dumps(result) + "\n")
            stream.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate", type=Path, help="Replay defence choices from a previous screen")
    args = parser.parse_args()
    if args.validate:
        families = json.loads(args.validate.read_text(encoding="utf-8"))["families"]
        with args.output.open("x", encoding="utf-8") as stream:
            validate_replies(families, args.budget, stream)
    else:
        with patch.object(Tablut, "TIME_BUDGET", args.budget):
            data = screen(args.budget)
        args.output.write_text(json.dumps(dict(budget=args.budget, families=data), indent=2), encoding="utf-8")
