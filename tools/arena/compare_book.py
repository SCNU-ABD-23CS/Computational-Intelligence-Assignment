"""Paired standard-opening book ablation using the unchanged arena referee."""

import argparse
import json
from functools import partial
from pathlib import Path

from Team19 import Player
from . import openings, runner


def compare(opponents, budget, stream):
    points = {False: 0, True: 0}
    for opponent in opponents:
        name = runner.TEAM_NAMES[opponent.casefold()]
        if name in ("Team19-v4", "Team19-New"):
            raise ValueError("Choose an independent opponent or a historical snapshot")
        for attack in (True, False):
            teams = ("Team19-v4", name) if attack else (name, "Team19-v4")
            match = runner.Match(*teams, budget, openings.standard())
            # Identical match IDs give the two arms identical opponent seeds.
            for enabled in ((False, True) if attack else (True, False)):
                factories = {"Team19-v4": partial(Player, use_book=enabled)}
                game = runner.run_match(match, factories=factories)
                side = "A" if attack else "D"
                score = 1 if game["winner"] is None else 2 * (game["winner"] == side)
                points[enabled] += score
                turns = [t for t in game["attempts"] if t["side"] == side]
                result = dict(book=enabled, id=game["id"], side=side, points=score,
                              reason=game["reason"], hits=sum(bool(t.get("bookHit")) for t in turns),
                              fallbacks=sum(bool(t.get("fallback")) for t in turns),
                              errors=sum("error" in t or "exception" in t for t in turns),
                              openingSeconds=turns[0]["elapsed"] if turns else None)
                print(json.dumps(result), flush=True)
                if stream:
                    game["bookEnabled"] = enabled
                    stream.write(json.dumps(game, separators=(",", ":")) + "\n")
                    stream.flush()
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponents", nargs="+", default=["qs", "v3"])
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--output", type=Path, help="New JSONL file for the complete replays")
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error("budget must be positive")
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            scores = compare(args.opponents, args.budget, stream)
    else:
        scores = compare(args.opponents, args.budget, None)
    print(f"bare={scores[False]} book={scores[True]}")


if __name__ == "__main__":
    main()
