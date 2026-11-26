"""Summarise paired arena logs and decide whether to enable the candidate by default."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median


def summarize(journal):
    opener = gzip.open if journal.suffix == ".gz" else open
    with opener(journal, "rt", encoding="utf-8") as stream:
        games = [json.loads(row) for row in stream if row.strip()]
    pairs = {}
    for game in games:
        side = next(s for s, team in game["teams"].items() if team == "Team19-v4")
        other = "A" if side == "D" else "D"
        turns = [a for a in game["attempts"] if a["side"] == side]
        enabled = game["bookEnabled"]
        winner = game["winner"]
        result = dict(points=1 if winner is None else 2 * int(winner == side),
                      reason=game["reason"], hits=sum(bool(t.get("bookHit")) for t in turns),
                      firstSeconds=turns[0]["elapsed"], fallbacks=sum(bool(t.get("fallback")) for t in turns),
                      errors=sum("error" in t or "exception" in t for t in turns),
                      refereeLoss=winner == other and game["reason"] in ("timeout", "exception", "illegal move"))
        key = (game["teams"][other], side)
        group = pairs.setdefault(key, {})
        arm = "book" if enabled else "bare"
        if arm in group or game["budget"] != 60 or game["opening"] != "standard" or game["prefix"]:
            raise ValueError(f"Duplicate arm or unexpected match conditions: {game['id']}")
        group[arm] = result
    expected = {(team, side) for team in ("Team19-v3", "Grandmaster-Numba", "Grandmaster-Plus", "QSearch-PVS")
                for side in ("A", "D")}
    if set(pairs) != expected or any(set(pair) != {"bare", "book"} for pair in pairs.values()):
        raise ValueError("Expected all sixteen matches across four opponents, both roles and both arms")
    scores = {arm: sum(p[arm]["points"] for p in pairs.values()) for arm in ("bare", "book")}
    healthy = all(not (p["book"]["refereeLoss"] or p["book"]["fallbacks"] or p["book"]["errors"])
                  for p in pairs.values())
    savings = [p["bare"]["firstSeconds"] - p["book"]["firstSeconds"] for (team, s), p in pairs.items() if s == "A"]
    return dict(budget=60, games=len(games), points=scores, candidateHealthy=healthy,
                enableByDefault=healthy and scores["book"] > scores["bare"],
                medianAttackerFirstSecondsSaved=median(savings),
                matches=[dict(opponent=t, side=s, **p) for (t, s), p in pairs.items()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.journal), indent=2) + "\n", encoding="utf-8")
