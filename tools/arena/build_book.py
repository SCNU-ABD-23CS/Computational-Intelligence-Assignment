"""Export static book paths from frozen-v4 replays and adjudicated candidate trials."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STANDARD = "arena-standard-b2c5e0df9948-60s.jsonl"
ROOT = (0, 5, 1, 5)


def flat(move):
    return (*move[0], *move[1])


def transform(move, t):
    def point(r, c):
        if t >= 4:
            c = 8 - c
        for _ in range(t % 4):
            r, c = c, 8-r
        return r, c
    return (*point(*move[:2]), *point(*move[2:]))


def key(path):
    return min(tuple(transform(m, t) for m in path) for t in range(8))


def historical(journal):
    groups = defaultdict(lambda: defaultdict(list))
    roots = set()
    for raw in journal.read_text(encoding="utf-8-sig").splitlines():
        r = json.loads(raw)
        if r["opening"] != "standard" or r["prefix"] or r["budget"] != 60:
            continue
        if "Team19-v4" not in r["teams"].values():
            continue
        path = ()
        for i, attempt in enumerate(r["attempts"][:4]):
            side = "A" if i % 2 == 0 else "D"
            if "event" not in attempt or attempt["side"] != side:
                raise ValueError(f"Invalid replay: {r['id']}")
            path += (flat(attempt["event"]["move"]),)
            if r["teams"][side] == "Team19-v4":
                if i == 0 and path == (ROOT,):
                    roots.add(r["teams"]["D"])
                groups[key(path[:-1])][key(path)].append((path, r["winner"] in (side, None)))
    if len(roots) != 13:
        raise ValueError("Expected frozen v4 to play the preferred root against 13 distinct opponents")
    result = {(): (ROOT,)}
    for prefix, choices in groups.items():
        if prefix and len(choices) == 1:
            observations = next(iter(choices.values()))
            if all(safe for _, safe in observations):
                result[prefix] = min(path for path, _ in observations)
    return result


def screened(lines, journal):
    groups = defaultdict(dict)
    for raw in journal.read_text(encoding="utf-8").splitlines():
        r = json.loads(raw)
        path = tuple(flat(m) for m in r["prefix"])
        groups[path][r["opponent"]] = r
    for path, opponents in groups.items():
        if set(opponents) != {"Grandmaster-Plus", "Team19-v3"}:
            continue
        rows = list(opponents.values())
        if any(r["errors"] or r["fallbacks"] or r["winner"] == "A" or
               r["reason"] not in ("king escaped to the edge", "fewer than two attackers remain",
                                   "no legal move", "repeated position") for r in rows):
            continue
        if not any(r["winner"] == "D" for r in rows):
            continue
        chosen = lines.setdefault(key(path[:-1]), path)
        if key(chosen) != key(path):
            continue
        for r in rows:
            if len(r["nextMoves"]) == 2 and r["winner"] == "D":
                fourth = path + tuple(flat(m) for m in r["nextMoves"])
                lines.setdefault(key(fourth[:-1]), fourth)


def export(results, output):
    lines = historical(results / STANDARD)
    screened(lines, HERE / "book_reply_trials.jsonl")
    decisions = json.loads((HERE / "book_decisions.json").read_text())
    for item in decisions:
        path = tuple(tuple(m) for m in item["line"])
        lines[key(path[:-1])] = path
    ordered = sorted(lines.values(), key=lambda path: (len(path), path))
    output.write_text('"""Verified opening paths. Regenerate with tools.arena.build_book."""\n\n'
                      + "BOOK_LINES = (\n" + "".join(f"    {path!r},\n" for path in ordered)
                      + ")\n", encoding="utf-8")
    return len(ordered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE.parents[2] / "results")
    parser.add_argument("--output", type=Path, default=HERE.parents[1] / "Team19_book_data.py")
    args = parser.parse_args()
    print(export(args.results, args.output), "book lines")
