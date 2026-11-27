"""Export static book paths from frozen-v4 replays and adjudicated candidate trials."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STANDARD = "arena-standard-b2c5e0df9948-60s.jsonl"
ROOT = (0, 5, 1, 5)
DEFAULT_EVIDENCE_DIR = HERE / "results"
_ALLOWED_REASONS = frozenset((
    "king escaped to the edge",
    "fewer than two attackers remain",
    "no legal move",
    "repeated position",
))


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
    groups = defaultdict(list)
    for raw in journal.read_text(encoding="utf-8").splitlines():
        r = json.loads(raw)
        path = tuple(flat(m) for m in r["prefix"])
        groups[key(path)].append((path, r))

    supported = {}
    for whole_path, samples in groups.items():
        rows = [row for _, row in samples]
        prefix = key(whole_path[:-1])
        existing = lines.get(prefix)
        if any(not _safe(row) for row in rows):
            if existing is not None and key(existing) == whole_path:
                lines.pop(prefix)
            continue
        if len({row["opponent"] for row in rows}) < 2 or not any(
            row["winner"] == "D" for row in rows
        ):
            continue
        supported[whole_path] = samples

    by_prefix = defaultdict(list)
    for whole_path, samples in supported.items():
        by_prefix[key(whole_path[:-1])].append((whole_path, samples))

    accepted = set()
    for prefix, candidates in by_prefix.items():
        if len(candidates) != 1:
            lines.pop(prefix, None)
            continue
        whole_path, samples = candidates[0]
        existing = lines.get(prefix)
        if existing is not None and key(existing) != whole_path:
            lines.pop(prefix)
            continue
        if existing is None:
            lines[prefix] = min(path for path, _ in samples)
        accepted.add(whole_path)

    replies = defaultdict(lambda: defaultdict(list))
    for parent_path, samples in groups.items():
        for path, row in samples:
            if len(row["nextMoves"]) != 2:
                continue
            extended = path + tuple(flat(move) for move in row["nextMoves"])
            replies[key(extended[:-1])][key(extended)].append((extended, row, parent_path))

    for prefix, candidates in replies.items():
        if len(candidates) != 1:
            lines.pop(prefix, None)
            continue
        whole_path, samples = next(iter(candidates.items()))
        rows = [row for _, row, _ in samples]
        existing = lines.get(prefix)
        if any(not _safe(row) for row in rows):
            if existing is not None and key(existing) == whole_path:
                lines.pop(prefix)
            continue
        if (len({row["opponent"] for row in rows}) < 2
                or not any(row["winner"] == "D" for row in rows)
                or not any(parent in accepted for _, _, parent in samples)):
            continue
        if existing is not None and key(existing) != whole_path:
            lines.pop(prefix)
            continue
        if existing is None:
            lines[prefix] = min(path for path, _, _ in samples)


def _safe(row):
    return (not row["errors"] and not row["fallbacks"] and row["winner"] != "A"
            and row["reason"] in _ALLOWED_REASONS)


def export(results, output, evidence_dir=DEFAULT_EVIDENCE_DIR):
    lines = historical(results / STANDARD)
    screened(lines, evidence_dir / "book_reply_trials.jsonl")
    decisions_path = evidence_dir / "book_decisions.json"
    decisions = json.loads(decisions_path.read_text())
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
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--output", type=Path, default=HERE.parents[1] / "Team19_book_data.py")
    args = parser.parse_args()
    print(export(args.results, args.output, args.evidence_dir), "book lines")
