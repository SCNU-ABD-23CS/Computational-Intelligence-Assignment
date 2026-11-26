"""Audit local Tablut journals and extract reusable opening evidence.

The command is deliberately read-only with respect to the corpus.  It scans
``D:/计算智能/results`` by default (or an explicit directory), handles both
arena JSONL journals and v1 replay JSON files, and emits a compact JSON report
or Markdown summary.  It is intended to seed opening/search experiments from
already recorded games without treating archived files as independent games.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "results"
STRONG = ("Team19-v4", "Team19-v3", "Grandmaster-Plus", "Grandmaster-Numba", "QSearch-PVS")


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_no}: {exc}")
    return rows, errors


def _fingerprint(row: dict[str, Any]) -> str:
    """Stable content identity, excluding timestamps and mutable progress fields."""
    identity = {key: row.get(key) for key in ("budget", "teams", "winner", "reason", "initial")}
    identity["prefix"] = [turn.get("move") for turn in row.get("prefix", [])]
    identity["moves"] = [attempt.get("move") for attempt in row.get("attempts", [])]
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _archive(path: Path) -> bool:
    return ".before-" in path.name or "archive" in str(path.parent).lower()


def _symmetry_key(board: list[str]) -> str:
    grid = tuple(tuple(row) for row in board)
    variants: list[str] = []
    for _ in range(4):
        variants.append("".join("".join(row) for row in grid))
        variants.append("".join("".join(reversed(row)) for row in grid))
        grid = tuple(zip(*reversed(grid)))
    return min(variants)


def _front_moves(row: dict[str, Any], limit: int = 4) -> list[list[list[int]]]:
    moves: list[list[list[int]]] = []
    for item in row.get("prefix", []):
        if len(moves) >= limit:
            break
        moves.append(item.get("move"))
    if len(moves) >= limit:
        return moves
    for item in row.get("attempts", []):
        event = item.get("event")
        if not isinstance(event, dict) or event.get("move") is None:
            continue
        moves.append(event["move"])
        if len(moves) >= limit:
            break
    return moves


def _outcome(row: dict[str, Any], team: str) -> str:
    winner = row.get("winner")
    if winner is None:
        return "draw"
    side = next((side for side, name in row.get("teams", {}).items() if name == team), None)
    return "win" if side and winner == side else "loss"


def _collect(root: Path) -> dict[str, Any]:
    jsonl_files = sorted(root.rglob("*.jsonl"))
    json_files = sorted(root.rglob("*.json"))
    rows: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for path in jsonl_files:
        loaded, errors = _load_jsonl(path)
        parse_errors.extend(errors)
        for row in loaded:
            row = dict(row)
            row["_path"] = str(path.relative_to(root))
            row["_archived"] = _archive(path)
            rows.append(row)

    # v1 replay JSON is a separate schema and is used only for replay coverage.
    replays = 0
    replay_schemas: Counter[str] = Counter()
    replay_versions: Counter[str] = Counter()
    for path in json_files:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parse_errors.append(f"{path}: {exc}")
            continue
        if isinstance(item, dict) and item.get("schema") == "team19.tablut.replay":
            replays += 1
            replay_schemas[str(item.get("schema"))] += 1
            replay_versions[str(item.get("version"))] += 1

    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _fingerprint(row)
        by_key[key].append(row)
    canonical = [min(group, key=lambda r: (r["_archived"], r["_path"])) for group in by_key.values()]

    schemas = Counter(str(row.get("schema")) for row in rows)
    budgets = Counter(str(row.get("budget")) for row in rows)
    stages = Counter(str(row.get("stage")) for row in rows)
    archived = sum(row["_archived"] for row in rows)
    duplicates = sum(max(0, len(group) - 1) for group in by_key.values())

    roles: Counter[str] = Counter()
    outcomes: dict[str, Counter[str]] = {team: Counter() for team in STRONG}
    opening_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    symmetry_classes: dict[str, str] = {}
    branch_records: Counter[str] = Counter()
    for row in canonical:
        teams = row.get("teams", {})
        for side, team in teams.items():
            roles[f"{team}:{side}"] += 1
            if team in outcomes:
                outcomes[team][_outcome(row, team)] += 1
        opening = str(row.get("opening"))
        opening_rows[opening].append(row)
        # Opening labels share the standard initial board.  For book analysis
        # classify the four-ply position when available, otherwise fall back
        # to the initial board (for standard/partial journals).
        prefix = row.get("prefix", [])
        position = prefix[-1].get("positionAfter", {}) if prefix else {}
        board = position.get("board")
        if not isinstance(board, list):
            board = row.get("initial", {}).get("board")
        if isinstance(board, list):
            symmetry_classes[opening] = _symmetry_key(board)
        stage = str(row.get("stage", ""))
        for branch in ("v4", "v3", "v2", "91ffcc8", "be23011", "3edc85c"):
            if branch in stage or branch in json.dumps(teams):
                branch_records[branch] += 1

    exact_sequences: list[dict[str, Any]] = []
    for opening in sorted(opening_rows):
        if opening not in {f"o{i:02d}" for i in range(1, 5)}:
            continue
        for row in opening_rows[opening]:
            if row.get("teams", {}).get("A") not in STRONG and row.get("teams", {}).get("D") not in STRONG:
                continue
            exact_sequences.append({
                "id": row.get("id"), "path": row.get("_path"), "stage": row.get("stage"),
                "budget": row.get("budget"), "opening": opening, "teams": row.get("teams"),
                "winner": row.get("winner"), "reason": row.get("reason"),
                "moves": _front_moves(row),
            })

    return {
        "root": root.name,
        "jsonlFiles": len(jsonl_files), "jsonFiles": len(json_files),
        "jsonlRecords": len(rows), "canonicalRecords": len(canonical),
        "duplicateReplays": duplicates, "archivedRecords": archived,
        "replayFiles": replays, "parseErrors": parse_errors,
        "schemas": dict(schemas), "replaySchemas": dict(replay_schemas),
        "replayVersions": dict(replay_versions), "budgets": dict(budgets),
        "stages": dict(stages), "roles": dict(roles),
        "strongEngines": {team: dict(outcomes[team]) for team in STRONG},
        "openingCounts": {key: len(value) for key, value in sorted(opening_rows.items())},
        "symmetryClasses": {key: hashlib.sha256(value.encode()).hexdigest()[:12]
                            for key, value in sorted(symmetry_classes.items())},
        "symmetryClassCount": len(set(symmetry_classes.values())),
        "branchSignals": dict(branch_records),
        "frontFourSequences": exact_sequences[:24],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Local Tablut book corpus audit", "", f"Corpus: `{report['root']}`", "",
             f"- JSONL files: {report['jsonlFiles']}; records: {report['jsonlRecords']}; canonical records: {report['canonicalRecords']}",
             f"- Archived records: {report['archivedRecords']}; duplicate replays: {report['duplicateReplays']}",
             f"- Replay JSON files: {report['replayFiles']}; parse errors: {len(report['parseErrors'])}",
             f"- Symmetry classes observed among opening labels: {report['symmetryClassCount']}", "", "## Strong-engine outcomes", ""]
    for team, counts in report["strongEngines"].items():
        lines.append(f"- **{team}**: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    lines += ["", "## Front-four exact sequences", ""]
    for row in report["frontFourSequences"]:
        lines.append(f"- `{row['id']}` {row['teams']} winner={row['winner']} reason={row['reason']} moves={row['moves']}")
    lines += ["", "## Interpretation", "", "Counts describe the available schedules. Equal team labels, budget, initial state, prefix, played moves and result count once across filenames and fixture IDs. Labels may span revisions. Opponent pools and budgets differ between teams."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, help="write JSON report")
    parser.add_argument("--markdown", type=Path, help="write Markdown report")
    args = parser.parse_args()
    report = _collect(args.root.resolve())
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(_markdown(report), encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
