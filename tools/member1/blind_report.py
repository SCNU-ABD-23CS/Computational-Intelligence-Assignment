"""Summarise blind-evaluation journals into the numbers the report will quote.

Wins are worth 2, draws 1, losses 0, exactly like the course league. Every
number is reported separately for the two colours, because a candidate that
improves the attacker while quietly giving back points as the defender must not
pass the release gate.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path

from tools.arena import runner


CONFIG_PREFIX = "M1-"
BASELINE = "Team19-v4"


def config_label(teams: dict[str, str]) -> str | None:
    for name in teams.values():
        if name.startswith(CONFIG_PREFIX) or name == BASELINE:
            return name
    return None


def side_of(teams: dict[str, str], name: str) -> str:
    return "A" if teams["A"] == name else "D"


def faults(row: dict, name: str) -> Counter:
    """Count referee-visible faults and self-reported fallbacks for one player."""
    marks = Counter()
    if row.get("reason") in ("timeout", "illegal move", "exception"):
        offender = row["teams"].get("D" if row["winner"] == "A" else "A")
        marks[row["reason"]] += 1 if offender == name else 0
    for attempt in row.get("attempts", ()):
        if attempt.get("team") != name:
            continue
        marks["illegal attempt"] += "exception" not in attempt and attempt.get("move") == []
        marks["exception"] += "exception" in attempt
        marks["reported error"] += "error" in attempt
        marks["fallback"] += bool(attempt.get("fallback"))
        marks["no time left"] += (attempt.get("playerRemaining") or 0) <= 0
    return +marks


def summarise(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    buckets = defaultdict(lambda: {"games": 0, "points": 0, "w": 0, "d": 0, "l": 0,
                                   "A": Counter(), "D": Counter(), "faults": Counter()})
    for row in rows:
        name = config_label(row["teams"])
        if name is None:
            continue
        side = side_of(row["teams"], name)
        opponent = row["teams"]["D" if side == "A" else "A"]
        key = (name, opponent, row["budget"])
        bucket = buckets[key]
        bucket["games"] += 1
        bucket[side]["games"] += 1
        winner = row["winner"]
        if winner is None:
            bucket["d"] += 1
            bucket[side]["d"] += 1
            bucket["points"] += 1
            bucket[side]["points"] += 1
        elif winner == side:
            bucket["w"] += 1
            bucket[side]["w"] += 1
            bucket["points"] += 2
            bucket[side]["points"] += 2
        else:
            bucket["l"] += 1
            bucket[side]["l"] += 1
        bucket["faults"].update(faults(row, name))
    table = []
    for (name, opponent, budget), bucket in sorted(buckets.items()):
        table.append({
            "config": name, "opponent": opponent, "budget": budget,
            "games": bucket["games"], "points": bucket["points"],
            "w": bucket["w"], "d": bucket["d"], "l": bucket["l"],
            "asAttacker": dict(bucket["A"], points=bucket["A"]["points"],
                               games=bucket["A"]["games"]),
            "asDefender": dict(bucket["D"], points=bucket["D"]["points"],
                               games=bucket["D"]["games"]),
            "faults": dict(bucket["faults"]),
        })
    return {"rows": len(rows), "summary": table}


def to_markdown(data: dict) -> str:
    lines = ["| 配置 | 对手 | 预算 | 局数 | 得分 | 胜/和/负 | 执攻得分 | 执守得分 | 故障 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in data["summary"]:
        attack = f"{row['asAttacker']['points']}/{row['asAttacker']['games']}"
        defend = f"{row['asDefender']['points']}/{row['asDefender']['games']}"
        marks = ", ".join(f"{name}×{count}" for name, count in row["faults"].items()) or "—"
        lines.append(f"| {row['config']} | {row['opponent']} | {row['budget']:g}s | "
                     f"{row['games']} | {row['points']} | {row['w']}/{row['d']}/{row['l']} | "
                     f"{attack} | {defend} | {marks} |")
    return "\n".join(lines) + "\n"


def totals(data: dict) -> dict[str, Counter]:
    by_config = defaultdict(Counter)
    for row in data["summary"]:
        bucket = by_config[row["config"]]
        bucket["games"] += row["games"]
        bucket["points"] += row["points"]
        bucket["w"] += row["w"]
        bucket["d"] += row["d"]
        bucket["l"] += row["l"]
        bucket["attack points"] += row["asAttacker"]["points"]
        bucket["attack games"] += row["asAttacker"]["games"]
        bucket["defense points"] += row["asDefender"]["points"]
        bucket["defense games"] += row["asDefender"]["games"]
        for name, count in row["faults"].items():
            bucket[f"fault: {name}"] += count
    return dict(by_config)


def main(argv=None) -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="journal name or path, e.g. m1-blind-base-60s")
    parser.add_argument("--out", type=Path, help="write the markdown table here")
    parser.add_argument("--json", type=Path, help="write the raw summary here")
    args = parser.parse_args(argv)
    paths = []
    for token in args.runs:
        path = Path(token)
        if not path.is_file():
            path = runner.RESULTS / f"{token}.jsonl"
        if not path.is_file():
            raise SystemExit(f"no journal at {path}")
        paths.append(path)
    data = summarise(paths)
    table = to_markdown(data)
    print(table)
    print(f"{data['rows']} games")
    for name, bucket in totals(data).items():
        print(f"{name}: {bucket['points']}/{2 * bucket['games']} points "
              f"({bucket['w']}/{bucket['d']}/{bucket['l']}), "
              f"attack {bucket['attack points']}/{2 * bucket['attack games']}, "
              f"defense {bucket['defense points']}/{2 * bucket['defense games']}")
    if args.out:
        args.out.write_text("# 盲评汇总\n\n" + table, encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
