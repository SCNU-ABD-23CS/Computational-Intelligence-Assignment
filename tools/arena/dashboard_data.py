"""Read the arena's existing reports without loading the game or running matches."""

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
from math import isclose, isfinite
from pathlib import Path
import re


STAGES = {"official": "60 秒官方原味赛", "fast": "10 秒快棋全循环", "final": "60 秒强敌对照"}
FORFEITS = ("timeout", "illegal move", "exception")
STABILITY_FIELDS = (*FORFEITS, "fallbacks", "internalErrors")
TABLES = ("standings", "roles", "matrix", "stability")
MATRIX_CELL = re.compile(r"(\d+)/(\d+)/(\d+) \(([+-]\d+)\)")


@dataclass(frozen=True)
class Score:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @classmethod
    def from_points(cls, counts):
        return cls(counts[2], counts[1], counts[0])

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> int:
        return 2 * self.wins + self.draws

    @property
    def point_rate(self) -> float:
        return self.points / (2 * self.games) if self.games else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def record(self) -> str:
        return f"{self.wins}–{self.draws}–{self.losses}"


@dataclass(frozen=True)
class Team:
    name: str
    rank: int
    score: Score
    elo: float
    marks: float | None


@dataclass
class Stage:
    name: str
    games: int
    scheduled: int
    plies: int
    teams: list[Team]
    roles: dict[tuple[str, str], Score]
    pair_roles: dict[tuple[str, str, str], Score]
    matrix: dict[tuple[str, str], Score]
    reasons: Counter
    stability: dict[str, Counter]


def _read_table(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    result = {row["team"]: row for row in rows}
    if not result or len(result) != len(rows) or "" in result:
        raise ValueError(f"Expected unique teams in {path.name}")
    return result


def _verify_score(row, score: Score, prefix="") -> None:
    if int(row[f"{prefix}games"]) != score.games or int(row[f"{prefix}points"]) != score.points:
        raise ValueError("CSV game or point totals differ from JSONL")
    value = row[f"{prefix}pointRate"]
    if (score.games or value) and not isclose(float(value), round(score.point_rate, 4), abs_tol=1e-9):
        raise ValueError("CSV point rate differs from JSONL")


def _load_stage(root: Path, name: str) -> Stage:
    tables = {table: _read_table(root / name / f"{table}.csv") for table in TABLES}
    ids, schedules, schedule_sizes = set(), set(), set()
    plies = 0
    reasons = Counter()
    totals, roles, pairs, pair_roles, stability = (defaultdict(Counter) for _ in range(5))
    with (root / f"{name}.jsonl").open(encoding="utf-8") as source:
        for line in source:
            game = json.loads(line)
            uid, winner, reason = game["id"], game["winner"], game["reason"]
            seats = game["teams"]
            if (uid in ids or game["stage"] != name or winner not in (None, "A", "D")
                    or seats["A"] == seats["D"]):
                raise ValueError(f"Invalid game record {uid}")
            ids.add(uid)
            schedules.add(game["schedule"])
            schedule_sizes.add(game["scheduledGames"])
            reasons[reason] += 1
            plies += len(game["prefix"])
            for attempt in game["attempts"]:
                team = seats[attempt["side"]]
                plies += attempt.get("event") is not None
                stability[team]["fallbacks"] += bool(attempt.get("fallback"))
                stability[team]["internalErrors"] += "error" in attempt
            for side, other in (("A", "D"), ("D", "A")):
                team, opponent = seats[side], seats[other]
                points = 1 if winner is None else 2 * (winner == side)
                totals[team][points] += 1
                roles[team, side][points] += 1
                pairs[team, opponent][points] += 1
                pair_roles[team, opponent, side][points] += 1
                if reason in FORFEITS and points == 0:
                    stability[team][reason] += 1

    if (not ids or len(schedules) != 1 or len(schedule_sizes) != 1 or any(
            table.keys() != totals.keys() for table in tables.values())):
        raise ValueError("The CSV tables and JSONL must describe one nonempty schedule with the same teams")
    scheduled = schedule_sizes.pop()
    if scheduled < len(ids):
        raise ValueError("More recorded games than scheduled games")

    teams = []
    standings = tables["standings"]
    for team, row in standings.items():
        score = Score.from_points(totals[team])
        _verify_score(row, score)
        if (int(row["wins"]), int(row["draws"]), int(row["losses"])) != (
            score.wins, score.draws, score.losses
        ):
            raise ValueError(f"Win, draw and loss totals differ for {team}")
        for side in ("A", "D"):
            side_score = Score.from_points(roles[team, side])
            _verify_score(tables["roles"][team], side_score, f"{side} ")
        for field in STABILITY_FIELDS:
            if int(tables["stability"][team][field]) != stability[team][field]:
                raise ValueError(f"Stability counts differ for {team}")
        marks = float(row["marks"]) if row["marks"] else None
        elo = float(row["BT Elo"])
        if not isfinite(elo) or (marks is not None and not isfinite(marks)):
            raise ValueError(f"Non-finite standing for {team}")
        teams.append(Team(team, int(row["rank"]), score, elo, marks))

    matrix = {}
    for team, row in tables["matrix"].items():
        for opponent in standings:
            entry = row[opponent]
            score = Score.from_points(pairs.get((team, opponent), Counter()))
            if not entry and not score.games:
                continue
            match = MATRIX_CELL.fullmatch(entry)
            if not match or tuple(map(int, match.groups())) != (
                score.wins, score.draws, score.losses, 2 * (score.wins - score.losses)
            ):
                raise ValueError(f"Head-to-head counts differ for {team} and {opponent}")
            matrix[team, opponent] = score

    return Stage(name, len(ids), scheduled, plies, sorted(teams, key=lambda team: (team.rank, team.name)),
                 {key: Score.from_points(value) for key, value in roles.items()},
                 {key: Score.from_points(value) for key, value in pair_roles.items()},
                 matrix, reasons, dict(stability))


def load_stages(root: Path) -> dict[str, Stage]:
    stages = {}
    for name in STAGES:
        path = root / f"{name}.jsonl"
        if not path.is_file() or not all((root / name / f"{table}.csv").is_file() for table in TABLES):
            continue
        try:
            stages[name] = _load_stage(root, name)
        except (KeyError, TypeError, ValueError, csv.Error) as error:
            raise ValueError(f"{name}: {error}") from error
    return stages
