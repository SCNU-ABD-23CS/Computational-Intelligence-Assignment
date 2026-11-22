"""Read complete records from the two top-six JSONL journals."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path

from . import HERE, TOP6_BUDGETS, TOP6_GAMES, TOP6_OPENINGS, TOP6_TEAMS
from .report import FORFEITS, elo, points


STAGES = {name: f"{budget} 秒 6 强全循环" for name, budget in TOP6_BUDGETS.items()}


@dataclass(frozen=True)
class Score:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @classmethod
    def from_counts(cls, counts):
        return cls(counts[2], counts[1], counts[0])

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> int:
        return 2 * self.wins + self.draws

    @property
    def point_rate(self) -> Fraction:
        return Fraction(self.points, 2 * self.games) if self.games else Fraction(0)

    @property
    def record(self) -> str:
        return f"{self.wins}–{self.draws}–{self.losses}"


@dataclass(frozen=True)
class Team:
    name: str
    rank: int | None
    score: Score
    elo: float | None
    marks: float | None


@dataclass
class Stage:
    name: str
    budget: int
    games: int
    plies: int
    reused: int
    teams: list[Team]
    roles: dict[tuple[str, str], Score]
    matrix: dict[tuple[str, str], Score]
    reasons: Counter
    stability: dict[str, Counter]

    @property
    def complete(self) -> bool:
        return self.games == TOP6_GAMES


def _load_records(path: Path, name: str, budget: int, openings: set[str]) -> list[dict]:
    if not path.is_file():
        return []
    records, ids, schedules = [], set(), set()
    with path.open("rb") as source:
        for line in source:
            if not line.endswith(b"\n"):
                # The running tournament may be appending its final line.
                break
            row = json.loads(line)
            attack, defence = row["teams"]["A"], row["teams"]["D"]
            opening = row["opening"]
            expected = f"{budget}-{opening}-{attack}-A-{defence}"
            if (row["stage"] != name or row["budget"] != budget or row["scheduledGames"] != TOP6_GAMES
                    or attack not in TOP6_TEAMS or defence not in TOP6_TEAMS or attack == defence
                    or opening not in openings or row["winner"] not in ("A", "D", None)
                    or row["id"] != expected or expected in ids):
                raise ValueError(f"Invalid or duplicate fixture {row['id']}")
            ids.add(expected)
            schedules.add(row["schedule"])
            records.append(row)
    if len(schedules) > 1:
        raise ValueError("Mixed schedules")
    return records


def _standings(records, totals):
    ratings = elo(records, TOP6_TEAMS) if records else {}
    names = sorted(TOP6_TEAMS, key=lambda team: (
        not totals[team].games, -totals[team].point_rate, team,
    ))
    complete = len(records) == TOP6_GAMES
    top = totals[names[0]].points if complete else None
    teams = []
    rank, previous_rate = None, None
    for index, name in enumerate(names, 1):
        score = totals[name]
        if score.games and score.point_rate != previous_rate:
            rank = index
            previous_rate = score.point_rate
        marks = 30 * score.points / top if top else None
        teams.append(Team(name, rank if score.games else None, score,
                          ratings.get(name) if score.games else None, marks))
    return teams


def _summarize(name: str, budget: int, records: list[dict]) -> Stage:
    totals, roles, matrix = (defaultdict(Counter) for _ in range(3))
    reasons = Counter()
    stability = {team: Counter() for team in TOP6_TEAMS}
    plies = 0
    for row in records:
        reasons[row["reason"]] += 1
        plies += len(row["prefix"])
        seats = row["teams"]
        for turn in row["attempts"]:
            team = seats[turn["side"]]
            plies += turn.get("event") is not None
            stability[team]["fallbacks"] += bool(turn.get("fallback"))
            stability[team]["internalErrors"] += "error" in turn
        for side, other in (("A", "D"), ("D", "A")):
            team, rival = seats[side], seats[other]
            score = points(row["winner"], side)
            totals[team][score] += 1
            roles[team, side][score] += 1
            matrix[team, rival][score] += 1
            if row["reason"] in FORFEITS and score == 0:
                stability[team][row["reason"]] += 1

    team_scores = {team: Score.from_counts(totals[team]) for team in TOP6_TEAMS}
    return Stage(name, budget, len(records), plies, sum("reusedFrom" in row for row in records),
                 _standings(records, team_scores),
                 {key: Score.from_counts(counts) for key, counts in roles.items()},
                 {key: Score.from_counts(counts) for key, counts in matrix.items()}, reasons, stability)


def load_stages(root: Path) -> dict[str, Stage]:
    data = json.loads((HERE / "openings.json").read_text(encoding="utf-8"))
    openings = {entry["name"] for entry in data["openings"][:TOP6_OPENINGS]}
    stages = {}
    for name, budget in TOP6_BUDGETS.items():
        path = root / f"{name}.jsonl"
        try:
            stages[name] = _summarize(name, budget, _load_records(path, name, budget, openings))
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"{path.name}: {error}") from error
    return stages
