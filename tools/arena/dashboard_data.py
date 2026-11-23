"""Read complete records from saved tournament journals."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path

from . import HERE, TOP6_BUDGETS, TOP6_OPENINGS, TOP6_TEAMS
from .report import FORFEITS, elo, points
from .schedule import Schedule, fixture_id


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
    schedule: Schedule
    games: int
    plies: int
    reused: int
    teams: list[Team]
    roles: dict[tuple[str, str], Score]
    matrix: dict[tuple[str, str], Score]
    reasons: Counter
    stability: dict[str, Counter]

    @property
    def name(self) -> str:
        return self.schedule.name

    @property
    def budget(self) -> float:
        return self.schedule.budget

    @property
    def title(self) -> str:
        return STAGES.get(self.name, self.schedule.title)

    @property
    def complete(self) -> bool:
        return self.games == self.schedule.games


def _load_records(path: Path, schedule: Schedule) -> list[dict]:
    if not path.is_file():
        return []
    records, ids, schedules = [], set(), set()
    fixtures = schedule.fixture_ids()
    total = schedule.games
    with path.open("rb") as source:
        for line in source:
            if not line.endswith(b"\n"):
                # The running tournament may be appending its final line.
                break
            row = json.loads(line)
            attack, defence = row["teams"]["A"], row["teams"]["D"]
            opening = row["opening"]
            expected = fixture_id(schedule.budget, opening, attack, defence)
            if (row["stage"] != schedule.name or row["budget"] != schedule.budget
                    or row["scheduledGames"] != total or row["winner"] not in ("A", "D", None)
                    or expected not in fixtures or row["id"] != expected or expected in ids):
                raise ValueError(f"Invalid or duplicate fixture {row['id']}")
            ids.add(expected)
            schedules.add(row["schedule"])
            records.append(row)
    if len(schedules) > 1:
        raise ValueError("Mixed schedules")
    return records


def _standings(schedule, records, totals):
    ratings = elo(records, schedule.teams) if records else {}
    names = sorted(schedule.teams, key=lambda team: (
        not totals[team].games, -totals[team].point_rate, team,
    ))
    top = totals[names[0]].points if not schedule.anchor and len(records) == schedule.games else None
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


def _summarize(schedule: Schedule, records: list[dict]) -> Stage:
    totals, roles, matrix = (defaultdict(Counter) for _ in range(3))
    reasons = Counter()
    stability = {team: Counter() for team in schedule.teams}
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

    team_scores = {team: Score.from_counts(totals[team]) for team in schedule.teams}
    return Stage(schedule, len(records), plies, sum("reusedFrom" in row for row in records),
                 _standings(schedule, records, team_scores),
                 {key: Score.from_counts(counts) for key, counts in roles.items()},
                 {key: Score.from_counts(counts) for key, counts in matrix.items()}, reasons, stability)


def load_stages(root: Path, schedules: tuple[Schedule, ...] | None = None) -> dict[str, Stage]:
    if schedules is None:
        data = json.loads((HERE / "openings.json").read_text(encoding="utf-8"))
        openings = tuple(entry["name"] for entry in data["openings"][:TOP6_OPENINGS])
        schedules = tuple(Schedule(name, TOP6_TEAMS, budget, openings)
                          for name, budget in TOP6_BUDGETS.items())
    stages = {}
    for schedule in schedules:
        path = root / f"{schedule.name}.jsonl"
        try:
            stages[schedule.name] = _summarize(schedule, _load_records(path, schedule))
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"{path.name}: {error}") from error
    return stages
