"""Fixture identity shared by the runner and its offline dashboard."""

from dataclasses import dataclass
from itertools import combinations


def fixture_id(budget, opening, attacker, defender):
    return f"{budget:g}-{opening}-{attacker}-A-{defender}"


@dataclass(frozen=True)
class Schedule:
    name: str
    teams: tuple[str, ...]
    budget: float
    openings: tuple[str, ...]
    anchor: str | None = None
    standard_opening: bool = False

    def pairs(self):
        if self.anchor is None:
            return combinations(self.teams, 2)
        return ((self.anchor, team) for team in self.teams if team != self.anchor)

    @property
    def games_per_pair(self):
        return 2 * len(self.openings)

    @property
    def games(self):
        return self.games_per_pair * sum(1 for _ in self.pairs())

    @property
    def format_label(self):
        return f"1v{len(self.teams) - 1} 车轮战" if self.anchor else f"{len(self.teams)} 队全循环"

    @property
    def opening_label(self):
        if self.standard_opening:
            return "标准初始盘面 × 换边（预走 0 手）"
        return f"前 {len(self.openings)} 个固定开局 × 换边"

    @property
    def title(self):
        return f"{self.budget:g} 秒 {self.format_label}"

    def fixture_ids(self):
        return {fixture_id(self.budget, opening, attacker, defender)
                for first, second in self.pairs() for opening in self.openings
                for attacker, defender in ((first, second), (second, first))}
