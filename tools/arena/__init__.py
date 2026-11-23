"""Offline JC4004 tournaments and replay diagnostics."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TOP6_BUDGETS = {"top6_60s": 60, "top6_10s": 10}
TOP6_TEAMS = (
    "Team19-v2", "Grandmaster-Numba", "QSearch-PVS",
    "Team19-91ffcc8", "BFS-Strategist", "Guided-MCTS",
)
TOP6_OPENINGS = 6
TOP6_PAIR_GAMES = 2 * TOP6_OPENINGS
TOP6_GAMES = len(TOP6_TEAMS) * (len(TOP6_TEAMS) - 1) * TOP6_OPENINGS
