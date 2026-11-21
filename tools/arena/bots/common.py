"""Shared clock and history for arena entrants using the Python engine."""

from __future__ import annotations

from dataclasses import dataclass
import random
from time import perf_counter

import Tablut
import Team19_engine as engine


Board = list[list[str]]
Move = list[list[int]]
MATE = 100_000


@dataclass
class Telemetry:
    depth: int | None = None
    nodes: int | None = None
    simulations: int | None = None
    fallback: bool = False
    score: float | None = None


class ArenaPlayer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.initial_hash = Tablut.Tablut().compute_zobrist_hash(True)
        self.reset_game()

    def reset_game(self) -> None:
        self.remaining = float(Tablut.TIME_BUDGET)
        self.history = {self.initial_hash}
        self.last_search = Telemetry()

    def seed_history(self, history: list[int]) -> None:
        self.history.update(history)

    def play_attacker(self, board: Board) -> Move:
        return self._play(board, True)

    def play_defender(self, board: Board) -> Move:
        return self._play(board, False)

    def _play(self, board: Board, attacker: bool) -> Move:
        start = perf_counter()
        try:
            root = engine._search_position(board)
            self.last_search = Telemetry()
            self.history.add(root.attacker_hash if attacker else root.defender_hash)
            moves = engine.get_legal_moves(board, attacker)
            if not moves:
                raise ValueError("No move on an ongoing board")
            available = max(0.0, self.remaining - 0.025)
            hand_count = max(12, int(1.55 * (root.attackers + root.defenders)))
            allowance = min(available / 10, max(available / hand_count, min(0.008, available)))
            deadline = start + allowance - min(0.006, allowance * 0.08)
            move = self.choose(root, attacker, moves, deadline)
            if move is None:
                self.last_search.fallback = True
                move = moves[0]
            child = engine._apply_legal_move(root, move)
            self.history.add(child.defender_hash if attacker else child.attacker_hash)
            return move
        finally:
            self.remaining -= perf_counter() - start

    def choose(self, root: engine._SearchPosition, attacker: bool, moves: list[Move],
               deadline: float) -> Move | None:
        raise NotImplementedError


def king_edge(king: tuple[int, int]) -> int:
    row, col = king
    return min(row, col, 8 - row, 8 - col)


def simple_score(board: Board, weights: list[float] | None = None) -> float:
    attackers = sum(row.count("A") for row in board)
    guards = sum(row.count("D") for row in board)
    king = next(((r, row.index("K")) for r, row in enumerate(board) if "K" in row), (4, 4))
    return 100 * (attackers - 2 * guards) + 80 * king_edge(king)


def key(move: Move) -> tuple[int, int, int, int]:
    (row, col), (rr, cc) = move
    return row, col, rr, cc
