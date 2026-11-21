"""Uniform seeded legal moves for the smoke test."""

from .common import ArenaPlayer


class Player(ArenaPlayer):
    def choose(self, root, attacker, moves, deadline):
        return self.rng.choice(moves)
