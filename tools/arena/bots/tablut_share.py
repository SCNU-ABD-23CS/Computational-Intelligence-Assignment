"""Adapt the locally installed Tablut share bundle to the Arena interface."""

import Tablut

from ..local_opponents.tablut_share.TeamXX import Player as SharePlayer
from ..local_opponents.tablut_share.config.settings import GAME_TIME_LIMIT, TIME_SAFETY_MARGIN
from ..local_opponents.tablut_share.src.game import get_state_key, parse_board
from ..local_opponents.tablut_share.src.time_control import AdaptiveTimeManager
from .common import Telemetry


class Player(SharePlayer):
    def __init__(self, seed=0):
        super().__init__(record_search=True)
        self.rng.seed(seed)
        budget = float(Tablut.TIME_BUDGET)
        margin = TIME_SAFETY_MARGIN * min(1.0, budget / GAME_TIME_LIMIT)
        self.clock = AdaptiveTimeManager(limit=budget, margin=margin)
        self.last_search = None

    @property
    def remaining(self):
        return max(0.0, self.clock.limit - self.clock.spent)

    def seed_positions(self, positions, is_attacker):
        # The first attacker callback is on the final setup board. For the
        # defender, that board is already in the past when their turn arrives.
        past = positions[:-1] if is_attacker else positions
        self.history = [
            get_state_key(parse_board(position["board"]), position["sideToMove"])
            for position in past
        ]

    def _play(self, board, side):
        self.last_search = None
        move = super()._play(board, side)
        stats = self.last_search_stats
        depth, score = stats["completed_depth"], stats["score"]
        self.last_search = Telemetry(
            depth=depth,
            nodes=stats["nodes"],
            fallback=depth == 0 and stats["fallback_used"],
            score=-score if side == "D" and score is not None else score,
        )
        return move
