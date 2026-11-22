"""Players that inject member 5's candidate weights without editing Team19 code."""

from __future__ import annotations

from copy import deepcopy

import Team19
import Team19_engine as engine
import Team19_eval as evaluation
import Team19_search as search

from .bots.common import ArenaPlayer, Telemetry


FUNCTIONS = search.SearchFunctions(
    engine.get_legal_moves,
    engine.apply_move,
    engine.check_winner,
    evaluation.evaluate_board,
    engine.compute_hash,
)


class FixedDepthWeightPlayer(ArenaPlayer):
    """A fixed-depth Alpha-Beta player used for controlled weight experiments."""

    def __init__(self, weights, seed: int = 0, depth: int = 2,
                 tactical_ordering: bool = True):
        if type(depth) is not int or depth < 1:
            raise ValueError("Experiment depth must be a positive integer")
        self.weights = evaluation.normalize_weights(weights)
        self.depth = depth
        self.tactical_ordering = bool(tactical_ordering)
        super().__init__(seed=seed)

    def choose(self, root, attacker, moves, deadline):
        try:
            result = search.search_fixed_depth(
                root.board,
                attacker,
                self.depth,
                functions=FUNCTIONS,
                weights=self.weights,
                deadline=deadline,
                game_history=self.history,
                tactical_ordering=self.tactical_ordering,
            )
        except search.SearchTimeout as error:
            self.last_search = Telemetry(
                depth=self.depth,
                nodes=error.nodes_visited,
                fallback=True,
            )
            return None
        self.last_search = Telemetry(
            depth=self.depth,
            nodes=result.nodes_visited,
            score=result.score,
        )
        return result.move


class IterativeWeightPlayer(Team19.Player):
    """The production Team19 player with an explicitly supplied weight vector."""

    def __init__(self, weights, seed: int = 0):
        # Team19 loads its normal modules and clock once. The seed is accepted
        # so all experiment players share one factory interface.
        super().__init__()
        self.weights = evaluation.normalize_weights(weights)


def make_weight_player(weights, seed: int, depth: int, mode: str,
                       tactical_ordering: bool = True):
    """Build a candidate or default-weight player for one official game."""
    if mode == "fixed":
        return FixedDepthWeightPlayer(
            weights, seed=seed, depth=depth, tactical_ordering=tactical_ordering,
        )
    if mode == "iterative":
        return IterativeWeightPlayer(weights, seed=seed)
    raise ValueError("Mode must be 'fixed' or 'iterative'")


def prepare_player_for_opening(player, starting, is_attacker: bool) -> None:
    """Restore the same real-game history and adapter state as arena.runner."""
    if hasattr(player, "seed_history"):
        player.seed_history(starting.referee.past_positions)
    elif hasattr(player, "history"):
        player.history.update(starting.referee.past_positions)

    if isinstance(player, Team19.Player) and starting.turns:
        player._last_board = deepcopy(starting.last_board[is_attacker])
        player._last_is_attacker = is_attacker
        if not is_attacker:
            player._opening_replies.add(tuple(
                tuple(row) for row in starting.positions[1]["board"]
            ))
