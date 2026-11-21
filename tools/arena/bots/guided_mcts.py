"""UCT with tactical pruning and a bounded heuristic at each new leaf."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter

import Team19_engine as engine
import Team19_eval as evaluation
from Team19_search import SearchTimeout, _tactical_order, _load_functions

from .common import ArenaPlayer, Move, Telemetry
from .tactics import poll, winning_move


@dataclass(slots=True)
class Node:
    state: engine._SearchPosition
    attacker: bool
    move: Move | None = None
    visits: int = 0
    total: float = 0.0
    prior: float = 0.0
    terminal: float | None = None
    pending: list[Move] | None = None
    children: list[Node] = field(default_factory=list)

    def score(self, sign: int, log_parent: float) -> float:
        return (sign * self.total / self.visits
                + 0.95 * math.sqrt(log_parent / self.visits)
                + 0.30 * sign * self.prior / (1 + self.visits))


class Player(ArenaPlayer):
    def __init__(self, seed=0):
        super().__init__(seed)
        self.weights = tuple(evaluation.normalize_weights(evaluation.DEFAULT_WEIGHTS))
        self.functions = _load_functions()

    def _prior(self, state):
        return math.tanh(evaluation._weighted_score(
            self.weights, evaluation._features_from_counts(
                state.board, state.attackers, state.defenders, state.king,
            ),
        ) / 6500)

    def _evaluate(self, node, ancestors):
        state = node.state
        winner = state.winner()
        position_hash = state.attacker_hash if node.attacker else state.defender_hash
        if winner is not None:
            node.terminal = 1.0 if winner == "A" else -1.0
        elif (not engine._has_legal_move(state.board, node.attacker)
              or position_hash in ancestors or position_hash in self.history):
            node.terminal = 0.0
        elif winning_move(state, node.attacker) is not None:
            node.terminal = 1.0 if node.attacker else -1.0
        node.prior = self._prior(state) if node.terminal is None else node.terminal
        return node.prior

    def choose(self, root, attacker, moves, deadline):
        if move := winning_move(root, attacker):
            self.last_search = Telemetry(depth=1, nodes=1, simulations=0)
            return move
        sign = 1 if attacker else -1
        root_hash = root.attacker_hash if attacker else root.defender_hash
        candidates = {}
        try:
            for move in moves:
                poll(deadline)
                state = engine._apply_legal_move(root, move)
                node = Node(state, not attacker, move)
                self._evaluate(node, {root_hash})
                candidates[tuple(map(tuple, move))] = node
        except SearchTimeout:
            pass
        if not candidates:
            return None
        ranked = sorted(candidates.values(), key=lambda node: sign * node.prior, reverse=True)
        # Prune an immediately losing continuation whenever a safe one is available.
        safe = [node for node in ranked if node.terminal != -sign]
        ranked = safe or ranked
        top = ranked[0].move
        tree = Node(root, attacker, pending=[node.move for node in reversed(ranked)])
        simulations = 0
        nodes = 1
        try:
            while perf_counter() < deadline:
                path = [tree]
                ancestors = {root_hash}
                node = tree
                while node.terminal is None:
                    poll(deadline)
                    if node.pending is None:
                        legal = engine.get_legal_moves(node.state.board, node.attacker)
                        ordered = _tactical_order(node.state.board, legal, node.attacker,
                                                  self.functions, deadline)
                        node.pending = list(reversed(ordered))
                    width = 1 + int(2 * math.sqrt(node.visits))
                    if node.pending and len(node.children) < width:
                        move = node.pending.pop()
                        if node is tree:
                            child = candidates[tuple(map(tuple, move))]
                        else:
                            state = engine._apply_legal_move(node.state, move)
                            child = Node(state, not node.attacker, move)
                            self._evaluate(child, ancestors)
                        node.children.append(child)
                        node = child
                        path.append(node)
                        nodes += 1
                        break
                    if not node.children:
                        break
                    side = 1 if node.attacker else -1
                    log_parent = math.log(max(2, node.visits))
                    node = max(node.children, key=lambda item: item.score(side, log_parent))
                    path.append(node)
                    ancestors.add(node.state.attacker_hash if node.attacker else node.state.defender_hash)
                value = node.prior
                for visited in path:
                    visited.visits += 1
                    visited.total += value
                simulations += 1
        except SearchTimeout:
            pass
        if tree.children:
            top = max(tree.children, key=lambda node: (node.visits, sign * node.total / node.visits)).move
        self.last_search = Telemetry(nodes=nodes, simulations=simulations)
        return top
