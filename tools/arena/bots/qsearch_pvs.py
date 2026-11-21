"""Principal variation search with a four-ply tactical horizon."""

from math import inf, nextafter

import Team19
import Team19_engine as engine
import Team19_eval as evaluation
import Team19_search as search

from .common import ArenaPlayer, MATE, Telemetry
from .tactics import poll, winning_move


QUIESCENCE_PLIES = 4


class PrincipalSearch:
    def __init__(self, evaluate, history, deadline):
        self.evaluate = evaluate
        self.history = history
        self.deadline = deadline
        self.path = set()
        self.ordering = search._MoveOrdering()
        self.nodes = 0

    def visit(self, state, attacker, depth, qleft, alpha, beta, ply=0, root_hint=None):
        poll(self.deadline)
        self.nodes += 1
        sign = 1 if attacker else -1
        winner = state.winner()
        if winner is not None:
            return sign * (MATE - ply if winner == "A" else -MATE + ply), None
        if not engine._has_legal_move(state.board, attacker):
            return 0.0, None
        position_hash = state.attacker_hash if attacker else state.defender_hash
        if ply and (position_hash in self.path or position_hash in self.history):
            return 0.0, None
        if depth == 0 and qleft == 0:
            return sign * self.evaluate(state), None
        if move := winning_move(state, attacker):
            return MATE - ply - 1, move

        forcing = depth == 0
        threatened = forcing and winning_move(state, not attacker) is not None
        best_score = -inf
        if forcing and not threatened:
            best_score = sign * self.evaluate(state)
            if best_score >= beta:
                return best_score, None
            alpha = max(alpha, best_score)

        self.path.add(position_hash)
        try:
            moves = engine.get_legal_moves(state.board, attacker)
            scope = search._FastSearch(state, engine._apply_legal_move, engine._has_legal_move,
                                       self.evaluate)
            if forcing and not threatened:
                targets = search._capture_targets(state.board, attacker, self.deadline)
                row, col = state.king
                moves = [move for move in moves if
                         tuple(move[1]) in targets or
                         (attacker and abs(move[1][0] - row) + abs(move[1][1] - col) == 1)]
            if root_hint is not None and root_hint in moves:
                self.ordering.record_hint(position_hash, root_hint)
            candidates = search._ordered_children(
                state, moves, attacker, position_hash, scope, self.deadline,
                self.ordering if depth else None, ply,
            )
            best_move = None
            first = True
            next_depth = max(0, depth - 1)
            next_qleft = qleft - forcing
            for move, child in candidates:
                poll(self.deadline)
                child = engine._apply_legal_move(state, move) if child is None else child
                quiet = (child.attackers == state.attackers and child.defenders == state.defenders
                         and child.king is not None)
                if forcing and not threatened and quiet:
                    if not attacker or winning_move(child, True) is None:
                        continue
                if first:
                    score = -self.visit(child, not attacker, next_depth, next_qleft,
                                        -beta, -alpha, ply + 1)[0]
                    first = False
                else:
                    score = -self.visit(child, not attacker, next_depth, next_qleft,
                                        -nextafter(alpha, inf), -alpha, ply + 1)[0]
                    if alpha < score < beta:
                        score = -self.visit(child, not attacker, next_depth, next_qleft,
                                            -beta, -alpha, ply + 1)[0]
                if score > best_score:
                    best_score, best_move = score, move
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    if depth and quiet:
                        self.ordering.record_cutoff(move, attacker, ply, depth)
                    break
            if depth and best_move is not None:
                self.ordering.record_hint(position_hash, best_move)
            return best_score, best_move
        finally:
            self.path.remove(position_hash)


class Player(ArenaPlayer):
    def __init__(self, seed=0):
        super().__init__(seed)
        self.weights = tuple(Team19._load_weights() or evaluation.DEFAULT_WEIGHTS)
        self.functions = search._load_functions()

    def evaluate(self, state):
        return evaluation._weighted_score(self.weights, evaluation._features_from_counts(
            state.board, state.attackers, state.defenders, state.king,
        ))

    def choose(self, root, attacker, moves, deadline):
        if move := winning_move(root, attacker):
            self.last_search = Telemetry(depth=1, nodes=1, score=MATE if attacker else -MATE)
            return move
        best = None
        quick_nodes = 0
        pvs = PrincipalSearch(self.evaluate, self.history, deadline)
        fast = search._FastSearch(root, engine._apply_legal_move, engine._has_legal_move, self.evaluate)
        try:
            quick, _ = search._search_depth(
                root.board, attacker, 1, self.functions, None, deadline, self.history,
                playable_root_moves=moves, fast=fast,
            )
            best = quick.move
            quick_nodes = quick.nodes_visited
            self.last_search = Telemetry(depth=1, nodes=quick_nodes, score=quick.score)
            for depth in range(1, 65):
                score, move = pvs.visit(root, attacker, depth, QUIESCENCE_PLIES, -inf, inf,
                                        root_hint=best)
                if move is not None:
                    best = move
                self.last_search = Telemetry(depth=depth, nodes=quick_nodes + pvs.nodes,
                                             score=score if attacker else -score)
                if abs(score) >= MATE - depth:
                    break
        except search.SearchTimeout as error:
            self.last_search.nodes = quick_nodes + pvs.nodes + error.nodes_visited
        return best
