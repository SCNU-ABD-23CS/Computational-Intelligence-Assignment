"""One move lookahead with captures and king progress."""

from time import perf_counter

import Team19_engine as engine

from .common import ArenaPlayer, king_edge, simple_score, Telemetry


class Player(ArenaPlayer):
    def choose(self, root, attacker, moves, deadline):
        side = "A" if attacker else "D"
        ahead = simple_score(root.board) * (1 if attacker else -1) > 0
        best_move, best_value = None, None
        visited = 0
        for move in moves:
            if best_move is not None and perf_counter() >= deadline:
                break
            child = engine._apply_legal_move(root, move)
            visited += 1
            if child.winner() == side:
                best_move = move
                break
            captures = ((root.defenders - child.defenders) if attacker else
                        (root.attackers - child.attackers))
            repeated = (child.defender_hash if attacker else child.attacker_hash) in self.history
            if attacker:
                rr, cc = move[1]
                r, c = child.king
                progress = -(abs(rr - r) + abs(cc - c))
            else:
                progress = -king_edge(child.king)
            value = (not (ahead and repeated), captures, progress)
            if best_value is None or value > best_value:
                best_move, best_value = move, value
        self.last_search = Telemetry(depth=1, nodes=visited)
        return best_move
