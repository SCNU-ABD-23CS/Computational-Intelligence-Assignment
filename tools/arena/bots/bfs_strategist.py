"""Iterative alpha-beta with a geometric king escape distance."""

import Team19_engine as engine
import Team19_search as search

from .common import ArenaPlayer, Telemetry
from .tactics import EDGE, RAYS


FLAT_RAYS = tuple(tuple(tuple(9 * r + c for r, c in ray) for ray in rays) for rays in RAYS)
NEIGHBOURS = tuple(tuple(ray[0] for ray in rays if ray) for rays in FLAT_RAYS)
KING_RING = tuple(tuple(
    (9 * rr + cc, 2 if abs(r - rr) + abs(c - cc) == 1 else 1)
    for rr in range(max(0, r - 2), min(9, r + 3))
    for cc in range(max(0, c - 2), min(9, c + 3))
    if 0 < abs(r - rr) + abs(c - cc) <= 2
) for r in range(9) for c in range(9))


def escape_distance(board: str, king: int) -> tuple[int | None, int]:
    """Shortest rook path to the edge with other pieces held in place."""
    if EDGE[king]:
        return 0, 0
    queue = [(king, 0)]
    seen = {king}
    mobility = 0
    for square, depth in queue:
        escape = False
        for ray in FLAT_RAYS[square]:
            for target in ray:
                if target == king or board[target] == "X":
                    continue
                if board[target] != ".":
                    break
                mobility += depth == 0
                if EDGE[target]:
                    escape = True
                elif target not in seen:
                    seen.add(target)
                    queue.append((target, depth + 1))
        if escape:
            return depth + 1, mobility
    return None, mobility


def evaluate_state(state: engine._SearchPosition) -> float:
    flat = "".join(map("".join, state.board))
    row, col = state.king
    king = 9 * row + col
    distance, mobility = escape_distance(flat, king)
    escape = 5000 if distance is None else (-7200, -3700, -1850, -780, 150)[min(distance, 4)]
    attack_ring = sum(value for square, value in KING_RING[king] if flat[square] == "A")
    guards = sum(1 for square, cell in enumerate(flat)
                 if cell == "D" and any(flat[n] in (".", "X") for n in NEIGHBOURS[square]))
    return (600 * state.attackers - 1050 * state.defenders + escape
            + 280 * attack_ring - 120 * mobility - 110 * guards)


def _evaluate(board, weights=None):
    return evaluate_state(engine._search_position(board))


FUNCTIONS = search.SearchFunctions(engine.get_legal_moves, engine.apply_move, engine.check_winner,
                                   _evaluate, engine.compute_hash)


class Player(ArenaPlayer):
    def choose(self, root, attacker, moves, deadline):
        fast = search._FastSearch(root, engine._apply_legal_move, engine._has_legal_move, evaluate_state)
        ordering = search._MoveOrdering()
        best = None
        nodes = 0
        try:
            ordered, children = search._order_legal_moves(root, moves, attacker, fast, deadline)
            root_moves = ordered
            for depth in range(1, 65):
                result, horizon = search._search_depth(
                    root.board, attacker, depth, FUNCTIONS, None, deadline, self.history,
                    playable_root_moves=root_moves, tactical_ordering=True, fast=fast,
                    root_children=children, ordering=ordering,
                )
                nodes += result.nodes_visited
                best = result.move
                self.last_search = Telemetry(depth=depth, nodes=nodes, score=result.score)
                if (result.score == (search.WIN_SCORE if attacker else -search.WIN_SCORE)
                        or not horizon):
                    break
                root_moves = [best, *(move for move in ordered if move != best)]
        except search.SearchTimeout as error:
            self.last_search.nodes = nodes + error.nodes_visited
        return best
