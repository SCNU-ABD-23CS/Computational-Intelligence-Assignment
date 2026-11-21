"""Fixed two-ply alpha-beta with a material and king-distance estimate."""

import Team19_engine as engine
import Team19_search as search

from .common import ArenaPlayer, simple_score, Telemetry


FUNCTIONS = search.SearchFunctions(engine.get_legal_moves, engine.apply_move, engine.check_winner,
                                   simple_score, engine.compute_hash)


class Player(ArenaPlayer):
    def choose(self, root, attacker, moves, deadline):
        best = None
        nodes = 0
        for depth in (1, 2):
            try:
                result = search.search_fixed_depth(root.board, attacker, depth, functions=FUNCTIONS,
                                                   deadline=deadline, game_history=self.history)
            except search.SearchTimeout as error:
                self.last_search.nodes = nodes + error.nodes_visited
                break
            nodes += result.nodes_visited
            best = result.move
            self.last_search = Telemetry(depth=depth, nodes=nodes, score=result.score)
            if result.score == (search.WIN_SCORE if attacker else -search.WIN_SCORE):
                break
        return best
