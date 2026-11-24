"""Score-table regressions and virtual-clock endurance for the two upgrades."""

from math import inf
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import Tablut
import Team19_engine as engine
import Team19_search as search
import Team19_tt as tt
from Team19_weights import native_weights

from . import runner
from .bots import grandmaster_numba as gm, grandmaster_plus as gmp
from .test_team19_v2 import _escape_threat


MATE = 100_000
WINDOW = 2 * MATE
EMPTY = np.empty(0, dtype=np.uint64)


class EntryTests(unittest.TestCase):
    def test_flags_accept_deeper_entries_but_require_the_same_search_mode(self):
        table = tt.allocate_table(8)
        tt.advance_generation(table)
        key = np.uint64(35)
        for flag, alpha, beta, score, cuts in (
                (tt.EXACT, 10, 20, 15, True),
                (tt.LOWER, 10, 20, 22, True),
                (tt.LOWER, 10, 20, 15, False),
                (tt.UPPER, 10, 20, 7, True),
                (tt.UPPER, 10, 20, 15, False),
                (tt.HINT, 10, 20, 15, False)):
            with self.subTest(flag=flag, score=score):
                tt.remember(table, key, 6, 3, 4, score, flag, 1)
                entry = tt.lookup(table, key)
                self.assertEqual(tt.hint_move(entry), 6)
                for depth in (3, 2):
                    self.assertEqual(tt.cutoff(entry, depth, 4, 1, alpha, beta)[0], cuts)
                for depth, mode in ((4, 4), (3, 0), (3, 12), (3, 20)):
                    self.assertFalse(tt.cutoff(entry, depth, mode, 1, alpha, beta)[0])
        self.assertEqual(tt.lookup(table, np.uint64(43)), 0)
        table.generation[1] = table.generation[0] + 1
        self.assertEqual(tt.lookup(table, key), 0)

    def test_generation_rollover_does_not_restore_stale_entries(self):
        table = tt.allocate_table(8)
        tt.advance_generation(table)
        key = np.uint64(35)
        tt.remember(table, key, 6, 3, 4, 30, tt.EXACT, 1)
        table.generation[0] = (1 << 16) - 1
        tt.advance_generation(table)
        self.assertEqual(tt.lookup(table, key), 0)
        tt.remember(table, key, 8, 3, 4, 30, tt.EXACT, 1)
        self.assertEqual(tt.hint_move(tt.lookup(table, key)), 8)

    def test_mates_keep_the_same_distance_from_the_cached_position(self):
        table = tt.allocate_table(8)
        tt.advance_generation(table)
        key = np.uint64(13)
        for sign in (1, -1):
            for flag in (tt.EXACT, tt.LOWER, tt.UPPER):
                with self.subTest(sign=sign, flag=flag):
                    tt.remember(table, key, 5, 8, 4, sign * (MATE - 9), flag, 5)
                    entry = tt.lookup(table, key)
                    value = sign * (MATE - 7)
                    alpha = value if flag == tt.UPPER else -WINDOW
                    beta = value if flag == tt.LOWER else WINDOW
                    self.assertEqual(tt.cutoff(entry, 6, 4, 3, alpha, beta), (True, value))


def _native_cases(board):
    flat, state = search._native_position(board)
    work = search._native_work(native_weights())
    draw = search._ATTACKER_REPETITION_SCORE

    def visit_current(depth, ply, played, *, qleft=0, deadline=inf, lmr=True, alpha=-WINDOW, beta=WINDOW):
        return search._native_visit(flat, True, *state, depth, qleft, ply, alpha, beta,
                                    deadline, played, work, lmr, draw, False)

    yield "v4", flat, state, work, 16, draw, search._REDUCTIONS, search._TT_CUTOFFS, visit_current

    gm_board = gmp._flat(board)
    gm_state = gm._root_state(gm_board)
    gm_work = gmp._work()

    def visit_gmp(depth, ply, played, *, qleft=0, deadline=inf, lmr=True, alpha=-WINDOW, beta=WINDOW):
        return gmp._visit(gm_board, True, *gm_state, depth, qleft, ply, alpha, beta,
                          deadline, played, gm_work, lmr)

    yield "gmp", gm_board, gm_state, gm_work, 0, 0, gmp._REDUCTIONS, gmp._TT_CUTOFFS, visit_gmp


class SearchTableTests(unittest.TestCase):
    def test_quiet_transposition_cuts_off_across_move_order_but_never_overrides_repetition(self):
        def quiet_line(first, second):
            game = Tablut.Tablut()
            game.past_positions = [game.compute_zobrist_hash(True)]
            attacker = True
            for move in (first, [[2, 4], [2, 3]], second, [[6, 4], [6, 5]]):
                self.assertTrue(game.make_move(attacker, move))
                attacker = not attacker
                self.assertFalse(game.check_if_game_over(attacker))
            self.assertEqual(sum(row.count("A") + row.count("D") for row in game.board), 24)
            return game.board, game.past_positions[:-1]

        move_a, move_b = [[0, 3], [0, 2]], [[0, 5], [0, 6]]
        board, first_path = quiet_line(move_a, move_b)
        other_board, other_path = quiet_line(move_b, move_a)
        self.assertEqual(board, other_board)
        self.assertNotEqual(set(first_path), set(other_path))
        ply = len(first_path)
        for name, _, state, work, _, draw, _, cutoffs, visit in _native_cases(board):
            with self.subTest(kernel=name):
                stamp = state[0]
                table = work.tt
                self.assertEqual(table.entries.shape, (1 << 19, 2))
                self.assertEqual(table.entries.nbytes, 8 * (1 << 20))
                work.path[:ply] = np.asarray(first_path, dtype=np.uint64)
                tt.advance_generation(table)
                first = visit(1, ply, EMPTY)
                self.assertEqual(tt.entry_flag(tt.lookup(table, stamp)), tt.EXACT)
                self.assertGreater(work.counts[0], 1)

                tt.advance_generation(table)
                work.path[:ply] = np.asarray(other_path, dtype=np.uint64)
                work.counts.fill(0)
                self.assertEqual(visit(1, ply, EMPTY), first)
                self.assertEqual((work.counts[0], work.counts[cutoffs]), (1, 1))

                work.counts.fill(0)
                self.assertEqual(visit(1, ply, np.array([stamp], dtype=np.uint64)), (draw, -1))
                self.assertEqual((work.counts[0], work.counts[cutoffs]), (1, 0))
                work.path[0] = stamp
                self.assertEqual(visit(1, ply, EMPTY), (draw, -1))
                self.assertEqual(work.counts[cutoffs], 0)

    def test_visited_repetition_keeps_only_hint_but_lmr_can_store_a_score(self):
        board = Tablut.Tablut().board
        moved = engine.apply_move(board, engine.get_legal_moves(board, True)[0], True)
        history = np.array([engine.compute_hash(moved, False)], dtype=np.uint64)
        for name, _, state, work, _, _, reductions, _, visit in _native_cases(board):
            with self.subTest(kernel=name):
                stamp, prior = state[:2]
                table = work.tt
                work.path[0] = prior
                tt.advance_generation(table)
                visit(1, 1, history)
                self.assertGreater(work.counts[0], 1)
                entry = tt.lookup(table, stamp)
                self.assertGreaterEqual(tt.hint_move(entry), 0)
                self.assertEqual(tt.entry_flag(entry), tt.HINT)

                work.counts.fill(0)
                visit(3, 1, EMPTY, alpha=20_000, beta=20_001)
                self.assertGreater(work.counts[reductions], 0)
                self.assertEqual(tt.entry_flag(tt.lookup(table, stamp)), tt.UPPER)

    def test_root_quiescence_certification_and_interruption_use_no_bound(self):
        for name, flat, state, work, mode, _, _, cutoffs, visit in _native_cases(Tablut.Tablut().board):
            with self.subTest(kernel=name):
                stamp = state[0]
                table = work.tt
                work.path[0] = state[1]
                tt.advance_generation(table)
                hint = 3 * 81 + 2
                score = 81_111
                tt.remember(table, stamp, hint, 1, mode, score, tt.EXACT, 1)
                self.assertNotEqual(visit(0, 1, EMPTY)[0], score)
                work.counts.fill(0)
                self.assertNotEqual(visit(1, 1, EMPTY, lmr=False)[0], score)
                self.assertGreater(work.counts[0], 1)

                tt.remember(table, stamp, hint, 1, mode, score, tt.EXACT, 1)
                work.counts.fill(0)
                self.assertNotEqual(visit(1, 0, EMPTY)[0], score)
                self.assertEqual(tt.entry_flag(tt.lookup(table, stamp)), tt.HINT)
                self.assertEqual(work.counts[cutoffs], 0)

                tt.remember(table, stamp, hint, 2, mode, score, tt.EXACT, 1)
                original_entry = tt.lookup(table, stamp)
                work.counts.fill(0)
                work.counts[0] = 510
                original = flat.copy()
                self.assertEqual(visit(2, 0, EMPTY, deadline=-inf), (0, -1))
                self.assertEqual(tt.lookup(table, stamp), original_entry)
                np.testing.assert_array_equal(flat, original)

    def test_entrants_keep_their_evaluation_and_frozen_versions(self):
        self.assertIs(gmp._evaluate, gm._evaluate)
        self.assertEqual(runner._select_teams("v4,v3,v2,gmp,gm"),
                         ("Team19-v4", "Team19-v3", "Team19-v2", "Grandmaster-Plus", "Grandmaster-Numba"))
        self.assertEqual(runner.TEAMS["Team19-v4"].module, "Team19")
        self.assertIn(".v_be23011.", runner.TEAMS["Team19-v3"].module)
        self.assertIn(".v_3edc85c.", runner.TEAMS["Team19-v2"].module)

    def test_grandmaster_plus_deadline_poll_stays_inside_native_search(self):
        board = gmp._flat(Tablut.Tablut().board)
        state = gm._root_state(board)
        work = gmp._work()
        gmp._visit(board, True, *state, 1, 0, 0, -WINDOW, WINDOW, inf, EMPTY, work, False)
        work.counts.fill(0)
        work.counts[gmp._NODES] = 511
        deadline = gmp._read_native_clock(work.clock) - 0.01
        with patch.object(gmp.time, "perf_counter", side_effect=RuntimeError("Python timer called")) as timer:
            self.assertEqual(gmp._visit(board, True, *state, 5, 4, 0,
                                        -WINDOW, WINDOW, deadline, EMPTY, work, True), (0, -1))
        timer.assert_not_called()
        self.assertTrue(work.counts[gmp._STOPPED])

    def test_grandmaster_plus_keeps_a_legal_depth_one_over_520_calls(self):
        board = _escape_threat()
        for budget in (10.0, 60.0):
            with self.subTest(budget=budget), patch.object(Tablut, "TIME_BUDGET", budget):
                player = gmp.Player()
                iteration = player._iteration
                clock = SimpleNamespace(now=0.0)
                referee_remaining = budget
                referee = Tablut.Tablut()

                def charge(flat, attacker, state, depth, *args):
                    clock.now += max(0.0004, player._step_budget(state[2], state[3]))
                    return iteration(flat, attacker, state, depth, *args)

                player._iteration = charge
                with patch.object(gmp, "time", SimpleNamespace(perf_counter=lambda: clock.now)):
                    for _ in range(520):
                        started = clock.now
                        move = player.play_attacker(board)
                        referee_remaining -= clock.now - started + 0.00015
                        self.assertGreater(referee_remaining, 0.05)
                        self.assertLess(player.remaining, referee_remaining)
                        self.assertEqual(player.last_search.depth, 1)
                        self.assertFalse(player.last_search.fallback)
                        referee.board = [row[:] for row in board]
                        self.assertTrue(referee.make_move(True, move))
                self.assertEqual(player.moves_played, 520)
                self.assertEqual(player.remaining, 0.0)
