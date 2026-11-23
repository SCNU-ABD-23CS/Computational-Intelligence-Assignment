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
    def test_flags_require_same_context_and_horizon(self):
        table = tt.allocate_table(8)
        table.generation[0] = 1
        key = np.uint64(35)
        context = np.uint64(17)
        for flag, alpha, beta, score, cuts in (
                (tt.EXACT, 10, 20, 15, True),
                (tt.LOWER, 10, 20, 22, True),
                (tt.LOWER, 10, 20, 15, False),
                (tt.UPPER, 10, 20, 7, True),
                (tt.UPPER, 10, 20, 15, False),
                (tt.HINT, 10, 20, 15, False)):
            with self.subTest(flag=flag, score=score):
                tt.remember(table, key, 6, context, 3, 4, score, flag, 1)
                slot = tt.lookup(table, key)
                self.assertGreaterEqual(slot, 0)
                self.assertEqual(tt.cutoff(table, slot, context, 3, 4, 1, alpha, beta)[0], cuts)
                for ctx, depth, mode in ((np.uint64(19), 3, 4), (context, 2, 4), (context, 3, 0)):
                    self.assertFalse(tt.cutoff(table, slot, ctx, depth, mode, 1, alpha, beta)[0])
        self.assertEqual(tt.lookup(table, np.uint64(43)), -1)
        table.generation[1] = table.generation[0] + 1
        self.assertEqual(tt.lookup(table, key), -1)

    def test_mates_keep_the_same_distance_from_the_cached_position(self):
        table = tt.allocate_table(8)
        table.generation[0] = 1
        key, context = np.uint64(13), np.uint64(24)
        for sign in (1, -1):
            for flag in (tt.EXACT, tt.LOWER, tt.UPPER):
                with self.subTest(sign=sign, flag=flag):
                    tt.remember(table, key, 5, context, 8, 4, sign * (MATE - 9), flag, 5)
                    slot = tt.lookup(table, key)
                    self.assertEqual(table.scores[slot], sign * (MATE - 4))
                    value = sign * (MATE - 7)
                    alpha = value if flag == tt.UPPER else -WINDOW
                    beta = value if flag == tt.LOWER else WINDOW
                    self.assertEqual(tt.cutoff(table, slot, context, 8, 4, 3, alpha, beta), (True, value))


def _native_cases(board):
    flat, state = search._native_position(board)
    work = search._native_work(native_weights())
    draw = search._ATTACKER_REPETITION_SCORE

    def visit_v3(depth, ply, played, *, qleft=0, deadline=inf, lmr=True):
        return search._native_visit(flat, True, *state, depth, qleft, ply, -WINDOW, WINDOW,
                                    deadline, played, work, lmr, draw, False)

    yield "v3", flat, state, work, 16 * draw, draw, visit_v3

    gm_board = gmp._flat(board)
    gm_state = gm._root_state(gm_board)
    gm_work = gmp._work()

    def visit_gmp(depth, ply, played, *, qleft=0, deadline=inf, lmr=True):
        return gmp._visit(gm_board, True, *gm_state, depth, qleft, ply, -WINDOW, WINDOW,
                          deadline, played, gm_work, lmr)

    yield "gmp", gm_board, gm_state, gm_work, 0, 0, visit_gmp


class SearchTableTests(unittest.TestCase):
    def test_real_search_reuses_a_bound_but_current_and_descendant_repetitions_win(self):
        board = Tablut.Tablut().board
        for name, _, state, work, mode, draw, visit in _native_cases(board):
            with self.subTest(kernel=name):
                stamp, prior = state[:2]
                table = work.tt
                self.assertEqual(len(table.keys), 1 << 20)
                work.path[0] = prior
                work.contexts[1] = tt.next_context(EMPTY, work.contexts, prior, 0)
                table.generation[0] = 1
                first = visit(1, 1, EMPTY)
                slot = tt.lookup(table, stamp)
                self.assertGreaterEqual(slot, 0)
                self.assertEqual(table.flags[slot], tt.EXACT)
                self.assertGreater(work.counts[0], 1)

                table.generation[0] += 1
                work.counts.fill(0)
                self.assertEqual(visit(1, 1, EMPTY), first)
                self.assertEqual(work.counts[0], 1)

                work.counts.fill(0)
                self.assertEqual(visit(1, 1, np.array([stamp], dtype=np.uint64)), (draw, -1))
                self.assertEqual(work.counts[0], 1)

                # A different actual history forbids a descendant even when
                # the current board is new. That node must search again.
                moved = engine.apply_move(board, engine.get_legal_moves(board, True)[0], True)
                history = np.array([engine.compute_hash(moved, False)], dtype=np.uint64)
                work.contexts[1] = tt.next_context(history, work.contexts, prior, 0)
                work.counts.fill(0)
                visit(1, 1, history)
                self.assertGreater(work.counts[0], 1)
                self.assertEqual(table.flags[tt.lookup(table, stamp)], tt.HINT)

    def test_root_quiescence_certification_and_interruption_use_no_bound(self):
        for name, flat, state, work, mode, _, visit in _native_cases(Tablut.Tablut().board):
            with self.subTest(kernel=name):
                stamp = state[0]
                table = work.tt
                table.generation[0] = 1
                hint = 3 * 81 + 2
                score = 81_111
                tt.remember(table, stamp, hint, np.uint64(0), 1, mode, score, tt.EXACT, 1)
                self.assertNotEqual(visit(0, 1, EMPTY)[0], score)
                work.counts.fill(0)
                self.assertNotEqual(visit(1, 1, EMPTY, lmr=False)[0], score)
                self.assertGreater(work.counts[0], 1)

                tt.remember(table, stamp, hint, np.uint64(0), 1, mode, score, tt.EXACT, 1)
                work.counts.fill(0)
                self.assertNotEqual(visit(1, 0, EMPTY)[0], score)
                slot = tt.lookup(table, stamp)
                self.assertEqual(table.flags[slot], tt.HINT)

                tt.remember(table, stamp, hint, np.uint64(0), 2, mode, score, tt.EXACT, 1)
                work.counts.fill(0)
                work.counts[0] = 510
                original = flat.copy()
                self.assertEqual(visit(2, 0, EMPTY, deadline=-inf), (0, -1))
                self.assertEqual(table.flags[slot], tt.EXACT)
                self.assertEqual(table.scores[slot], score)
                np.testing.assert_array_equal(flat, original)

    def test_entrants_keep_their_evaluation_and_frozen_versions(self):
        self.assertIs(gmp._evaluate, gm._evaluate)
        self.assertEqual(runner._select_teams("v3,v2,gmp,gm"),
                         ("Team19-v3", "Team19-v2", "Grandmaster-Plus", "Grandmaster-Numba"))
        self.assertEqual(runner.TEAMS["Team19-v3"].module, "Team19")
        self.assertIn(".v_3edc85c.", runner.TEAMS["Team19-v2"].module)

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
