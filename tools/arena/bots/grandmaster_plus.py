"""Grandmaster's original evaluator with a larger, path-safe search."""

from dataclasses import dataclass
from math import inf
import time
from typing import NamedTuple

import numpy as np
from numba import njit

import Tablut
from Team19_search import _CLOCK_DTYPE, _native_now, _read_native_clock
from Team19_tt import (
    BoundTable, HINT, advance_generation, allocate_table, classify_bound, cutoff, hint_move,
    lookup, remember, seen,
)

from .common import MATE, Telemetry
from .grandmaster_numba import (
    _DEPTH_LIMIT, _EDGE_DISTANCE, _PLY_SLOTS, _Q_LIMIT,
    _capture_priority, _distance, _evaluate, _king_win, _root_state,
)
from .numba_rules import _A, _AREA, _K, _MOVE_CAPACITY, _flat, _has_legal, _legal, _move_list, _pop, _push, _winner


_RETURN_RESERVE = 0.225
_CALL_OVERHEAD = 0.00125
_LOW_CLOCK = 0.5
_NODES, _REDUCTIONS, _REPETITIONS, _STOPPED, _TT_CUTOFFS = range(5)


@dataclass
class _PlusTelemetry(Telemetry):
    tt_cutoffs: int = 0


class Work(NamedTuple):
    moves: np.ndarray
    priorities: np.ndarray
    captured_at: np.ndarray
    captured_pieces: np.ndarray
    path: np.ndarray
    tt: BoundTable
    killers: np.ndarray
    history: np.ndarray
    queue: np.ndarray
    seen: np.ndarray
    clock: np.ndarray
    counts: np.ndarray


def _work() -> Work:
    return Work(
        np.empty((_PLY_SLOTS, _MOVE_CAPACITY), dtype=np.int16),
        np.empty((_PLY_SLOTS, _MOVE_CAPACITY), dtype=np.int32),
        np.empty((_PLY_SLOTS, 4), dtype=np.int16),
        np.empty((_PLY_SLOTS, 4), dtype=np.int8),
        np.empty(_PLY_SLOTS, dtype=np.uint64),
        allocate_table(),
        np.full((_PLY_SLOTS, 2), -1, dtype=np.int16),
        np.zeros((2, _AREA * _AREA), dtype=np.int32),
        np.empty(_AREA, dtype=np.int16),
        np.empty(_AREA, dtype=np.bool_),
        np.empty(2, dtype=_CLOCK_DTYPE),
        np.zeros(5, dtype=np.int64),
    )


@njit(no_cfunc_wrapper=True, _nrt=False)
def _visit(board, attacker, hash_a, hash_d, attack_count, defend_count, king,
           depth, qleft, ply, alpha, beta, deadline, played, work, lmr):
    counts = work.counts
    counts[_NODES] += 1
    if counts[_NODES] & 511 == 0 and deadline < inf and _native_now(work.clock) >= deadline:
        counts[_STOPPED] = 1
    if counts[_STOPPED]:
        return 0, -1
    sign = 1 if attacker else -1
    winner = _winner(attack_count, king)
    if winner:
        return sign * (MATE - ply if winner == _A else -MATE + ply), -1
    stamp = hash_a if attacker else hash_d
    if ply and seen(played, work.path, stamp, ply):
        counts[_REPETITIONS] += 1
        return 0, -1

    normal = depth > 0
    table = work.tt
    entry = lookup(table, stamp) if normal else np.uint64(0)
    hint = hint_move(entry)
    use_bounds = normal and ply > 0 and lmr
    if use_bounds:
        hit, value = cutoff(entry, depth, qleft, ply, alpha, beta)
        if hit:
            counts[_TT_CUTOFFS] += 1
            return value, hint

    # Normal nodes already generate the full legal list below.
    if not normal and not _has_legal(board, attacker):
        return 0, -1
    if normal or qleft or ply == 1:
        immediate = _king_win(board, attacker, king)
        if immediate >= 0:
            return MATE - ply - 1, immediate
    if not normal and qleft == 0:
        return sign * _evaluate(board, attacker, attack_count, defend_count, king, work), -1

    can_reduce = lmr and depth >= 3 and ply > 0 and beta == alpha + 1
    danger = (not normal or can_reduce) and _king_win(board, not attacker, king) >= 0
    best, best_move = -2 * MATE, -1
    if not normal and not danger:
        best = sign * _evaluate(board, attacker, attack_count, defend_count, king, work)
        if best >= beta:
            return best, -1
        alpha = max(alpha, best)

    work.path[ply] = stamp
    moves = work.moves[ply]
    scores = work.priorities[ply]
    count = _legal(board, attacker, moves)
    if normal and count == 0:
        return 0, -1
    side = 0 if attacker else 1
    useful = 0
    for index in range(count):
        move = moves[index]
        source, target = divmod(move, _AREA)
        tactical = _capture_priority(board, source, target, attacker)
        king_contact = attacker and _distance(target, king) == 1
        if not normal and not danger and tactical == 0 and not king_contact:
            continue
        priority = tactical
        if board[source] == _K:
            priority += 220 * (_EDGE_DISTANCE[source] - _EDGE_DISTANCE[target])
        if normal:
            if move == hint:
                priority += 2_000_000
            elif move == work.killers[ply, 0]:
                priority += 5000
            elif move == work.killers[ply, 1]:
                priority += 4000
            priority += min(3500, work.history[side, move])
        moves[useful], scores[useful] = move, priority
        useful += 1

    child_depth = max(0, depth - 1)
    child_qleft = qleft if normal else qleft - 1
    before_alpha = alpha
    before_repetitions = counts[_REPETITIONS]
    first = True
    captured_squares = work.captured_at[ply]
    captured_pieces = work.captured_pieces[ply]
    for index in range(useful):
        rank = index
        for sibling in range(index + 1, useful):
            if scores[sibling] > scores[rank]:
                rank = sibling
        moves[index], moves[rank] = moves[rank], moves[index]
        scores[index], scores[rank] = scores[rank], scores[index]
        move = np.int64(moves[index])
        source, target = divmod(move, _AREA)
        ha, hd, a, d, k, captured = _push(board, move, hash_a, hash_d, attack_count,
                                           defend_count, king, captured_squares, captured_pieces)
        if not normal and not danger and captured == 0:
            if not attacker or _king_win(board, True, k) < 0:
                _pop(board, move, captured_squares, captured_pieces, captured)
                continue
        trial_depth = child_depth
        if (can_reduce and index >= 4 and not danger and captured == 0 and k == king
                and move != hint and move != work.killers[ply, 0] and move != work.killers[ply, 1]
                and _distance(king, source) > 2 and _distance(king, target) > 2
                and abs(alpha) < MATE - _PLY_SLOTS and _king_win(board, attacker, k) < 0):
            counts[_REDUCTIONS] += 1
            trial_depth -= 1

        narrow = not first
        while True:
            child, _ = _visit(board, not attacker, ha, hd, a, d, k, trial_depth, child_qleft,
                              ply + 1, -alpha - 1 if narrow else -beta, -alpha, deadline, played, work, lmr)
            if counts[_STOPPED]:
                break
            if trial_depth < child_depth and -child > alpha:
                trial_depth = child_depth
            elif narrow and alpha < -child < beta:
                narrow = False
            else:
                break
        first = False
        _pop(board, move, captured_squares, captured_pieces, captured)
        if counts[_STOPPED]:
            return 0, -1
        value = -child
        if value > best:
            best, best_move = value, move
        alpha = max(alpha, best)
        if alpha >= beta or best == MATE - ply - 1:
            if normal and captured == 0:
                if move != work.killers[ply, 0]:
                    work.killers[ply, 1] = work.killers[ply, 0]
                    work.killers[ply, 0] = move
                work.history[side, move] += depth * depth
            break
    if normal and best_move >= 0:
        flag = HINT
        if use_bounds and counts[_REPETITIONS] == before_repetitions:
            flag = classify_bound(best, before_alpha, beta)
        remember(table, stamp, best_move, depth, qleft, best, flag, ply)
    return best, best_move


class Player:
    def __init__(self, seed=0):
        initial = Tablut.Tablut()
        self.initial_hash = initial.compute_zobrist_hash(True)
        self.work = _work()
        flat = _flat(initial.board)
        started = time.perf_counter()
        advance_generation(self.work.tt)
        self._iteration(flat, True, _root_state(flat), 1, 0, -2 * MATE, 2 * MATE,
                        inf, np.empty(0, dtype=np.uint64), False)
        self.warmup_seconds = time.perf_counter() - started
        self.reset_game()

    def reset_game(self):
        self.remaining = float(Tablut.TIME_BUDGET)
        self.history = {self.initial_hash}
        self.moves_played = 0
        self.last_search = _PlusTelemetry()
        self.work.tt.generation[1] = self.work.tt.generation[0] + 1

    def seed_history(self, history):
        self.history.update(history)

    def _iteration(self, flat, attacker, state, depth, qleft, alpha, beta, deadline, history, lmr):
        return _visit(flat, attacker, *state, depth, qleft, 0, alpha, beta,
                      deadline, history, self.work, lmr)

    def _step_budget(self, attackers, defenders):
        available = self.remaining - _RETURN_RESERVE - _CALL_OVERHEAD
        if available <= 0.02:
            return 0.0
        shares = max(12, int(1.45 * (attackers + defenders)) + max(0, 12 - self.moves_played))
        allowance = min(available / 10, max(0.012, available / shares))
        return min(0.008, allowance) if self.remaining < _LOW_CLOCK else allowance

    def play_attacker(self, board):
        return self._play(board, True)

    def play_defender(self, board):
        return self._play(board, False)

    def _play(self, board, attacker):
        started = time.perf_counter()
        self.last_search = _PlusTelemetry()
        try:
            flat = _flat(board)
            state = _root_state(flat)
            ha, hd, attackers, defenders, _ = state
            allowance = self._step_budget(attackers, defenders)
            deadline = started + allowance - min(0.004, allowance * 0.10)
            self.history.add(int(ha if attacker else hd))
            played = np.array(sorted(self.history), dtype=np.uint64)
            work = self.work
            available = max(0.0, deadline - time.perf_counter())
            native_deadline = _read_native_clock(work.clock) + available
            work.counts.fill(0)
            advance_generation(work.tt)
            work.killers.fill(-1)
            work.history.fill(0)
            best, score, depth_done = -1, 0, 0
            for depth in range(1, _DEPTH_LIMIT + 1):
                first = depth == 1
                if not first and time.perf_counter() >= deadline:
                    break
                # Keep one complete, bounded tactical pass even on a low clock.
                stop_at = inf if first else native_deadline
                qleft = 0 if first else _Q_LIMIT
                reduced = work.counts[_REDUCTIONS]
                cached = work.counts[_TT_CUTOFFS]
                alpha, beta = (-2 * MATE, 2 * MATE) if first else (score - 850, score + 850)
                value, move = self._iteration(flat, attacker, state, depth, qleft, alpha, beta,
                                              stop_at, played, True)
                if not work.counts[_STOPPED] and (value <= alpha or value >= beta):
                    value, move = self._iteration(flat, attacker, state, depth, qleft, -2 * MATE,
                                                  2 * MATE, stop_at, played, True)
                if (not work.counts[_STOPPED] and abs(value) >= MATE - _PLY_SLOTS
                        and (work.counts[_REDUCTIONS] > reduced or work.counts[_TT_CUTOFFS] > cached)):
                    value, move = self._iteration(flat, attacker, state, depth, qleft, -2 * MATE,
                                                  2 * MATE, stop_at, played, False)
                if work.counts[_STOPPED]:
                    break
                score, best, depth_done = value, move, depth
                if abs(score) >= MATE - depth:
                    break
            fallback = best < 0
            if fallback:
                length = _legal(flat, attacker, work.moves[0])
                if not length:
                    raise ValueError("No move on an ongoing board")
                best = int(work.moves[0, 0])
            ha, hd, _, _, _, _ = _push(flat, best, *state, work.captured_at[0], work.captured_pieces[0])
            self.history.add(int(hd if attacker else ha))
            self.moves_played += 1
            self.last_search = _PlusTelemetry(
                depth=depth_done, nodes=int(work.counts[_NODES]), score=score if attacker else -score,
                fallback=fallback, tt_cutoffs=int(work.counts[_TT_CUTOFFS]),
            )
            return _move_list(best)
        finally:
            self.remaining = max(0.0, self.remaining - (time.perf_counter() - started) - _CALL_OVERHEAD)
