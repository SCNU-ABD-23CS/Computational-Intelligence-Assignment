"""CPU challenger with a compiled PVS tree and private incremental state."""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np
from numba import njit, objmode

import Tablut

from .common import MATE, Telemetry
from .numba_rules import (
    _A, _AREA, _BEYOND, _CASTLE, _D, _EDGE, _K, _KING_CASTLE, _MOVE_CAPACITY, _NEAR, _X,
    _describe, _flat, _has_legal, _legal, _move_list, _pop, _push, _winner,
)


_DEPTH_LIMIT = 64
_Q_LIMIT = 4
_PLY_SLOTS = _DEPTH_LIMIT + _Q_LIMIT + 2
_TT_SIZE = 16_384
_OPPOSITE = (2, 3, 0, 1)
_EDGE_DISTANCE = np.array([min(r, c, 8 - r, 8 - c) for r in range(9) for c in range(9)],
                          dtype=np.int16)
_ATTACK_PST = np.array([35 * min(r, c, 8 - r, 8 - c) - 15 * (r in (0, 8) or c in (0, 8))
                        for r in range(9) for c in range(9)], dtype=np.int16)
_ESCAPE_POINTS = (0, -7300, -4300, -2100, -900, 160)


class Work(NamedTuple):
    moves: np.ndarray
    priorities: np.ndarray
    captured_at: np.ndarray
    captured_pieces: np.ndarray
    path: np.ndarray
    tt_keys: np.ndarray
    tt_moves: np.ndarray
    killers: np.ndarray
    history: np.ndarray
    queue: np.ndarray
    seen: np.ndarray
    nodes: np.ndarray
    stop: np.ndarray


def _work() -> Work:
    return Work(
        np.empty((_PLY_SLOTS, _MOVE_CAPACITY), dtype=np.int16),
        np.empty((_PLY_SLOTS, _MOVE_CAPACITY), dtype=np.int32),
        np.empty((_PLY_SLOTS, 4), dtype=np.int16),
        np.empty((_PLY_SLOTS, 4), dtype=np.int8),
        np.empty(_PLY_SLOTS, dtype=np.uint64),
        np.zeros(_TT_SIZE, dtype=np.uint64),
        np.full(_TT_SIZE, -1, dtype=np.int16),
        np.full((_PLY_SLOTS, 2), -1, dtype=np.int16),
        np.zeros((2, _AREA * _AREA), dtype=np.int32),
        np.empty(_AREA, dtype=np.int16),
        np.empty(_AREA, dtype=np.bool_),
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.bool_),
    )


@njit(cache=True, inline="always")
def _distance(first, second):
    return abs(first // 9 - second // 9) + abs(first % 9 - second % 9)


@njit(cache=True)
def _escape(board, king, queue, seen):
    seen.fill(False)
    seen[king] = True
    queue[0] = king
    length = 1
    head = 0
    distance = 1
    mobility = 0
    while head < length:
        end = length
        route = False
        while head < end:
            square = queue[head]
            head += 1
            for direction in range(4):
                target = _NEAR[square, direction]
                while target >= 0:
                    cell = board[target]
                    if cell == 0:
                        mobility += distance == 1
                        if _EDGE[target]:
                            route = True
                            if distance > 1:
                                return distance, mobility
                        elif not seen[target]:
                            seen[target] = True
                            queue[length] = target
                            length += 1
                    elif cell != _X and target != king:
                        break
                    target = _NEAR[target, direction]
        if route:
            return distance, mobility
        distance += 1
    return 0, mobility


@njit(cache=True)
def _evaluate(board, attacker, attackers, defenders, king, work):
    distance, mobility = _escape(board, king, work.queue, work.seen)
    escape = 2750 if distance == 0 else _ESCAPE_POINTS[min(5, distance)]
    material = (520 if defenders >= 5 else 625) * attackers - 1050 * defenders
    score = material + escape - 160 * mobility - 450 * (4 - _EDGE_DISTANCE[king])
    if attackers <= 5:
        score -= 400 * (6 - attackers)
    if distance == 1:
        score -= 1400 if not attacker else 200
    for square in range(_AREA):
        cell = board[square]
        if cell == _A:
            dist = _distance(king, square)
            score += _ATTACK_PST[square] + max(0, 4 - dist) * 100
            if dist == 1:
                score += 330
        elif cell == _D:
            dist = _distance(king, square)
            score -= max(0, 4 - dist) * 75
            for direction in range(4):
                target = _NEAR[square, direction]
                if target >= 0 and board[target] == 0:
                    score -= 105
                    break
    return score


@njit(cache=True)
def _king_win(board, attacker, king):
    if not attacker:
        for direction in range(4):
            target = _NEAR[king, direction]
            while target >= 0:
                occupant = board[target]
                if occupant == 0 and _EDGE[target]:
                    return king * _AREA + target
                if occupant != 0 and occupant != _X:
                    break
                target = _NEAR[target, direction]
        return -1
    for direction in range(4):
        target = _NEAR[king, direction]
        opposite = _NEAR[king, _OPPOSITE[direction]]
        if target < 0 or opposite < 0 or board[target] != 0:
            continue
        if _KING_CASTLE[king]:
            supported = True
            for other in range(4):
                if other != direction:
                    occupant = board[_NEAR[king, other]]
                    if occupant != _A and occupant != _X:
                        supported = False
                        break
            if not supported:
                continue
        elif board[opposite] != _A:
            continue
        for ray in range(4):
            source = _NEAR[target, ray]
            while source >= 0:
                occupant = board[source]
                if occupant == _A:
                    return source * _AREA + target
                if occupant != 0 and occupant != _X:
                    break
                source = _NEAR[source, ray]
    return -1


@njit(cache=True)
def _capture_priority(board, source, target, attacker):
    priority = 0
    for direction in range(4):
        far = _BEYOND[target, direction]
        if far < 0:
            continue
        victim = board[_NEAR[target, direction]]
        ally = board[far] if far != source else _X if source == _CASTLE else 0
        if attacker:
            if victim == _K:
                if ally == _A or _KING_CASTLE[_NEAR[target, direction]]:
                    priority += 12000
            elif victim == _D and (ally == _A or ally == _X or ally == _K and far == _CASTLE):
                priority += 6500
        elif victim == _A and (ally == _D or ally == _K or ally == _X):
            priority += 6000
    return priority


@njit(cache=True)
def _history_contains(history, stamp):
    low, high = 0, len(history)
    while low < high:
        mid = (low + high) // 2
        if history[mid] < stamp:
            low = mid + 1
        else:
            high = mid
    return low < len(history) and history[low] == stamp


@njit(cache=True)
def _visit(board, attacker, hash_a, hash_d, attack_count, defend_count, king,
           depth, qleft, ply, alpha, beta, deadline, played, work):
    work.nodes[0] += 1
    if work.nodes[0] % 2048 == 0:
        with objmode(now="float64"):
            now = time.perf_counter()
        if now >= deadline:
            work.stop[0] = True
    if work.stop[0]:
        return 0, -1
    sign = 1 if attacker else -1
    winner = _winner(attack_count, king)
    if winner:
        return sign * (MATE - ply if winner == _A else -MATE + ply), -1
    if not _has_legal(board, attacker):
        return 0, -1
    stamp = hash_a if attacker else hash_d
    if ply:
        if _history_contains(played, stamp):
            return 0, -1
        for earlier in range(ply):
            if work.path[earlier] == stamp:
                return 0, -1
    if depth == 0 and qleft == 0:
        return sign * _evaluate(board, attacker, attack_count, defend_count, king, work), -1
    immediate = _king_win(board, attacker, king)
    if immediate >= 0:
        return MATE - ply - 1, immediate

    quiescent = depth == 0
    danger = quiescent and _king_win(board, not attacker, king) >= 0
    best, best_move = -2 * MATE, -1
    if quiescent and not danger:
        best = sign * _evaluate(board, attacker, attack_count, defend_count, king, work)
        if best >= beta:
            return best, -1
        alpha = max(alpha, best)

    work.path[ply] = stamp
    moves = work.moves[ply]
    scores = work.priorities[ply]
    count = _legal(board, attacker, moves)
    slot = int(stamp & np.uint64(_TT_SIZE - 1))
    hint = work.tt_moves[slot] if work.tt_keys[slot] == stamp else -1
    side_index = 0 if attacker else 1
    useful = 0
    for index in range(count):
        move = moves[index]
        source, target = divmod(move, _AREA)
        tactical = _capture_priority(board, source, target, attacker)
        king_contact = attacker and _distance(target, king) == 1
        if quiescent and not danger and tactical == 0 and not king_contact:
            continue
        priority = tactical
        if board[source] == _K:
            priority += 220 * (_EDGE_DISTANCE[source] - _EDGE_DISTANCE[target])
        if depth:
            if move == hint:
                priority += 2_000_000
            elif move == work.killers[ply, 0]:
                priority += 5000
            elif move == work.killers[ply, 1]:
                priority += 4000
            priority += min(3500, work.history[side_index, move])
        moves[useful], scores[useful] = move, priority
        useful += 1

    next_depth = max(0, depth - 1)
    next_qleft = qleft - quiescent
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
        ha, hd, a, d, k, captured = _push(board, move, hash_a, hash_d, attack_count,
                                           defend_count, king, captured_squares, captured_pieces)
        if quiescent and not danger and captured == 0:
            if not attacker or _king_win(board, True, k) < 0:
                _pop(board, move, captured_squares, captured_pieces, captured)
                continue
        if first:
            child, _ = _visit(board, not attacker, ha, hd, a, d, k, next_depth, next_qleft,
                             ply + 1, -beta, -alpha, deadline, played, work)
            first = False
        else:
            child, _ = _visit(board, not attacker, ha, hd, a, d, k, next_depth, next_qleft,
                             ply + 1, -alpha - 1, -alpha, deadline, played, work)
            if alpha < -child < beta and not work.stop[0]:
                child, _ = _visit(board, not attacker, ha, hd, a, d, k, next_depth, next_qleft,
                                 ply + 1, -beta, -alpha, deadline, played, work)
        _pop(board, move, captured_squares, captured_pieces, captured)
        if work.stop[0]:
            return 0, -1
        score = -child
        if score > best:
            best, best_move = score, move
        alpha = max(alpha, best)
        if alpha >= beta or best == MATE - ply - 1:
            if depth and captured == 0:
                if move != work.killers[ply, 0]:
                    work.killers[ply, 1] = work.killers[ply, 0]
                    work.killers[ply, 0] = move
                work.history[side_index, move] += depth * depth
            break
    if depth and best_move >= 0:
        work.tt_keys[slot] = stamp
        work.tt_moves[slot] = best_move
    return best, best_move


def _root_state(flat):
    a_hash, d_hash, attackers, defenders, king = _describe(flat)
    return np.uint64(a_hash), np.uint64(d_hash), attackers, defenders, king


class Player:
    def __init__(self, seed=0):
        initial = Tablut.Tablut()
        self.initial_hash = initial.compute_zobrist_hash(True)
        self.work = _work()
        root = _flat(initial.board)
        began = time.perf_counter()
        self._iteration(root, True, _root_state(root), 1, -2 * MATE, 2 * MATE,
                        float("inf"), np.empty(0, dtype=np.uint64))
        self.warmup_seconds = time.perf_counter() - began
        self.reset_game()

    def reset_game(self):
        self.remaining = float(Tablut.TIME_BUDGET)
        self.history = {self.initial_hash}
        self.moves_played = 0
        self.last_search = Telemetry()

    def seed_history(self, history):
        self.history.update(history)

    def _iteration(self, root, attacker, state, depth, alpha, beta, deadline, history):
        return _visit(root, attacker, *state, depth, _Q_LIMIT, 0, alpha, beta, deadline,
                      history, self.work)

    def play_attacker(self, board):
        return self._play(board, True)

    def play_defender(self, board):
        return self._play(board, False)

    def _play(self, board, attacker):
        started = time.perf_counter()
        try:
            flat = _flat(board)
            state = _root_state(flat)
            ha, hd, a, d, king = state
            available = max(0.0, self.remaining - 0.06)
            shares = max(12, int(1.45 * (a + d)) + max(0, 12 - self.moves_played))
            deadline = started + min(available / 10, max(0.012, available / shares))
            self.history.add(int(ha if attacker else hd))
            played = np.array(sorted(self.history), dtype=np.uint64)
            self.work.nodes[0] = 0
            self.work.stop[0] = False
            self.work.tt_moves.fill(-1)
            self.work.killers.fill(-1)
            self.work.history.fill(0)
            best, score, depth_done = -1, 0, 0
            for depth in range(1, _DEPTH_LIMIT + 1):
                if time.perf_counter() >= deadline:
                    break
                margin = 850
                alpha, beta = (score - margin, score + margin) if depth > 1 else (-2 * MATE, 2 * MATE)
                value, move = self._iteration(flat, attacker, state, depth, alpha, beta,
                                              deadline, played)
                if self.work.stop[0]:
                    break
                if value <= alpha or value >= beta:
                    value, move = self._iteration(flat, attacker, state, depth, -2 * MATE,
                                                  2 * MATE, deadline, played)
                    if self.work.stop[0]:
                        break
                score, best, depth_done = value, move, depth
                if score >= MATE - depth or score <= -MATE + depth:
                    break
            fallback = best < 0
            if fallback:
                length = _legal(flat, attacker, self.work.moves[0])
                if not length:
                    raise ValueError("No move on an ongoing board")
                best = int(self.work.moves[0, 0])
            ha, hd, _, _, _, _ = _push(flat, best, *state, self.work.captured_at[0],
                                        self.work.captured_pieces[0])
            self.history.add(int(hd if attacker else ha))
            self.moves_played += 1
            self.last_search = Telemetry(depth=depth_done, nodes=int(self.work.nodes[0]),
                                         score=score if attacker else -score, fallback=fallback)
            return _move_list(best)
        finally:
            self.remaining -= time.perf_counter() - started
