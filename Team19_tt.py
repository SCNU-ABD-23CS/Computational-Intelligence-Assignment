"""Native transposition bounds shared by the compiled search implementations."""

from typing import NamedTuple

import numpy as np
from numba import njit


TT_SIZE = 1 << 19
HINT, EXACT, LOWER, UPPER = range(4)
_MATE_FLOOR = 100_000 - 128
_MOVE_MASK = (1 << 13) - 1
_SCORE_SHIFT = 13
_SCORE_MASK = (1 << 19) - 1
_SCORE_BIAS = 1 << 18
_DEPTH_SHIFT = 32
_DEPTH_MASK = (1 << 7) - 1
_MODE_SHIFT = 39
_MODE_MASK = (1 << 5) - 1
_FLAG_SHIFT = 44
_GENERATION_SHIFT = 48
_LAST_GENERATION = (1 << 16) - 1


class BoundTable(NamedTuple):
    # Each slot is one cache-local key/payload pair. The payload stores the
    # move, signed score, depth, search mode, bound flag and game generation.
    entries: np.ndarray
    generation: np.ndarray


def allocate_table(size: int = TT_SIZE) -> BoundTable:
    return BoundTable(np.zeros((size, 2), dtype=np.uint64), np.array([0, 1], dtype=np.int64))


def advance_generation(table):
    if table.generation[0] == _LAST_GENERATION:
        table.entries[:, 1].fill(0)
        table.generation[:] = (0, 1)
    table.generation[0] += 1


@njit(inline="always")
def history_contains(played, key):
    lo, hi = 0, len(played)
    while lo < hi:
        mid = (lo + hi) // 2
        if played[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(played) and played[lo] == key


@njit(no_cpython_wrapper=True, no_cfunc_wrapper=True)
def seen(played, path, key, ply):
    if history_contains(played, key):
        return True
    for earlier in range(ply):
        if path[earlier] == key:
            return True
    return False


@njit(inline="always")
def _slot(table, key):
    return np.int64(key & np.uint64(len(table.entries) - 1))


@njit(inline="always")
def lookup(table, key):
    slot = _slot(table, key)
    if table.entries[slot, 0] == key:
        entry = table.entries[slot, 1]
        if entry >> np.uint64(_GENERATION_SHIFT) >= np.uint64(table.generation[1]):
            return entry
    return np.uint64(0)


@njit(inline="always")
def hint_move(entry):
    return np.int64(entry & np.uint64(_MOVE_MASK)) - 1


@njit(inline="always")
def entry_flag(entry):
    return (entry >> np.uint64(_FLAG_SHIFT)) & np.uint64(3)


@njit(inline="always")
def classify_bound(score, alpha, beta):
    if score <= alpha:
        return UPPER
    if score >= beta:
        return LOWER
    return EXACT


@njit(inline="always")
def _store_score(score, ply):
    if score >= _MATE_FLOOR:
        return score + ply
    if score <= -_MATE_FLOOR:
        return score - ply
    return score


@njit(inline="always")
def _load_score(score, ply):
    if score >= _MATE_FLOOR:
        return score - ply
    if score <= -_MATE_FLOOR:
        return score + ply
    return score


@njit(inline="always")
def cutoff(entry, depth, mode, ply, alpha, beta):
    flag = entry_flag(entry)
    if flag == HINT:
        return False, 0
    stored_depth = (entry >> np.uint64(_DEPTH_SHIFT)) & np.uint64(_DEPTH_MASK)
    stored_mode = (entry >> np.uint64(_MODE_SHIFT)) & np.uint64(_MODE_MASK)
    if stored_depth < depth or stored_mode != mode:
        return False, 0
    score = (entry >> np.uint64(_SCORE_SHIFT)) & np.uint64(_SCORE_MASK)
    value = _load_score(np.int64(score) - _SCORE_BIAS, ply)
    return flag == EXACT or flag == LOWER and value >= beta or flag == UPPER and value <= alpha, value


@njit(inline="always")
def remember(table, key, move, depth, mode, score, flag, ply):
    slot = _slot(table, key)
    payload = np.uint64(move + 1) | (np.uint64(table.generation[0]) << np.uint64(_GENERATION_SHIFT))
    if flag != HINT:
        score_bits = np.uint64(_store_score(score, ply) + _SCORE_BIAS)
        payload |= score_bits << np.uint64(_SCORE_SHIFT)
        payload |= np.uint64(depth) << np.uint64(_DEPTH_SHIFT)
        payload |= np.uint64(mode) << np.uint64(_MODE_SHIFT)
        payload |= np.uint64(flag) << np.uint64(_FLAG_SHIFT)
    table.entries[slot, 0] = key
    table.entries[slot, 1] = payload
