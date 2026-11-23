"""Native transposition bounds shared by the compiled search implementations."""

from typing import NamedTuple

import numpy as np
from numba import njit


TT_SIZE = 1 << 20
HINT, EXACT, LOWER, UPPER = range(4)
_MATE_FLOOR = 100_000 - 128


class BoundTable(NamedTuple):
    keys: np.ndarray
    contexts: np.ndarray
    scores: np.ndarray
    modes: np.ndarray
    moves: np.ndarray
    depths: np.ndarray
    flags: np.ndarray
    generations: np.ndarray
    generation: np.ndarray


def allocate_table(size: int = TT_SIZE) -> BoundTable:
    return BoundTable(
        np.empty(size, dtype=np.uint64),
        np.empty(size, dtype=np.uint64),
        np.empty(size, dtype=np.int32),
        np.empty(size, dtype=np.int32),
        np.empty(size, dtype=np.int16),
        np.empty(size, dtype=np.uint8),
        np.empty(size, dtype=np.uint8),
        np.zeros(size, dtype=np.uint32),
        np.array([0, 1], dtype=np.int64),
    )


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
def _history_token(key):
    # Mix each Zobrist key before XOR so the linear board hashes do not
    # cancel predictably across sets of distinct positions.
    value = np.uint64(key) ^ np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


@njit(inline="always")
def next_context(played, contexts, key, ply):
    if ply == 0:
        prefix = np.uint64(0)
        for prior in played:
            prefix ^= _history_token(prior)
        contexts[0] = prefix
        if history_contains(played, key):
            return prefix
    return contexts[ply] ^ _history_token(key)


@njit(inline="always")
def _slot(table, key):
    return np.int64(key & np.uint64(len(table.keys) - 1))


@njit(inline="always")
def lookup(table, key):
    slot = _slot(table, key)
    if table.generations[slot] >= table.generation[1] and table.keys[slot] == key:
        return slot
    return -1


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
def cutoff(table, slot, context, depth, mode, ply, alpha, beta):
    if (slot < 0 or table.flags[slot] == HINT or table.contexts[slot] != context
            or table.depths[slot] != depth or table.modes[slot] != mode):
        return False, 0
    value = _load_score(int(table.scores[slot]), ply)
    flag = table.flags[slot]
    return flag == EXACT or flag == LOWER and value >= beta or flag == UPPER and value <= alpha, value


@njit(inline="always")
def remember(table, key, move, context, depth, mode, score, flag, ply):
    slot = _slot(table, key)
    table.keys[slot] = key
    table.moves[slot] = move
    table.flags[slot] = flag
    table.generations[slot] = table.generation[0]
    if flag != HINT:
        table.contexts[slot] = context
        table.depths[slot] = depth
        table.modes[slot] = mode
        table.scores[slot] = _store_score(score, ply)
