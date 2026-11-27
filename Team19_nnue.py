"""Allocation-free-in-recursion Numba NNUE residuals for the frozen v4 search.

The accumulator stores hidden preactivations, not clipped hidden outputs. A root
is built once per fixed-depth iteration. Each child frame updates only the moved
and captured features in its own preallocated stack row; popping a search move
requires no NNUE undo because the next sibling overwrites the same child row.

Only NumPy, Numba and the static root weight module are used here. There is no
file I/O, cache configuration, threading, process creation, or tools import.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit

from Team19_nnue_weights import (
    HEAD_BIAS, HEAD_WEIGHT, HIDDEN_BIAS, HIDDEN_WEIGHT_BY_FEATURE, RESIDUAL_CLIP,
)


FEATURES = 243
HIDDEN = 32
AREA = 81
# Team19_search: maximum iterative depth 64 + quiescence 4 + two spare slots.
MAX_PLY = 70


class NNUEWork(NamedTuple):
    """Per-player storage; no array is allocated by the recursive search."""

    hidden_weight: np.ndarray  # [243,32], contiguous by input feature
    hidden_bias: np.ndarray    # [32]
    head_weight: np.ndarray    # [2,32], side 0=D and side 1=A
    head_bias: np.ndarray      # [2]
    residual_clip: np.float32
    inv_residual_clip: np.float32
    acc_stack: np.ndarray      # [MAX_PLY,32] float32 hidden preactivations
    enabled: bool


def _checked_array(value, shape) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"NNUE array has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError("NNUE arrays must contain only finite values")
    return np.ascontiguousarray(array)


def make_nnue_work(
    enabled: bool = False,
    *,
    hidden_weight: np.ndarray = HIDDEN_WEIGHT_BY_FEATURE,
    hidden_bias: np.ndarray = HIDDEN_BIAS,
    head_weight: np.ndarray = HEAD_WEIGHT,
    head_bias: np.ndarray = HEAD_BIAS,
) -> NNUEWork:
    """Allocate one fixed stack at player construction, never during search."""
    return NNUEWork(
        _checked_array(hidden_weight, (FEATURES, HIDDEN)),
        _checked_array(hidden_bias, (HIDDEN,)),
        _checked_array(head_weight, (2, HIDDEN)),
        _checked_array(head_bias, (2,)),
        np.float32(RESIDUAL_CLIP),
        np.float32(1.0) / np.float32(RESIDUAL_CLIP),
        np.zeros((MAX_PLY, HIDDEN), dtype=np.float32),
        bool(enabled),
    )


@njit(no_cfunc_wrapper=True, fastmath=True)
def _nnue_full(board, hidden_weight, hidden_bias, out):
    """Recompute hidden preactivations from a flat int8[81] board."""
    for hidden in range(HIDDEN):
        out[hidden] = hidden_bias[hidden]
    for square in range(AREA):
        piece = board[square]
        # Codes 0 (empty) and 4 (empty castle) intentionally have no features.
        if 1 <= piece <= 3:
            feature = square * 3 + piece - 1
            for hidden in range(HIDDEN):
                out[hidden] += hidden_weight[feature, hidden]


@njit(inline="always", fastmath=True)
def _nnue_root(board, work):
    if work.enabled:
        _nnue_full(board, work.hidden_weight, work.hidden_bias, work.acc_stack[0])


@njit(inline="always", fastmath=True)
def _nnue_push(board, move, ply, captured_at, captured_pieces, captured, work):
    """Write child ply+1 from parent ply after Team19_engine._push."""
    if not work.enabled:
        return
    source, target = divmod(move, AREA)
    moved = board[target]
    parent = work.acc_stack[ply]
    child = work.acc_stack[ply + 1]
    source_feature = source * 3 + moved - 1
    target_feature = target * 3 + moved - 1
    for hidden in range(HIDDEN):
        child[hidden] = (parent[hidden] - work.hidden_weight[source_feature, hidden]
                         + work.hidden_weight[target_feature, hidden])
    for index in range(captured):
        piece = captured_pieces[index]
        feature = captured_at[index] * 3 + piece - 1
        for hidden in range(HIDDEN):
            child[hidden] -= work.hidden_weight[feature, hidden]


@njit(inline="always", fastmath=True)
def _nnue_residual(attacker, ply, work):
    """Evaluate one static leaf; both heads are attacker-positive."""
    if not work.enabled:
        return np.float32(0.0)
    side = 1 if attacker else 0
    raw = np.float32(0.0)
    for hidden in range(HIDDEN):
        activation = work.acc_stack[ply, hidden]
        if activation < np.float32(0.0):
            activation = np.float32(0.0)
        elif activation > np.float32(1.0):
            activation = np.float32(1.0)
        raw += activation * work.head_weight[side, hidden]
    raw += work.head_bias[side]
    # Multiplication by a precomputed reciprocal avoids a scalar division per leaf.
    # fastmath lets LLVM vectorize the independent 32-wide reduction portably.
    bound = work.residual_clip
    return np.float32(bound * np.tanh(raw * work.inv_residual_clip))


def residual_flat(board: np.ndarray, attacker: bool, work: NNUEWork | None = None) -> float:
    """Small Python adapter used by offline alignment checks, not by recursion."""
    flat = np.asarray(board, dtype=np.int8)
    if flat.shape != (AREA,):
        raise ValueError("board must be flat int8[81]")
    active = make_nnue_work(True) if work is None else work
    _nnue_root(flat, active)
    return float(_nnue_residual(bool(attacker), 0, active))
