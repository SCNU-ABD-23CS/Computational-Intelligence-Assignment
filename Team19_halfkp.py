"""Incremental HalfKP evaluation for the CPU search."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic, overload


AREA = 81
KING_STRIDE = 2 * AREA
HIDDEN = 64
HIDDEN2 = 32
RESIDUAL_CLIP = np.float32(750.0)
_W2 = HIDDEN
_B2 = _W2 + HIDDEN * HIDDEN2
_W3 = _B2 + HIDDEN2
_B3 = _W3 + 2 * HIDDEN2


class HalfKPWork(NamedTuple):
    embedding: np.ndarray    # [13122,64] contiguous by feature
    parameters: np.ndarray   # b1, [64,32] W2, b2, [2,32] W3 (D/A), b3
    acc_stack: np.ndarray      # [ply,64] unclipped preactivations
    ksq_stack: np.ndarray      # [ply] int32, -1 after king capture


def make_halfkp_work(ply_slots: int, *, zero_head: bool = False) -> HalfKPWork:
    """Share decoded weights and allocate all writable workspace up front."""
    from Team19_halfkp_weights import load_weights

    embedding, parameters = load_weights()
    if zero_head:
        parameters = parameters.copy()
        parameters[_W3:] = 0
        parameters.flags.writeable = False
    return HalfKPWork(embedding, parameters,
                      np.empty((ply_slots, HIDDEN), dtype=np.float32),
                      np.empty(ply_slots, dtype=np.int32))


@njit(no_cfunc_wrapper=True, fastmath=True)
def _halfkp_full(board, king, embedding, parameters, out):
    """Build the accumulator from the current king and non-king pieces."""
    for hidden in range(HIDDEN):
        out[hidden] = np.float32(0.0)
    if king >= 0:
        base = king * KING_STRIDE
        for square in range(AREA):
            piece = board[square]
            if piece == 1 or piece == 2:
                feature = base + (piece - 1) * AREA + square
                for hidden in range(HIDDEN):
                    out[hidden] += embedding[feature, hidden]
    # Match the offline summation order: EmbeddingBag first, then its bias.
    for hidden in range(HIDDEN):
        out[hidden] += parameters[hidden]


@njit(inline="always", fastmath=True)
def _halfkp_root(board, king, work):
    work.ksq_stack[0] = king
    _halfkp_full(board, king, work.embedding, work.parameters, work.acc_stack[0])


@intrinsic
def _slide_accumulator(typingctx, stack, embedding, ply, removed, added):
    """Update one row with full-width loads and a single store."""
    def codegen(context, builder, signature, args):
        index = context.get_value_type(types.intp)
        vector = ir.VectorType(ir.FloatType(), HIDDEN)
        acc, weights = (context.make_array(t)(context, builder, value).data
                        for t, value in zip(signature.args[:2], args[:2]))

        def row(array, number):
            offset = builder.mul(number, index(HIDDEN))
            return builder.bitcast(builder.gep(array, (offset,)), vector.as_pointer())

        frame, source, target = (context.cast(builder, value, t, types.intp)
                                 for t, value in zip(signature.args[2:], args[2:]))
        parent = builder.load(row(acc, frame), align=4)
        source_weights = builder.load(row(weights, source), align=4)
        target_weights = builder.load(row(weights, target), align=4)
        child = builder.fadd(builder.fsub(parent, source_weights), target_weights)
        builder.store(child, row(acc, builder.add(frame, index(1))), align=4)
        return context.get_dummy_value()

    return types.void(stack, embedding, ply, removed, added), codegen


@njit(inline="always", fastmath=True)
def _halfkp_push(board, source, target, king, ply, captured_at, captured_pieces, captured, work):
    """Update the child row after the rules engine commits the move."""
    work.ksq_stack[ply + 1] = king
    if king != work.ksq_stack[ply]:
        _halfkp_full(board, king, work.embedding, work.parameters, work.acc_stack[ply + 1])
        return

    base = king * KING_STRIDE
    offset = base + (board[target] - 1) * AREA
    embedding = work.embedding
    _slide_accumulator(work.acc_stack, embedding, ply, offset + source, offset + target)
    # A captured king takes the rebuild branch above, so these are A/D features.
    for capture in range(captured):
        feature = base + (captured_pieces[capture] - 1) * AREA + captured_at[capture]
        for hidden in range(HIDDEN):
            work.acc_stack[ply + 1, hidden] -= embedding[feature, hidden]


@intrinsic
def _second_layer_dot(typingctx, stack, parameters, ply, side):
    """Lower the tiny matvec to register accumulators at JIT compile time."""
    def codegen(context, builder, signature, args):
        f32 = ir.FloatType()
        index = context.get_value_type(types.intp)
        vector = ir.VectorType(f32, HIDDEN2)
        zero, one = f32(0.0), f32(1.0)
        zeros, ones = vector([zero] * HIDDEN2), vector([one] * HIDDEN2)
        broadcast = ir.VectorType(ir.IntType(32), HIDDEN2)([0] * HIDDEN2)
        stack_data, packed = (context.make_array(typ)(context, builder, value).data
                              for typ, value in zip(signature.args[:2], args[:2]))
        frame, head = (context.cast(builder, value, t, types.intp)
                       for t, value in zip(signature.args[2:], args[2:]))
        inputs = builder.gep(stack_data, (builder.mul(frame, index(HIDDEN)),))

        def clip(value, lo, hi):
            low = builder.fcmp_ordered("<", value, lo, flags=("fast",))
            value = builder.select(low, lo, value)
            high = builder.fcmp_ordered(">", value, hi, flags=("fast",))
            return builder.select(high, hi, value)

        def load_row(offset):
            pointer = builder.bitcast(builder.gep(packed, (offset,)), vector.as_pointer())
            return builder.load(pointer, align=4)

        def multiply_row(row):
            scalar = builder.load(builder.gep(inputs, (row,)))
            lane = builder.insert_element(zeros, clip(scalar, zero, one), index(0))
            activation = builder.shuffle_vector(lane, zeros, broadcast)
            offset = builder.add(index(_W2), builder.mul(row, index(HIDDEN2)))
            return builder.fmul(activation, load_row(offset), flags=("fast",))

        # The two independent vector accumulators let LLVM overlap the FMA
        # chains. The decoder places each group of 32 coefficients contiguously.
        even, odd = (cgutils.alloca_once(builder, vector) for _ in range(2))
        builder.store(zeros, even)
        builder.store(zeros, odd)
        with cgutils.for_range(builder, index(HIDDEN // 2)) as loop:
            row = builder.mul(loop.index, index(2))
            add_even = builder.fadd(builder.load(even), multiply_row(row), flags=("fast",))
            add_odd = builder.fadd(builder.load(odd), multiply_row(builder.add(row, index(1))),
                                   flags=("fast",))
            builder.store(add_even, even)
            builder.store(add_odd, odd)

        total = builder.fadd(builder.load(even), builder.load(odd), flags=("fast",))
        hidden2 = clip(builder.fadd(total, load_row(index(_B2)), flags=("fast",)), zeros, ones)
        head_offset = builder.add(index(_W3), builder.mul(head, index(HIDDEN2)))
        terms = builder.fmul(hidden2, load_row(head_offset), flags=("fast",))
        reduce_sum = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(f32, (f32, vector)),
            f"llvm.vector.reduce.fadd.v{HIDDEN2}f32",
        )
        output_bias = builder.load(builder.gep(packed, (builder.add(index(_B3), head),)))
        return builder.fadd(builder.call(reduce_sum, (zero, terms), fastmath=("fast",)), output_bias)

    return types.float32(stack, parameters, ply, side), codegen


@njit(inline="always", fastmath=True)
def _halfkp_residual(attacker, ply, work):
    """Evaluate a leaf using the D/A head in attackers' score units."""
    side = int(attacker)
    raw = _second_layer_dot(work.acc_stack, work.parameters, ply, side)
    return np.float32(RESIDUAL_CLIP * math.tanh(raw))


def push_halfkp(board, source, target, king, ply, squares, pieces, captured, work):
    if work is not None:
        _halfkp_push(board, source, target, king, ply, squares, pieces, captured, work)


@overload(push_halfkp, inline="always")
def _compile_push(board, source, target, king, ply, squares, pieces, captured, work):
    # Numba still types the unused branch of the baseline search. Give that
    # specialization a no-op so it never needs a HalfKP workspace or weights.
    if work is types.none:
        def push(board, source, target, king, ply, squares, pieces, captured, work):
            return
    else:
        def push(board, source, target, king, ply, squares, pieces, captured, work):
            _halfkp_push(board, source, target, king, ply, squares, pieces, captured, work)
    return push


def halfkp_residual(attacker, ply, work):
    return _halfkp_residual(attacker, ply, work) if work is not None else np.float32(0.0)


@overload(halfkp_residual, inline="always")
def _compile_residual(attacker, ply, work):
    if work is types.none:
        def residual(attacker, ply, work):
            return np.float32(0.0)
    else:
        def residual(attacker, ply, work):
            return _halfkp_residual(attacker, ply, work)
    return residual
