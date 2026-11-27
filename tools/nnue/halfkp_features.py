"""Tablut HalfKP feature layout shared by training and portable inference."""

from __future__ import annotations

import numpy as np

BOARD_SIZE = 9
SQUARES = BOARD_SIZE ** 2
KING_STRIDE = 2 * SQUARES
FEATURES = SQUARES * KING_STRIDE
PADDING_PIECE = 255
# Feature 0 would place an attacker and the king on the same square.
PADDING_INDEX = 0


def _permutations() -> np.ndarray:
    grid = np.arange(SQUARES, dtype=np.int64).reshape(BOARD_SIZE, BOARD_SIZE)
    mappings = []
    for turns in range(4):
        rotated = np.rot90(grid, turns)
        for view in (rotated, rotated[::-1]):
            mappings.append(np.argsort(view.ravel()))
    return np.stack(mappings)


# PERM[view, square] maps an original square to its square in that view.
PERM = _permutations()
PERM.setflags(write=False)


def pack_boards(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 king squares and padded rows of piece_type * 81 + square."""
    if boards.ndim != 2 or boards.shape[1] != SQUARES or boards.dtype != np.int8:
        raise ValueError("HalfKP boards must be int8[N,81]")
    if np.any((boards < 0) | (boards > 4)):
        raise ValueError("HalfKP board codes must be in [0,4]")
    kings = boards == 3
    if np.any(kings.sum(axis=1) != 1):
        raise ValueError("HalfKP requires exactly one king per board")

    pieces = (boards == 1) | (boards == 2)
    counts = pieces.sum(axis=1)
    starts = np.cumsum(counts) - counts
    rows, squares = np.nonzero(pieces)
    codes = (boards[rows, squares].astype(np.int64) - 1) * SQUARES + squares
    packed = np.full((len(boards), max(1, int(counts.max(initial=0)))),
                     PADDING_PIECE, dtype=np.uint8)
    packed[rows, np.arange(len(rows)) - starts[rows]] = codes
    return kings.argmax(axis=1).astype(np.uint8), packed


def feature_indices(kings: np.ndarray, packed: np.ndarray,
                    views: np.ndarray | int = 0) -> np.ndarray:
    """Produce fixed-width HalfKP bags for a scalar or per-board D4 view."""
    views = np.broadcast_to(views, kings.shape)
    codes = packed.astype(np.int64)
    squares = PERM[views[:, None], codes % SQUARES]
    result = PERM[views, kings][:, None] * KING_STRIDE + codes // SQUARES * SQUARES + squares
    return np.where(codes == PADDING_PIECE, PADDING_INDEX, result)


def d4_overlap(boards: np.ndarray, sides: np.ndarray,
               heldout: np.ndarray, heldout_sides: np.ndarray) -> int:
    """Count matching (D4 board, side to move) positions as in the dataset split."""
    heldout = np.where(heldout == 4, 0, heldout)
    keys = {(row.tobytes(), side) for permutation in PERM
            for row, side in zip(heldout[:, permutation], heldout_sides)}
    return sum((row.tobytes(), side) in keys
               for row, side in zip(np.where(boards == 4, 0, boards), sides))
