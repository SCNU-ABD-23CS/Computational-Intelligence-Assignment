"""Read and partition the HalfKP offline data without accessing ranking questions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .common import NNUE_RESULTS
from .halfkp_features import SQUARES, d4_overlap
from .model import deep_regression_mask

DEFAULT_DATA = NNUE_RESULTS / "volume_halfkp"
DEFAULT_RUN = NNUE_RESULTS.parent / "runs" / "halfkp_v1"
MAX_TRAIN_DEPTH = 32
SPLIT_FILES = {"train": "team19_nnue_train.npz", "validation": "team19_nnue_val.npz"}
_COLUMN_DTYPES = {
    "sides": np.dtype(np.int8),
    "base_evals": np.dtype(np.float32),
    "deep_evals": np.dtype(np.float32),
    "outcomes": np.dtype(np.float32),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(path: Path, *, training: bool = False) -> tuple[dict[str, np.ndarray], int]:
    """Require depths for training and filter before packing, augmentation or grouping."""
    required = {"boards", "game_ids", *_COLUMN_DTYPES}
    if training:
        required.add("depths")
    with np.load(path, allow_pickle=False) as dataset:
        missing = required.difference(dataset.files)
        if missing:
            raise ValueError(f"{path.name} is missing fields: {sorted(missing)}")
        fields = required | ({"depths"} & set(dataset.files))
        arrays = {field: dataset[field] for field in fields}

    boards = arrays["boards"]
    if boards.ndim != 2 or boards.shape[1] != SQUARES or boards.dtype != np.int8:
        raise ValueError(f"{path.name}: boards must be int8[N,81]")
    rows = len(boards)
    for name, dtype in _COLUMN_DTYPES.items():
        column = arrays[name]
        if column.shape != (rows,) or column.dtype != dtype or not np.isfinite(column).all():
            raise ValueError(f"{path.name}: {name} must be finite {dtype}[N]")
    if not np.isin(arrays["sides"], [0, 1]).all():
        raise ValueError(f"{path.name}: sides must be 0 (D) or 1 (A)")
    if not np.isin(arrays["outcomes"], [0.0, 0.5, 1.0]).all():
        raise ValueError(f"{path.name}: outcomes must be 0, 0.5 or 1")
    game_ids = arrays["game_ids"]
    if game_ids.shape != (rows,) or game_ids.dtype.kind != "U" or (game_ids == "").any():
        raise ValueError(f"{path.name}: game_ids must be non-empty Unicode strings")

    if "depths" in arrays:
        depths = arrays["depths"]
        if depths.shape != (rows,) or depths.dtype != np.int16:
            raise ValueError(f"{path.name}: depths must be int16[N]")
    if training:
        keep = arrays["depths"] <= MAX_TRAIN_DEPTH
        arrays = {field: values[keep] for field, values in arrays.items()}
    if not len(arrays["boards"]):
        raise ValueError(f"{path.name}: no usable rows")
    return arrays, rows


def split_audit(train: dict[str, np.ndarray], validation: dict[str, np.ndarray]) -> dict:
    shared_games = len(set(train["game_ids"]).intersection(validation["game_ids"]))
    shared_positions = d4_overlap(train["boards"], train["sides"],
                                  validation["boards"], validation["sides"])
    if shared_games or shared_positions:
        raise ValueError(f"Training overlaps locked validation: {shared_games} game IDs, "
                         f"{shared_positions} D4-equivalent positions")
    return {"shared_game_ids": shared_games, "shared_d4_positions": shared_positions}


def ranking_groups(train: dict[str, np.ndarray], *, seed: int,
                   size: int = 3) -> np.ndarray:
    """Disjoint groups of positions sampled in the same game by the same side."""
    if size < 2:
        raise ValueError("A ranking group needs at least two positions")
    eligible = np.flatnonzero(deep_regression_mask(train["deep_evals"]))
    order = np.random.default_rng(seed).permutation(eligible)
    ids, sides = train["game_ids"], train["sides"]
    order = order[np.lexsort((sides[order], ids[order]))]
    change = (ids[order[1:]] != ids[order[:-1]]) | (sides[order[1:]] != sides[order[:-1]])
    boundaries = np.concatenate(([0], np.flatnonzero(change) + 1, [len(order)]))
    groups = [order[start:start + (stop - start) // size * size]
              for start, stop in zip(boundaries, boundaries[1:]) if stop - start >= size]
    if not groups:
        return np.empty((0, size), dtype=np.int64)
    groups = np.concatenate(groups).reshape(-1, size)
    return groups[np.ptp(train["deep_evals"][groups], axis=1) > 1e-6]
