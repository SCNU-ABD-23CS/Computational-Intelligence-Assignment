"""Framework-free NNUE encoding, target masking and exported-weight inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np


FEATURES = 243
HIDDEN = 32
HEADS = 2
RESIDUAL_CLIP = 750.0
EVAL_SCALE = 6000.0
MATE_THRESHOLD = 90_000.0
REPETITION_SCORE = 3_000.0
WEIGHT_FIELDS = (
    "hidden_weight", "hidden_bias", "head_weight", "head_bias",
    "residual_clip", "eval_scale",
)


def encode_boards(boards: np.ndarray) -> np.ndarray:
    """Encode int8[N,81] as float32[N,243] in cell-major A/D/K plane order."""
    flat = np.asarray(boards)
    if flat.ndim != 2 or flat.shape[1] != 81:
        raise ValueError(f"boards must have shape [N,81], got {flat.shape}")
    if not np.isin(flat, np.arange(5)).all():
        raise ValueError("board codes must be in 0..4")
    grid = flat.reshape(-1, 9, 9)
    encoded = np.zeros((*grid.shape, 3), dtype=np.float32)
    for plane, code in enumerate((1, 2, 3)):
        encoded[..., plane] = grid == code
    # Square 0 features are 0,1,2; square 1 features are 3,4,5, etc.
    # Codes 0 and 4 deliberately produce three zero features.
    return encoded.reshape(-1, FEATURES)


def d4_boards(boards: np.ndarray) -> np.ndarray:
    """Return eight transform-major copies; labels must be tiled, not repeated."""
    flat = np.asarray(boards)
    if flat.ndim != 2 or flat.shape[1] != 81:
        raise ValueError(f"boards must have shape [N,81], got {flat.shape}")
    grid = flat.reshape(-1, 9, 9)
    variants = []
    for turns in range(4):
        rotated = np.rot90(grid, turns, axes=(1, 2))
        variants.extend((rotated, np.fliplr(rotated)))
    return np.concatenate(variants, axis=0).reshape(-1, 81).astype(np.int8, copy=False)


def augment_train(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply D4 exactly once to training arrays; never call this for validation."""
    augmented = dict(arrays)
    augmented["boards"] = d4_boards(arrays["boards"])
    for key in ("sides", "base_evals", "deep_evals", "outcomes", "game_ids"):
        augmented[key] = np.tile(arrays[key], 8)
    return augmented


def deep_regression_mask(deep_evals: np.ndarray) -> np.ndarray:
    """Retain outcome learning while suppressing mate and repetition deep terms."""
    deep = np.abs(np.asarray(deep_evals, dtype=np.float64))
    mate = deep >= MATE_THRESHOLD
    repetition = np.isclose(deep, REPETITION_SCORE, rtol=0.0, atol=1e-6)
    return ~(mate | repetition)


def mixed_targets(
    deep_evals: np.ndarray,
    outcomes: np.ndarray,
    *,
    scale: float = EVAL_SCALE,
    outcome_weight: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 targets and the mask controlling the deep-search term."""
    if not 0.0 <= outcome_weight <= 1.0:
        raise ValueError("outcome_weight must be in [0,1]")
    if scale <= 0:
        raise ValueError("scale must be positive")
    deep = np.asarray(deep_evals, dtype=np.float64)
    outcome = np.asarray(outcomes, dtype=np.float64)
    mask = deep_regression_mask(deep)
    y_deep = 1.0 / (1.0 + np.exp(-np.clip(deep / scale, -60.0, 60.0)))
    mixed = outcome_weight * outcome + (1.0 - outcome_weight) * y_deep
    # Masked rows become outcome-only instead of silently becoming zero-weight rows.
    target = np.where(mask, mixed, outcome)
    return target.astype(np.float32), mask


def load_weights(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the small, pickle-free deployment artifact."""
    with np.load(Path(path), allow_pickle=False) as data:
        missing = set(WEIGHT_FIELDS) - set(data.files)
        if missing:
            raise ValueError(f"Missing NNUE weight fields: {sorted(missing)}")
        weights = {key: np.array(data[key], dtype=np.float32, copy=True) for key in WEIGHT_FIELDS}
        # Artifacts before the scale fix have no activation field and used hard clip.
        weights["output_activation"] = np.array(
            str(data["output_activation"]) if "output_activation" in data.files else "clip"
        )
    expected = {
        "hidden_weight": (HIDDEN, FEATURES),
        "hidden_bias": (HIDDEN,),
        "head_weight": (HEADS, HIDDEN),
        "head_bias": (HEADS,),
        "residual_clip": (),
        "eval_scale": (),
    }
    for key, shape in expected.items():
        if weights[key].shape != shape:
            raise ValueError(f"{key} has shape {weights[key].shape}, expected {shape}")
        if not np.isfinite(weights[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    if float(weights["residual_clip"]) <= 0 or float(weights["eval_scale"]) <= 0:
        raise ValueError("residual_clip and eval_scale must be positive")
    if str(weights["output_activation"]) not in {"clip", "tanh"}:
        raise ValueError("output_activation must be 'clip' or 'tanh'")
    return weights


def numpy_hidden(weights: dict[str, np.ndarray], boards: np.ndarray) -> np.ndarray:
    """Expose clipped hidden activations for non-training saturation diagnostics."""
    features = encode_boards(boards)
    return np.clip(
        features @ weights["hidden_weight"].T + weights["hidden_bias"], 0.0, 1.0,
    ).astype(np.float32)


def numpy_residual(weights: dict[str, np.ndarray], boards: np.ndarray,
                   sides: np.ndarray) -> np.ndarray:
    """Evaluate exported weights without PyTorch; used by the sealed-question check."""
    side = np.asarray(sides, dtype=np.int64)
    if side.shape != (len(boards),) or not np.isin(side, (0, 1)).all():
        raise ValueError("sides must have shape [N] and contain only 0/1")
    hidden = numpy_hidden(weights, boards)
    both_heads = hidden @ weights["head_weight"].T + weights["head_bias"]
    selected = both_heads[np.arange(len(boards)), side]
    bound = float(weights["residual_clip"])
    if str(weights["output_activation"]) == "tanh":
        return (bound * np.tanh(selected / bound)).astype(np.float32)
    return np.clip(selected, -bound, bound).astype(np.float32)
