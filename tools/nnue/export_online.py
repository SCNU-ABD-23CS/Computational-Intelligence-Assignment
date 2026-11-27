"""Export the sealed NNUE artifact as read-only, import-time NumPy constants.

This is an offline development tool. The generated root module never reads NPZ
files, imports tools, or touches the filesystem when the player is running.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from .model import load_weights


EXPECTED_SHA256 = "cf27439fdce79c34c05fdc831b755d9c210cd4019dbc41d12057c584150d3809"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "tools/nnue/results/volume1200" / (
    "member5-c750-k6000-l0p5-tanh-precalib5-center1e3-listwise20-seed20261127"
    "/best_weights.npz"
)
DEFAULT_OUTPUT = ROOT / "Team19_nnue_weights.py"


def _literal(array: np.ndarray) -> str:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    lines = []
    for start in range(0, len(values), 6):
        row = ", ".join(repr(float(value)) for value in values[start:start + 6])
        lines.append(f"        {row},")
    return "\n".join(lines)


def render(weights: dict[str, np.ndarray], source_hash: str) -> str:
    if str(weights["output_activation"]) != "tanh":
        raise ValueError("Online export requires the sealed tanh artifact")
    hidden = np.asarray(weights["hidden_weight"], dtype=np.float32)
    by_feature = np.ascontiguousarray(hidden.T)
    head = np.ascontiguousarray(weights["head_weight"], dtype=np.float32)
    arrays = (
        ("HIDDEN_WEIGHT_BY_FEATURE", by_feature),
        ("HIDDEN_BIAS", weights["hidden_bias"]),
        ("HEAD_WEIGHT", head),
        ("HEAD_BIAS", weights["head_bias"]),
    )
    blocks = []
    for name, array in arrays:
        blocks.append(
            f"{name} = np.array([\n{_literal(array)}\n], dtype=np.float32).reshape{array.shape}\n"
            f"{name}.flags.writeable = False\n"
        )
    return f'''"""Read-only static NNUE parameters; generated offline, do not edit manually.

Source best_weights.npz SHA-256: {source_hash}
Feature order is square*3+plane with attacker, defender and king planes.
HIDDEN_WEIGHT_BY_FEATURE is the contiguous transpose of offline [32,243].
"""

import numpy as np


SOURCE_SHA256 = {source_hash!r}
FEATURES = 243
HIDDEN = 32
HEADS = 2
RESIDUAL_CLIP = np.float32({float(weights["residual_clip"])!r})
EVAL_SCALE = np.float32({float(weights["eval_scale"])!r})

{chr(10).join(blocks)}
for _array in (HIDDEN_WEIGHT_BY_FEATURE, HIDDEN_BIAS, HEAD_WEIGHT, HEAD_BIAS):
    if not _array.flags.c_contiguous or _array.flags.writeable:
        raise RuntimeError("Static NNUE arrays must be contiguous and read-only")
del _array
'''


def verify_module(path: Path, weights: dict[str, np.ndarray]) -> None:
    spec = importlib.util.spec_from_file_location("_nnue_weight_verification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comparisons = (
        (module.HIDDEN_WEIGHT_BY_FEATURE.T, weights["hidden_weight"]),
        (module.HIDDEN_BIAS, weights["hidden_bias"]),
        (module.HEAD_WEIGHT, weights["head_weight"]),
        (module.HEAD_BIAS, weights["head_bias"]),
    )
    for actual, expected in comparisons:
        if not np.array_equal(actual, expected):
            raise ValueError("Generated static arrays differ from the sealed NPZ")
        if actual.flags.writeable:
            raise ValueError("Generated static arrays must be read-only")
    if module.RESIDUAL_CLIP != weights["residual_clip"]:
        raise ValueError("Residual bound differs from the sealed NPZ")
    if module.EVAL_SCALE != weights["eval_scale"]:
        raise ValueError("Evaluation scale differs from the sealed NPZ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if source_hash != EXPECTED_SHA256:
        raise ValueError(f"Refusing unapproved NNUE artifact: {source_hash}")
    weights = load_weights(args.weights)
    text = render(weights, source_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(args.output)
    verify_module(args.output, weights)
    print(f"exported={args.output}")
    print(f"source_sha256={source_hash}")
    print("arrays_exact=True read_only=True runtime_filesystem_access=False")


if __name__ == "__main__":
    main()
