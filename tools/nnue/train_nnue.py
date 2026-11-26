"""Train the 243->32->2 bounded residual NNUE on the sealed volume split."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

# Keep offline training single-process and single-threaded as well as the player.
import os
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import NNUE_RESULTS, write_json
from .model import (
    EVAL_SCALE, FEATURES, HEADS, HIDDEN, RESIDUAL_CLIP, augment_train, deep_regression_mask,
    encode_boards, mixed_targets,
)


FIELDS = {"boards", "sides", "base_evals", "deep_evals", "outcomes", "game_ids"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != FIELDS:
            raise ValueError(f"{path} fields differ from the six-field contract")
        arrays = {key: np.array(data[key], copy=True) for key in FIELDS}
    rows = len(arrays["boards"])
    if arrays["boards"].shape != (rows, 81):
        raise ValueError(f"Invalid boards shape in {path}")
    for key in FIELDS - {"boards"}:
        if arrays[key].shape != (rows,):
            raise ValueError(f"Invalid {key} shape in {path}")
    return arrays


class ResidualNNUE(nn.Module):
    """Two side-selected heads, both always producing attacker-perspective residuals."""

    def __init__(self, residual_clip: float, eval_scale: float):
        super().__init__()
        self.hidden = nn.Linear(FEATURES, HIDDEN)
        self.heads = nn.Linear(HIDDEN, HEADS)
        # Training/validation-only global calibration in logit space. This absorbs
        # side-specific base bias without consuming the bounded residual budget.
        self.side_calib = nn.Parameter(torch.zeros(HEADS))
        self.residual_clip = float(residual_clip)
        self.eval_scale = float(eval_scale)
        # Dimensionless heads start near zero without tying all 32 hidden units
        # to exactly the same gradient. export_weights applies residual_clip so
        # deployment keeps the original score-domain NumPy/Numba contract.
        nn.init.normal_(self.heads.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.heads.bias)

    def forward(self, features, sides, base_evals):
        hidden = torch.clamp(self.hidden(features), 0.0, 1.0)
        both_heads = self.heads(hidden)
        selected = both_heads.gather(1, sides.reshape(-1, 1)).squeeze(1)
        # A smooth bound avoids the zero-gradient region reached by the first
        # linear-scale experiment, whose candidate residuals all clipped to -C.
        residual = self.residual_clip * torch.tanh(selected)
        prediction = torch.sigmoid(
            (base_evals + residual) / self.eval_scale + self.side_calib[sides]
        )
        return prediction, residual


def batches(arrays: dict[str, np.ndarray], indices: list[int], batch_size: int):
    for start in range(0, len(indices), batch_size):
        selected = np.asarray(indices[start:start + batch_size], dtype=np.int64)
        features = torch.from_numpy(encode_boards(arrays["boards"][selected]))
        sides = torch.from_numpy(arrays["sides"][selected].astype(np.int64))
        base = torch.from_numpy(arrays["base_evals"][selected].astype(np.float32))
        target = torch.from_numpy(arrays["targets"][selected].astype(np.float32))
        yield features, sides, base, target


@torch.no_grad()
def evaluate(model: ResidualNNUE, arrays: dict[str, np.ndarray], batch_size: int = 4096) -> dict:
    model.eval()
    total = len(arrays["boards"])
    smooth_l1 = 0.0
    nll = 0.0
    residual_absolute = 0.0
    indices = list(range(total))
    for features, sides, base, target in batches(arrays, indices, batch_size):
        prediction, residual = model(features, sides, base)
        smooth_l1 += float(F.smooth_l1_loss(prediction, target, reduction="sum"))
        clamped = torch.clamp(prediction, 1e-7, 1.0 - 1e-7)
        nll += float(F.binary_cross_entropy(clamped, target, reduction="sum"))
        residual_absolute += float(torch.sum(torch.abs(residual)))
    return {
        "rows": total,
        "smooth_l1": smooth_l1 / total,
        "nll": nll / total,
        "mean_abs_residual": residual_absolute / total,
    }


def calibration_report(model: ResidualNNUE) -> dict:
    """Report D/A calibration in logits and equivalent raw-score units."""
    values = model.side_calib.detach().cpu().numpy()
    return {
        "scope": "training_and_validation_only",
        "exported": False,
        "used_by_rank_questions": False,
        "logits": {"D": float(values[0]), "A": float(values[1])},
        "score_equivalent": {
            "D": float(values[0] * model.eval_scale),
            "A": float(values[1] * model.eval_scale),
        },
    }


def export_weights(model: ResidualNNUE, path: Path) -> None:
    arrays = {
        "hidden_weight": model.hidden.weight.detach().cpu().numpy().astype(np.float32),
        "hidden_bias": model.hidden.bias.detach().cpu().numpy().astype(np.float32),
        "head_weight": (
            model.heads.weight.detach().cpu().numpy() * model.residual_clip
        ).astype(np.float32),
        "head_bias": (
            model.heads.bias.detach().cpu().numpy() * model.residual_clip
        ).astype(np.float32),
        "residual_clip": np.array(model.residual_clip, dtype=np.float32),
        "eval_scale": np.array(model.eval_scale, dtype=np.float32),
        "feature_layout": np.array("cell-major square*3+plane; planes attacker,defender,king"),
        "output_activation": np.array("tanh"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _mask_report(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    deep = np.abs(arrays["deep_evals"].astype(np.float64))
    return {
        "rows": len(deep),
        "deep_regression_rows": int(np.sum(mask)),
        "outcome_only_mate_rows": int(np.sum(deep >= 90_000.0)),
        "outcome_only_abs_3000_rows": int(np.sum(np.isclose(deep, 3_000.0, rtol=0.0, atol=1e-6))),
        "depth_field_available": False,
    }


def train(args: argparse.Namespace) -> dict:
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience < 0:
        raise ValueError("epochs/batch_size must be positive and patience non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid AdamW settings")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    directory = NNUE_RESULTS / args.tag
    train_path = directory / "team19_nnue_train.npz"
    val_path = directory / "team19_nnue_val.npz"
    raw_train = load_split(train_path)
    validation = load_split(val_path)
    if set(raw_train["game_ids"]) & set(validation["game_ids"]):
        raise ValueError("Game ids cross the train/validation split")

    train_arrays = augment_train(raw_train)
    train_arrays["targets"], train_mask = mixed_targets(
        train_arrays["deep_evals"], train_arrays["outcomes"],
        scale=args.eval_scale, outcome_weight=args.outcome_weight,
    )
    validation["targets"], val_mask = mixed_targets(
        validation["deep_evals"], validation["outcomes"],
        scale=args.eval_scale, outcome_weight=args.outcome_weight,
    )
    # Validation is intentionally not augmented.
    if len(validation["boards"]) != len(load_split(val_path)["boards"]):
        raise AssertionError("Validation D4 augmentation is forbidden")

    output = Path(args.output) if args.output else (
        directory / f"member5-{args.run_id}"
    )
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {output}; choose a new --run-id")
    output.mkdir(parents=True, exist_ok=True)

    model = ResidualNNUE(args.residual_clip, args.eval_scale)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    best_state = copy.deepcopy(model.state_dict())
    best_nll = math.inf
    best_epoch = 0
    stale_epochs = 0
    history_path = output / "history.jsonl"
    history_path.write_text("", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_arrays["boards"]), generator=generator).tolist()
        train_loss = 0.0
        for features, sides, base, target in batches(
                train_arrays, order, args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(features, sides, base)
            loss = F.smooth_l1_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(target)
        metrics = evaluate(model, validation)
        calibration = calibration_report(model)
        record = {
            "epoch": epoch,
            "train_smooth_l1": train_loss / len(order),
            "validation": metrics,
            "side_calibration": calibration,
            "selected_by": "validation_nll_only",
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            f"epoch={epoch:02d} train={record['train_smooth_l1']:.7f} "
            f"val={metrics['smooth_l1']:.7f} nll={metrics['nll']:.7f} "
            f"mean|residual|={metrics['mean_abs_residual']:.3f} "
            f"calibD={calibration['score_equivalent']['D']:.1f} "
            f"calibA={calibration['score_equivalent']['A']:.1f}", flush=True,
        )
        if metrics["nll"] < best_nll - args.min_delta:
            best_nll = metrics["nll"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs > args.patience:
                break

    model.load_state_dict(best_state)
    weights_path = output / "best_weights.npz"
    export_weights(model, weights_path)
    final_metrics = evaluate(model, validation)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True,
    ).strip()
    config = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_train_start": commit,
        "seed": args.seed,
        "device": "cpu",
        "threads": 1,
        "network": {
            "input": FEATURES, "hidden": HIDDEN, "heads": HEADS,
            "activation": "clipped_relu_[0,1]",
            "output": "side_selected_attacker_residual",
            "training_parameterization": "sigmoid((base + C*tanh(head))/K + side_calib[side])",
            "deployment_parameterization": "residual_clip * tanh(score_domain_head / residual_clip)",
            "export": "head_weight and head_bias multiplied by residual_clip",
            "head_initialization": "Normal(mean=0, std=0.001); zero bias",
        },
        "feature_layout": "cell-major square*3+plane; planes attacker,defender,king; codes 0/4 empty",
        "residual_clip": args.residual_clip,
        "eval_scale": args.eval_scale,
        "side_calibration": calibration_report(model),
        "outcome_weight": args.outcome_weight,
        "loss": "SmoothL1",
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate,
                      "weight_decay": args.weight_decay},
        "batch_size": args.batch_size,
        "requested_epochs": args.epochs,
        "completed_epochs": len(history_path.read_text(encoding="utf-8").splitlines()),
        "selected_epoch": best_epoch,
        "early_stopping": {"metric": "validation NLL", "patience": args.patience,
                           "min_delta": args.min_delta, "questions_used": False},
        "d4": {"train_copies": 8, "validation_copies": 1, "questions_copies": 0},
        "class_weights": False,
        "resampling": False,
        "label_policy": (
            "abs(deep)>=90000 and abs(deep)==3000 retain outcome only; "
            "NPZ has no depth field, so the abs==3000 mask is deliberately conservative"
        ),
        "masks": {"train_before_d4": _mask_report(raw_train, deep_regression_mask(raw_train["deep_evals"])),
                  "validation": _mask_report(validation, val_mask)},
        "data": {
            "tag": args.tag,
            "train": {"path": str(train_path), "sha256": sha256(train_path),
                      "rows": len(raw_train["boards"])},
            "validation": {"path": str(val_path), "sha256": sha256(val_path),
                           "rows": len(validation["boards"])},
        },
        "environment": {"python": os.sys.version.split()[0], "numpy": np.__version__,
                        "torch": torch.__version__},
        "weights": {"file": weights_path.name, "sha256": sha256(weights_path)},
        "best_validation": final_metrics,
    }
    write_json(output / "training_config.json", config)
    write_json(output / "validation_metrics.json", {
        "selected_epoch": best_epoch,
        "selection_metric": "validation_nll_only",
        "best": final_metrics,
        "side_calibration": calibration_report(model),
        "train_rows_after_d4": len(train_arrays["boards"]),
        "validation_rows_after_no_d4": len(validation["boards"]),
        "weights_sha256": sha256(weights_path),
    })
    print(json.dumps({"selected_epoch": best_epoch, "best_validation": final_metrics,
                      "side_calibration": calibration_report(model),
                      "weights": str(weights_path)}, indent=2), flush=True)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="volume1200")
    parser.add_argument("--run-id", default="c750-k6000-l0p5-tanh-sidecalib-seed20261126")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-clip", type=float, default=RESIDUAL_CLIP)
    parser.add_argument("--eval-scale", type=float, default=EVAL_SCALE)
    parser.add_argument("--outcome-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    args = parser.parse_args()
    if args.residual_clip <= 0 or args.eval_scale <= 0:
        parser.error("residual clip and eval scale must be positive")
    train(args)


if __name__ == "__main__":
    main()
