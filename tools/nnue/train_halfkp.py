"""Train the offline HalfKP residual with CUDA or an explicit CPU device."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import write_json
from .halfkp_data import (
    DEFAULT_DATA, DEFAULT_RUN, MAX_TRAIN_DEPTH, SPLIT_FILES, load_split, ranking_groups,
    sha256, split_audit,
)
from .halfkp_features import PERM, pack_boards
from .halfkp_model import HalfKP, export_weights, torch_indices
from .model import EVAL_SCALE, RESIDUAL_CLIP, mixed_targets


class DeviceSplit:
    def __init__(self, arrays: dict[str, np.ndarray], device: torch.device) -> None:
        kings, pieces = pack_boards(arrays["boards"])
        target, _ = mixed_targets(arrays["deep_evals"], arrays["outcomes"])
        self.kings = torch.from_numpy(kings).to(device)
        self.pieces = torch.from_numpy(pieces).to(device)
        self.sides = torch.from_numpy(arrays["sides"]).to(device=device, dtype=torch.long)
        self.base = torch.from_numpy(arrays["base_evals"]).to(device)
        self.deep = torch.from_numpy(arrays["deep_evals"]).to(device)
        self.target = torch.from_numpy(target).to(device)
        self.permutations = torch.tensor(PERM, device=device)

    def __len__(self) -> int:
        return len(self.base)

    def indices(self, rows: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
        return torch_indices(self.kings[rows], self.pieces[rows], views, self.permutations)


def augmented_batches(rows: int, batch_size: int, generator: torch.Generator,
                      views: int = len(PERM)):
    order = torch.randperm(rows * views, device=generator.device, generator=generator)
    for batch in order.split(batch_size):
        yield batch.remainder(rows), batch.div(rows, rounding_mode="floor")


def _optimizer(parameters, lr: float, decay: float, device: torch.device):
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=decay, fused=device.type == "cuda")


def _stable_bce(logits, target, reduction="mean"):
    # Same clipped-probability validation metric as the legacy NNUE trainer.
    prediction = logits.sigmoid().clamp(1e-7, 1.0 - 1e-7)
    return F.binary_cross_entropy(prediction, target, reduction=reduction)


@torch.no_grad()
def validation_nll(model: HalfKP, data: DeviceSplit, batch_size: int,
                   *, calibrating: bool = False) -> float:
    model.eval()
    nll = data.base.new_zeros(())
    for rows in torch.arange(len(data), device=data.base.device).split(batch_size):
        base, sides = data.base[rows], data.sides[rows]
        if not calibrating:
            base = base + model(data.indices(rows, torch.zeros_like(rows)), sides)
        nll += _stable_bce(model.logits(base, sides), data.target[rows], reduction="sum")
    return float(nll) / len(data)


def _record(history: list, stage: str, epoch: int, values: dict) -> None:
    if not all(math.isfinite(value) for value in values.values() if value is not None):
        raise FloatingPointError(f"Non-finite training metric at {stage} epoch {epoch}")
    history.append({"stage": stage, "epoch": epoch, **values})
    validation = "n/a" if values["val_nll"] is None else f'{values["val_nll"]:.6f}'
    print(f"{stage} epoch {epoch:02d}  train_loss={values['train_loss']:.6f}  "
          f"val_nll={validation}  delta_mean={values['delta_mean']:+.2f}  "
          f"delta_std={values['delta_std']:.2f}  sat_rate={values['sat_rate']:.2%}", flush=True)


def calibrate(model, train, validation, args, generator, history) -> None:
    model.requires_grad_(False)
    model.side_calib.requires_grad_(True)
    optimizer = _optimizer([model.side_calib], args.calibration_lr, 0.0, generator.device)
    for epoch in range(1, args.calibration_epochs + 1):
        total = train.base.new_zeros(())
        # D4 does not affect the base scores or calibration logits.
        for rows, _ in augmented_batches(len(train), args.batch_size, generator, views=1):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model.logits(train.base[rows], train.sides[rows]), train.target[rows],
            )
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(rows)
        _record(history, "calibration", epoch, {
            "train_loss": float(total) / len(train),
            "val_nll": validation_nll(model, validation, args.batch_size, calibrating=True),
            "delta_mean": 0.0, "delta_std": 0.0, "sat_rate": 0.0,
        })
    model.requires_grad_(True)


def _train_epoch(model, data, args, generator, optimizer, scaler,
                 groups: torch.Tensor | None = None) -> dict:
    model.train()
    amp = scaler.is_enabled()
    totals = data.base.new_zeros(4)
    group_width = 1 if groups is None else groups.shape[1]
    pool_size = len(data) if groups is None else len(groups)
    for rows, views in augmented_batches(pool_size, args.batch_size // group_width, generator):
        if groups is not None:
            rows = groups[rows].reshape(-1)
            views = views.repeat_interleave(group_width)
        indices = data.indices(rows, views)
        sides, base, target = data.sides[rows], data.base[rows], data.target[rows]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=generator.device.type, dtype=torch.float16, enabled=amp):
            residual = model(indices, sides)
        scores = base + residual
        bce = F.binary_cross_entropy_with_logits(model.logits(scores, sides), target)
        loss = bce + args.lambda_zc * (residual / model.cap).square().mean()
        if groups is not None:
            logits = scores.reshape(-1, group_width) / args.temperature
            labels = data.deep[rows].reshape(-1, group_width) / args.temperature
            rank = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(labels, dim=1),
                            reduction="batchmean")
            loss = loss + args.listwise_weight * rank
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        delta = residual.detach()
        totals += torch.stack((loss.detach() * len(rows), delta.sum(), delta.square().sum(),
                               (delta.abs() > 0.9 * model.cap).sum(dtype=torch.float32)))
    loss_mean, mean, mean_square, saturation = (totals / (pool_size * group_width * len(PERM))).tolist()
    return {"train_loss": loss_mean, "delta_mean": mean,
            "delta_std": math.sqrt(max(0.0, mean_square - mean ** 2)), "sat_rate": saturation}


def fit_pointwise(model, train, validation, args, generator, scaler, history) -> int:
    optimizer = _optimizer(model.parameters(), args.lr, args.weight_decay, generator.device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05,
    )
    best_nll = math.inf
    best_epoch, best_state = 0, None
    for epoch in range(1, args.epochs + 1):
        values = _train_epoch(model, train, args, generator, optimizer, scaler)
        values["val_nll"] = validation_nll(model, validation, args.batch_size)
        _record(history, "stage2", epoch, values)
        if values["val_nll"] < best_nll:
            best_nll, best_epoch = values["val_nll"], epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        scheduler.step()
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    return best_epoch


def fit_listwise(model, train, groups, args, generator, scaler, history) -> None:
    """Use fixed epochs and training groups only. Validation is reported after freezing."""
    optimizer = _optimizer(model.parameters(), args.listwise_lr, args.weight_decay, generator.device)
    for epoch in range(1, args.listwise_epochs + 1):
        values = _train_epoch(model, train, args, generator, optimizer, scaler, groups)
        _record(history, "stage3", epoch, {**values, "val_nll": None})


def _save(run_dir: Path, stage: str, model: HalfKP) -> dict:
    checkpoint, weights = run_dir / f"{stage}.pt", run_dir / f"{stage}.npz"
    torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "hidden": model.embedding.embedding_dim,
                "hidden2": model.hidden2.out_features if model.hidden2 is not None else 0,
                "cap": model.cap, "eval_scale": EVAL_SCALE}, checkpoint)
    export_weights(weights, model)
    return {kind: {"file": path.name, "sha256": sha256(path)}
            for kind, path in (("checkpoint", checkpoint), ("weights", weights))}


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch environment. Use a CUDA build "
                           "or pass --device cpu explicitly.")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {args.run_dir}. Choose a new --run-dir.")

    train_path = args.data_dir / SPLIT_FILES["train"]
    validation_path = args.data_dir / SPLIT_FILES["validation"]
    fingerprints = {name: sha256(path)
                    for name, path in (("train", train_path), ("validation", validation_path))}
    raw_train, original_count = load_split(train_path, training=True)
    raw_validation, _ = load_split(validation_path)
    audit = split_audit(raw_train, raw_validation)
    train_data = DeviceSplit(raw_train, device)
    validation = DeviceSplit(raw_validation, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    model = HalfKP(args.hidden, args.hidden2, args.cap).to(device)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"{name}  AMP={amp}  K={EVAL_SCALE:g}  C={args.cap:g}", flush=True)
    print(f"train={len(train_data):,}/{original_count:,}  "
          f"filtered={original_count - len(train_data):,}  "
          f"D4={len(train_data) * len(PERM):,}  validation={len(validation):,}", flush=True)

    calibrate(model, train_data, validation, args, generator, history)
    calibration = model.side_calib.detach().tolist()
    best_epoch = fit_pointwise(model, train_data, validation, args, generator, scaler, history)
    models = {"stage2": _save(args.run_dir, "stage2", model)}
    count = 0
    if args.listwise_epochs:
        groups = ranking_groups(raw_train, seed=args.seed)
        count = len(groups)
        if not count:
            raise ValueError("No eligible training game groups for listwise fine-tuning")
        print(f"stage3 groups={count:,} from filtered train only", flush=True)
        fit_listwise(model, train_data, torch.from_numpy(groups).to(device), args,
                     generator, scaler, history)
        models["stage3"] = _save(args.run_dir, "stage3", model)

    metrics = {
        "schema": 1, "training_complete": True, "device": name, "torch": torch.__version__,
        "config": {name: str(value) if isinstance(value, Path) else value
                   for name, value in vars(args).items()},
        "eval_scale": EVAL_SCALE, "amp": amp, "calibration_logits_D_A": calibration,
        "data": {"train_rows_original": original_count, "train_rows": len(train_data),
                 "filtered_rows": original_count - len(train_data), "max_depth": MAX_TRAIN_DEPTH,
                 "train_sha256": fingerprints["train"], "d4_views": len(PERM),
                 "validation_rows": len(validation), "validation_sha256": fingerprints["validation"],
                 **audit},
        "selection": {"stage2": {"best_epoch": best_epoch, "metric": "locked validation NLL"},
                      "stage3": {"training_game_groups": count, "selection": "fixed last epoch",
                                 "validation_used_during_fit": False},
                      "sealed_questions_used": False},
        "models": models, "epochs": history,
    }
    write_json(args.run_dir / "metrics.json", metrics)
    print(f"Frozen checkpoints: {args.run_dir.resolve()} (stage2 epoch {best_epoch})", flush=True)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true", help="use full float32 on CUDA")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32, help="0 disables the second hidden layer")
    parser.add_argument("--cap", type=float, default=RESIDUAL_CLIP)
    parser.add_argument("--epochs", type=int, default=16, help="maximum pointwise epochs")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--calibration-epochs", type=int, default=5)
    parser.add_argument("--calibration-lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-zc", type=float, default=0.025)
    parser.add_argument("--listwise-epochs", type=int, default=0)
    parser.add_argument("--listwise-lr", type=float, default=1e-4)
    parser.add_argument("--listwise-weight", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=20261127)
    args = parser.parse_args(argv)
    positive = (args.hidden, args.epochs, args.patience, args.calibration_epochs, args.cap,
                args.lr, args.calibration_lr, args.listwise_lr, args.temperature)
    nonnegative = (args.hidden2, args.listwise_epochs, args.weight_decay, args.lambda_zc,
                   args.listwise_weight)
    if (any(not math.isfinite(value) or value <= 0 for value in positive)
            or any(not math.isfinite(value) or value < 0 for value in nonnegative)
            or args.batch_size < 3):
        parser.error("Use positive dimensions, epoch limits and learning rates, with batch size >= 3.")
    try:
        device_type = torch.device(args.device).type
    except RuntimeError:
        device_type = None
    if device_type not in {"cpu", "cuda"}:
        parser.error("--device must be cpu, cuda or cuda:<index>")
    return args


if __name__ == "__main__":
    train(parse_args())
