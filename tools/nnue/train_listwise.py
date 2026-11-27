"""Fine-tune the bounded residual with train-only three-candidate listwise labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import Team19_engine as engine

from .common import NNUE_RESULTS, board_rows, encode_board, write_json
from .model import (
    EVAL_SCALE, HEADS, RESIDUAL_CLIP, augment_train, d4_boards, encode_boards,
    load_weights, mixed_targets, numpy_residual,
)
from .rank_questions import metric_group
from .train_nnue import ResidualNNUE, batches, evaluate, export_weights, load_split, sha256


def questions_to_arrays(questions: list[dict]) -> dict[str, np.ndarray]:
    boards = []
    static = []
    deep = []
    for question in questions:
        root = board_rows(question["board"])
        for candidate in question["candidates"]:
            child = engine.apply_move(root, candidate["move"], True)
            boards.append(encode_board(child))
            static.append(float(candidate["static_attacker_eval"]))
            deep.append(float(candidate["deep_attacker_eval"]))
    return {
        "boards": np.stack(boards).astype(np.int8).reshape(-1, 3, 81),
        "static": np.asarray(static, dtype=np.float32).reshape(-1, 3),
        "deep": np.asarray(deep, dtype=np.float32).reshape(-1, 3),
        "game_ids": np.asarray([row["game_id"] for row in questions], dtype=str),
    }


def split_by_game(questions: list[dict], seed: int, dev_fraction: float):
    games = sorted({row["game_id"] for row in questions})
    ordered = sorted(games, key=lambda game: hashlib.sha256(
        f"listwise-split:{seed}:{game}".encode("utf-8")
    ).hexdigest())
    dev_count = max(1, round(len(games) * dev_fraction))
    if dev_count >= len(games):
        raise ValueError("Not enough games for a disjoint listwise split")
    dev_games = set(ordered[:dev_count])
    train_questions = [row for row in questions if row["game_id"] not in dev_games]
    dev_questions = [row for row in questions if row["game_id"] in dev_games]
    if not train_questions or not dev_questions:
        raise ValueError("Empty listwise train or development split")
    return train_questions, dev_questions


def augment_listwise(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = len(arrays["game_ids"])
    flat = arrays["boards"].reshape(-1, 81)
    boards = d4_boards(flat).reshape(8, count, 3, 81).reshape(-1, 3, 81)
    return {
        "boards": boards,
        "static": np.tile(arrays["static"], (8, 1)),
        "deep": np.tile(arrays["deep"], (8, 1)),
        "game_ids": np.tile(arrays["game_ids"], 8),
    }


def source_model(weights_path: Path, validation_path: Path) -> ResidualNNUE:
    weights = load_weights(weights_path)
    model = ResidualNNUE(float(weights["residual_clip"]), float(weights["eval_scale"]))
    with torch.no_grad():
        model.hidden.weight.copy_(torch.from_numpy(weights["hidden_weight"]))
        model.hidden.bias.copy_(torch.from_numpy(weights["hidden_bias"]))
        model.heads.weight.copy_(torch.from_numpy(
            weights["head_weight"] / float(weights["residual_clip"])
        ))
        model.heads.bias.copy_(torch.from_numpy(
            weights["head_bias"] / float(weights["residual_clip"])
        ))
        calibration = json.loads(validation_path.read_text(encoding="utf-8"))["side_calibration"]
        model.side_calib.copy_(torch.tensor([
            calibration["logits"]["D"], calibration["logits"]["A"],
        ], dtype=torch.float32))
    return model


@torch.no_grad()
def listwise_scores(model: ResidualNNUE, arrays: dict[str, np.ndarray], batch_size=256):
    output = []
    for start in range(0, len(arrays["game_ids"]), batch_size):
        boards = arrays["boards"][start:start + batch_size]
        base = arrays["static"][start:start + batch_size]
        count = len(base)
        features = torch.from_numpy(encode_boards(boards.reshape(-1, 81)))
        sides = torch.zeros(count * 3, dtype=torch.int64)
        flat_base = torch.from_numpy(base.reshape(-1))
        _, residual = model(features, sides, flat_base)
        scores = (flat_base + residual).reshape(count, 3).numpy()
        output.append(scores)
    return np.concatenate(output, axis=0)


def development_metrics(model, questions, arrays):
    scores = listwise_scores(model, arrays)
    indices = [index for index, row in enumerate(questions) if not row["tie_top2"]]
    selected_questions = [questions[index] for index in indices]
    return metric_group(
        selected_questions, arrays["static"][indices], scores[indices], arrays["deep"][indices],
    )


def train(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    directory = NNUE_RESULTS / args.tag
    questions_path = Path(args.questions)
    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = payload["questions"]
    if not payload.get("train_npz_only") or payload.get("sealed_questions_used"):
        raise ValueError("Listwise questions must be train-only and sealed-question-free")
    train_questions, dev_questions = split_by_game(questions, args.seed, args.dev_fraction)
    train_arrays = augment_listwise(questions_to_arrays(train_questions))
    dev_arrays = questions_to_arrays(dev_questions)
    pointwise = augment_train(load_split(directory / "team19_nnue_train.npz"))
    pointwise["targets"], _ = mixed_targets(
        pointwise["deep_evals"], pointwise["outcomes"], scale=args.eval_scale,
        outcome_weight=args.outcome_weight,
    )
    validation = load_split(directory / "team19_nnue_val.npz")
    validation["targets"], _ = mixed_targets(
        validation["deep_evals"], validation["outcomes"], scale=args.eval_scale,
        outcome_weight=args.outcome_weight,
    )
    weights_path = Path(args.weights)
    source_validation_path = Path(args.source_validation)
    model = source_model(weights_path, source_validation_path)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    output = Path(args.output) if args.output else directory / (
        "member5-c750-k6000-l0p5-tanh-precalib5-center1e3-listwise20-seed20261127"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Listwise output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    best_state = None
    best_epoch = 0
    best_key = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_arrays["game_ids"]), generator=generator).tolist()
        totals = np.zeros(4, dtype=np.float64)
        for start in range(0, len(order), args.batch_size):
            selected_indices = np.asarray(order[start:start + args.batch_size], dtype=np.int64)
            boards = train_arrays["boards"][selected_indices]
            static = train_arrays["static"][selected_indices]
            deep = train_arrays["deep"][selected_indices]
            count = len(selected_indices)
            features = torch.from_numpy(encode_boards(boards.reshape(-1, 81)))
            sides = torch.zeros(count * 3, dtype=torch.int64)
            flat_static = torch.from_numpy(static.reshape(-1))
            _, residual, selected = model(
                features, sides, flat_static, return_preactivation=True,
            )
            scores = (flat_static + residual).reshape(count, 3) / args.temperature
            labels = torch.from_numpy(deep)
            log_targets = (labels - torch.max(labels, dim=1, keepdim=True).values) / args.temperature
            target = torch.softmax(log_targets, dim=1)
            listwise_ce = -torch.sum(target * F.log_softmax(scores, dim=1), dim=1).mean()
            replay_indices = torch.randint(
                len(pointwise["boards"]), (count,), generator=generator,
            ).tolist()
            replay_features, replay_sides, replay_base, replay_target = next(
                batches(pointwise, replay_indices, count)
            )
            replay_prediction, _, replay_selected = model(
                replay_features, replay_sides, replay_base, return_preactivation=True,
            )
            replay_loss = F.smooth_l1_loss(replay_prediction, replay_target)
            center = torch.cat((selected, replay_selected)).square().mean()
            objective = (
                listwise_ce + args.replay_weight * replay_loss + args.center_weight * center
            )
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Non-finite listwise loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            totals += np.array([
                float(objective.detach()), float(listwise_ce.detach()),
                float(replay_loss.detach()), float(center.detach()),
            ]) * count
        dev = development_metrics(model, dev_questions, dev_arrays)
        val = evaluate(model, validation)
        eligible = dev["delta"]["top1"] > 0.0 and dev["delta"]["pairwise"] > 0.0
        record = {
            "epoch": epoch,
            "train_objective": float(totals[0] / len(order)),
            "train_listwise_ce": float(totals[1] / len(order)),
            "train_replay_smooth_l1": float(totals[2] / len(order)),
            "train_center_mse": float(totals[3] / len(order)),
            "development": dev,
            "original_validation": val,
            "development_eligible": eligible,
            "sealed_400_used": False,
        }
        with history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(
            f"epoch={epoch:02d} objective={record['train_objective']:.6f} "
            f"dev_top1={dev['nnue']['top1_correct']}/{dev['questions']} "
            f"dev_pair={dev['nnue']['pairwise']['correct_credit']:.1f} "
            f"eligible={eligible} val_nll={val['nll']:.7f}", flush=True,
        )
        if eligible:
            key = (
                dev["nnue"]["top1_correct"], dev["nnue"]["pairwise"]["correct_credit"],
                -dev["delta"]["top1"],
            )
            if best_key is None or key[:2] > best_key[:2]:
                best_key = key
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
    source_validation = json.loads(source_validation_path.read_text(encoding="utf-8"))
    config = {
        "schema": 1,
        "seed": args.seed,
        "source_weights": {"path": str(weights_path), "sha256": sha256(weights_path)},
        "source_validation_nll": source_validation["best"]["nll"],
        "questions": {"path": str(questions_path), "sha256": sha256(questions_path),
                      "train_npz_only": True, "sealed_400_used": False},
        "split": {"method": "whole game_id SHA-256 order", "dev_fraction": args.dev_fraction,
                  "train_questions": len(train_questions), "dev_questions": len(dev_questions),
                  "train_after_d4": len(train_arrays["game_ids"]),
                  "dev_after_no_d4": len(dev_arrays["game_ids"])},
        "loss": {
            "listwise": "cross_entropy(softmax((static+residual)/T), softmax(deep/T))",
            "temperature": args.temperature,
            "replay_weight": args.replay_weight,
            "replay": "original D4 pointwise SmoothL1",
            "center": "center_weight * mean(concat(listwise,replay selected_pre_tanh)^2)",
            "center_weight": args.center_weight,
        },
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate,
                      "weight_decay": args.weight_decay},
        "batch_size": args.batch_size,
        "completed_epochs": args.epochs,
        "selection": "strict dev Top-1 and Pairwise improvement; then max Top-1 and Pairwise",
        "selected_epoch": best_epoch or None,
        "residual_clip": RESIDUAL_CLIP,
        "eval_scale": EVAL_SCALE,
        "heads": HEADS,
    }
    write_json(output / "training_config.json", config)
    if best_state is None:
        write_json(output / "validation_metrics.json", {
            "selected_epoch": None, "reason": "no epoch strictly improved both development metrics",
            "sealed_400_used": False,
        })
        print(json.dumps({"selected_epoch": None, "output": str(output)}, indent=2))
        raise SystemExit(2)
    model.load_state_dict(best_state)
    export_weights(model, output / "best_weights.npz")
    final_dev = development_metrics(model, dev_questions, dev_arrays)
    final_val = evaluate(model, validation)
    write_json(output / "validation_metrics.json", {
        "selected_epoch": best_epoch,
        "selection_metric": "train_only_development_top1_then_pairwise",
        "development": final_dev,
        "original_validation": final_val,
        "side_calibration_scope": "training_and_validation_only_not_exported",
        "weights_sha256": sha256(output / "best_weights.npz"),
        "sealed_400_used": False,
    })
    print(json.dumps({"selected_epoch": best_epoch, "development": final_dev,
                      "original_validation": final_val,
                      "weights": str(output / "best_weights.npz")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="volume1200")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source-validation", required=True)
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1000.0)
    parser.add_argument("--replay-weight", type=float, default=1.0)
    parser.add_argument("--center-weight", type=float, default=1e-3)
    parser.add_argument("--eval-scale", type=float, default=EVAL_SCALE)
    parser.add_argument("--outcome-weight", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.dev_fraction < 1.0:
        parser.error("dev fraction must be in (0,1)")
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size and learning rate must be positive")
    if args.temperature <= 0 or args.replay_weight < 0 or args.center_weight < 0:
        parser.error("invalid listwise loss settings")
    train(args)


if __name__ == "__main__":
    main()
