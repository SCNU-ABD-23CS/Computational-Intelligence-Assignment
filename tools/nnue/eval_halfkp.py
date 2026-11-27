"""Evaluate all frozen HalfKP checkpoints in one sealed ranking report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .common import encode_board, write_json
from .halfkp_data import DEFAULT_DATA, DEFAULT_RUN, SPLIT_FILES, load_split, sha256
from .halfkp_features import d4_overlap, feature_indices, pack_boards
from .halfkp_model import HalfKP, load_weights, numpy_residual
from .model import EVAL_SCALE, mixed_targets
from .rank_questions import metric_group, prepare_candidates

_REPORT = "final_evaluation.json"
_QUESTIONS = "questions_400.json"
_LEGACY = {"val_nll": 0.5751187776, "top1": "133/377", "pairwise": "537.0/1065",
           "source": "tools/nnue/MEMBER5_RESULTS.md"}


def _nll(scores: np.ndarray, target: np.ndarray,
         calibration: np.ndarray | float = 0.0) -> float:
    logits = scores.astype(np.float64) / EVAL_SCALE + calibration
    prediction = np.clip(1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60))), 1e-7, 1.0 - 1e-7)
    return float(np.mean(-target * np.log(prediction) - (1.0 - target) * np.log1p(-prediction)))


def _frozen_models(run_dir: Path, metrics: dict):
    expected = {"stage2", "stage3"} if metrics["config"]["listwise_epochs"] else {"stage2"}
    if set(metrics["models"]) != expected:
        raise ValueError("Training metrics do not have all configured frozen checkpoints")
    for stage, files in metrics["models"].items():
        for file in files.values():
            if sha256(run_dir / file["file"]) != file["sha256"]:
                raise ValueError(f"{stage}: {file['file']} does not match the frozen training run")
        checkpoint = torch.load(run_dir / files["checkpoint"]["file"],
                                map_location="cpu", weights_only=True)
        if checkpoint["eval_scale"] != EVAL_SCALE:
            raise ValueError(f"{stage}: unexpected K in checkpoint")
        model = HalfKP(checkpoint["hidden"], checkpoint["hidden2"], checkpoint["cap"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        yield stage, model, load_weights(run_dir / files["weights"]["file"])


@torch.no_grad()
def _checked_residual(model, weights, boards, sides) -> tuple[np.ndarray, float]:
    indices = feature_indices(*pack_boards(boards))
    exported = numpy_residual(weights, boards, sides)
    tensor = model(torch.from_numpy(indices), torch.from_numpy(sides.astype(np.int64))).numpy()
    difference = float(np.max(np.abs(exported - tensor)))
    if difference > 0.002:
        raise ValueError(f"NumPy export disagrees with PyTorch checkpoint by {difference:g}")
    return exported, difference


def _ranking(questions, static, scores, deep) -> dict:
    main = [index for index, row in enumerate(questions) if not row["tie_top2"]]
    ties = [index for index, row in enumerate(questions) if row["tie_top2"]]
    return {
        name: metric_group([questions[index] for index in indices],
                           static[indices], scores[indices], deep[indices])
        for name, indices in (("main", main), ("tie_top2_information_only", ties)) if indices
    }


def _sealed_questions(data_dir: Path) -> tuple[list[dict], str]:
    raw = (data_dir / _QUESTIONS).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        locked = manifest.get("files", {}).get("questions", {}).get("sha256")
        if locked and digest != locked:
            raise ValueError("The sealed question file does not match the dataset manifest")
    payload = json.loads(raw)
    if payload.get("count") != 400 or len(payload["questions"]) != 400:
        raise ValueError("The sealed ranking protocol expects exactly 400 questions")
    return payload["questions"], digest


def _evaluate(data_dir: Path, run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Complete training before sealed evaluation: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("training_complete") is not True or metrics["selection"]["sealed_questions_used"]:
        raise ValueError("The run must be frozen before evaluation")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    models = list(_frozen_models(run_dir, metrics))
    train_path = data_dir / SPLIT_FILES["train"]
    validation_path = data_dir / SPLIT_FILES["validation"]
    for name, path in (("train", train_path), ("validation", validation_path)):
        if sha256(path) != metrics["data"][f"{name}_sha256"]:
            raise ValueError(f"{name} data does not match the frozen training run")
    train, _ = load_split(train_path, training=True)
    validation, _ = load_split(validation_path)
    sides = validation["sides"]
    base = validation["base_evals"]
    targets, _ = mixed_targets(validation["deep_evals"], validation["outcomes"])
    base_val = {
        "rows": len(base), "raw_static_nll": _nll(base, targets),
        "static_with_stage1_calibration_nll": _nll(
            base, targets, np.asarray(metrics["calibration_logits_D_A"])[sides],
        ),
        "provided_v4_nll_reference": 0.5739,
        "reference_note": "MEMBER5_RESULTS.md attributes 0.5738794516 to the old pointwise NNUE",
    }
    result = {}
    for stage, model, weights in models:
        residual, difference = _checked_residual(model, weights, validation["boards"], sides)
        scores = base + residual
        result[stage] = {
            "validation": {
                "val_nll": _nll(scores, targets, model.side_calib.detach().numpy()[sides]),
                "uncalibrated_nll": _nll(scores, targets),
                "delta_mean": float(residual.mean()), "delta_std": float(residual.std()),
                "sat_rate": float(np.mean(np.abs(residual) > 0.9 * model.cap)),
                "max_pt_npz_delta_difference": difference,
            },
        }

    # Access the sealed file only after checkpoint integrity and validation are complete.
    questions, digest = _sealed_questions(data_dir)
    child_boards, static, deep, audit = prepare_candidates(questions)
    roots = np.stack([encode_board(q["board"]) for q in questions]).astype(np.int8)
    root_sides = np.ones(len(questions), dtype=np.int8)
    child_sides = np.zeros(len(child_boards), dtype=np.int8)
    audit.update(
        train_game_ids_matching_question_games=len(
            set(train["game_ids"]).intersection(q["game_id"] for q in questions)
        ),
        train_rows_matching_question_roots=d4_overlap(
            train["boards"], train["sides"], roots, root_sides,
        ),
        train_rows_matching_question_candidates=d4_overlap(
            train["boards"], train["sides"], child_boards, child_sides,
        ),
        question_roots_present_in_locked_validation=d4_overlap(
            roots, root_sides, validation["boards"], validation["sides"],
        ),
    )
    for stage, model, weights in models:
        residual, difference = _checked_residual(model, weights, child_boards, child_sides)
        result[stage].update(
            _ranking(questions, static, static + residual.reshape(-1, 3), deep),
            max_candidate_pt_npz_delta_difference=difference,
        )
    return {
        "schema": 1,
        "questions": {"file": _QUESTIONS, "sha256": digest, "count": len(questions),
                      "training_or_model_selection_use": False, "d4_augmentation": False},
        "audit": audit, "static_validation": base_val, "legacy_reference": _LEGACY,
        "models": result,
    }


def final_report(data_dir: Path, run_dir: Path) -> dict:
    output = run_dir / _REPORT
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    if not (run_dir / "metrics.json").is_file():
        raise FileNotFoundError(f"Complete training before sealed evaluation: {run_dir}")
    lock = run_dir / ".final_evaluation.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        if output.is_file():
            return json.loads(output.read_text(encoding="utf-8"))
        report = _evaluate(data_dir, run_dir)
        write_json(output, report)
        return report
    finally:
        lock.unlink()


def _print_report(report: dict, path: Path) -> None:
    print("Model                         val NLL   Top-1     Pairwise")
    print("---------------------------------------------------------------")

    def row(name, nll, ranking, count):
        pairs = ranking["pairwise"]
        print(f"{name:28s} {nll:.6f}   {ranking['top1_correct']}/{count}   "
              f"{pairs['correct_credit']:.1f}/{pairs['compared_pairs']}")

    main = report["models"]["stage2"]["main"]
    row("Team19-v4 (raw static)", report["static_validation"]["raw_static_nll"],
        main["static"], main["questions"])
    print(f"{'Member 5 NNUE (reference)':28s} {_LEGACY['val_nll']:.6f}   "
          f"{_LEGACY['top1']}   {_LEGACY['pairwise']}")
    for stage, result in report["models"].items():
        main = result["main"]
        row(f"HalfKP {stage}", result["validation"]["val_nll"], main["nnue"], main["questions"])
    print("The quoted 0.5739 NLL is the legacy pointwise NNUE reference.")
    print(f"Report saved to {path.resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    _print_report(final_report(args.data_dir, args.run_dir), args.run_dir / _REPORT)


if __name__ == "__main__":
    main()
