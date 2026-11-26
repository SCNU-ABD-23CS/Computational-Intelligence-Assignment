"""Evaluate exported residual weights on the sealed 400-question ranking set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import Team19_engine as engine

from .common import NNUE_RESULTS, NativeEvaluator, board_rows, encode_board, write_json
from .model import load_weights, numpy_hidden, numpy_residual


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_candidates(questions: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply each attacker move and evaluate the child with the defender head."""
    boards = []
    static = []
    deep = []
    evaluator = NativeEvaluator()
    max_static_discrepancy = 0.0
    for question in questions:
        if question["side_to_move"] != "A":
            raise ValueError("The sealed ranking contract currently requires attacker roots")
        root = board_rows(question["board"])
        for candidate in question["candidates"]:
            child = engine.apply_move(root, candidate["move"], True)
            boards.append(encode_board(child))
            static.append(float(candidate["static_attacker_eval"]))
            deep.append(float(candidate["deep_attacker_eval"]))
            recomputed = evaluator.evaluate(child, "D")
            max_static_discrepancy = max(
                max_static_discrepancy, abs(recomputed - float(candidate["static_attacker_eval"])),
            )
    audit = {
        "candidate_positions": len(boards),
        "side_selected_for_every_child": "D",
        "training_only_side_calibration_applied": False,
        "side_calibration_note": "constant within each question; deliberately excluded from ranking",
        "max_recomputed_static_discrepancy": max_static_discrepancy,
    }
    if max_static_discrepancy > 0.01:
        raise ValueError("Question static scores do not match the frozen native base")
    return (
        np.stack(boards).astype(np.int8),
        np.asarray(static, dtype=np.float32).reshape(-1, 3),
        np.asarray(deep, dtype=np.float32).reshape(-1, 3),
        audit,
    )


def _top1(scores: np.ndarray, questions: list[dict]) -> tuple[float, int]:
    correct = 0
    for row, question in zip(scores, questions):
        # Candidate order is the deterministic tie-break; no question is used for tuning.
        predicted = max(range(3), key=lambda index: (float(row[index]), -index))
        correct += predicted == question["ranking"][0]
    return correct / len(questions), correct


def _pairwise(scores: np.ndarray, deep: np.ndarray) -> dict:
    correct_credit = 0.0
    compared = 0
    predicted_ties = 0
    excluded_label_ties = 0
    for row, labels in zip(scores, deep):
        for first in range(3):
            for second in range(first + 1, 3):
                label_delta = float(labels[first] - labels[second])
                if math.isclose(label_delta, 0.0, rel_tol=0.0, abs_tol=1e-6):
                    excluded_label_ties += 1
                    continue
                prediction_delta = float(row[first] - row[second])
                compared += 1
                if math.isclose(prediction_delta, 0.0, rel_tol=0.0, abs_tol=1e-7):
                    predicted_ties += 1
                    correct_credit += 0.5
                else:
                    correct_credit += prediction_delta * label_delta > 0.0
    return {
        "accuracy": correct_credit / compared if compared else None,
        "correct_credit": correct_credit,
        "compared_pairs": compared,
        "excluded_label_ties": excluded_label_ties,
        "predicted_ties_half_credit": predicted_ties,
    }


def metric_group(questions: list[dict], static: np.ndarray, nnue: np.ndarray,
                 deep: np.ndarray) -> dict:
    static_top1, static_correct = _top1(static, questions)
    nnue_top1, nnue_correct = _top1(nnue, questions)
    static_pairs = _pairwise(static, deep)
    nnue_pairs = _pairwise(nnue, deep)
    return {
        "questions": len(questions),
        "static": {"top1": static_top1, "top1_correct": static_correct,
                   "pairwise": static_pairs},
        "nnue": {"top1": nnue_top1, "top1_correct": nnue_correct,
                 "pairwise": nnue_pairs},
        "delta": {"top1": nnue_top1 - static_top1,
                  "pairwise": nnue_pairs["accuracy"] - static_pairs["accuracy"]},
    }


def evaluate(weights_path: Path, questions_path: Path) -> dict:
    payload = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = payload["questions"]
    if len(questions) != 400 or payload.get("count") != 400:
        raise ValueError("This gate expects exactly the sealed 400 questions")
    flat_boards, static, deep, audit = prepare_candidates(questions)
    weights = load_weights(weights_path)
    # Every child has defender to move (side code 0), but both heads remain attacker-positive.
    residual = numpy_residual(weights, flat_boards, np.zeros(len(flat_boards), dtype=np.int8))
    bounded = residual.reshape(-1, 3)
    hidden = numpy_hidden(weights, flat_boards)
    nnue = static + bounded
    main_indices = [index for index, row in enumerate(questions) if not row["tie_top2"]]
    tie_indices = [index for index, row in enumerate(questions) if row["tie_top2"]]
    main_questions = [questions[index] for index in main_indices]
    tie_questions = [questions[index] for index in tie_indices]
    main = metric_group(main_questions, static[main_indices], nnue[main_indices], deep[main_indices])
    ties = metric_group(tie_questions, static[tie_indices], nnue[tie_indices], deep[tie_indices])
    gate = {
        "top1_strictly_better": main["delta"]["top1"] > 0.0,
        "pairwise_strictly_better": main["delta"]["pairwise"] > 0.0,
    }
    gate["passed"] = all(gate.values())
    records = []
    for index, question in enumerate(questions):
        records.append({
            "id": question["id"],
            "tie_top2": question["tie_top2"],
            "label_ranking": question["ranking"],
            "static_order": sorted(range(3), key=lambda j: (-float(static[index, j]), j)),
            "nnue_order": sorted(range(3), key=lambda j: (-float(nnue[index, j]), j)),
            "residuals": [float(value) for value in bounded[index]],
        })
    residual_ranges = np.ptp(bounded, axis=1)
    one_by_unit = np.mean(hidden >= 1.0, axis=0)
    diagnostics = {
        "mean_abs_residual": float(np.mean(np.abs(bounded))),
        "residual_range_per_question": {
            "median": float(np.median(residual_ranges)),
            "p90": float(np.percentile(residual_ranges, 90)),
            "maximum": float(np.max(residual_ranges)),
            "zero_range_questions": int(np.sum(residual_ranges == 0.0)),
            "nonzero_range_questions": int(np.sum(residual_ranges != 0.0)),
        },
        "order_changed_vs_static_questions": sum(
            row["nnue_order"] != row["static_order"] for row in records
        ),
        "hidden_activation": {
            "values_at_one_fraction": float(np.mean(hidden >= 1.0)),
            "values_at_zero_fraction": float(np.mean(hidden <= 0.0)),
            "positions_all_32_at_one": int(np.sum(np.all(hidden >= 1.0, axis=1))),
            "positions_all_32_at_zero": int(np.sum(np.all(hidden <= 0.0, axis=1))),
            "per_unit_one_saturation_fraction": [float(value) for value in one_by_unit],
        },
        "usage": "diagnostic only; never a model-selection or gate metric",
    }
    return {
        "schema": 1,
        "weights": {"path": str(weights_path), "sha256": file_sha256(weights_path)},
        "questions": {"path": str(questions_path), "sha256": file_sha256(questions_path),
                      "training_or_model_selection_use": False, "d4_augmentation": False},
        "candidate_audit": audit,
        "model_diagnostics": diagnostics,
        "tie_policy": {
            "main_excludes_tie_top2": True,
            "main_questions": len(main_questions),
            "reported_separately": len(tie_questions),
            "pairwise_label_ties_excluded": True,
            "pairwise_prediction_ties": "half credit",
        },
        "main": main,
        "tie_top2_information_only": ties,
        "gate": gate,
        "question_results": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="volume1200")
    parser.add_argument("--questions", default="questions_400.json")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    directory = NNUE_RESULTS / args.tag
    weights_path = Path(args.weights)
    result = evaluate(weights_path, directory / args.questions)
    output = Path(args.output) if args.output else weights_path.parent / "ranking_400.json"
    write_json(output, result)
    summary = {
        "main": result["main"],
        "tie_top2": result["tie_top2_information_only"],
        "gate": result["gate"],
        "model_diagnostics": result["model_diagnostics"],
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not result["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
