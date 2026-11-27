"""Select one residual strength on train-only fixed-depth development questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import Team19_engine as engine
from Team19 import _INITIAL_BOARD
from Team19_search import (
    _ATTACKER_REPETITION_SCORE, _MATE, _Q_LIMIT, _NativeSearch, _native_position,
)
from Team19_tt import advance_generation

from .common import NNUE_RESULTS, NativeEvaluator, board_rows, d4_key, write_json
from .model import load_weights, numpy_residual, scale_tanh_weights
from .questions import _candidate_pool, _move_key
from .rank_questions import metric_group, prepare_candidates


SCALES = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FixedDepthNative:
    """Deterministic frozen-v4 search for development labels, not the player."""

    def __init__(self):
        self.search = _NativeSearch(_INITIAL_BOARD)

    def choose(self, board, attacker: bool, depth: int, history=()):
        search = self.search
        search.reset_game()
        work = search.work
        advance_generation(work.tt)
        work.killers.fill(-1)
        work.history.fill(0)
        work.counts.fill(0)
        flat, state = _native_position(board)
        played = np.asarray(sorted(history), dtype=np.uint64)
        repetition = _ATTACKER_REPETITION_SCORE if attacker else 0
        siege = attacker and state[3] <= 6 and state[2] >= state[3] + 4
        value, move = search._iteration(
            flat, attacker, state, depth, _Q_LIMIT, -2 * _MATE, 2 * _MATE,
            math.inf, played, False, repetition, siege,
        )
        score = float(value if attacker else -value)
        return (None if move < 0 else engine._move_list(move)), score


def choose_candidates(board, evaluator, fixed: FixedDepthNative, root_depth: int):
    sample = SimpleNamespace(board=board, history=())
    pool = _candidate_pool(sample, evaluator)
    if pool is None:
        return None
    scored, usable = pool
    if len(usable) < 3 or any(item["lost"] for item in usable):
        return None
    ordered = sorted(usable, key=lambda item: (-item["static"], item["move"]))
    best, _ = fixed.choose(board_rows(board), True, root_depth)
    if best is None:
        return None
    best = _move_key(best)
    best_item = next((item for item in scored if item["move"] == best), None)
    if best_item is None or best_item["lost"]:
        return None
    candidates = [best_item]
    candidates[0]["source"] = "v4_search"
    first_source = best[0]
    for item in ordered:
        if item["move"] != best and item["move"][0] != first_source:
            item["source"] = "static_different_source"
            candidates.append(item)
            break
    if len(candidates) < 2:
        item = next(item for item in ordered if item["move"] != best)
        item["source"] = "static_different_source_fallback"
        candidates.append(item)
    capture_class = candidates[1]["captures"] > 0
    for item in ordered:
        if item["move"] in {entry["move"] for entry in candidates}:
            continue
        if (item["captures"] > 0) != capture_class:
            item["source"] = "static_different_capture_class"
            candidates.append(item)
            break
    for item in ordered:
        if len(candidates) == 3:
            break
        if item["move"] not in {entry["move"] for entry in candidates}:
            item["source"] = "static_fallback"
            candidates.append(item)
    return candidates if len(candidates) == 3 else None


def development_order(arrays, attempts: int) -> list[int]:
    eligible = np.flatnonzero(
        (arrays["sides"] == 1)
        & (np.abs(arrays["deep_evals"]) < 90_000.0)
        & ~np.isclose(np.abs(arrays["deep_evals"]), 3_000.0, rtol=0.0, atol=1e-6)
    )
    by_game = {}
    for index in eligible:
        board = arrays["boards"][index]
        digest = hashlib.sha256(
            str(arrays["game_ids"][index]).encode("utf-8") + board.tobytes()
        ).hexdigest()
        by_game.setdefault(str(arrays["game_ids"][index]), []).append((digest, int(index)))
    buckets = {
        game: [index for _, index in sorted(rows)] for game, rows in by_game.items()
    }
    order = []
    while len(order) < attempts:
        added = False
        for game in sorted(buckets):
            bucket = buckets[game]
            if bucket:
                order.append(bucket.pop(0))
                added = True
                if len(order) == attempts:
                    break
        if not added:
            break
    return order


def build_questions(arrays, count: int, root_depth: int, label_depth: int):
    evaluator = NativeEvaluator()
    fixed = FixedDepthNative()
    questions = []
    failures = []
    seen_d4 = set()
    for index in development_order(arrays, count * 4):
        board = tuple(engine._nested(arrays["boards"][index]))
        key = d4_key(board, "A")
        if key in seen_d4:
            continue
        try:
            candidates = choose_candidates(board, evaluator, fixed, root_depth)
            if candidates is None:
                failures.append({"index": index, "error": "no_three_safe_candidates"})
                continue
            labels = []
            for candidate in candidates:
                child = candidate["child"]
                history = {int(engine.compute_hash(board_rows(child), False))}
                _, score = fixed.choose(child, False, label_depth, history)
                labels.append({
                    "move": candidate["move"],
                    "generated_by": candidate["source"],
                    "static_attacker_eval": round(candidate["static"], 6),
                    "captures": int(candidate["captures"]),
                    "deep_attacker_eval": score,
                    "label_depth": label_depth,
                    "terminal_root_win": False,
                })
            ranking = sorted(range(3), key=lambda pos: (
                -labels[pos]["deep_attacker_eval"], labels[pos]["move"],
            ))
            best = labels[ranking[0]]["deep_attacker_eval"]
            second = labels[ranking[1]]["deep_attacker_eval"]
            question = {
                "id": f"d{len(questions) + 1:04d}",
                "game_id": str(arrays["game_ids"][index]),
                "source": "train_npz_fixed_depth_development",
                "board": [list(row) for row in board],
                "side_to_move": "A",
                "position_d4": key,
                "candidates": labels,
                "ranking": ranking,
                "best_move": labels[ranking[0]]["move"],
                "margin_top2": best - second,
                "tie_top2": math.isclose(best, second, abs_tol=1e-6),
            }
            questions.append(question)
            seen_d4.add(key)
            if len(questions) == count:
                break
        except Exception as error:
            failures.append({"index": index, "error": f"{type(error).__name__}: {error}"})
    if len(questions) != count:
        raise RuntimeError(f"Only {len(questions)} development questions succeeded")
    return questions, failures


def run(args: argparse.Namespace):
    directory = NNUE_RESULTS / args.tag
    train_path = directory / "team19_nnue_train.npz"
    with np.load(train_path, allow_pickle=False) as data:
        arrays = {key: np.array(data[key], copy=True) for key in data.files}
    output = Path(args.output) if args.output else directory / (
        "member5-c750-k6000-l0p5-tanh-precalib5-center1e3-top1dev-seed20261126"
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Development output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    questions, failures = build_questions(
        arrays, args.count, args.root_depth, args.label_depth,
    )
    payload = {
        "schema": 1,
        "purpose": "train-only residual-scale development; never the sealed 400-question gate",
        "validation_only": False,
        "train_npz_only": True,
        "sealed_questions_used": False,
        "d4_augmentation": False,
        "history_limitation": "NPZ has no full game history; labels include only the child hash",
        "root_depth": args.root_depth,
        "label_depth": args.label_depth,
        "count": len(questions),
        "failures": failures,
        "questions": questions,
    }
    questions_path = output / "development_questions.json"
    write_json(questions_path, payload)
    if args.generate_only:
        print(json.dumps({
            "count": len(questions), "failures": len(failures),
            "path": str(questions_path), "sha256": file_sha256(questions_path),
            "sealed_questions_used": False,
        }, indent=2), flush=True)
        return
    boards, static, deep, audit = prepare_candidates(questions)
    source_path = Path(args.weights)
    source = load_weights(source_path)
    residual = numpy_residual(source, boards, np.zeros(len(boards), dtype=np.int8)).reshape(-1, 3)
    main_indices = [index for index, row in enumerate(questions) if not row["tie_top2"]]
    main_questions = [questions[index] for index in main_indices]
    rows = []
    eligible = []
    for scale in SCALES:
        metrics = metric_group(
            main_questions, static[main_indices],
            (static + scale * residual)[main_indices], deep[main_indices],
        )
        row = {"scale": scale, "main": metrics}
        rows.append(row)
        if metrics["delta"]["top1"] > 0.0 and metrics["delta"]["pairwise"] > 0.0:
            eligible.append(row)
    report = {
        "schema": 1,
        "source_weights": {"path": str(source_path), "sha256": file_sha256(source_path)},
        "development_questions": {"path": str(questions_path), "sha256": file_sha256(questions_path)},
        "selection_rule": "among strict dev Top-1 and Pairwise improvements: max Top-1, max Pairwise, then smaller scale",
        "candidate_audit": audit,
        "rows": rows,
        "eligible_scales": [row["scale"] for row in eligible],
        "sealed_400_used_for_selection": False,
    }
    if not eligible:
        report["selected_scale"] = None
        write_json(output / "development_scaling.json", report)
        print(json.dumps({"eligible_scales": [], "output": str(output)}, indent=2))
        raise SystemExit(2)
    selected = max(eligible, key=lambda row: (
        row["main"]["nnue"]["top1_correct"],
        row["main"]["nnue"]["pairwise"]["correct_credit"],
        -row["scale"],
    ))
    scale = selected["scale"]
    report["selected_scale"] = scale
    report["selected_development_metrics"] = selected
    scaled = scale_tanh_weights(source, scale)
    scaled["feature_layout"] = np.array("cell-major square*3+plane; planes attacker,defender,king")
    weights_path = output / "best_weights.npz"
    with weights_path.open("wb") as stream:
        np.savez_compressed(stream, **scaled)
    report["scaled_weights"] = {"path": str(weights_path), "sha256": file_sha256(weights_path)}
    write_json(output / "development_scaling.json", report)
    write_json(output / "scaling_config.json", {
        "source_weights_sha256": file_sha256(source_path),
        "selected_scale": scale,
        "export_identity": "scaled_residual = scale * source_residual",
        "sealed_400_used_for_selection": False,
        "development_questions_sha256": file_sha256(questions_path),
    })
    print(json.dumps({
        "selected_scale": scale,
        "eligible_scales": report["eligible_scales"],
        "development": selected["main"],
        "weights": str(weights_path),
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="volume1200")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--count", type=int, default=192)
    parser.add_argument("--root-depth", type=int, default=2)
    parser.add_argument("--label-depth", type=int, default=4)
    parser.add_argument("--output")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if args.count <= 0 or args.root_depth <= 0 or args.label_depth <= 0:
        parser.error("count and fixed depths must be positive")
    run(args)


if __name__ == "__main__":
    main()
