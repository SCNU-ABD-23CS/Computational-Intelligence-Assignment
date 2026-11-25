"""Build the sealed-ranking validation set from validation-only v4 decisions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

import Team19_engine as engine
from Team19 import _INITIAL_BOARD
from Team19_search import _NativeSearch

from .common import (
    MATE_THRESHOLD, NNUE_RESULTS, Sample, board_rows, d4_key, write_json,
)
from .dataset import load_collection, load_p1_samples, merge_samples, split_families
from .common import NativeEvaluator


SIDE = "A"


def _move_key(move) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(tuple(int(value) for value in square) for square in move)


def _material(board) -> int:
    return sum(piece in "ADK" for row in board for piece in row)


def _winner(board, next_attacker: bool) -> str | None:
    return engine.check_winner(board_rows(board), next_attacker)


def _immediate_loss(child, defender_to_move: bool = True) -> bool:
    if _winner(child, not defender_to_move) == "D":
        return True
    for reply in engine.get_legal_moves(board_rows(child), not defender_to_move):
        after = engine.apply_move(board_rows(child), reply, not defender_to_move)
        if _winner(after, defender_to_move) == "D":
            return True
    return False


def _candidate_pool(sample: Sample, evaluator: NativeEvaluator):
    rows = board_rows(sample.board)
    legal = engine.get_legal_moves(rows, True)
    if len(legal) < 3:
        return None
    scored = []
    for move in legal:
        child = engine.apply_move(rows, move, True)
        winner = _winner(child, False)
        if winner == "A":
            return None
        lost = winner == "D" or _immediate_loss(child)
        value = evaluator.evaluate(child, "D")
        captured = _material(rows) - _material(child)
        scored.append({"move": _move_key(move), "child": child, "static": value,
                       "lost": lost, "captures": captured})
    safe = [item for item in scored if not item["lost"]]
    return scored, (safe if len(safe) >= 3 else scored)


def _choose_candidates(sample: Sample, evaluator: NativeEvaluator, search: _NativeSearch,
                       candidate_seconds: float):
    pool = _candidate_pool(sample, evaluator)
    if pool is None:
        return None
    scored, usable = pool
    usable_sorted = sorted(usable, key=lambda item: (-item["static"], item["move"]))
    if len(usable_sorted) < 3:
        return None
    rows = board_rows(sample.board)
    search.reset_game()
    best = _move_key(search.choose(rows, True, time.perf_counter() + candidate_seconds,
                                   set(sample.history)))
    best_item = next((item for item in scored if item["move"] == best), None)
    if best_item is None or best_item["lost"]:
        return None
    candidates = [best_item]
    candidates[0]["source"] = "v4_search"
    first_source = best_item["move"][0]
    for item in usable_sorted:
        if item["move"] == best:
            continue
        if item["move"][0] != first_source:
            item["source"] = "static_different_source"
            candidates.append(item)
            break
    if len(candidates) < 2:
        item = next(item for item in usable_sorted if item["move"] != best)
        item["source"] = "static_different_source_fallback"
        candidates.append(item)
    capture_class = candidates[1]["captures"] > 0
    for item in usable_sorted:
        if item["move"] in {entry["move"] for entry in candidates}:
            continue
        if (item["captures"] > 0) != capture_class:
            item["source"] = "static_different_capture_class"
            candidates.append(item)
            break
    for item in usable_sorted:
        if len(candidates) == 3:
            break
        if item["move"] not in {entry["move"] for entry in candidates}:
            item["source"] = "static_fallback"
            candidates.append(item)
    if len(candidates) != 3:
        return None
    return candidates


def _label_candidate(sample: Sample, candidate: dict, search: _NativeSearch,
                     label_seconds: float):
    rows = board_rows(sample.board)
    child = candidate["child"]
    if _winner(child, False) == "A":
        return 100_000.0, 1, True
    history = set(sample.history)
    history.add(int(engine.compute_hash(board_rows(child), False)))
    search.reset_game()
    search.choose(board_rows(child), False, time.perf_counter() + label_seconds, history)
    telemetry = search.last_search
    return float(telemetry.score), int(telemetry.depth), False


def _question(payload):
    index, sample, candidate_seconds, label_seconds = payload
    try:
        evaluator = NativeEvaluator()
        search = _NativeSearch(_INITIAL_BOARD)
        candidates = _choose_candidates(sample, evaluator, search, candidate_seconds)
        if candidates is None:
            return {"index": index, "error": "no_three_competitive_candidates"}
        labels = []
        for candidate in candidates:
            score, depth, terminal = _label_candidate(sample, candidate, search, label_seconds)
            labels.append({
                "move": candidate["move"],
                "generated_by": candidate["source"],
                "static_attacker_eval": round(candidate["static"], 6),
                "captures": int(candidate["captures"]),
                "deep_attacker_eval": score,
                "label_depth": depth,
                "terminal_root_win": terminal,
            })
        order = sorted(range(3), key=lambda pos: (-labels[pos]["deep_attacker_eval"],
                                                 labels[pos]["move"]))
        best = labels[order[0]]["deep_attacker_eval"]
        second = labels[order[1]]["deep_attacker_eval"]
        margin = best - second
        if abs(best) >= MATE_THRESHOLD and abs(second) >= MATE_THRESHOLD:
            margin = 0.0
        return {
            "index": index,
            "question": {
                "id": f"q{index:04d}",
                "game_id": sample.game_id,
                "source": sample.source,
                "ply": sample.ply,
                "board": list(sample.board),
                "side_to_move": "A",
                "deep_eval": sample.deep_eval,
                "search_depth": sample.depth,
                "position_d4": d4_key(sample.board, sample.side),
                "candidates": labels,
                "ranking": order,
                "best_move": labels[order[0]]["move"],
                "margin_top2": margin,
                "tie_top2": math.isclose(best, second, abs_tol=1e-6),
            },
        }
    except Exception as error:
        return {"index": index, "error": f"{type(error).__name__}: {error}"}


def validation_pool(tag: str) -> tuple[list[Sample], list[Sample]]:
    samples, _ = merge_samples([load_p1_samples(), load_collection(tag)])
    families, _ = split_families(samples)
    train = [sample for sample in samples if sample.opening_family not in families]
    val = [sample for sample in samples if sample.opening_family in families]
    return train, val


def select_positions(train: list[Sample], val: list[Sample], count: int) -> list[Sample]:
    train_sym = {sample.symmetry_key for sample in train}
    fresh = [
        sample for sample in val
        if sample.side == SIDE and sample.depth >= 5
        and abs(sample.deep_eval) < MATE_THRESHOLD
        and sample.symmetry_key not in train_sym
    ]
    by_game = {}
    for sample in sorted(fresh, key=lambda row: (
            not row.source.startswith("collection:"), -row.depth,
            abs(row.deep_eval), row.game_id, row.ply)):
        by_game.setdefault(sample.game_id, []).append(sample)
    ordered = []
    for cap in (2, 3, 4, 6, 8):
        ordered = []
        buckets = {game: list(rows[:cap]) for game, rows in by_game.items()}
        while len(ordered) < count:
            added = False
            for game_id in sorted(buckets):
                bucket = buckets[game_id]
                if bucket:
                    ordered.append(bucket.pop(0))
                    added = True
                    if len(ordered) == count:
                        break
            if not added:
                break
        if len(ordered) >= count:
            break
    if len(ordered) < count:
        raise ValueError(f"Only {len(ordered)} validation attacker positions are available; "
                         f"requested {count}")
    seen = set()
    unique = []
    for sample in ordered:
        if sample.symmetry_key in seen:
            continue
        seen.add(sample.symmetry_key)
        unique.append(sample)
    if len(unique) < count:
        raise ValueError("Validation position pool is smaller than requested after D4 deduplication")
    return unique[:count]


def build(tag: str, count: int, workers: int, candidate_seconds: float,
          label_seconds: float) -> dict:
    train, val = validation_pool(tag)
    pool_size = count + max(8, count // 4)
    samples = select_positions(train, val, pool_size)
    payloads = [(index, sample, candidate_seconds, label_seconds)
                for index, sample in enumerate(samples, 1)]
    print(f"questions={count} evaluatedPositions={pool_size} workers={workers} "
          f"label={label_seconds}s validation_pool={len(val)}", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        raw = pool.map(_question, payloads)
    successes = sorted((row["question"] for row in raw if "question" in row),
                       key=lambda row: row["id"])
    failures = [row for row in raw if "error" in row]
    if len(successes) < count:
        raise RuntimeError(f"Only {len(successes)} questions succeeded; {len(failures)} failed")
    successes = successes[:count]
    ids = {row["id"] for row in successes}
    if len(ids) != count:
        raise ValueError("Duplicate question ids")
    if any(row["side_to_move"] != "A" for row in successes):
        raise ValueError("Every sealed question must be an attacker decision")
    directory = NNUE_RESULTS / tag
    question_name = f"questions_{count}.json"
    output = {
        "schema": 1,
        "status": "pilot_field_check" if count <= 40 else "sealed_validation_set",
        "purpose": "Detect whether residual NNUE learns attacker blockade choice, not just outcome smoothing",
        "source_tag": tag,
        "validation_only": True,
        "train_d4_excluded": True,
        "label_engine": "frozen v4 native search",
        "label_seconds_per_candidate": label_seconds,
        "candidate_generation_seconds": candidate_seconds,
        "candidate_policy": {
            "candidate_1": "frozen v4 best move",
            "candidate_2": "highest static safe move with a different source square",
            "candidate_3": "highest static safe move with a different capture class, otherwise the next-highest safe move",
            "safety": "reject moves allowing an immediate defender win; reject positions without three safe candidates",
        },
        "count": count,
        "evaluated_positions": len(payloads),
        "failures": failures,
        "ranking_distinct_margins": {
            "non_ties": sum(not row["tie_top2"] for row in successes),
            "ties": sum(row["tie_top2"] for row in successes),
        },
        "questions": successes,
    }
    path = directory / question_name
    write_json(path, output)
    print(json.dumps({"count": len(successes), "failures": len(failures),
                      "ties": output["ranking_distinct_margins"]["ties"],
                      "path": str(path)}, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="pilot300")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--candidate-seconds", type=float, default=0.4)
    parser.add_argument("--label-seconds", type=float, default=6.0)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.label_seconds <= 0 or args.candidate_seconds <= 0:
        parser.error("search times must be positive")
    build(args.tag, args.count, args.workers, args.candidate_seconds, args.label_seconds)


if __name__ == "__main__":
    main()
