"""Export an exact-schema NNUE dataset from audited samples and one collection tag."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from Team19_weights import DEFAULT_WEIGHTS

from .common import (
    FROZEN_TEAMS, MATE_THRESHOLD, MIN_BUDGET, MIN_DEPTH, MIN_PLY, NNUE_RESULTS,
    Sample, WHITELIST, encode_board, eligible_game, game_samples, read_jsonl,
    stable_digest, write_json,
)
from .common import NativeEvaluator


SPLIT_SEED = 20261125
SIDE_CODES = {"A": 1, "D": 0}
BOARD_ENCODING = {
    "0": "empty '.'",
    "1": "attacker 'A'",
    "2": "defender 'D'",
    "3": "king 'K'",
    "4": "empty castle/throne 'X'",
}


def load_p1_samples() -> list[Sample]:
    path = NNUE_RESULTS / "p1_samples.jsonl"
    if not path.exists():
        raise FileNotFoundError("Run tools.nnue.audit first; p1_samples.jsonl is missing")
    check_path = NNUE_RESULTS / "version_check.json"
    if not check_path.exists():
        raise FileNotFoundError("Run tools.nnue.audit --check-version before exporting labels")
    check = json.loads(check_path.read_text(encoding="utf-8"))
    if not check.get("accept_legacy_scores"):
        raise ValueError("Legacy labels failed the frozen-v4 version check; relabel before export")
    return [Sample.from_json(row) for row in read_jsonl(path)]


def load_collection(tag: str) -> list[Sample]:
    directory = NNUE_RESULTS / tag
    path = directory / "games.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Complete collection {tag!r} first; {path} is missing")
    summary_path = directory / "collection_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError(f"Collection {tag!r} is incomplete")
    if summary.get("v4_errors_or_exceptions") or summary.get("v4_fallbacks"):
        raise ValueError(f"Collection {tag!r} contains fallbacks or errors")
    samples = []
    for game in read_jsonl(path):
        eligible, provenance = eligible_game(game, f"collection:{tag}")
        if not eligible:
            raise ValueError(f"Collected game is outside the frozen-v4 rules: {game.get('id')}")
        samples.extend(game_samples(game, f"collection:{tag}", provenance))
    return samples


def merge_samples(groups: list[list[Sample]]) -> tuple[list[Sample], dict]:
    exact: dict[tuple, Sample] = {}
    duplicates = 0
    for group in groups:
        for sample in group:
            rank = (sample.depth, sample.budget, sample.source, sample.game_id, sample.ply)
            previous = exact.get(sample.exact_key)
            if previous is None:
                exact[sample.exact_key] = sample
                continue
            duplicates += 1
            old_rank = (previous.depth, previous.budget, previous.source,
                        previous.game_id, previous.ply)
            if rank > old_rank:
                exact[sample.exact_key] = sample
    ordered = sorted(exact.values(), key=lambda row: (
        -row.depth, -row.budget, row.source, row.game_id, row.ply))
    symmetry: dict[str, Sample] = {}
    d4_removed = 0
    for sample in ordered:
        key = sample.symmetry_key
        if key in symmetry:
            d4_removed += 1
            continue
        symmetry[key] = sample
    survivors = sorted(symmetry.values(), key=lambda row: (
        row.source, row.game_id, row.ply, row.side))
    return survivors, {"exact_duplicates_replaced": duplicates, "d4_duplicates_removed": d4_removed}


def split_families(samples: list[Sample], fraction: float = 0.15) -> tuple[set[str], dict]:
    counts = Counter(sample.opening_family for sample in samples)
    total = len(samples)
    target = fraction * total
    ordered = sorted(counts, key=lambda family: stable_digest(f"split:{SPLIT_SEED}:{family}"))
    validation: set[str] = set()
    selected = 0
    for family in ordered:
        count = counts[family]
        if selected >= target:
            break
        if selected + count <= 0.20 * total:
            validation.add(family)
            selected += count
    if selected < 0.08 * total:
        for family in ordered:
            if family in validation:
                continue
            validation.add(family)
            selected += counts[family]
            if selected >= 0.12 * total:
                break
    if not validation or selected == total:
        raise ValueError("Unable to build a non-empty, non-whole validation split")
    return validation, {
        "method": "whole opening families chosen by SHA-256 order; no game crosses splits",
        "seed": SPLIT_SEED,
        "target_fraction": fraction,
        "families_total": len(counts),
        "families_validation": len(validation),
        "samples_validation": selected,
        "samples_train": total - selected,
    }


def dataset_arrays(rows: list[Sample], evaluator: NativeEvaluator) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Empty dataset split")
    boards = np.stack([encode_board(sample.board) for sample in rows]).astype(np.int8, copy=False)
    sides = np.array([SIDE_CODES[sample.side] for sample in rows], dtype=np.int8)
    base = np.array([evaluator.evaluate(sample.board, sample.side) for sample in rows],
                    dtype=np.float32)
    deep = np.array([sample.deep_eval for sample in rows], dtype=np.float32)
    outcomes = np.array([sample.outcome for sample in rows], dtype=np.float32)
    game_ids = np.array([sample.game_id for sample in rows], dtype=str)
    values = np.unique(boards)
    if not np.isin(values, np.arange(5)).all():
        raise ValueError(f"Unexpected board codes: {values.tolist()}")
    for name, array in (("sides", sides), ("base_evals", base), ("deep_evals", deep),
                        ("outcomes", outcomes)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    if not np.isin(sides, (0, 1)).all():
        raise ValueError("sides must contain only 0 and 1")
    if not np.isin(outcomes, (0.0, 0.5, 1.0)).all():
        raise ValueError("outcomes must contain only 0.0, 0.5 and 1.0")
    return {
        "boards": boards,
        "sides": sides,
        "base_evals": base,
        "deep_evals": deep,
        "outcomes": outcomes,
        "game_ids": game_ids,
    }


def describe(rows: list[Sample]) -> dict:
    depths = Counter(sample.depth for sample in rows)
    return {
        "rows": len(rows),
        "games": len({sample.game_id for sample in rows}),
        "sides": {side: sum(sample.side == side for sample in rows) for side in ("A", "D")},
        "outcomes": {str(value): sum(sample.outcome == value for sample in rows)
                     for value in (1.0, 0.5, 0.0)},
        "depths": dict(sorted(depths.items())),
        "mean_depth": sum(depth * count for depth, count in depths.items()) / len(rows),
        "mate_labels": sum(abs(sample.deep_eval) >= MATE_THRESHOLD for sample in rows),
        "sources": dict(Counter(sample.source for sample in rows)),
        "budgets": dict(Counter(str(sample.budget) for sample in rows)),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(tag: str, fraction: float) -> dict:
    old = load_p1_samples()
    new = load_collection(tag)
    samples, duplicate_report = merge_samples([old, new])
    validation_families, split_report = split_families(samples, fraction)
    train = [sample for sample in samples if sample.opening_family not in validation_families]
    val = [sample for sample in samples if sample.opening_family in validation_families]
    train_games = {sample.game_id for sample in train}
    val_games = {sample.game_id for sample in val}
    if train_games & val_games:
        raise ValueError("A game crosses the train/validation split")
    train_sym = {sample.symmetry_key for sample in train}
    val_sym = {sample.symmetry_key for sample in val}
    if train_sym & val_sym:
        raise ValueError("A D4-symmetric position crosses the train/validation split")
    evaluator = NativeEvaluator()
    directory = NNUE_RESULTS / tag
    version = json.loads((NNUE_RESULTS / "version_check.json").read_text(encoding="utf-8"))
    train_arrays = dataset_arrays(train, evaluator)
    val_arrays = dataset_arrays(val, evaluator)
    train_path = directory / "team19_nnue_train.npz"
    val_path = directory / "team19_nnue_val.npz"
    np.savez_compressed(train_path, **train_arrays)
    np.savez_compressed(val_path, **val_arrays)
    reloaded = np.load(train_path)
    if set(reloaded.files) != set(train_arrays):
        raise ValueError("Saved npz fields do not match the agreed schema")
    manifest = {
        "schema": 1,
        "status": "pilot_for_member5_field_check",
        "collection_tag": tag,
        "files": {
            "train": {"path": str(train_path), "sha256": file_sha256(train_path),
                       "bytes": train_path.stat().st_size},
            "validation": {"path": str(val_path), "sha256": file_sha256(val_path),
                           "bytes": val_path.stat().st_size},
        },
        "fields": {
            "boards": "int8[N,81], row-major, codes from Team19_engine._flat()",
            "sides": "int8[N], 1=attacker to move, 0=defender to move",
            "base_evals": "float32[N], attacker perspective, raw 22-feature native score before the 7500 siege bonus",
            "deep_evals": "float32[N], attacker perspective, frozen-v4 root score from the source journal",
            "outcomes": "float32[N], attacker perspective: win 1.0, draw 0.5, loss 0.0",
            "game_ids": "str[N], source game id for leakage auditing",
        },
        "board_encoding": BOARD_ENCODING,
        "member5_input_note": (
            "For the 9x9x3 input, map code 0 (empty) and code 4 (empty castle) "
            "both to the empty plane; 1=attacker, 2=defender, 3=king."
        ),
        "base_eval": {
            "function": "Team19_eval._native_evaluate",
            "weights": "Team19_weights.DEFAULT_WEIGHTS (via native_weights(None))",
            "weights_sha256": hashlib.sha256(json.dumps(DEFAULT_WEIGHTS).encode()).hexdigest(),
            "perspective": "attackers positive",
            "siege_bonus_included": False,
            "siege_threshold_owner": "Team19_search._native_score",
        },
        "filters": {
            "teams": sorted(FROZEN_TEAMS), "min_budget_seconds": MIN_BUDGET,
            "min_completed_depth": MIN_DEPTH, "min_ply": MIN_PLY,
            "previous_move_capture_excluded": True, "whitelist": sorted(WHITELIST),
        },
        "deduplication": {"exact": "board + side", "symmetry": "D4 board + side",
                           **duplicate_report},
        "split": {**split_report, "train_games": len(train_games),
                  "validation_games": len(val_games),
                  "train": describe(train), "validation": describe(val)},
        "mate_policy": "retained; member 5 to confirm filtering or clipping before training",
        "version_check": {key: version[key] for key in (
            "method", "samples", "consistent", "consistency_fraction",
            "move_matches", "accept_legacy_scores",
        )},
        "d4_augmentation": "not applied; member 5 applies eight-board augmentation to train only",
        "source_tag_summary": json.loads((directory / "collection_summary.json").read_text(
            encoding="utf-8")),
    }
    write_json(directory / "dataset_manifest.json", manifest)
    print(json.dumps({"train": describe(train), "validation": describe(val),
                      "split": split_report, "dedup": duplicate_report,
                      "files": manifest["files"]}, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="pilot300")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if not 0.05 <= args.val_fraction <= 0.30:
        parser.error("--val-fraction must be between 0.05 and 0.30")
    export(args.tag, args.val_fraction)


if __name__ == "__main__":
    main()
