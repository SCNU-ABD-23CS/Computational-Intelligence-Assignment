"""Verify the exported npz contract and the sealed-question structure end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import Team19_engine as engine

from .common import NNUE_RESULTS, d4_key, write_json


FIELDS = {
    "boards": np.dtype("int8"),
    "sides": np.dtype("int8"),
    "base_evals": np.dtype("float32"),
    "deep_evals": np.dtype("float32"),
    "outcomes": np.dtype("float32"),
    "game_ids": None,
}
OPTIONAL_FIELDS = {"depths": np.dtype("int16")}
ALLOWED_SOURCES = {
    "v4_search", "static_different_source", "static_different_capture_class",
    "static_different_source_fallback", "static_fallback",
}


def _board_text(row) -> list[str]:
    return ["".join(engine._CELLS[int(code)] for code in row[start:start + 9])
            for start in range(0, 81, 9)]


def _split_keys(data) -> set[str]:
    keys = set()
    for board, side in zip(data["boards"], data["sides"]):
        keys.add(d4_key(_board_text(board), "A" if int(side) == 1 else "D"))
    return keys


def verify(tag: str, question_name: str) -> dict:
    directory = NNUE_RESULTS / tag
    manifest = json.loads((directory / "dataset_manifest.json").read_text(encoding="utf-8"))
    train = np.load(directory / "team19_nnue_train.npz")
    val = np.load(directory / "team19_nnue_val.npz")
    checks = {}
    unknown_depths = {}
    for name, data in (("train", train), ("validation", val)):
        rows = len(data["boards"])
        expected = set(FIELDS) | ({"depths"} if "depths" in data.files else set())
        checks[f"{name}_fields"] = set(data.files) == expected
        checks[f"{name}_shapes"] = all(
            data[key].shape == ((rows, 81) if key == "boards" else (rows,))
            for key in expected)
        checks[f"{name}_dtypes"] = all(
            (dtype is None or data[key].dtype == dtype)
            for key, dtype in {**FIELDS, **OPTIONAL_FIELDS}.items()
            if key in expected)
        checks[f"{name}_board_codes"] = set(np.unique(data["boards"]).tolist()) <= {0, 1, 2, 3, 4}
        checks[f"{name}_sides"] = set(np.unique(data["sides"]).tolist()) <= {0, 1}
        checks[f"{name}_outcomes"] = set(np.unique(data["outcomes"]).tolist()) <= {0.0, 0.5, 1.0}
        checks[f"{name}_finite"] = all(bool(np.isfinite(data[key]).all())
                                       for key in ("base_evals", "deep_evals", "outcomes"))
        checks[f"{name}_game_ids"] = data["game_ids"].dtype.kind == "U"
        # -1 marks rows whose root depth is unknown (legacy volume1200 rows may
        # legitimately carry it); anything below -1 or above 128 is corrupt.
        depths = data["depths"] if "depths" in data.files else None
        checks[f"{name}_has_depths"] = depths is None or (
            depths.dtype == np.dtype("int16")
            and depths.shape == (rows,)
            and int(depths.min()) >= -1
            and int(depths.max()) <= 128
        )
        if depths is not None:
            unknown_depths[name] = int((depths == -1).sum())
    checks["game_ids_disjoint"] = not (set(train["game_ids"]) & set(val["game_ids"]))
    train_keys = _split_keys(train)
    val_keys = _split_keys(val)
    checks["d4_split_disjoint"] = not (train_keys & val_keys)
    questions = json.loads((directory / question_name).read_text(encoding="utf-8"))
    checks["question_count"] = len(questions["questions"]) == questions["count"]
    checks["question_validation_only"] = all(
        question["position_d4"] in val_keys and question["position_d4"] not in train_keys
        for question in questions["questions"])
    question_ids = [question["id"] for question in questions["questions"]]
    checks["question_ids_unique"] = len(set(question_ids)) == len(question_ids)
    checks["question_d4_unique"] = len({question["position_d4"]
                                        for question in questions["questions"]}) == len(question_ids)
    for question in questions["questions"]:
        candidates = question["candidates"]
        moves = [tuple(tuple(int(value) for value in square) for square in candidate["move"])
                 for candidate in candidates]
        if len(candidates) != 3 or len(set(moves)) != 3:
            raise ValueError(f"{question['id']} does not contain three distinct candidates")
        board = question["board"]
        legal = {tuple(tuple(int(value) for value in square)
                       for square in move)
                 for move in engine.get_legal_moves([list(row) for row in board], True)}
        if any(move not in legal for move in moves):
            raise ValueError(f"{question['id']} contains an illegal candidate")
        if sorted(question["ranking"]) != [0, 1, 2]:
            raise ValueError(f"{question['id']} has an invalid ranking permutation")
        values = [candidate["deep_attacker_eval"] for candidate in candidates]
        order = question["ranking"]
        if any(values[order[pos]] < values[order[pos + 1]] for pos in range(2)):
            raise ValueError(f"{question['id']} ranking is not descending")
        if any(candidate["generated_by"] not in ALLOWED_SOURCES for candidate in candidates):
            raise ValueError(f"{question['id']} has an unknown candidate source")
    result = {
        "schema": 1,
        "tag": tag,
        "question_file": question_name,
        "checks": checks,
        "rows": {"train": len(train["boards"]), "validation": len(val["boards"])},
        "depth_unknown": unknown_depths,
        "d4_keys": {"train": len(train_keys), "validation": len(val_keys)},
        "questions": len(questions["questions"]),
        "sha256": {
            "train": hashlib.sha256((directory / "team19_nnue_train.npz").read_bytes()).hexdigest(),
            "validation": hashlib.sha256((directory / "team19_nnue_val.npz").read_bytes()).hexdigest(),
            "questions": hashlib.sha256((directory / question_name).read_bytes()).hexdigest(),
        },
        "passed": all(checks.values()),
    }
    write_json(directory / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="pilot300")
    parser.add_argument("--questions", default="questions_40.json")
    args = parser.parse_args()
    verify(args.tag, args.questions)


if __name__ == "__main__":
    main()
