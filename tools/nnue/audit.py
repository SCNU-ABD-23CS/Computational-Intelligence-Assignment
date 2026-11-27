"""Audit frozen-v4 samples and optionally recheck legacy labels with the current engine."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
import subprocess
import time

from Team19 import _INITIAL_BOARD
from Team19_search import _NativeSearch

from .common import (
    ARENA_RESULTS, MATE_THRESHOLD, MIN_BUDGET, MIN_DEPTH, MIN_PLY, NNUE_RESULTS,
    Sample, V4_COMMIT, V4_CUTOFF, WHITELIST, board_rows, d4_key, eligible_game,
    game_samples, read_jsonl, write_json, write_jsonl,
)


def collect_samples(paths) -> tuple[dict, dict]:
    selected = {}
    rejection = Counter()
    accepted_games = Counter()
    raw_candidates = Counter()
    duplicates = 0
    for path in paths:
        for game in read_jsonl(path):
            eligible, provenance = eligible_game(game, path.name)
            rejection[provenance] += 1
            if not eligible:
                continue
            accepted_games[provenance] += 1
            for sample in game_samples(game, path.name, provenance):
                raw_candidates[sample.source] += 1
                previous = selected.get(sample.exact_key)
                rank = (sample.depth, sample.budget, sample.source, sample.game_id, sample.ply)
                if previous is None:
                    selected[sample.exact_key] = sample
                else:
                    duplicates += 1
                    old_rank = (previous.depth, previous.budget, previous.source,
                                previous.game_id, previous.ply)
                    if rank > old_rank:
                        selected[sample.exact_key] = sample
    samples = sorted(selected.values(), key=lambda row: (row.source, row.game_id, row.ply))
    symmetry_groups = Counter(row.symmetry_key for row in samples)
    report = {
        "schema": 1,
        "frozen_commit": V4_COMMIT,
        "frozen_cutoff_utc": V4_CUTOFF.isoformat(),
        "filters": {
            "teams": ["Team19-v4", "Team19-New"],
            "min_budget_seconds": MIN_BUDGET,
            "min_completed_depth": MIN_DEPTH,
            "min_ply": MIN_PLY,
            "previous_move_capture_excluded": True,
            "deduplication": "exact board + side to move; retain deepest/budget-largest source",
            "whitelist": sorted(WHITELIST),
        },
        "games_by_provenance": dict(rejection),
        "accepted_games": dict(accepted_games),
        "samples": len(samples),
        "unique_games": len({row.game_id for row in samples}),
        "unique_opening_families": len({row.opening_family for row in samples}),
        "unique_d4_positions": len(symmetry_groups),
        "exact_duplicates_replaced": duplicates,
        "d4_collision_groups": sum(count > 1 for count in symmetry_groups.values()),
        "sides": dict(Counter(row.side for row in samples)),
        "depths": dict(sorted(Counter(row.depth for row in samples).items())),
        "outcomes": dict(Counter(str(row.outcome) for row in samples)),
        "mate_labels": {
            "threshold": MATE_THRESHOLD,
            "count": sum(abs(row.deep_eval) >= MATE_THRESHOLD for row in samples),
            "policy": "retained; member 5 must confirm whether to filter or clip during training",
        },
        "sources": {
            source: {
                "samples": sum(row.source == source for row in samples),
                "games": len({row.game_id for row in samples if row.source == source}),
                "raw_candidates_before_global_dedup": raw_candidates[source],
            }
            for source in sorted({row.source for row in samples})
        },
    }
    total = len(samples)
    if total:
        attackers = sum(row.side == "A" for row in samples)
        report["attacker_fraction"] = attackers / total
        report["mate_fraction"] = report["mate_labels"]["count"] / total
    return {row.sample_id: row for row in samples}, report


def audit() -> dict:
    paths = sorted(path for path in ARENA_RESULTS.glob("*.jsonl") if "0p5s" not in path.name)
    samples, report = collect_samples(paths)
    report["git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ARENA_RESULTS.parents[2], text=True,
    ).strip()
    write_jsonl(NNUE_RESULTS / "p1_samples.jsonl",
                (row.to_json() for row in sorted(samples.values(), key=lambda row: row.sample_id)))
    write_json(NNUE_RESULTS / "p1_audit.json", report)
    print(json.dumps({key: report[key] for key in (
        "samples", "unique_games", "unique_d4_positions", "sides", "attacker_fraction",
        "mate_labels", "sources",
    )}, ensure_ascii=False, indent=2), flush=True)
    return report


def version_check(per_file: int = 10) -> dict:
    path = NNUE_RESULTS / "p1_samples.jsonl"
    if not path.exists():
        audit()
    samples = [Sample.from_json(row) for row in read_jsonl(path)]
    groups = defaultdict(list)
    for sample in samples:
        if sample.source in WHITELIST:
            groups[sample.source].append(sample)
    rng = random.Random(20261125)
    chosen = []
    for source in sorted(groups):
        group = sorted(groups[source], key=lambda row: row.sample_id)
        chosen.extend(rng.sample(group, min(per_file, len(group))))
    search = _NativeSearch(_INITIAL_BOARD)
    records = []
    for index, sample in enumerate(chosen, 1):
        search.reset_game()
        # Allow timing jitter to finish the original depth. A shallower rerun
        # would otherwise look like a code-version change even when moves agree.
        seconds = max(0.075, min(1.5 * (sample.elapsed or 0.25), 6.0))
        started = time.perf_counter()
        search.choose(board_rows(sample.board), sample.side == "A",
                      started + seconds, set(sample.history))
        telemetry = search.last_search
        iterations = {depth: result for depth, result, _, _ in search.iterations}
        original = iterations.get(sample.depth)
        reached_original_depth = original is not None
        same_depth_score = float(original.score) if original is not None else None
        same_depth_move = ([list(square) for square in original.move]
                           if original is not None else None)
        old_score = sample.deep_eval
        if original is None:
            score_close = move_match = consistent = False
        else:
            both_mate = abs(old_score) >= MATE_THRESHOLD and abs(same_depth_score) >= MATE_THRESHOLD
            score_close = (old_score * same_depth_score > 0 if both_mate
                           else abs(same_depth_score - old_score) <= 50.0)
            move_match = [list(square) for square in sample.move] == same_depth_move
            consistent = score_close and move_match
        record = {
            "source": sample.source,
            "game_id": sample.game_id,
            "ply": sample.ply,
            "old_score": old_score,
            "old_depth": sample.depth,
            "old_move": sample.move,
            "reached_original_depth": reached_original_depth,
            "same_depth_score": same_depth_score,
            "same_depth_move": same_depth_move,
            "same_depth_score_delta": (None if same_depth_score is None
                                       else same_depth_score - old_score),
            "final_depth": int(telemetry.depth),
            "final_score": float(telemetry.score),
            "move_match": move_match,
            "score_close": score_close,
            "consistent": consistent,
            "search_seconds": seconds,
        }
        records.append(record)
        print(f"version-check {index}/{len(chosen)} {sample.source} ply={sample.ply} "
              f"depth={sample.depth} same_depth_score={same_depth_score} "
              f"move_match={move_match} final_depth={telemetry.depth}", flush=True)
    consistent = sum(row["consistent"] for row in records)
    result = {
        "schema": 1,
        "method": "Frozen v4 native search; original history; compare the completed iteration at the original depth (1.5x time ensures that depth is reached)",
        "score_tolerance": 50.0,
        "acceptance_fraction": 0.8,
        "samples": len(records),
        "consistent": consistent,
        "consistency_fraction": consistent / len(records) if records else 0.0,
        "reached_original_depth": sum(row["reached_original_depth"] for row in records),
        "move_matches": sum(row["move_match"] for row in records),
        "accept_legacy_scores": bool(records and consistent / len(records) >= 0.8),
        "records": records,
    }
    write_json(NNUE_RESULTS / "version_check.json", result)
    print(json.dumps({key: result[key] for key in (
        "samples", "consistent", "consistency_fraction", "move_matches", "accept_legacy_scores",
    )}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-version", action="store_true")
    parser.add_argument("--per-file", type=int, default=10)
    args = parser.parse_args()
    if args.check_version:
        version_check(args.per_file)
    else:
        audit()


if __name__ == "__main__":
    main()
