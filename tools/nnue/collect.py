"""Collect diverse frozen-v4 games with random prefixes, outside the arena lock."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time

import Tablut
import Team19_engine as engine

from tools.arena.openings import Opening, start
from tools.arena.runner import Match, run_match

from .common import (
    FROZEN_TEAMS, NNUE_RESULTS, d4_key, game_samples, read_jsonl, write_json,
)


SEED = 20261125
NON_SELF = ("Team19-v3", "Grandmaster-Plus", "Grandmaster-Numba")
OPPONENTS = ("Team19-v4", *NON_SELF)
PREFIX_MIN, PREFIX_MAX = 2, 6


def random_opening(index: int, used_families: set[str], seed: int = SEED) -> Opening:
    """Reject prefix families that are reachable by board symmetry."""
    rng = random.Random(seed + index)
    for _ in range(200):
        game = Tablut.Tablut()
        game.past_positions = [game.compute_zobrist_hash(True)]
        plies = rng.randint(PREFIX_MIN, PREFIX_MAX)
        moves = []
        for ply in range(plies):
            attacker = ply % 2 == 0
            legal = engine.get_legal_moves(game.board, attacker)
            if not legal:
                break
            move = rng.choice(legal)
            with redirect_stdout(StringIO()):
                accepted = game.make_move(attacker, move)
                finished = accepted and game.check_if_game_over(not attacker)
            if not accepted or finished:
                break
            moves.append(move)
        if len(moves) < PREFIX_MIN:
            continue
        name = f"r{index:04d}-p{len(moves)}"
        candidate = Opening(name, moves, "")
        try:
            started = start(candidate)
        except ValueError:
            continue
        family = d4_key(started.positions[-1]["board"])
        if family in used_families:
            continue
        used_families.add(family)
        return Opening(name, moves, started.positions[-1]["hash"])
    raise RuntimeError(f"Unable to build a distinct random prefix for game {index}")


def build_matches(games: int, budget: float, shard_id: int = 0,
                  seed: int = SEED) -> list[Match]:
    """50% self-play, 50% against three opponents; each opponent A/D balanced.

    Even global indices are v4 self-play. Odd indices rotate through the three
    non-self opponents; each six-game block uses one v4 side, so every opponent
    gets both sides after two blocks.
    """
    used: set[str] = set()
    matches = []
    offset = max(0, shard_id - 1) * games
    for local_index in range(games):
        global_index = offset + local_index
        opening = random_opening(global_index, used, seed)
        if global_index % 2 == 0:
            attacker = defender = "Team19-v4"
        else:
            pair_index = global_index // 2
            opponent = NON_SELF[pair_index % len(NON_SELF)]
            if (pair_index // len(NON_SELF)) % 2 == 0:
                attacker, defender = "Team19-v4", opponent
            else:
                attacker, defender = opponent, "Team19-v4"
        matches.append(Match(attacker, defender, budget, opening))
    return matches


def _v4_depth_summary(row: dict) -> dict:
    """Per-game row summary, separate from the per-attempt depth already recorded."""
    depths = [int(attempt["depth"]) for attempt in row.get("attempts", ())
              if attempt.get("team") in FROZEN_TEAMS and attempt.get("depth") is not None]
    if not depths:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(depths), "mean": sum(depths) / len(depths), "max": max(depths)}


def _worker(payload) -> dict:
    worker, tag, matches, budget, shard_id, seed = payload
    directory = NNUE_RESULTS / tag
    if shard_id:
        directory = directory / f"shard_{shard_id:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"worker_{worker}.jsonl"
    done = set()
    if path.exists():
        done = {row["id"] for row in read_jsonl(path)}
    pending = [match for match in matches if match.id not in done]
    began = time.perf_counter()
    written = 0
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for number, match in enumerate(pending, 1):
            row = run_match(match)
            row["collector"] = {
                "tag": tag, "seed": seed, "worker": worker, "shard": shard_id,
                "prefixPlies": len(match.opening.moves),
                "budget": budget,
                "depth": _v4_depth_summary(row),
            }
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                    allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            written += 1
            if number == 1 or number % 5 == 0 or number == len(pending):
                print(f"[worker {worker}] {number}/{len(pending)} {match.id} "
                      f"winner={row['winner']} reason={row['reason']}", flush=True)
    return {
        "worker": worker, "path": str(path), "written": written, "resumed": len(done),
        "seconds": time.perf_counter() - began,
    }


def _summary_from_rows(rows: list[dict], tag: str, budget: float, requested: int,
                       workers: int, directory: Path, shard_id: int = 0,
                       seed: int = SEED, write_games: bool = True) -> dict:
    unique = {row["id"]: row for row in rows}
    if len(unique) != len(rows):
        raise ValueError("Duplicate game ids across worker journals")
    ordered = sorted(unique.values(), key=lambda row: row["id"])
    depth_counts = Counter()
    side_counts = Counter()
    reasons = Counter()
    winners = Counter()
    teams = Counter()
    fallbacks = exceptions = 0
    v4_seconds = []
    v4_turns = 0
    filtered_depths = []
    filtered_samples = 0
    game_seconds = []
    for game in ordered:
        winners[str(game["winner"])] += 1
        reasons[game["reason"]] += 1
        game_seconds.append(float(game.get("elapsed") or 0.0))
        for name in game["teams"].values():
            teams[name] += 1
        for sample in game_samples(game, f"collection:{tag}", "post_frozen_v4"):
            filtered_samples += 1
            filtered_depths.append(sample.depth)
        for attempt in game["attempts"]:
            if attempt.get("team") not in FROZEN_TEAMS:
                continue
            if attempt.get("depth") is not None:
                depth_counts[int(attempt["depth"])] += 1
                v4_turns += 1
                side_counts[attempt["side"]] += 1
            v4_seconds.append(float(attempt.get("elapsed") or 0.0))
            fallbacks += bool(attempt.get("fallback"))
            exceptions += "exception" in attempt or "error" in attempt
    mean_depth = (sum(depth * count for depth, count in depth_counts.items()) / v4_turns
                  if v4_turns else 0.0)
    filtered_mean_depth = (sum(filtered_depths) / len(filtered_depths)
                           if filtered_depths else 0.0)
    summary = {
        "schema": 1,
        "tag": tag,
        "shard_id": shard_id,
        "budget_seconds": budget,
        "workers": workers,
        "requested_games": requested,
        "recorded_games": len(ordered),
        "complete": len(ordered) == requested,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "prefix_plies": {"min": PREFIX_MIN, "max": PREFIX_MAX},
        "opponents": list(NON_SELF),
        "self_play_fraction": 0.5,
        "team_appearances": dict(teams),
        "winners": dict(winners),
        "terminal_reasons": dict(reasons),
        "v4_turns": v4_turns,
        "v4_side_turns": dict(side_counts),
        "v4_depth_histogram": dict(sorted(depth_counts.items())),
        "v4_mean_depth": mean_depth,
        "depth_target": 4.5,
        "depth_ok": mean_depth >= 4.5,
        "filtered_samples": filtered_samples,
        "filtered_samples_per_game": filtered_samples / len(ordered) if ordered else 0.0,
        "filtered_mean_depth": filtered_mean_depth,
        "filtered_depth_histogram": dict(sorted(Counter(filtered_depths).items())),
        "filtered_depth_ok": filtered_mean_depth >= 4.5,
        "mean_game_seconds": sum(game_seconds) / len(game_seconds) if game_seconds else 0.0,
        "max_game_seconds": max(game_seconds) if game_seconds else 0.0,
        "v4_mean_move_seconds": (sum(v4_seconds) / len(v4_seconds) if v4_seconds else 0.0),
        "v4_max_move_seconds": (max(v4_seconds) if v4_seconds else 0.0),
        "v4_fallbacks": fallbacks,
        "v4_errors_or_exceptions": exceptions,
        "paths": {"games": str(directory / "games.jsonl"),
                  "summary": str(directory / "collection_summary.json")},
    }
    write_json(directory / "collection_summary.json", summary)
    if write_games:
        with (directory / "games.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for game in ordered:
                stream.write(json.dumps(game, ensure_ascii=False, separators=(",", ":"),
                                        allow_nan=False) + "\n")
    return summary


def summarise(tag: str, budget: float, requested: int, workers: int,
              shard_id: int = 0, seed: int = SEED) -> dict:
    directory = NNUE_RESULTS / tag
    if shard_id:
        directory = directory / f"shard_{shard_id:02d}"
    rows = []
    for worker in range(workers):
        path = directory / f"worker_{worker}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing worker journal {path}")
        rows.extend(read_jsonl(path))
    return _summary_from_rows(rows, tag, budget, requested, workers, directory,
                              shard_id=shard_id, seed=seed)


def merge_shards(tag: str, shard_count: int, games_per_shard: int, budget: float,
                 seed: int = SEED) -> dict:
    """Merge shard journals into the root tag consumed by dataset.py."""
    root = NNUE_RESULTS / tag
    paths = sorted(root.glob("shard_*/worker_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No shard journals found under {root}")
    rows = []
    for path in paths:
        rows.extend(read_jsonl(path))
    expected = shard_count * games_per_shard
    summary = _summary_from_rows(rows, tag, budget, expected, workers=0,
                                 directory=root, seed=seed)
    summary["shard_count"] = shard_count
    summary["shard_files"] = len(paths)
    summary["shard_games_per_shard"] = games_per_shard
    unique = {row["id"]: row for row in rows}
    if len(unique) != len(rows):
        raise ValueError("Duplicate game ids across shard journals")
    ordered = sorted(unique.values(), key=lambda row: row["id"])
    with (root / "games.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for game in ordered:
            stream.write(json.dumps(game, ensure_ascii=False, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    write_json(root / "collection_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=300,
                        help="games for this run; per-shard games when --shard-id is used")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel processes (1-16; 8 only on a machine with enough physical cores)")
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--tag", default="pilot300")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="1-based shard id; output goes to <tag>/shard_XX/")
    parser.add_argument("--shard-count", type=int, default=12,
                        help="only used with --merge-shards")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--merge-shards", action="store_true",
                        help="merge <tag>/shard_*/worker_*.jsonl after all shards finish")
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.budget < 1.0:
        parser.error("--budget must be at least one second")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    if args.shard_id < 0:
        parser.error("--shard-id must be non-negative")
    if args.merge_shards and args.shard_id:
        parser.error("--merge-shards cannot be combined with --shard-id")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    if args.merge_shards:
        summary = merge_shards(args.tag, args.shard_count, args.games, args.budget, args.seed)
        print(json.dumps({key: summary[key] for key in (
            "tag", "recorded_games", "complete", "shard_count", "shard_files",
            "v4_mean_depth", "depth_ok", "filtered_samples",
            "filtered_samples_per_game", "filtered_mean_depth", "filtered_depth_ok",
            "mean_game_seconds", "v4_fallbacks", "v4_errors_or_exceptions",
        )}, ensure_ascii=False, indent=2), flush=True)
        return
    matches = build_matches(args.games, args.budget, args.shard_id, args.seed)
    if len({match.id for match in matches}) != len(matches):
        raise ValueError("Duplicate fixture ids in generated schedule")
    chunks = [matches[index::args.workers] for index in range(args.workers)]
    payloads = [(index, args.tag, chunk, args.budget, args.shard_id, args.seed)
                for index, chunk in enumerate(chunks)]
    print(f"tag={args.tag} shard={args.shard_id or 'none'} games={len(matches)} "
          f"workers={args.workers} budget={args.budget}s seed={args.seed}", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_worker, payloads)
    summary = summarise(args.tag, args.budget, args.games, args.workers,
                        shard_id=args.shard_id, seed=args.seed)
    print(json.dumps({"workers": results, **{key: summary[key] for key in (
        "recorded_games", "complete", "v4_mean_depth", "depth_ok", "v4_turns",
        "filtered_samples", "filtered_samples_per_game", "filtered_mean_depth",
        "filtered_depth_ok", "mean_game_seconds", "winners", "terminal_reasons",
        "v4_fallbacks", "v4_errors_or_exceptions",
    )}}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
