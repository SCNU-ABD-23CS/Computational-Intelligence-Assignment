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

from .common import FROZEN_TEAMS, NNUE_RESULTS, d4_key, read_jsonl, write_json


SEED = 20261125
OPPONENTS = ("Team19-v4", "Team19-v3", "Grandmaster-Plus", "Grandmaster-Numba")
PREFIX_MIN, PREFIX_MAX = 2, 6


def random_opening(index: int, used_families: set[str]) -> Opening:
    """Reject prefix families that are reachable by board symmetry."""
    rng = random.Random(SEED + index)
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


def build_matches(games: int, budget: float) -> list[Match]:
    used: set[str] = set()
    matches = []
    for index in range(games):
        opening = random_opening(index, used)
        opponent = OPPONENTS[index % len(OPPONENTS)]
        if opponent == "Team19-v4":
            attacker = defender = "Team19-v4"
        elif index % 2 == 0:
            attacker, defender = "Team19-v4", opponent
        else:
            attacker, defender = opponent, "Team19-v4"
        matches.append(Match(attacker, defender, budget, opening))
    return matches


def _worker(payload) -> dict:
    worker, tag, matches, budget = payload
    directory = NNUE_RESULTS / tag
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
                "tag": tag, "seed": SEED, "worker": worker,
                "prefixPlies": len(match.opening.moves),
                "budget": budget,
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


def summarise(tag: str, budget: float, requested: int, workers: int) -> dict:
    directory = NNUE_RESULTS / tag
    rows = []
    for worker in range(workers):
        path = directory / f"worker_{worker}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing worker journal {path}")
        rows.extend(read_jsonl(path))
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
    for game in ordered:
        winners[str(game["winner"])] += 1
        reasons[game["reason"]] += 1
        for name in game["teams"].values():
            teams[name] += 1
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
    summary = {
        "schema": 1,
        "tag": tag,
        "budget_seconds": budget,
        "workers": workers,
        "requested_games": requested,
        "recorded_games": len(ordered),
        "complete": len(ordered) == requested,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "prefix_plies": {"min": PREFIX_MIN, "max": PREFIX_MAX},
        "opponents": list(OPPONENTS),
        "team_appearances": dict(teams),
        "winners": dict(winners),
        "terminal_reasons": dict(reasons),
        "v4_turns": v4_turns,
        "v4_side_turns": dict(side_counts),
        "v4_depth_histogram": dict(sorted(depth_counts.items())),
        "v4_mean_depth": mean_depth,
        "depth_target": 5.0,
        "depth_ok": mean_depth >= 5.0,
        "v4_mean_move_seconds": (sum(v4_seconds) / len(v4_seconds) if v4_seconds else 0.0),
        "v4_max_move_seconds": (max(v4_seconds) if v4_seconds else 0.0),
        "v4_fallbacks": fallbacks,
        "v4_errors_or_exceptions": exceptions,
        "paths": {"games": str(directory / "games.jsonl"),
                  "summary": str(directory / "collection_summary.json")},
    }
    write_json(directory / "collection_summary.json", summary)
    with (directory / "games.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for game in ordered:
            stream.write(json.dumps(game, ensure_ascii=False, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--tag", default="pilot300")
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.budget < 1.0:
        parser.error("--budget must be at least one second")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    matches = build_matches(args.games, args.budget)
    if len({match.id for match in matches}) != len(matches):
        raise ValueError("Duplicate fixture ids in generated schedule")
    chunks = [matches[index::args.workers] for index in range(args.workers)]
    payloads = [(index, args.tag, chunk, args.budget) for index, chunk in enumerate(chunks)]
    print(f"tag={args.tag} games={len(matches)} workers={args.workers} budget={args.budget}s",
          flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(_worker, payloads)
    summary = summarise(args.tag, args.budget, args.games, args.workers)
    print(json.dumps({"workers": results, **{key: summary[key] for key in (
        "recorded_games", "complete", "v4_mean_depth", "depth_ok", "v4_turns",
        "winners", "terminal_reasons", "v4_fallbacks", "v4_errors_or_exceptions",
    )}}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
