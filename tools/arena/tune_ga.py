"""Tune the native evaluation at fixed depth and validate weights on held-out openings."""

from __future__ import annotations

from argparse import ArgumentParser
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

import Tablut
import Team19_engine as engine
import Team19_search as search
from Team19_eval import _native_piece_bonus
from Team19_weights import DEFAULT_WEIGHTS, FEATURE_NAMES, native_weights

from . import HERE, openings, runner
from .bots import grandmaster_numba as grandmaster


RESULTS = HERE / "results"
POSITIONS = HERE / "tuning_positions.json"
CANDIDATE_FILE = RESULTS / "team19-v2-candidate.json"
TRAIN_SEED = 20261122
MAX_PLIES = 48
ELITES = 4
STRONG, WARM = "Grandmaster-Numba", "hot-start"
DEFAULT = "DEFAULT_WEIGHTS"


class TrainingDeadline(Exception):
    """The current candidate did not finish within the training allowance."""


def load_weights(source=DEFAULT) -> np.ndarray:
    if str(source) == DEFAULT:
        return native_weights(DEFAULT_WEIGHTS)
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "features" in data and tuple(data["features"]) != FEATURE_NAMES:
            raise ValueError("The weight file uses a different feature order")
        data = data["weights"]
    return native_weights(data)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
                       + "\n", encoding="utf-8")
    pending.replace(path)


@dataclass(frozen=True)
class TrainingCase:
    opening: openings.Opening
    referee: Tablut.Tablut
    candidate_attacks: bool
    opponent: str


def training_cases() -> list[TrainingCase]:
    source = json.loads(POSITIONS.read_text(encoding="utf-8"))
    midgames = [openings.Opening(row["name"], row["moves"], row["signature"])
                for row in source["midgames"]]
    base = openings.load_suite()
    cases = []
    for index, (opening, middle) in enumerate(zip((base[0], base[3], base[8], base[11]), midgames,
                                                  strict=True)):
        opponent = STRONG if index % 2 == 0 else WARM
        for position in (opening, middle):
            game = openings.start(position).referee
            cases.append(TrainingCase(position, game, True, opponent))
        cases.append(TrainingCase(middle, game, False, opponent))
    return cases


def _native_move(workspace, board, attacker, depth, played, deadline):
    flat, state = search._native_position(board)
    work = workspace.work
    work.generation[0] += 1
    work.killers.fill(-1)
    work.history.fill(0)
    work.counts.fill(0)
    clock_deadline = search._read_native_clock(work.clock) + max(0.0, deadline - perf_counter())
    alpha, beta = -2 * search._MATE, 2 * search._MATE
    score, move = workspace._iteration(flat, attacker, state, depth, search._Q_LIMIT, alpha,
                                      beta, clock_deadline, played, True)
    if (not work.counts[search._STOPPED] and work.counts[search._REDUCTIONS]
            and abs(score) >= search._MATE - search._NATIVE_PLY_SLOTS):
        score, move = workspace._iteration(flat, attacker, state, depth, search._Q_LIMIT, alpha,
                                          beta, clock_deadline, played, False)
    if work.counts[search._STOPPED] or perf_counter() >= deadline:
        raise TrainingDeadline
    if move < 0:
        raise RuntimeError("Native search did not return a move on a playable position")
    return engine._move_list(move)


def _grandmaster_move(player, board, attacker, depth, played, deadline):
    flat = grandmaster._flat(board)
    state = grandmaster._root_state(flat)
    work = player.work
    work.nodes[0] = 0
    work.stop[0] = False
    work.tt_moves.fill(-1)
    work.killers.fill(-1)
    work.history.fill(0)
    _, move = player._iteration(flat, attacker, state, depth, -2 * grandmaster.MATE,
                                2 * grandmaster.MATE, deadline, played)
    if work.stop[0] or perf_counter() >= deadline:
        raise TrainingDeadline
    if move < 0:
        raise RuntimeError("Grandmaster did not return a move on a playable position")
    return grandmaster._move_list(move)


class FixedDepthTrainer:
    def __init__(self, cases, depth, baseline):
        self.cases, self.depth = cases, depth
        initial = Tablut.Tablut().board
        self.candidate = search._NativeSearch(initial, baseline)
        self.warm = search._NativeSearch(initial, baseline)
        self.grandmaster = grandmaster.Player()

    def play(self, case, deadline):
        game = deepcopy(case.referee)
        attacker = True
        self.candidate.reset_game()
        self.warm.reset_game()
        with redirect_stdout(StringIO()):
            for ply in range(1, MAX_PLIES + 1):
                if perf_counter() >= deadline:
                    raise TrainingDeadline
                played = np.asarray(sorted(set(game.past_positions)), dtype=np.uint64)
                if attacker == case.candidate_attacks:
                    move = _native_move(self.candidate, game.board, attacker, self.depth,
                                        played, deadline)
                elif case.opponent == STRONG:
                    move = _grandmaster_move(self.grandmaster, game.board, attacker, self.depth,
                                             played, deadline)
                else:
                    move = _native_move(self.warm, game.board, attacker, self.depth, played, deadline)
                if not game.make_move(attacker, move):
                    raise RuntimeError(f"Fixed-depth training returned an illegal move at {case.opening.name}")
                attacker = not attacker
                if game.check_if_game_over(attacker):
                    break
        winner = game.winner
        won = winner == ("A" if case.candidate_attacks else "D")
        return {"opening": case.opening.name, "opponent": case.opponent,
                "side": "A" if case.candidate_attacks else "D",
                "points": 1 if winner is None else 2 * won, "plies": ply}

    def score(self, weights, deadline):
        self.candidate.work.weights[:] = weights
        self.candidate.work.piece_bonus[:] = _native_piece_bonus(weights)
        games = [self.play(case, deadline) for case in self.cases]
        rates = {
            (side, opponent): mean(game["points"] / 2 for game in games
                                   if game["side"] == side and game["opponent"] == opponent)
            for side in ("A", "D") for opponent in (STRONG, WARM)
        }
        fitness = 0.7 * (0.65 * rates["A", STRONG] + 0.35 * rates["A", WARM])
        fitness += 0.3 * (0.65 * rates["D", STRONG] + 0.35 * rates["D", WARM])
        # At equal point rates, prefer converting attacker wins in fewer plies.
        conversion = mean((MAX_PLIES - game["plies"]) / MAX_PLIES
                          if game["points"] == 2 else 0.0 for game in games if game["side"] == "A")
        return fitness, conversion, games


def mutation_sigma(baseline):
    sigma = np.maximum(30.0, np.abs(baseline) * 0.14)
    sigma[-2:] = np.maximum(sigma[-2:], (100.0, 30.0))
    return sigma


def mutate(population, sigma, rng):
    parent = population[int(rng.integers(len(population)))]
    weights = np.asarray(parent["weights"])
    mask = rng.random(len(sigma)) < 0.45
    mask[-2:] |= rng.random(2) < 0.55
    noise = rng.normal(0, sigma * rng.uniform(0.5, 1.5))
    return np.clip(weights + mask * noise, -10_000.0, 10_000.0)


def _ranking(entry):
    return entry["fitness"], entry["conversion"]


def _checkpoint_config(depth, seed, cases):
    root = HERE.parents[1]
    paths = (root / "Team19_engine.py", root / "Team19_eval.py", root / "Team19_search.py",
             HERE / "bots" / "grandmaster_numba.py", HERE / "bots" / "numba_rules.py",
             Path(__file__))
    code = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return {"depth": depth, "seed": seed, "code": code,
            "cases": [[case.opening.signature, case.candidate_attacks, case.opponent] for case in cases]}


def train(args):
    began = perf_counter()
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        raise ValueError("Training time must be finite and positive")
    deadline = began + args.time_limit
    path = args.output
    checkpoint = args.resume if args.resume else path.with_suffix(".checkpoint.json")
    cases = training_cases()
    rng = np.random.default_rng()
    if args.resume:
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        depth = args.depth if args.depth is not None else state["config"]["depth"]
        seed = args.seed if args.seed is not None else state["config"]["seed"]
        if state["config"] != _checkpoint_config(depth, seed, cases) or args.init_weights:
            raise ValueError("Resume with the checkpoint's search, depth, seed and initial weights")
        rng.bit_generator.state = state["rng"]
    else:
        depth = 4 if args.depth is None else args.depth
        seed = TRAIN_SEED if args.seed is None else args.seed
        rng = np.random.default_rng(seed)
        baseline = load_weights()
        initial = load_weights(args.init_weights or DEFAULT)
        state = {"config": _checkpoint_config(depth, seed, cases), "baseline": baseline.tolist(),
                 "initial": initial.tolist(), "sigma": mutation_sigma(initial).tolist(), "history": [],
                 "rng": rng.bit_generator.state}

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with runner._writer_lock(checkpoint):
        if not args.resume and (checkpoint.exists() or path.exists()):
            raise ValueError("Use --resume or a new --output to preserve the prior run")
        trainer = FixedDepthTrainer(cases, depth, load_weights() if not args.resume else
                                    native_weights(state["baseline"]))
        history = state["history"]
        population = sorted(history, key=_ranking, reverse=True)[:ELITES]
        print(f"D{depth} | {len(cases)} training games per candidate | seed {seed} | "
              f"{args.time_limit:g}s | {len(history)} evaluations recorded", flush=True)
        while perf_counter() < deadline:
            weights = (mutate(population, np.asarray(state["sigma"]), rng) if population
                       else native_weights(state["initial"]))
            try:
                score, conversion, games = trainer.score(weights, deadline)
            except TrainingDeadline:
                break
            entry = {"index": len(history), "fitness": score, "conversion": conversion,
                     "weights": weights.tolist(), "games": games, "elapsed": perf_counter() - began}
            history.append(entry)
            population = sorted([*population, entry], key=_ranking, reverse=True)[:ELITES]
            state["rng"] = rng.bit_generator.state
            write_json(checkpoint, state)
            print(f"{entry['index']:03d} | {score:.4f} | best {population[0]['fitness']:.4f} | "
                  f"ray weights {weights[-2]:+.1f}, {weights[-1]:+.1f} | {entry['elapsed']:.1f}s",
                  flush=True)
        mutants = [entry for entry in history if entry["weights"] != state["baseline"]]
        if not mutants:
            raise ValueError("No non-baseline candidate completed its fixture list")
        chosen = max(mutants, key=_ranking)
        write_json(path, {"features": FEATURE_NAMES, "weights": chosen["weights"],
                          "selection": chosen["index"], "elapsed": perf_counter() - began,
                          "baseline": state["baseline"], "config": state["config"],
                          "sigma": state["sigma"], "history": history})
    print(path, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evolve = subparsers.add_parser("train", help="evolve one candidate using the training suite")
    evolve.add_argument("--resume", type=Path)
    evolve.add_argument("--init-weights")
    evolve.add_argument("--seed", type=int)
    evolve.add_argument("--depth", choices=(3, 4), type=int)
    evolve.add_argument("--time-limit", type=float, default=1560)
    evolve.add_argument("--output", type=Path, default=CANDIDATE_FILE)
    verify = subparsers.add_parser("validate", help="run the unused opening suite serially at both budgets")
    verify.add_argument("--candidate", default=CANDIDATE_FILE)
    verify.add_argument("--baseline", default=DEFAULT)
    verify.add_argument("--run-id", default="team19-v2-validation")
    verify.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "train":
            train(args)
        else:
            from .validate_v2 import validate

            validate(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
