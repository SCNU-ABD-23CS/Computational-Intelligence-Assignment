"""Reproducible weight tournaments, GA tuning, validation and ablation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import argparse
import csv
import hashlib
from importlib import import_module
from io import StringIO
import json
import math
from pathlib import Path
import random
import statistics
import time
from unittest.mock import patch

import Tablut
import Team19_eval as evaluation

from . import HERE
from .openings import Opening, load_suite, start
from .runner import _reason
from .weight_player import make_weight_player, prepare_player_for_opening


RESULTS = HERE / "results" / "tuning"
FORFEITS = {"timeout", "illegal move", "exception"}
BOT_MODULES = {
    "random": ".bots.random_bot",
    "greedy": ".bots.greedy_bot",
    "naive": ".bots.naive_ab_d2",
    "mcts": ".bots.guided_mcts",
    "bfs": ".bots.bfs_strategist",
    "pvs": ".bots.qsearch_pvs",
    "numba": ".bots.grandmaster_numba",
}


@dataclass(frozen=True)
class Settings:
    depth: int = 2
    budget: float = 10.0
    mode: str = "fixed"
    tactical_ordering: bool = True

    def __post_init__(self):
        if type(self.depth) is not int or self.depth < 1:
            raise ValueError("depth must be a positive integer")
        if not math.isfinite(self.budget) or self.budget <= 0:
            raise ValueError("budget must be positive and finite")
        if self.mode not in ("fixed", "iterative"):
            raise ValueError("mode must be fixed or iterative")


@dataclass(frozen=True)
class Opponent:
    name: str
    kind: str
    weights: tuple[float, ...] | None = None


def opponent_pool(names: str, incumbent: list[float] | None = None) -> list[Opponent]:
    """Build a fixed named opponent pool; default/incumbent use the same search."""
    pool = []
    for raw_name in names.split(","):
        name = raw_name.strip().lower()
        if not name:
            continue
        if name == "default":
            pool.append(Opponent("default", "weights",
                                 tuple(evaluation.DEFAULT_WEIGHTS)))
        elif name == "incumbent":
            if incumbent is None:
                raise ValueError("The incumbent opponent requires --incumbent")
            pool.append(Opponent("incumbent", "weights",
                                 tuple(evaluation.normalize_weights(incumbent))))
        elif name in BOT_MODULES:
            pool.append(Opponent(name, "bot"))
        else:
            choices = ["default", "incumbent", *BOT_MODULES]
            raise ValueError(f"Unknown opponent {name!r}; choose from {choices}")
    if not pool or len({item.name for item in pool}) != len(pool):
        raise ValueError("Opponent pool must contain distinct, nonempty names")
    return pool


def select_openings(count: int, start_index: int = 0) -> list[Opening]:
    suite = load_suite()
    if count < 1 or start_index < 0 or start_index + count > len(suite):
        raise ValueError(f"Need 1..{len(suite)} openings within the fixed suite")
    return suite[start_index:start_index + count]


def _match_seed(seed: int, opening: Opening, opponent: Opponent,
                candidate_attacker: bool) -> int:
    text = f"{seed}|{opening.name}|{opponent.name}|{candidate_attacker}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _build_opponent(opponent: Opponent, seed: int, settings: Settings):
    if opponent.kind == "weights":
        return make_weight_player(
            opponent.weights, seed, settings.depth, settings.mode,
            settings.tactical_ordering,
        )
    module = import_module(BOT_MODULES[opponent.name], __package__)
    return module.Player(seed=seed)


def play_weight_game(weights: list[float], opponent: Opponent, opening: Opening,
                     candidate_attacker: bool, settings: Settings,
                     seed: int) -> dict:
    """Play one game through the unchanged official referee."""
    game = Tablut.Tablut()
    starting = start(opening, game)
    game_seed = _match_seed(seed, opening, opponent, candidate_attacker)
    output = StringIO()
    with patch.object(Tablut, "TIME_BUDGET", settings.budget), redirect_stdout(output):
        candidate = make_weight_player(
            weights, game_seed, settings.depth, settings.mode,
            settings.tactical_ordering,
        )
        other = _build_opponent(opponent, game_seed + 1, settings)
        prepare_player_for_opening(candidate, starting, candidate_attacker)
        prepare_player_for_opening(other, starting, not candidate_attacker)
        attacker, defender = ((candidate, other) if candidate_attacker
                              else (other, candidate))
        began = time.perf_counter()
        game.play(attacker, defender)
        elapsed = time.perf_counter() - began
    side = "A" if candidate_attacker else "D"
    points = 1 if game.winner is None else 2 * (game.winner == side)
    reason = _reason(game, output.getvalue())
    candidate_forfeit = reason in FORFEITS and game.winner != side
    return {
        "opening": opening.name,
        "opponent": opponent.name,
        "candidateSide": side,
        "winner": game.winner,
        "reason": reason,
        "points": points,
        "candidateForfeit": candidate_forfeit,
        "gameSeconds": round(elapsed, 6),
        "plies": game.rounds_played,
        "candidateRemaining": round(max(0.0, getattr(candidate, "remaining", 0.0)), 6),
    }


def aggregate_games(games: list[dict]) -> dict:
    if not games:
        raise ValueError("Cannot aggregate an empty game list")
    points = sum(game["points"] for game in games)
    wins = sum(game["points"] == 2 for game in games)
    draws = sum(game["points"] == 1 for game in games)
    return {
        "games": len(games),
        "points": points,
        "maxPoints": 2 * len(games),
        "pointRate": points / (2 * len(games)),
        "wins": wins,
        "draws": draws,
        "losses": len(games) - wins - draws,
        "forfeits": sum(game["candidateForfeit"] for game in games),
        "meanGameSeconds": statistics.fmean(game["gameSeconds"] for game in games),
    }


def evaluate_weights(weights: list[float], opponents: list[Opponent],
                     openings: list[Opening], settings: Settings,
                     seed: int) -> dict:
    """Evaluate one vector with paired sides on every opponent/opening pair."""
    active_weights = evaluation.normalize_weights(weights)
    games = [
        play_weight_game(active_weights, opponent, opening, candidate_attacker,
                         settings, seed)
        for opponent in opponents
        for opening in openings
        for candidate_attacker in (True, False)
    ]
    result = {
        "weights": active_weights,
        "weightDigest": weight_digest(active_weights),
        **aggregate_games(games),
        "byOpponent": {
            opponent.name: aggregate_games([
                game for game in games if game["opponent"] == opponent.name
            ])
            for opponent in opponents
        },
        "gameLog": games,
    }
    return result


def _evaluate_worker(payload: dict) -> dict:
    return evaluate_weights(
        payload["weights"],
        [Opponent(**item) for item in payload["opponents"]],
        [Opening(**item) for item in payload["openings"]],
        Settings(**payload["settings"]),
        payload["seed"],
    )


def evaluate_population(population: list[list[float]], opponents: list[Opponent],
                        openings: list[Opening], settings: Settings, seed: int,
                        workers: int = 1) -> list[dict]:
    """Evaluate independent candidates in worker processes without shared writes."""
    payloads = [{
        "weights": weights,
        "opponents": [asdict(item) for item in opponents],
        "openings": [asdict(item) for item in openings],
        "settings": asdict(settings),
        "seed": seed,
    } for weights in population]
    if workers == 1:
        return [_evaluate_worker(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_evaluate_worker, payloads))


def weight_digest(weights: list[float]) -> str:
    normalized = evaluation.normalize_weights(weights)
    payload = json.dumps(normalized, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")
    return path


def write_weight_file(path: Path, weights: list[float], metadata: dict | None = None) -> dict:
    document = {
        "format": "team19-weight-experiment-v1",
        "featureNames": list(evaluation.FEATURE_NAMES),
        "weights": evaluation.normalize_weights(weights),
        "weightDigest": weight_digest(weights),
        "createdUtc": datetime.now(UTC).isoformat(),
        "metadata": metadata or {},
    }
    write_json(path, document)
    return document


def load_weight_file(path: str | Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != "team19-weight-experiment-v1":
        raise ValueError("Not a Team19 weight experiment JSON file")
    if document.get("featureNames") != list(evaluation.FEATURE_NAMES):
        raise ValueError("Weight feature order does not match Team19_eval")
    weights = evaluation.normalize_weights(document["weights"])
    if document.get("weightDigest") != weight_digest(weights):
        raise ValueError("Weight digest does not match the vector")
    document["weights"] = weights
    return document


def _mutate(weights: list[float], rng: random.Random, mutation: float) -> list[float]:
    sigma = 10_000.0 * mutation
    return [
        float(max(-10_000.0, min(10_000.0,
            value + (rng.gauss(0.0, sigma) if rng.random() < 0.5 else 0.0))))
        for value in weights
    ]


def _crossover(first: list[float], second: list[float], rng: random.Random) -> list[float]:
    return [a + rng.random() * (b - a) for a, b in zip(first, second)]


def _initial_population(size: int, rng: random.Random, mutation: float) -> list[list[float]]:
    population = [list(evaluation.DEFAULT_WEIGHTS)]
    while len(population) < size:
        population.append(_mutate(list(evaluation.DEFAULT_WEIGHTS), rng, mutation * 2))
    return population


def _rank(results: list[dict]) -> list[dict]:
    # Wall time is reported but never breaks ties: deterministic selection makes
    # repeated runs comparable across fast and slow machines.
    return sorted(results, key=lambda item: (
        -item["pointRate"], item["forfeits"],
        item["weights"] != evaluation.DEFAULT_WEIGHTS, item["weightDigest"],
    ))


def write_evolution_svg(records: list[dict], path: Path) -> Path:
    """Write a dependency-free generation curve for the project report."""
    width, height, pad = 840, 330, 55
    xs = list(range(len(records)))
    max_x = max(1, len(records) - 1)

    def points(key):
        return " ".join(
            f"{pad + index * (width - 2 * pad) / max_x:.2f},"
            f"{height - pad - record[key] * (height - 2 * pad):.2f}"
            for index, record in enumerate(records)
        )

    circles = "".join(
        f'<circle cx="{pad + index * (width - 2 * pad) / max_x:.2f}" '
        f'cy="{height - pad - record["bestPointRate"] * (height - 2 * pad):.2f}" r="4" />'
        for index, record in enumerate(records)
    )
    labels = "".join(
        f'<text x="{pad + index * (width - 2 * pad) / max_x:.2f}" y="{height - 20}" '
        f'text-anchor="middle">{record["generation"]}</text>'
        for index in xs for record in [records[index]]
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font:13px sans-serif;fill:#263238}} .axis{{stroke:#607d8b;stroke-width:1}} .best{{fill:none;stroke:#1565c0;stroke-width:3}} .mean{{fill:none;stroke:#ef6c00;stroke-width:2;stroke-dasharray:7 5}} circle{{fill:#1565c0}}</style>
<text x="{width / 2}" y="24" text-anchor="middle" font-size="18">GA training fitness by generation</text>
<line class="axis" x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}"/>
<line class="axis" x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}"/>
<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle">Paired point rate</text>
<text x="{width / 2}" y="{height - 3}" text-anchor="middle">Generation</text>
<text x="8" y="{pad + 4}">1.0</text><text x="8" y="{height - pad + 4}">0.0</text>
<polyline class="mean" points="{points('meanPointRate')}"/>
<polyline class="best" points="{points('bestPointRate')}"/>{circles}{labels}
<text x="{width - 210}" y="48" fill="#1565c0">— Best</text><text x="{width - 130}" y="48" fill="#ef6c00">- - Mean</text>
</svg>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path


def genetic_optimize(population_size: int, generations: int, mutation: float,
                     elitism: int, opponents: list[Opponent],
                     openings: list[Opening], settings: Settings, seed: int,
                     workers: int, output_dir: Path) -> dict:
    """Run a real-valued elitist GA and checkpoint every completed generation."""
    if population_size < 2 or generations < 1:
        raise ValueError("GA requires population >= 2 and generations >= 1")
    if not 0 < mutation <= 0.5:
        raise ValueError("mutation must be in (0, 0.5]")
    if not 1 <= elitism < population_size:
        raise ValueError("elitism must be at least 1 and smaller than population")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty experiment directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    population = _initial_population(population_size, rng, mutation)
    configuration = {
        "populationSize": population_size,
        "generations": generations,
        "mutation": mutation,
        "elitism": elitism,
        "opponents": [asdict(item) for item in opponents],
        "openings": [asdict(item) for item in openings],
        "settings": asdict(settings),
        "seed": seed,
        "workers": workers,
    }
    write_json(output_dir / "config.json", configuration)
    records = []
    best = None
    with (output_dir / "generations.jsonl").open("w", encoding="utf-8") as stream:
        for generation in range(generations):
            results = _rank(evaluate_population(
                population, opponents, openings, settings, seed, workers,
            ))
            best = results[0]
            record = {
                "generation": generation,
                "bestPointRate": best["pointRate"],
                "meanPointRate": statistics.fmean(item["pointRate"] for item in results),
                "bestWeights": best["weights"],
                "bestWeightDigest": best["weightDigest"],
                "candidates": results,
            }
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            write_evolution_svg(records, output_dir / "evolution.svg")
            write_weight_file(
                output_dir / "best_weights.json", best["weights"],
                {"generation": generation, "trainingPointRate": best["pointRate"],
                 "seed": seed, "settings": asdict(settings)},
            )
            print(f"generation {generation}: best={best['pointRate']:.4f} "
                  f"mean={record['meanPointRate']:.4f} weights={best['weights']}", flush=True)

            ranked_weights = [item["weights"] for item in results]
            next_population = [list(item) for item in ranked_weights[:elitism]]
            while len(next_population) < population_size - 1:
                def tournament():
                    contenders = rng.sample(results, min(3, len(results)))
                    return max(contenders, key=lambda item: item["pointRate"])["weights"]
                child = _crossover(tournament(), tournament(), rng) if rng.random() < 0.75 \
                    else list(tournament())
                next_population.append(_mutate(child, rng, mutation))
            # Keep the hand-set reference in every generation as a safety anchor.
            next_population.append(list(evaluation.DEFAULT_WEIGHTS))
            population = next_population
    return best


def run_validation(candidate: dict, opponents: list[Opponent], openings: list[Opening],
                   settings: Settings, seed: int, workers: int) -> dict:
    results = evaluate_population(
        [candidate["weights"], list(evaluation.DEFAULT_WEIGHTS)],
        opponents, openings, settings, seed, workers,
    )
    candidate_result, default_result = results
    passed = (candidate_result["pointRate"] > default_result["pointRate"]
              and candidate_result["forfeits"] == 0)
    return {
        "format": "team19-weight-validation-v1",
        "createdUtc": datetime.now(UTC).isoformat(),
        "candidateDigest": candidate["weightDigest"],
        "candidate": candidate_result,
        "default": default_result,
        "opponents": [asdict(item) for item in opponents],
        "openings": [asdict(item) for item in openings],
        "settings": asdict(settings),
        "seed": seed,
        "passed": passed,
        "criterion": "candidate pointRate must exceed default and candidate forfeits must be zero",
    }


def export_weights(candidate_path: Path, validation_path: Path, output: Path) -> Path:
    """Export only a matching, independently validated improvement."""
    candidate = load_weight_file(candidate_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("format") != "team19-weight-validation-v1":
        raise ValueError("Not a Team19 validation file")
    if validation.get("candidateDigest") != candidate["weightDigest"]:
        raise ValueError("Validation belongs to a different candidate")
    if validation["candidate"]["weightDigest"] != candidate["weightDigest"]:
        raise ValueError("Validation result digest does not match the candidate")
    passed = (validation["candidate"]["pointRate"] > validation["default"]["pointRate"]
              and validation["candidate"]["forfeits"] == 0)
    if not passed:
        raise ValueError("Export refused: candidate did not beat default without forfeits")
    lines = [
        '"""Offline-tuned Team19 weights; see the accompanying validation JSON."""',
        "",
        f"# Candidate SHA-256: {candidate['weightDigest']}",
        f"# Validation point rate: {validation['candidate']['pointRate']:.6f}",
        f"# Default point rate:   {validation['default']['pointRate']:.6f}",
        "WEIGHTS = [",
        *[f"    {value!r},  # {name}" for value, name in zip(
            candidate["weights"], evaluation.FEATURE_NAMES)],
        "]",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _write_csv(path: Path, rows: list[dict]) -> Path:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _summary_row(name: str, result: dict) -> dict:
    return {
        "variant": name,
        "games": result["games"],
        "W-D-L": f"{result['wins']}-{result['draws']}-{result['losses']}",
        "points": f"{result['points']}/{result['maxPoints']}",
        "pointRate": f"{result['pointRate']:.4f}",
        "forfeits": result["forfeits"],
        "meanGameSeconds": f"{result['meanGameSeconds']:.3f}",
    }


def _write_markdown_table(path: Path, title: str, rows: list[dict]) -> Path:
    keys = list(rows[0])
    lines = [f"# {title}", "", "| " + " | ".join(keys) + " |",
             "| " + " | ".join("---" for _ in keys) + " |"]
    lines += ["| " + " | ".join(str(row[key]) for key in keys) + " |" for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _safe_run_id(value: str) -> str:
    if not value or value in (".", "..") or Path(value).name != value:
        raise ValueError("run-id must be one safe path component")
    return value


def _settings(args) -> Settings:
    return Settings(args.depth, args.budget, args.mode, not args.engine_order)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--mode", choices=("fixed", "iterative"), default="fixed")
    parser.add_argument("--engine-order", action="store_true",
                        help="Disable tactical move ordering for controlled comparisons")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261122)


def _check_workers(args) -> None:
    if not 1 <= args.workers <= 32:
        raise ValueError("workers must be between 1 and 32")


def _command_smoke(args) -> None:
    _check_workers(args)
    output = RESULTS / _safe_run_id(args.run_id)
    opponents = opponent_pool("random,greedy")
    result = evaluate_weights(evaluation.DEFAULT_WEIGHTS, opponents,
                              select_openings(1), _settings(args), args.seed)
    write_json(output / "smoke.json", {
        "settings": asdict(_settings(args)), "seed": args.seed, "result": result,
    })
    _write_markdown_table(output / "summary.md", "Member 5 smoke test",
                          [_summary_row("default", result)])
    print(json.dumps(aggregate_games(result["gameLog"]), indent=2))


def _command_ga(args) -> None:
    _check_workers(args)
    settings = _settings(args)
    if settings.mode != "fixed":
        raise ValueError("GA training uses --mode fixed; use validate for iterative mode")
    output = RESULTS / _safe_run_id(args.run_id)
    incumbent = load_weight_file(args.incumbent)["weights"] if args.incumbent else None
    train_opponents = opponent_pool(args.train_opponents, incumbent)
    validation_opponents = opponent_pool(args.validation_opponents)
    train_openings = select_openings(args.train_openings, 0)
    validation_openings = select_openings(args.validation_openings, args.train_openings)
    best = genetic_optimize(
        args.population, args.generations, args.mutation, args.elitism,
        train_opponents, train_openings, settings, args.seed, args.workers, output,
    )
    candidate = load_weight_file(output / "best_weights.json")
    validation = run_validation(candidate, validation_opponents, validation_openings,
                                settings, args.seed + 1, args.workers)
    write_json(output / "validation.json", validation)
    rows = [_summary_row("candidate", validation["candidate"]),
            _summary_row("default", validation["default"])]
    _write_markdown_table(output / "validation.md", "Held-out weight validation", rows)
    print(f"best weights: {best['weights']}")
    print(f"held-out candidate={validation['candidate']['pointRate']:.4f} "
          f"default={validation['default']['pointRate']:.4f} passed={validation['passed']}")
    print(f"results: {output}")


def _command_validate(args) -> None:
    _check_workers(args)
    candidate = load_weight_file(args.candidate)
    openings = select_openings(args.openings, args.opening_start)
    validation = run_validation(candidate, opponent_pool(args.opponents), openings,
                                _settings(args), args.seed, args.workers)
    output = Path(args.output) if args.output else Path(args.candidate).parent / (
        f"validation-{args.mode}-d{args.depth}-{args.budget:g}s.json"
    )
    write_json(output, validation)
    _write_markdown_table(output.with_suffix(".md"), "Independent weight validation", [
        _summary_row("candidate", validation["candidate"]),
        _summary_row("default", validation["default"]),
    ])
    print(json.dumps({"passed": validation["passed"],
                      "candidate": _summary_row("candidate", validation["candidate"]),
                      "default": _summary_row("default", validation["default"])}, indent=2))
    print(f"validation: {output}")


def _command_ablate(args) -> None:
    _check_workers(args)
    candidate = load_weight_file(args.candidate)
    variants = [("full", candidate["weights"])]
    for index, name in enumerate(evaluation.FEATURE_NAMES):
        weights = list(candidate["weights"])
        weights[index] = 0.0
        variants.append((name, weights))
    results = evaluate_population(
        [weights for _, weights in variants], opponent_pool(args.opponents),
        select_openings(args.openings, args.opening_start), _settings(args),
        args.seed, args.workers,
    )
    full_rate = results[0]["pointRate"]
    rows = []
    for (name, _), result in zip(variants, results):
        rows.append({
            **_summary_row(name, result),
            "deltaPointRate": f"{result['pointRate'] - full_rate:.4f}",
        })
    output = Path(args.candidate).parent / "ablation"
    _write_csv(output / "ablation.csv", rows)
    write_json(output / "ablation.json", {
        "candidateDigest": candidate["weightDigest"],
        "settings": asdict(_settings(args)), "seed": args.seed,
        "results": rows,
    })
    _write_markdown_table(output / "ablation.md", "Single-feature weight ablation", rows)
    print(f"ablation: {output / 'ablation.csv'}")


def _command_depths(args) -> None:
    _check_workers(args)
    candidate = load_weight_file(args.candidate)
    depths = [int(value) for value in args.depths.split(",")]
    if not depths or len(set(depths)) != len(depths):
        raise ValueError("depths must be distinct comma-separated positive integers")
    rows = []
    raw = []
    openings = select_openings(args.openings, args.opening_start)
    opponents = opponent_pool(args.opponents)
    for depth in depths:
        settings = Settings(depth, args.budget, args.mode, not args.engine_order)
        results = evaluate_population(
            [candidate["weights"], list(evaluation.DEFAULT_WEIGHTS)],
            opponents, openings, settings, args.seed, args.workers,
        )
        for name, result in zip(("candidate", "default"), results):
            rows.append({"depth": depth, **_summary_row(name, result)})
        raw.append({"depth": depth, "candidate": results[0], "default": results[1]})
        print(f"depth {depth}: candidate={results[0]['pointRate']:.4f} "
              f"default={results[1]['pointRate']:.4f}", flush=True)
    output = Path(args.candidate).parent / "depth_comparison"
    _write_csv(output / "depths.csv", rows)
    write_json(output / "depths.json", raw)
    _write_markdown_table(output / "depths.md", "Search-depth comparison", rows)
    print(f"depth comparison: {output / 'depths.csv'}")


def _command_export(args) -> None:
    output = export_weights(Path(args.candidate), Path(args.validation), Path(args.output))
    print(f"exported: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser("smoke", help="Run default weights against Random and Greedy")
    _add_common(smoke)
    smoke.add_argument("--run-id", default="smoke")
    smoke.set_defaults(function=_command_smoke)

    ga = commands.add_parser("ga", help="Tune the seven weights with an elitist real-valued GA")
    _add_common(ga)
    ga.add_argument("--run-id", default="ga-d2")
    ga.add_argument("--population", type=int, default=8)
    ga.add_argument("--generations", type=int, default=8)
    ga.add_argument("--mutation", type=float, default=0.08)
    ga.add_argument("--elitism", type=int, default=1)
    ga.add_argument("--incumbent", help="Optional previous best_weights.json for the training pool")
    ga.add_argument("--train-opponents", default="default,greedy,random")
    ga.add_argument("--validation-opponents", default="naive,mcts")
    ga.add_argument("--train-openings", type=int, default=2)
    ga.add_argument("--validation-openings", type=int, default=2)
    ga.set_defaults(function=_command_ga)

    validate = commands.add_parser("validate", help="Compare candidate and default on held-out games")
    _add_common(validate)
    validate.add_argument("--candidate", required=True)
    validate.add_argument("--opponents", default="naive,mcts")
    validate.add_argument("--openings", type=int, default=2)
    validate.add_argument("--opening-start", type=int, default=2)
    validate.add_argument("--output")
    validate.set_defaults(function=_command_validate)

    ablate = commands.add_parser("ablate", help="Set each feature weight to zero in turn")
    _add_common(ablate)
    ablate.add_argument("--candidate", required=True)
    ablate.add_argument("--opponents", default="default,naive")
    ablate.add_argument("--openings", type=int, default=2)
    ablate.add_argument("--opening-start", type=int, default=2)
    ablate.set_defaults(function=_command_ablate)

    depths = commands.add_parser("depths", help="Compare candidate and default at several depths")
    _add_common(depths)
    depths.add_argument("--candidate", required=True)
    depths.add_argument("--depths", default="1,2,3")
    depths.add_argument("--opponents", default="default,naive")
    depths.add_argument("--openings", type=int, default=2)
    depths.add_argument("--opening-start", type=int, default=2)
    depths.set_defaults(function=_command_depths)

    export = commands.add_parser("export", help="Export Team19_weights.py only after a passed validation")
    export.add_argument("--candidate", required=True)
    export.add_argument("--validation", required=True)
    export.add_argument("--output", default="Team19_weights.py")
    export.set_defaults(function=_command_export)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; completed generation files remain available.\n")
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
