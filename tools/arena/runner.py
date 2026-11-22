"""Run serial, resumable matches through the official Tablut.play method."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
from importlib import import_module
from io import StringIO
from itertools import combinations
import json
import operator
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

import Tablut

from . import HERE, RESULTS, TOP6_BUDGETS, TOP6_OPENINGS, TOP6_TEAMS
from .openings import Opening, load_suite, move_event, start, standard
from .top6 import match_seconds, progress_text, reusable_games


SCHEMA = 1


@dataclass(frozen=True)
class Team:
    module: str
    search: str | None = None


TEAMS = {
    "Team19-New": Team("Team19", "Team19_search"),
    "Team19-PreFix": Team(".snapshots.v_e3d64c4.Team19", ".snapshots.v_e3d64c4.Team19_search"),
    "Team19-Old": Team(".snapshots.v_99671ae.Team19", ".snapshots.v_99671ae.Team19_search"),
    "GreedyBot": Team(".bots.greedy_bot"),
    "NaiveAB-D2": Team(".bots.naive_ab_d2"),
    "Guided-MCTS": Team(".bots.guided_mcts"),
    "BFS-Strategist": Team(".bots.bfs_strategist"),
    "QSearch-PVS": Team(".bots.qsearch_pvs"),
    "Grandmaster-Numba": Team(".bots.grandmaster_numba"),
    "RandomBot": Team(".bots.random_bot"),
}
LEAGUE = tuple(name for name in TEAMS if name != "RandomBot")
EXTERNAL = LEAGUE[3:]
TEAMS.update({
    "Team19-91ffcc8": Team(".snapshots.v_91ffcc8.Team19", ".snapshots.v_91ffcc8.Team19_search"),
    "Team19-v2": Team("Team19", "Team19_search"),
})


@dataclass(frozen=True)
class Match:
    attacker: str
    defender: str
    budget: float
    opening: Opening

    @property
    def id(self) -> str:
        return f"{self.budget:g}-{self.opening.name}-{self.attacker}-A-{self.defender}"


def paired(pairs, openings: list[Opening], budget: float) -> list[Match]:
    return [Match(attacker, defender, budget, opening)
            for a, b in pairs for opening in openings
            for attacker, defender in ((a, b), (b, a))]


class ObservedReferee(Tablut.Tablut):
    """Capture accepted move events after the official implementation applies them."""

    attempts: list[dict] | None = None
    prefix_plies: int = 0

    def make_move(self, is_attacker, moves):
        before = self.board
        accepted = super().make_move(is_attacker, moves)
        if accepted and self.attempts is not None:
            ply = self.prefix_plies + len(self.attempts)
            self.attempts[-1]["event"] = move_event(self, before, is_attacker, moves, ply)
        return accepted


class OfficialClock:
    """Pass real time.time() values to the referee and retain its exact readings."""

    def __init__(self):
        self.ticks = []

    def time(self) -> float:
        clock = time.time()
        self.ticks.append(clock)
        return clock


def _move_or_illegal(move):
    try:
        (r, c), (rr, cc) = move
        return [[operator.index(r), operator.index(c)], [operator.index(rr), operator.index(cc)]]
    except (TypeError, ValueError, OverflowError):
        # Tablut.make_move handles a list of the wrong length as an illegal move.
        return []


class ObservedPlayer:
    def __init__(self, team: str, player, attempts: list[dict], prefix_restore=None):
        self.team = team
        self.player = player
        self.attempts = attempts
        self.prefix_restore = prefix_restore
        self.depths = []
        self.partial_nodes = 0
        self.recording = False

    def observe_search(self, stack: ExitStack) -> None:
        relative_name = TEAMS[self.team].search
        if relative_name is None:
            return
        search = import_module(relative_name, __package__)
        original = search._search_depth

        def observe(board, side, depth, *args, **kwargs):
            if not self.recording:
                return original(board, side, depth, *args, **kwargs)
            try:
                result, horizon = original(board, side, depth, *args, **kwargs)
            except search.SearchTimeout as error:
                self.partial_nodes += getattr(error, "nodes_visited", 0)
                raise
            self.depths.append((depth, result))
            return result, horizon

        stack.enter_context(patch.object(search, "_search_depth", observe))

    def play_attacker(self, board):
        return self._play(board, True)

    def play_defender(self, board):
        return self._play(board, False)

    def _play(self, board, attacker):
        if self.prefix_restore is not None:
            game, history = self.prefix_restore
            game.past_positions[:] = history
            self.prefix_restore = None
        record = {"side": "A" if attacker else "D", "team": self.team}
        self.attempts.append(record)
        self.depths.clear()
        self.partial_nodes = 0
        previous_error = getattr(self.player, "last_error", None)
        self.recording = True
        try:
            move = (self.player.play_attacker(board) if attacker else self.player.play_defender(board))
            record["move"] = _move_or_illegal(move)
            return record["move"]
        except Exception as error:
            record["exception"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.recording = False
            telemetry = getattr(self.player, "last_search", None)
            latest_error = getattr(self.player, "last_error", None)
            if latest_error is not None and latest_error is not previous_error:
                record["error"] = str(latest_error)
            if telemetry is not None:
                record.update(asdict(telemetry))
            elif TEAMS[self.team].search is not None:
                depth, result = self.depths[-1] if self.depths else (0, None)
                record.update(
                    depth=depth, nodes=sum(item.nodes_visited for _, item in self.depths)
                    + self.partial_nodes,
                    score=result.score if result else None,
                    fallback="move" in record and (
                        result is None or record["move"] != result.move or "error" in record
                    ),
                )
            record["playerRemaining"] = getattr(self.player, "remaining", None)


def _player(name, seed, starting, side, factory=None):
    entrant = TEAMS[name]
    if factory is None:
        module = import_module(entrant.module, __package__)
        player = module.Player() if entrant.search else module.Player(seed=seed)
    else:
        player = factory()
    reset = getattr(player, "reset_game", None)
    if reset is not None:
        reset()
    if entrant.search:
        player.history.update(starting.referee.past_positions)
        if starting.turns:
            player._last_board = deepcopy(starting.last_board[side])
            player._last_is_attacker = side
            if not side:
                player._opening_replies.add(tuple(tuple(row) for row in starting.positions[1]["board"]))
    elif hasattr(player, "seed_history"):
        player.seed_history(starting.referee.past_positions)
    return player


def _reason(game, messages):
    if "exceeded time limit!" in messages:
        return "timeout"
    if "Illegal move by" in messages:
        return "illegal move"
    if "Exception caught for" in messages:
        return "exception"
    if game.winner == "A":
        return "king captured"
    if game.winner == "D":
        on_edge = any(cell == "K" and (r in (0, 8) or c in (0, 8))
                      for r, row in enumerate(game.board) for c, cell in enumerate(row))
        return "king escaped to the edge" if on_edge else "fewer than two attackers remain"
    if "Repeated position found" in messages:
        return "repeated position"
    if "Player cannot move" in messages:
        return "no legal move"
    raise RuntimeError(f"Unrecognized referee outcome: {messages!r}")


def run_match(match: Match, *, factories=None) -> dict:
    started = datetime.now(UTC).isoformat()
    game = ObservedReferee()
    opening = start(match.opening, game)
    preserved = game.past_positions[:]
    seed = int.from_bytes(hashlib.sha256(match.id.encode()).digest()[:8], "big")
    attempts = []
    clock = OfficialClock()
    output = StringIO()
    with ExitStack() as stack:
        stack.enter_context(patch.object(Tablut, "TIME_BUDGET", match.budget))
        init_time = {}
        observers = []
        for side, name in ((True, match.attacker), (False, match.defender)):
            begin = time.perf_counter()
            factory = factories.get(name) if factories else None
            player = _player(name, seed, opening, side, factory)
            init_time["A" if side else "D"] = time.perf_counter() - begin
            observer = ObservedPlayer(name, player, attempts, (game, preserved) if side else None)
            observer.observe_search(stack)
            observers.append(observer)
        game.attempts = attempts
        game.prefix_plies = len(opening.turns)
        stack.enter_context(patch.object(Tablut, "time", clock))
        stack.enter_context(redirect_stdout(output))
        begin = time.perf_counter()
        game.play(*observers)
        elapsed = time.perf_counter() - begin
    remaining = {"A": match.budget, "D": match.budget}
    for idx, attempt in enumerate(attempts):
        time_index = 2 * idx
        charged = (clock.ticks[time_index + 1] - clock.ticks[time_index]
                   if time_index + 1 < len(clock.ticks) else 0.0)
        side = attempt["side"]
        remaining[side] -= charged
        attempt.update(elapsed=charged, refereeRemaining=remaining[side])
    if len(clock.ticks) != 2 * len(attempts) - int("exception" in attempts[-1]):
        raise RuntimeError("Referee clock and player calls do not agree")
    reason = _reason(game, output.getvalue())
    return {
        "schema": SCHEMA, "id": match.id, "budget": match.budget, "opening": match.opening.name,
        "teams": {"A": match.attacker, "D": match.defender}, "winner": game.winner,
        "reason": reason, "started": started, "elapsed": elapsed, "remaining": remaining,
        "initial": opening.positions[0], "prefix": opening.turns, "attempts": attempts,
        "initSeconds": init_time,
    }


def _fingerprint(stage: str, matches: list[Match], extra=None) -> str:
    root = Path(Tablut.__file__).resolve().parent
    paths = [root / "Tablut.py", *root.glob("Team19*.py")]
    for name in sorted({name for game in matches for name in (game.attacker, game.defender)}):
        module = TEAMS[name].module
        if module.startswith("."):
            file = HERE.joinpath(*module[1:].split(".")).with_suffix(".py")
            paths.extend(file.parent.glob("Team19*.py") if TEAMS[name].search else [file])
    paths += sorted((HERE / "bots").glob("*common*.py"))
    advanced = {"Guided-MCTS", "BFS-Strategist", "QSearch-PVS"}
    if any(name in advanced for game in matches for name in (game.attacker, game.defender)):
        paths.append(HERE / "bots" / "tactics.py")
    if any("Grandmaster-Numba" in (game.attacker, game.defender) for game in matches):
        paths.append(HERE / "bots" / "numba_rules.py")
    payload = [SCHEMA, stage, [asdict(match) for match in matches],
               sorted({hashlib.sha256(file.read_bytes()).hexdigest() for file in paths})]
    if extra is not None:
        payload.append(extra)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@contextmanager
def _writer_lock(path: Path):
    with path.with_suffix(".lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _read_finished(stream, fingerprint: str, expected: set[str], records=None) -> set[str]:
    completed = set()
    stream.seek(0)
    while line := stream.readline():
        if not line.endswith(b"\n"):
            # A crash can leave the final append incomplete.
            stream.truncate(stream.tell() - len(line))
            break
        row = json.loads(line)
        game_id = row["id"]
        if row["schedule"] != fingerprint or game_id not in expected or game_id in completed:
            raise ValueError("JSONL belongs to a different schedule or contains a duplicate")
        completed.add(game_id)
        if records is not None:
            records[game_id] = row
    return completed


def fixture_for(args) -> list[Match]:
    if args.stage == "official":
        return paired(combinations(LEAGUE, 2), [standard()], 60.0)
    suite = load_suite()
    if args.stage in TOP6_BUDGETS:
        return paired(combinations(TOP6_TEAMS, 2), suite[:TOP6_OPENINGS], TOP6_BUDGETS[args.stage])
    if args.stage == "smoke":
        return paired(combinations(LEAGUE[:5], 2), suite[:args.openings or 1], 10.0)
    if args.stage == "random":
        return paired([("Team19-New", "RandomBot")], suite[:args.openings or 5], 10.0)
    if args.stage == "fast":
        return paired(combinations(LEAGUE, 2), suite[:args.openings or 6], 10.0)
    if args.stage == "duel":
        if args.opponent is None or args.opponent == "Team19-New":
            raise ValueError("duel requires an opponent")
        return paired([("Team19-New", args.opponent)], suite[:args.openings or 1], args.budget)
    bosses = select_bosses()
    return paired([("Team19-New", name) for name in bosses], suite, 60.0)


def select_bosses() -> list[str]:
    points = {}
    for stage, games in (("official", 72), ("fast", 432)):
        rows = [json.loads(line) for line in (RESULTS / f"{stage}.jsonl").read_text(
            encoding="utf-8").splitlines()]
        if len(rows) != games or len({row["id"] for row in rows}) != games:
            raise ValueError(f"Complete the {stage} schedule first")
        scores = Counter()
        for row in rows:
            for side, name in row["teams"].items():
                scores[name] += 1 if row["winner"] is None else 2 * (row["winner"] == side)
        points[stage] = scores
    return sorted(EXTERNAL, key=lambda name: (
        -points["official"][name], -points["fast"][name], name
    ))[:3]


def _append_result(stream, row, fingerprint, stage, total):
    row.update(schedule=fingerprint, stage=stage, scheduledGames=total)
    stream.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())
    stream.flush()
    os.fsync(stream.fileno())


def run_schedule(matches, run_id, stage, *, resume=False, limit=None, factories=None,
                 fingerprint_extra=None) -> Path:
    """Run one fixture list serially and keep its results resumable."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run identifier")
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{run_id}.jsonl"
    fingerprint = _fingerprint(stage, matches, fingerprint_extra)
    top6 = stage in TOP6_BUDGETS
    recorded = {} if top6 else None
    began = time.perf_counter()
    total = len(matches)
    with _writer_lock(RESULTS / "arena"), path.open("a+b") as stream:
        if stream.tell() and not resume:
            raise ValueError(f"{path.name} already exists. Use --resume or another --run-id.")
        done = _read_finished(stream, fingerprint, {game.id for game in matches}, recorded)
        if top6 and any((row["stage"], row["scheduledGames"]) != (stage, total)
                        for row in recorded.values()):
            raise ValueError(f"{path.name}: inconsistent stage or schedule size")
        if stage == "top6_10s":
            for row in reusable_games(RESULTS, matches, recorded):
                _append_result(stream, row, fingerprint, stage, total)
                recorded[row["id"]] = row
                done.add(row["id"])
        pending = [game for game in matches if game.id not in done]
        if top6:
            reused = sum("reusedFrom" in row for row in recorded.values())
            live_times = [match_seconds(row) for row in recorded.values() if "reusedFrom" not in row]
            print(f"{run_id} | 已复用 {reused} 局 · 待跑 {len(pending)} 局 | workers=1", flush=True)
            print(progress_text(len(done), total, time.perf_counter() - began, live_times,
                                pending[0] if pending else None), flush=True)
        else:
            print(f"{run_id} | {len(done)}/{total} recorded | workers=1", flush=True)
        for index, match in enumerate(pending[:limit]):
            row = run_match(match, factories=factories) if factories else run_match(match)
            _append_result(stream, row, fingerprint, stage, total)
            completed = len(done) + index + 1
            if top6:
                live_times.append(match_seconds(row))
                upcoming = pending[index + 1] if index + 1 < len(pending) else None
                print(progress_text(completed, total, time.perf_counter() - began, live_times,
                                    upcoming, row), flush=True)
            else:
                winner = row["teams"].get(row["winner"], "draw")
                print(f"{completed}/{total} {match.id} | {winner} | {row['reason']} | "
                      f"{row['remaining']['A']:.3f}/{row['remaining']['D']:.3f}s remaining", flush=True)
    return path


def run(args) -> None:
    run_id = args.run_id or (f"duel-{args.opponent}-{args.budget:g}" if args.stage == "duel"
                              else args.stage)
    run_schedule(fixture_for(args), run_id, args.stage, resume=args.resume, limit=args.limit)
    if args.stage in TOP6_BUDGETS and run_id == args.stage:
        from .dashboard import write_dashboard

        path, _ = write_dashboard(RESULTS)
        print(f"看板已更新 {path.resolve()}", flush=True)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "random", "duel", "official", "fast", "final",
                                            *TOP6_BUDGETS),
                        default="smoke")
    parser.add_argument("--openings", type=int, choices=range(1, 13))
    parser.add_argument("--opponent", choices=TEAMS)
    parser.add_argument("--budget", type=float, choices=(10.0, 60.0), default=10.0)
    parser.add_argument("--workers", type=int, choices=(1,), default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.openings and args.stage in ("official", "final", *TOP6_BUDGETS):
        parser.error("This stage uses its prescribed opening schedule")
    try:
        run(args)
    except KeyboardInterrupt:
        parser.exit(130, "Stopped. Resume with the same arguments and --resume.\n")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
