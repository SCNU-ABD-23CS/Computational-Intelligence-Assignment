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
from math import isfinite
import operator
import os
from pathlib import Path
import re
from shutil import copyfile
from tempfile import NamedTemporaryFile
import time
from unittest.mock import patch

import Tablut

from . import HERE, RESULTS, TOP6_BUDGETS, TOP6_OPENINGS, TOP6_TEAMS
from .openings import Opening, load_suite, move_event, start, standard
from .schedule import Schedule, fixture_id
from .top6 import duration, match_seconds, progress_text, reusable_games


SCHEMA = 1


@dataclass(frozen=True)
class Team:
    module: str
    alias: str
    description: str
    search: str | None = None
    use_book: bool = False
    use_nnue: bool = False


_V2_MODULE = ".snapshots.v_3edc85c.Team19"
_V2_SEARCH = ".snapshots.v_3edc85c.Team19_search"

TEAMS = {
    "Team19-New": Team("Team19", "new", "历史赛段入口，运行当前根目录版本", "Team19_search"),
    "Team19-PreFix": Team(".snapshots.v_e3d64c4.Team19", "prefix", "估值修复前快照 e3d64c4",
                          ".snapshots.v_e3d64c4.Team19_search"),
    "Team19-Old": Team(".snapshots.v_99671ae.Team19", "old", "三阶段优化前快照 99671ae",
                       ".snapshots.v_99671ae.Team19_search"),
    "GreedyBot": Team(".bots.greedy_bot", "greedy", "一步贪心基线"),
    "NaiveAB-D2": Team(".bots.naive_ab_d2", "ab2", "固定两层 Alpha-Beta 基线"),
    "Guided-MCTS": Team(".bots.guided_mcts", "mcts", "启发式引导蒙特卡洛树搜索"),
    "BFS-Strategist": Team(".bots.bfs_strategist", "bfs", "国王逃生路径与包围网估值"),
    "QSearch-PVS": Team(".bots.qsearch_pvs", "qs", "主变搜索与静态延伸"),
    "Grandmaster-Numba": Team(".bots.grandmaster_numba", "gm", "Numba 高速强敌"),
    "RandomBot": Team(".bots.random_bot", "random", "随机合法步冒烟基线"),
}
LEAGUE = tuple(name for name in TEAMS if name != "RandomBot")
EXTERNAL = LEAGUE[3:]
TEAMS.update({
    "Team19-91ffcc8": Team(".snapshots.v_91ffcc8.Team19", "91f", "Numba 重构前快照 91ffcc8",
                         ".snapshots.v_91ffcc8.Team19_search"),
    "Team19-v2": Team(_V2_MODULE, "v2", "时钟及收官补丁版快照 3edc85c", _V2_SEARCH),
    "Team19-v3": Team(".snapshots.v_be23011.Team19", "v3", "分值置换表版快照 be23011",
                      ".snapshots.v_be23011.Team19_search"),
    "Team19-v4": Team("Team19", "v4", "逃线战术搜索与关键回合加时", "Team19_search"),
    "Team19-NNUE": Team("Team19", "nnue", "v4＋有界残差 NNUE，不开局库", "Team19_search",
                         use_nnue=True),
    "Team19-Book-NNUE": Team("Team19", "booknnue", "开局库＋有界残差 NNUE", "Team19_search",
                              use_book=True, use_nnue=True),
    "Grandmaster-Plus": Team(".bots.grandmaster_plus", "gmp", "Grandmaster 原估值，增强搜索和时钟"),
})
TEAM_NAMES = {label.casefold(): name for name, team in TEAMS.items() for label in (name, team.alias)}
TEAM_PRESETS = {"top6": TOP6_TEAMS, "all": tuple(name for name in TEAMS if name != "Team19-New")}


@dataclass(frozen=True)
class Match:
    attacker: str
    defender: str
    budget: float
    opening: Opening

    @property
    def id(self) -> str:
        return fixture_id(self.budget, self.opening.name, self.attacker, self.defender)


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
            if getattr(self.player, "last_book_hit", False) and "move" in record and "error" not in record:
                record.update(bookHit=True, depth=0, nodes=0, score=None, fallback=False)
            elif telemetry is not None:
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
        if entrant.search and (entrant.use_book or entrant.use_nnue):
            player = module.Player(use_book=entrant.use_book, use_nnue=entrant.use_nnue)
        else:
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
    teams = {name for game in matches for name in (game.attacker, game.defender)}
    for name in sorted(teams):
        module = TEAMS[name].module
        if module.startswith("."):
            file = HERE.joinpath(*module[1:].split(".")).with_suffix(".py")
            paths.extend(file.parent.glob("Team19*.py") if TEAMS[name].search else [file])
    paths += sorted((HERE / "bots").glob("*common*.py"))
    advanced = {"Guided-MCTS", "BFS-Strategist", "QSearch-PVS"}
    if teams & advanced:
        paths.append(HERE / "bots" / "tactics.py")
    if teams & {"Grandmaster-Numba", "Grandmaster-Plus"}:
        paths.append(HERE / "bots" / "numba_rules.py")
    if "Grandmaster-Plus" in teams:
        paths.append(HERE / "bots" / "grandmaster_numba.py")
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


def _read_finished(stream, fingerprint: str | None, expected: set[str],
                   records=None, *, repair=True) -> set[str]:
    completed = set()
    stream.seek(0)
    while line := stream.readline():
        if not line.endswith(b"\n"):
            # A crash can leave the final append incomplete.
            if repair:
                stream.truncate(stream.tell() - len(line))
            break
        row = json.loads(line)
        game_id = row["id"]
        if (fingerprint is not None and row["schedule"] != fingerprint
                or game_id not in expected or game_id in completed):
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
    with _writer_lock(RESULTS / "arena"):
        _play_schedule(path, matches, stage, fingerprint, resume=resume, limit=limit, factories=factories)
    return path


def _play_schedule(path, matches, stage, fingerprint, *, resume, limit, factories=None,
                   progress=False):
    detailed = progress or stage in TOP6_BUDGETS
    recorded = {} if detailed else None
    began = time.perf_counter()
    total = len(matches)
    with path.open("a+b") as stream:
        if stream.tell() and not resume:
            raise ValueError(f"{path.name} already exists. Use --resume or another --run-id.")
        done = _read_finished(stream, fingerprint, {game.id for game in matches}, recorded)
        if detailed and any((row["stage"], row["scheduledGames"]) != (stage, total)
                            for row in recorded.values()):
            raise ValueError(f"{path.name}: inconsistent stage or schedule size")
        if stage == "top6_10s":
            for row in reusable_games(path.parent, matches, recorded):
                _append_result(stream, row, fingerprint, stage, total)
                recorded[row["id"]] = row
                done.add(row["id"])
        pending = [game for game in matches if game.id not in done]
        if detailed:
            reused = sum("reusedFrom" in row for row in recorded.values())
            live_times = [match_seconds(row) for row in recorded.values() if "reusedFrom" not in row]
            print(f"{path.stem} | 已复用 {reused} 局 · 待跑 {len(pending)} 局 | workers=1", flush=True)
            print(progress_text(len(done), total, time.perf_counter() - began, live_times,
                                pending[0] if pending else None), flush=True)
        else:
            print(f"{path.stem} | {len(done)}/{total} recorded | workers=1", flush=True)
        for index, match in enumerate(pending[:limit]):
            row = run_match(match, factories=factories) if factories else run_match(match)
            _append_result(stream, row, fingerprint, stage, total)
            completed = len(done) + index + 1
            if detailed:
                live_times.append(match_seconds(row))
                upcoming = pending[index + 1] if index + 1 < len(pending) else None
                print(progress_text(completed, total, time.perf_counter() - began, live_times,
                                    upcoming, row), flush=True)
            else:
                winner = row["teams"].get(row["winner"], "draw")
                print(f"{completed}/{total} {match.id} | {winner} | {row['reason']} | "
                      f"{row['remaining']['A']:.3f}/{row['remaining']['D']:.3f}s remaining", flush=True)


def _team_name(label: str, teams: tuple[str, ...] = ()) -> str:
    try:
        name = TEAM_NAMES[label.strip().casefold()]
    except KeyError:
        raise ValueError(f"未知队伍 {label!r}，用 --list-teams 查看简称。") from None
    if name == "Team19-New" and name not in teams and "Team19-v4" in teams:
        return "Team19-v4"
    return name


def _select_teams(value: str) -> tuple[str, ...]:
    teams = []
    for token in value.split(","):
        preset = TEAM_PRESETS.get(token.strip().casefold())
        teams.extend(preset if preset is not None else (_team_name(token),))
    selected = dict.fromkeys(teams)
    if "Team19-v4" in selected:
        selected.pop("Team19-New", None)
    if len(selected) < 2:
        raise ValueError("--teams 至少需要两个不同队伍。")
    return tuple(selected)


def _select_budgets(value: str) -> tuple[float, ...]:
    try:
        budgets = tuple(float(token) for token in value.split(","))
    except ValueError:
        raise ValueError("--budgets 需要逗号分隔的秒数，例如 60,10。") from None
    if any(not isfinite(budget) or budget <= 0 for budget in budgets):
        raise ValueError("--budgets 只接受有限的正数。")
    if len(budgets) != len(set(budgets)):
        raise ValueError("--budgets 中同一档预算只能指定一次。")
    return budgets


def _list_teams() -> None:
    print(f"{'简称':<8} {'算法全名':<21} 简介")
    for name, team in TEAMS.items():
        print(f"{team.alias:<10} {name:<25} {team.description}")
    for name, teams in TEAM_PRESETS.items():
        print(f"{name:<10} " + ",".join(TEAMS[team].alias for team in teams))


def _experiment_schedules(args):
    teams = _select_teams(args.teams)
    budgets = _select_budgets("60" if args.budgets is None else args.budgets)
    games = 2 if args.games_per_pair is None else args.games_per_pair
    if args.standard_opening:
        if games != 2:
            raise ValueError("--standard-opening 要求 --games-per-pair 为 2，攻守各打一局。")
        openings = [standard()]
    else:
        if games <= 0 or games % 2:
            raise ValueError("--games-per-pair 必须是正偶数，每个开局交换攻守各打一局。")
        suite = load_suite()
        if games > 2 * len(suite):
            raise ValueError(f"开局库有 {len(suite)} 个开局，每组最多 {2 * len(suite)} 局。")
        openings = suite[:games // 2]
    if args.anchor is not None and args.format != "1vn":
        raise ValueError("--anchor 只用于 --format 1vn。")
    anchor = None
    if args.format == "1vn":
        anchor = teams[0] if args.anchor is None else _team_name(args.anchor, teams)
    if anchor is not None and anchor not in teams:
        raise ValueError(f"擂主 {anchor} 必须在 --teams 中。")
    if args.run_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("--run-id 只接受英文字母、数字、连字符和下划线。")

    config_fields = [teams, anchor, games]
    opening_tag = ""
    if args.standard_opening:
        config_fields.append("standard")
        opening_tag = "standard-"
    config = hashlib.sha256(json.dumps(config_fields).encode()).hexdigest()[:12]
    prefix = f"{args.run_id or 'arena'}-{opening_tag}{config}"
    schedules = []
    for budget in budgets:
        budget_id = str(budget).removesuffix(".0").replace(".", "p").replace("+", "")
        schedule = Schedule(f"{prefix}-{budget_id}s", teams, budget,
                            tuple(opening.name for opening in openings), anchor,
                            standard_opening=args.standard_opening)
        schedules.append((schedule, paired(schedule.pairs(), openings, budget)))
    if not isfinite(sum(2 * schedule.budget * schedule.games for schedule, _ in schedules)):
        raise ValueError("--budgets 指定的总计时过大。")
    return schedules


@dataclass
class ExperimentPlan:
    schedule: Schedule
    matches: list[Match]
    path: Path
    fingerprint: str
    archive: bool
    reusable: list[dict]
    completed: int
    needs_mode: bool


def _plan_experiment(schedule, matches, args, rerun_team):
    path = RESULTS / f"{schedule.name}.jsonl"
    fingerprint = _fingerprint(schedule.name, matches)
    exists = path.is_file()
    archive = exists and (args.fresh or rerun_team is not None)
    needs_mode = bool(exists and path.stat().st_size and not (args.resume or archive))
    if needs_mode and not args.dry_run:
        raise ValueError(f"{path.name} 已有赛果，请使用 --resume、--fresh 或 --rerun-team。")
    if rerun_team is not None and not exists:
        raise ValueError(f"{path.name} 缺少同配置赛果。请先运行完整实验。")
    records = {}
    if exists and not args.fresh:
        try:
            with path.open("rb") as stream:
                _read_finished(stream, None if rerun_team else fingerprint,
                               {match.id for match in matches}, records, repair=False)
            if (len({row["schedule"] for row in records.values()}) > 1
                    or any((row["stage"], row["scheduledGames"]) != (schedule.name, len(matches))
                           for row in records.values())):
                raise ValueError("Mixed schedules")
        except (TypeError, KeyError, ValueError) as error:
            raise ValueError(f"{path.name} 与当前实验不匹配。可用 --fresh 归档并重跑。") from error
    reusable = []
    if rerun_team is not None:
        from .overview import audit_game

        changed = TEAMS[rerun_team].module
        for match in matches:
            row = records.get(match.id)
            if row and all(TEAMS[name].module != changed for name in (match.attacker, match.defender)):
                audit_game(row, match)
                reusable.append(row)
    return ExperimentPlan(schedule, matches, path, fingerprint, archive, reusable,
                          0 if archive else len(records), needs_mode)


def _archive_experiment(plan):
    path = plan.path
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archive = path.with_name(f"{path.stem}.before-{stamp}.jsonl")
    if archive.exists():
        raise FileExistsError(archive)
    temporary = None
    try:
        # Prepare all reused rows before replacing the journal users resume.
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            for row in plan.reusable:
                origin = {"file": archive.name, "id": row["id"], "schedule": row["schedule"]}
                _append_result(stream, dict(row, reusedFrom=origin), plan.fingerprint,
                               plan.schedule.name, len(plan.matches))
        copyfile(path, archive)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"已归档 {archive.resolve()}", flush=True)


def _describe_experiment(plans, *, dashboard):
    schedule = plans[0].schedule
    print("参赛队伍 " + "、".join(f"{TEAMS[name].alias} ({name})" for name in schedule.teams))
    anchor = f" · 擂主 {schedule.anchor}" if schedule.anchor else ""
    print(f"{schedule.format_label}{anchor} · {schedule.opening_label} · "
          f"每组 {schedule.games_per_pair} 局 · workers=1")
    total = pending = 0
    estimate = 0.0
    for plan in plans:
        games = len(plan.matches)
        left = games - plan.completed - len(plan.reusable)
        seconds = 2 * plan.schedule.budget * left
        total += games
        pending += left
        estimate += seconds
        print(f"{plan.schedule.budget:g} 秒档 · {games} 局 · 已落盘 {plan.completed} 局 · "
              f"可复用 {games - left} 局 · 待跑 {left} 局 · 预算估算 {duration(seconds)}")
        print(f"  {plan.path.resolve()}")
        if plan.archive:
            print("  开跑前将归档旧赛果。")
        if plan.needs_mode:
            print("  继续时请用 --resume，也可用 --fresh 或 --rerun-team。")
    print(f"合计 {total} 局 · 待跑 {pending} 局 · 预算估算 {duration(estimate)}（双方用满计时，另计初始化）")
    if dashboard:
        print(f"看板 {(RESULTS / 'dashboard.html').resolve()}")


def _resume_command(schedules, run_id):
    first = schedules[0]
    teams = ",".join(TEAMS[name].alias for name in first.teams)
    budgets = ",".join(str(schedule.budget).removesuffix(".0") for schedule in schedules)
    command = (f"python -X utf8 -B -m tools.arena.runner --teams {teams} "
               f"--games-per-pair {first.games_per_pair} --budgets {budgets} --resume")
    if first.standard_opening:
        command += " --standard-opening"
    if first.anchor:
        command += f" --format 1vn --anchor {TEAMS[first.anchor].alias}"
    if run_id:
        command += f" --run-id {run_id}"
    return command


def run_experiment(args) -> None:
    entries = _experiment_schedules(args)
    teams = entries[0][0].teams
    rerun_team = _team_name(args.rerun_team, teams) if args.rerun_team is not None else None
    if rerun_team is not None and rerun_team not in teams:
        raise ValueError(f"{rerun_team} 必须在 --teams 中才能局部重跑。")
    if args.dry_run:
        plans = [_plan_experiment(schedule, matches, args, rerun_team) for schedule, matches in entries]
        _describe_experiment(plans, dashboard=not args.no_dashboard)
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    with _writer_lock(RESULTS / "arena"):
        plans = [_plan_experiment(schedule, matches, args, rerun_team) for schedule, matches in entries]
        _describe_experiment(plans, dashboard=not args.no_dashboard)
        for plan in plans:
            if plan.archive:
                _archive_experiment(plan)
        for plan in plans:
            _play_schedule(plan.path, plan.matches, plan.schedule.name, plan.fingerprint,
                           resume=args.resume or plan.archive, limit=args.limit, progress=True)
        if not args.no_dashboard:
            from .dashboard import write_dashboard

            schedules = tuple(plan.schedule for plan in plans)
            path, _ = write_dashboard(RESULTS, schedules=schedules,
                                      resume_command=_resume_command(schedules, args.run_id))
            print(f"看板已更新 {path.resolve()}", flush=True)


def run(args) -> None:
    run_id = args.run_id or (f"duel-{args.opponent}-{args.budget:g}" if args.stage == "duel"
                              else args.stage)
    run_schedule(fixture_for(args), run_id, args.stage, resume=args.resume, limit=args.limit)
    if args.stage in TOP6_BUDGETS and run_id == args.stage and not getattr(args, "no_dashboard", False):
        from .dashboard import write_dashboard

        path, _ = write_dashboard(RESULTS)
        print(f"看板已更新 {path.resolve()}", flush=True)


def main(argv=None) -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "random", "duel", "official", "fast", "final",
                                            *TOP6_BUDGETS), help="运行已有固定赛段")
    parser.add_argument("--openings", type=int, choices=range(1, 13))
    parser.add_argument("--opponent", choices=TEAMS)
    parser.add_argument("--budget", type=float, choices=(10.0, 60.0), help="历史 duel 赛段的时间档")
    parser.add_argument("--workers", type=int, choices=(1,), default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--list-teams", action="store_true", help="列出算法全名、简称和预设")
    parser.add_argument("--teams", help="逗号分隔的队名、简称或预设，如 top6")
    parser.add_argument("--standard-opening", action="store_true",
                        help="配合 --teams 使用标准初始盘面，每组攻守各 1 局")
    parser.add_argument("--games-per-pair", type=int, help="每组场次，默认 2 局")
    parser.add_argument("--budgets", help="每方整局秒数，逗号分隔，默认 60")
    parser.add_argument("--format", choices=("rr", "1vn"), help="全循环 rr（默认）或车轮战 1vn")
    parser.add_argument("--anchor", help="1vn 的擂主，默认第一个队伍")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true", help="从已落盘的赛程续跑")
    modes.add_argument("--fresh", action="store_true", help="归档并全新运行参数化实验")
    modes.add_argument("--rerun-team", help="归档旧赛果并只重跑指定队伍参与的对局")
    parser.add_argument("--dry-run", action="store_true", help="只预览参数化赛程")
    parser.add_argument("--no-dashboard", action="store_true", help="保留当前看板文件")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.list_teams:
        _list_teams()
        return
    if args.standard_opening and args.teams is None:
        parser.error("--standard-opening 需要搭配 --teams 使用。")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.openings and args.stage in ("official", "final", *TOP6_BUDGETS):
        parser.error("This stage uses its prescribed opening schedule")
    try:
        if args.teams is not None:
            if any(value is not None for value in (args.stage, args.openings, args.budget, args.opponent)):
                raise ValueError("--teams 使用参数化赛程，请使用 --games-per-pair 和 --budgets。")
            run_experiment(args)
        else:
            if (args.fresh or args.dry_run or any(value is not None for value in (
                    args.rerun_team, args.budgets, args.games_per_pair, args.format, args.anchor))):
                raise ValueError("使用参数化选项时请通过 --teams 指定参赛队伍。")
            args.stage = args.stage or "smoke"
            args.budget = args.budget or 10.0
            run(args)
    except KeyboardInterrupt:
        parser.exit(130, "已停止。保留赛程参数，用 --resume 续跑。\n")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
