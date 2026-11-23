"""Play a frozen three-way weight comparison on independent openings."""

from collections import Counter
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import re
from unittest.mock import patch

from . import openings, report, runner
from .overview import audit_game


HOT, CANDIDATE, STRONG = "Team19-v2-Hot", "Team19-v2-Candidate", "Grandmaster-Numba"
VALIDATION_SEED = 20261124
STAGE = "v2-validation"
MATCHUPS = ((CANDIDATE, HOT), (CANDIDATE, STRONG), (HOT, STRONG))


def held_out_openings(path: Path):
    from .tune_ga import write_json

    known = openings.load_suite()
    if not path.exists():
        fresh = openings.generate(VALIDATION_SEED, exclude=tuple(known))
        entries = [dict(asdict(opening), name=f"v{idx:02d}") for idx, opening in enumerate(fresh, 1)]
        write_json(path, {"seed": VALIDATION_SEED, "openings": entries})
    source = json.loads(path.read_text(encoding="utf-8"))
    suite = [openings.Opening(**row) for row in source["openings"]]
    training_keys = {openings._symmetry_key(openings.start(item).referee.board) for item in known}
    keys = {openings._symmetry_key(openings.start(item).referee.board) for item in suite}
    if (source["seed"] != VALIDATION_SEED or len(suite) != len(keys) or len(suite) != 12
            or len({item.name for item in suite}) != 12 or keys & training_keys
            or any(len(item.moves) != 4 or not item.signature for item in suite)):
        raise ValueError("The validation set must contain 12 independent four-ply openings")
    return suite


def weighted_player(weights):
    from .snapshots.v_3edc85c import Team19, Team19_weights

    # The production constructor and timed callbacks are used on both seats.
    with patch.object(Team19_weights, "DEFAULT_WEIGHTS", tuple(weights)):
        player = Team19.Player()
    if player._native is None:
        raise RuntimeError("The validation Player failed to initialize") from player.native_init_error
    return player


def matchup(games, team, opponent):
    outcomes = Counter()
    roles = Counter()
    paired = Counter()
    for game in games:
        if set(game["teams"].values()) != {team, opponent}:
            continue
        side = next(seat for seat, name in game["teams"].items() if name == team)
        score = report.points(game["winner"], side)
        outcomes[score] += 1
        roles[side] += score
        paired[game["opening"]] += score - 1
    count = sum(outcomes.values())
    return {"team": team, "opponent": opponent, "games": count,
            "points": 2 * outcomes[2] + outcomes[1], "attackPoints": roles["A"],
            "defencePoints": roles["D"], "wins": outcomes[2], "draws": outcomes[1],
            "losses": outcomes[0], "positivePairs": sum(value > 0 for value in paired.values()),
            "negativePairs": sum(value < 0 for value in paired.values())}


def _eligible(result):
    return result["games"] == 24 and result["points"] > 24 and (
        result["positivePairs"] > result["negativePairs"]
    )


def _stable(stability):
    return not any(stability[field] for field in (*report.FORFEITS, "fallbacks", "internalErrors"))


def _write_comparison(path, batches, eligible):
    lines = [
        "# Team19-v2 独立开局权重验证",
        "",
        "训练开局与验证开局没有相同盘面或对称盘面。每组 12 个未见开局换边各一局，串行执行。",
        "积分按胜 2、和 1、负 0 计算。每组满分 48 分，执攻与执守各 24 分。",
        "",
        "| 预算 | 权重 / 队伍 | 对手 | 积分 | 胜–和–负 | 攻方积分 | 守方积分 | 正分开局对 / 负分开局对 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for batch in batches:
        for row in batch["matchups"]:
            lines.append(
                f"| {batch['budget']:g}s | {row['team']} | {row['opponent']} | "
                f"{row['points']}∶{48 - row['points']} | {row['wins']}–{row['draws']}–{row['losses']} | "
                f"{row['attackPoints']}/24 | {row['defencePoints']}/24 | "
                f"{row['positivePairs']} / {row['negativePairs']} |"
            )
    lines += [
        "",
        "晋级条件要求候选在两档分别战胜热启动版与 Grandmaster，且每次对照赢得的开局对多于输掉的开局对。"
        "候选须全程零超时、非法步、异常及兜底。",
        "",
        f"**{'候选达到默认权重更新门槛' if eligible else '本轮保留热启动默认权重'}。**",
        "",
        "| 预算 | 队伍 | 超时 | 非法 | 裁判异常 | 内部异常 | 兜底 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for batch in batches:
        for row in batch["stability"]:
            lines.append(f"| {batch['budget']:g}s | {row['team']} | {row['timeout']} | "
                         f"{row['illegal move']} | {row['exception']} | {row['internalErrors']} | "
                         f"{row['fallbacks']} |")
    lines += ["", *[f"- [{row['budget']:g} 秒原始棋谱]({row['runId']}.jsonl)" for row in batches], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def validate(args):
    from Team19_weights import FEATURE_NAMES
    from .tune_ga import load_weights, write_json

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Invalid validation run identifier")
    root = runner.RESULTS
    suite = held_out_openings(root / f"{args.run_id}-openings.json")
    hot, trial = load_weights(args.baseline).tolist(), load_weights(args.candidate).tolist()
    if hot == trial:
        raise ValueError("Supply a candidate distinct from the baseline")
    fingerprints = {"features": FEATURE_NAMES, "hot": hot, "candidate": trial}
    frozen = root / f"{args.run_id}-weights.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != dict(
            fingerprints, features=list(FEATURE_NAMES)):
        raise ValueError("Use a new run ID if either weight vector has changed")
    if not frozen.exists():
        write_json(frozen, fingerprints)

    v2 = runner.TEAMS["Team19-v2"]
    batches = []
    with patch.dict(runner.TEAMS, {HOT: v2, CANDIDATE: v2}):
        for budget in (10.0, 60.0):
            run_id = f"{args.run_id}-{budget:g}"
            schedule = runner.paired(combinations((HOT, CANDIDATE, STRONG), 2), suite, budget)
            factories = {HOT: lambda: weighted_player(hot), CANDIDATE: lambda: weighted_player(trial)}
            path = runner.run_schedule(schedule, run_id, STAGE, resume=args.resume,
                                       factories=factories, fingerprint_extra=fingerprints)
            games = report.load(path)
            expected = {match.id: match for match in schedule}
            if {game["id"] for game in games} != expected.keys():
                raise ValueError("The validation fixture list is incomplete")
            plies = sum(audit_game(game, expected[game["id"]]) for game in games)
            stats = report.tables(games)
            pairings = [matchup(games, *pair) for pair in MATCHUPS]
            batches.append({"budget": budget, "runId": run_id, "games": len(games), "verifiedPlies": plies,
                            "matchups": pairings, "stability": stats["stability"]})
            report.write_report(path)
            print(f"{budget:g}s audited {len(games)} games | " +
                  " | ".join(f"{row['team']} vs {row['opponent']}: "
                             f"{row['points']}:{48 - row['points']}" for row in pairings), flush=True)

    eligible = all(
        all(_eligible(row) for row in batch["matchups"] if row["team"] == CANDIDATE)
        and _stable(next(row for row in batch["stability"] if row["team"] == CANDIDATE))
        for batch in batches
    )
    write_json(root / f"{args.run_id}.json",
               {"eligible": eligible, "batches": batches, **fingerprints})
    path = root / f"{args.run_id}.md"
    _write_comparison(path, batches, eligible)
    print(path, flush=True)
