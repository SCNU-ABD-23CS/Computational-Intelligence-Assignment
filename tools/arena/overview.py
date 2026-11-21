"""Audit the completed league against Tablut and write a combined report."""

from argparse import ArgumentParser
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from math import isclose, isfinite
from types import SimpleNamespace

from .openings import move_event, start
from .report import FORFEITS, load, points, tables, write_report
from .runner import RESULTS, SCHEMA, _fingerprint, _reason, fixture_for, select_bosses


STAGES = {"official": "60 秒标准开局", "fast": "10 秒快棋", "final": "60 秒强敌对照"}
NEW = "Team19-New"


def audit_game(row, match) -> int:
    """Reconstruct each accepted position and adjudicate the recorded clocks."""
    def check(condition, description):
        if not condition:
            raise ValueError(f"{match.id}: {description}")

    check(row["schema"] == SCHEMA and row["id"] == match.id, "invalid schema or ID")
    teams = {"A": match.attacker, "D": match.defender}
    check(row["teams"] == teams and row["budget"] == match.budget
          and row["opening"] == match.opening.name, "unexpected setup")
    setup = start(match.opening)
    game = setup.referee
    check(row["initial"] == setup.positions[0] and row["prefix"] == setup.turns,
          "opening board or history mismatch")
    clocks = dict.fromkeys(teams, match.budget)
    attempts = row["attempts"]
    output = StringIO()
    reason = None
    applied = len(setup.turns)
    with redirect_stdout(output):
        for index, attempt in enumerate(attempts):
            check(reason is None, "attempt recorded after the result")
            attacker = index % 2 == 0
            side, other = ("A", "D") if attacker else ("D", "A")
            check(attempt["side"] == side and attempt["team"] == teams[side],
                  f"wrong actor at attempt {index + 1}")
            elapsed = attempt["elapsed"]
            check(isfinite(elapsed) and elapsed >= 0, "invalid move time")
            clocks[side] -= elapsed
            check(isclose(attempt["refereeRemaining"], clocks[side], rel_tol=0, abs_tol=1e-7),
                  f"clock mismatch at attempt {index + 1}")
            before = game.board
            if "exception" in attempt:
                check(elapsed == 0, "referee charged an exception")
                reason = "exception"
            elif clocks[side] < 0:
                reason = "timeout"
            elif not game.make_move(attacker, attempt["move"]):
                reason = "illegal move"
            else:
                applied += 1
                check(attempt.get("event") == move_event(game, before, attacker, attempt["move"], applied),
                      f"board, capture or hash mismatch at ply {applied}")
                if game.check_if_game_over(not attacker):
                    reason = _reason(game, output.getvalue())
            if reason in FORFEITS:
                game.winner = other
                check("event" not in attempt, "forfeited move changed the position")
    check(reason is not None and (game.winner, reason) == (row["winner"], row["reason"]),
          "referee result mismatch")
    check(row["remaining"] == clocks, "incorrect final clocks")
    return applied


def audit_stage(stage):
    matches = fixture_for(SimpleNamespace(stage=stage, openings=None))
    expected = {match.id: match for match in matches}
    path = RESULTS / f"{stage}.jsonl"
    games = load(path)
    if {row["id"] for row in games} != expected.keys() or not path.read_bytes().endswith(b"\n"):
        raise ValueError(f"{stage}: incomplete or unexpected schedule")
    fingerprint = _fingerprint(stage, matches)
    plies = 0
    for row in games:
        if (row["stage"], row["scheduledGames"], row["schedule"]) != (stage, len(matches), fingerprint):
            raise ValueError(f"{row['id']}: schedule or source fingerprint mismatch")
        plies += audit_game(row, expected[row["id"]])
    return games, plies


def _one_on_one(games, opponent):
    seats = [(game, side) for game in games for side, name in game["teams"].items()
             if name == NEW and opponent in game["teams"].values()]
    outcome = Counter(points(game["winner"], side) for game, side in seats)
    roles = Counter()
    for game, side in seats:
        roles[side] += points(game["winner"], side)
    total = roles["A"] + roles["D"]
    return (len(seats), roles["A"], roles["D"], total, 2 * len(seats) - total,
            f"{outcome[2]} / {outcome[1]} / {outcome[0]}")


def _table(header, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |"
                     for row in (header, ("---",) * len(header), *rows))


def write_overview(completed):
    views = {stage: tables(games) for stage, (games, _) in completed.items()}
    games = [game for rows, _ in completed.values() for game in rows]
    counts = tables(games)["stability"]
    highlights = []
    for stage in ("official", "fast"):
        standings = next(row for row in views[stage]["standings"] if row["team"] == NEW)
        role = next(row for row in views[stage]["roles"] if row["team"] == NEW)
        highlights.append((STAGES[stage], standings["rank"], standings["points"], role["A points"],
                           role["D points"], f'{standings["wins"]} / {standings["draws"]} / '
                           f'{standings["losses"]}', standings["marks"] if stage == "official" else "—"))
    finals = completed["final"][0]
    final_scores = [(name, *_one_on_one(finals, name)) for name in select_bosses()]
    replays = {stage: sum(NEW in row["teams"].values()
                          and row["teams"].get(row["winner"]) != NEW for row in rows)
               for stage, (rows, _) in completed.items()}
    results = Counter(game["reason"] for game in games)
    sections = [
        "# JC4004 九队离线联赛",
        f"共 {len(games)} 局。逐手用官方裁判复放了 {sum(plies for _, plies in completed.values()):,} 手，"
        "并核对开局历史、合法步、吃子、局面哈希、两方剩时和终局判定。",
        _table(("赛段", "已核对场次", "有效棋步含开局", "败局及和局棋谱"),
               [(f"[{STAGES[stage]}]({stage}/report.md)", len(rows), plies, replays[stage])
                for stage, (rows, plies) in completed.items()]),
        "## Team19-New 总积分",
        _table(("赛段", "排名", "总积分", "执攻积分", "执守积分", "胜 / 和 / 负", "课程折算分"), highlights),
        "标准开局赛每队 16 场，满分 32。快棋每队 96 场，满分 192。课程折算采用标准开局赛榜首积分。",
        "## Team19-New 的 60 秒强敌对照",
        "外部六队按 60 秒标准开局赛积分选出前三。若同分则看 10 秒快棋积分，再按队名排序。"
        "每个对手在 12 个开局交换攻守，积分按对阵分别计算。",
        _table(("对手", "场次", "新版执攻", "新版执守", "新版积分", "对手积分", "新版胜 / 和 / 负"),
               final_scores),
        "## 正赛稳定性",
        _table(("队伍", "超时负", "非法步负", "异常负", "内部错误", "兜底次数"),
               [(row["team"], row["timeout"], row["illegal move"], row["exception"],
                 row["internalErrors"], row["fallbacks"]) for row in counts]),
        "Team19-Old 按历史实现参赛。其合法零层兜底单独计数。",
        _table(("终局原因", "场次"), sorted(results.items(), key=lambda row: (-row[1], row[0]))),
    ]
    if (RESULTS / "qualification.md").is_file():
        sections.append("[赛前验收与强敌换边数据](qualification.md)")
    path = RESULTS / "overview.md"
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    completed = {}
    for stage in (args.stage,) if args.stage else STAGES:
        completed[stage] = audit_stage(stage)
        rows, plies = completed[stage]
        write_report(RESULTS / f"{stage}.jsonl")
        print(f"{stage} | {len(rows)} games, {plies} plies verified", flush=True)
    if args.stage is None:
        print(write_overview(completed))


if __name__ == "__main__":
    main()
