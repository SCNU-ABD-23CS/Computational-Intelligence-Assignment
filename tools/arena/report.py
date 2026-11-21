"""Write league standings, role splits, cross tables and replay exports."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import csv
from fractions import Fraction
import json
from math import exp, log
from pathlib import Path

from .runner import RESULTS


FORFEITS = ("timeout", "illegal move", "exception")
WIN_REASONS = ("king captured", "king escaped to the edge", "fewer than two attackers remain")
DRAW_REASONS = ("repeated position", "no legal move")


def load(path: Path) -> list[dict]:
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1
    rows = [json.loads(line) for line in data[:end].splitlines()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected a nonempty schedule with distinct game IDs")
    if len({row["schedule"] for row in rows}) != 1:
        raise ValueError("Mixed schedules")
    return rows


def points(winner: str | None, side: str) -> int:
    return 1 if winner is None else 2 * (winner == side)


def elo(games, teams):
    """Regularised Bradley–Terry estimate with a fitted attacker advantage."""
    ratings = dict.fromkeys(teams, 0.0)
    appearances = Counter(team for game in games for team in game["teams"].values())
    advantage = 0.0
    prior = 0.5
    for _ in range(300):
        gradient = {team: -prior * rating for team, rating in ratings.items()}
        side_error = -prior * advantage
        for game in games:
            attack, defend = game["teams"]["A"], game["teams"]["D"]
            logit = ratings[attack] - ratings[defend] + advantage
            estimate = 1 / (1 + exp(-logit))
            error = points(game["winner"], "A") / 2 - estimate
            gradient[attack] += error
            gradient[defend] -= error
            side_error += error
        changes = {team: 1.5 * value / (appearances[team] + prior)
                   for team, value in gradient.items()}
        for team, change in changes.items():
            ratings[team] += change
        adjustment = 1.5 * side_error / (len(games) + prior)
        advantage += adjustment
        if max(abs(adjustment), *(abs(change) for change in changes.values())) < 1e-7:
            break
    shift = sum(ratings.values()) / len(ratings)
    scale = 400 / log(10)
    return {team: round(1500 + scale * (value - shift), 1) for team, value in ratings.items()}


def _mean(values):
    return round(sum(values) / len(values), 3) if values else ""


def tables(games):
    teams = sorted({name for game in games for name in game["teams"].values()})
    ratings = elo(games, teams)
    standings, roles, stability, matrix = [], [], [], []
    for team in teams:
        own = [(game, side) for game in games for side, name in game["teams"].items() if name == team]
        outcomes = Counter(points(game["winner"], side) for game, side in own)
        total = 2 * outcomes[2] + outcomes[1]
        standings.append({"rank": 0, "team": team, "games": len(own), "wins": outcomes[2],
                          "draws": outcomes[1], "losses": outcomes[0], "points": total,
                          "pointRate": round(total / (2 * len(own)), 4), "marks": 0,
                          "BT Elo": ratings[team]})
        by_side = {"team": team}
        for side in ("A", "D"):
            records = [points(game["winner"], side) for game, seat in own if seat == side]
            by_side.update({f"{side} games": len(records), f"{side} points": sum(records),
                            f"{side} winRate": round(records.count(2) / len(records), 4) if records else "",
                            f"{side} pointRate": round(sum(records) / (2 * len(records)), 4)
                            if records else ""})
        roles.append(by_side)
        turns = [turn for game, _ in own for turn in game["attempts"] if turn["team"] == team]
        metrics = {"team": team}
        for reasons, required in ((WIN_REASONS, 2), (DRAW_REASONS, 1), (FORFEITS, 0)):
            for reason in reasons:
                metrics[reason] = sum(game["reason"] == reason and points(game["winner"], side) == required
                                      for game, side in own)
        metrics.update({
            "gameSeconds": _mean([game["elapsed"] for game, _ in own]),
            "remainingSeconds": _mean([game["remaining"][side] for game, side in own]),
            "meanDepth": _mean([turn["depth"] for turn in turns if turn.get("depth") is not None]),
            "meanSimulations": _mean([turn["simulations"] for turn in turns
                                      if turn.get("simulations") is not None]),
            "fallbacks": sum(bool(turn.get("fallback")) for turn in turns),
            "internalErrors": sum("error" in turn for turn in turns),
        })
        stability.append(metrics)
        versus = {"team": team}
        for opponent in teams:
            results = Counter(points(game["winner"], side) for game, side in own
                              if game["teams"].get("D" if side == "A" else "A") == opponent)
            versus[opponent] = (f"{results[2]}/{results[1]}/{results[0]} "
                               f"({2 * (results[2] - results[0]):+d})" if results else "")
        matrix.append(versus)
    equal_games = len({row["games"] for row in standings}) == 1

    def rank_value(row):
        return row["points"] if equal_games else Fraction(row["points"], row["games"])

    standings.sort(key=lambda row: (-rank_value(row), row["team"]))
    top = standings[0]["points"]
    rank = 0
    last = None
    for i, row in enumerate(standings, 1):
        value = rank_value(row)
        if value != last:
            rank = i
        last = value
        row["rank"] = rank
        row["marks"] = round(30 * row["points"] / top, 3) if equal_games else ""
    order = {row["team"]: idx for idx, row in enumerate(standings)}
    return {
        "standings": standings,
        "roles": sorted(roles, key=lambda row: order[row["team"]]),
        "matrix": matrix,
        "stability": sorted(stability, key=lambda row: order[row["team"]]),
    }


def _analysis(attempt: dict, ply: int) -> dict:
    return {"ply": ply, "side": attempt["side"], "depth": attempt.get("depth"),
            "nodes": attempt.get("nodes"), "score": attempt.get("score"),
            "simulations": attempt.get("simulations"), "elapsed": attempt["elapsed"],
            "fallback": attempt.get("fallback", False)}


def replay(game: dict) -> dict:
    """Export the practice board's v1 replay schema with arena metadata."""
    ai = "A" if game["teams"]["A"] == "Team19-New" else "D"
    opponent = "D" if ai == "A" else "A"
    turns = [{**event, "actor": "opening", "clocksAfter": None, "analysis": None}
             for event in game["prefix"]]
    clocks = {side: game["budget"] for side in ("A", "D")}
    failed = None
    for attempt in game["attempts"]:
        side = attempt["side"]
        clocks[side] = attempt["refereeRemaining"]
        if event := attempt.get("event"):
            turns.append({**event, "actor": "ai" if side == ai else "human",
                          "clocksAfter": {"refereeRemaining": clocks[ai],
                                          "playerRemaining": clocks[ai], "bySide": clocks.copy()},
                          "analysis": _analysis(attempt, event["ply"])})
        else:
            failed = {key: value for key, value in attempt.items() if key != "event"}
    final_position = turns[-1]["positionAfter"] if turns else game["initial"]
    same_position = None
    if game["reason"] == "repeated position":
        for ply, position in enumerate((game["initial"], *(t["positionAfter"] for t in turns[:-1]))):
            if position == final_position:
                same_position = {"firstPly": ply, "repeatedPly": len(turns),
                                 "sideToMove": final_position["sideToMove"]}
                break
    stats = [t for t in game["attempts"] if t["side"] == ai]
    ai_stats = [{"elapsed": t["elapsed"], "depth": t.get("depth") or 0, "nodes": t.get("nodes") or 0,
                 "fallback": t.get("fallback", False), "error": "error" in t or "exception" in t}
                for t in stats]
    return {
        "schema": "team19.tablut.replay", "version": 1, "source": "arena",
        "recordingStartedUtc": game["started"], "resumedFrom": None,
        "humanSide": opponent, "budgetSeconds": game["budget"],
        "scorePerspective": "positive favours attackers", "initial": game["initial"],
        "turns": turns,
        "arena": {"id": game["id"], "teams": game["teams"], "opening": game["opening"]},
        "final": {
            "position": final_position, "finished": True, "winner": game["winner"],
            "reason": game["reason"], "repetition": same_position,
            "forfeit": game["reason"] if game["reason"] in FORFEITS else None,
            "clocks": {"refereeRemaining": clocks[ai], "playerRemaining": clocks[ai], "bySide": clocks},
            "points": {"ai": points(game["winner"], ai), "human": points(game["winner"], opponent)},
            "aiTurnStats": ai_stats, "failedAiAttempt": failed,
            "summary": {
                "aiTurns": len(ai_stats), "depths": dict(Counter(t["depth"] for t in ai_stats)),
                "fallbacks": sum(t["fallback"] for t in ai_stats),
                "errors": sum(t["error"] for t in ai_stats),
                "slowest": max((t["elapsed"] for t in ai_stats), default=0),
            },
        },
    }


def _markdown(rows) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    lines.extend("| " + " | ".join(str(row[k]) for k in keys) + " |" for row in rows)
    return "\n".join(lines)


def write_report(path: Path) -> Path:
    games = load(path)
    dest = path.with_suffix("")
    dest.mkdir(exist_ok=True)
    views = tables(games)
    sections = [
        f"# {path.stem}",
        f"{len(games)} of {games[0]['scheduledGames']} games. Win 2, draw 1, loss 0.",
        "Balanced schedules show `30 × points / leader points` for reference. The `official` "
        "schedule replicates the coursework tournament. The cross table uses W/D/L and net points.",
    ]
    if games[0]["stage"] == "final":
        sections.append("The standings rank by point rate because Team19-New plays three times as "
                        "many games. The cross table shows the results of each paired matchup.")
    for section, rows in views.items():
        with (dest / f"{section}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        sections.extend((f"## {section.capitalize()}", _markdown(rows)))
    notable = []
    for game in games:
        for side, team in game["teams"].items():
            if team == "Team19-New" and points(game["winner"], side) < 2:
                filename = f"{game['id']}.json"
                target = dest / "replays" / filename
                target.parent.mkdir(exist_ok=True)
                target.write_text(json.dumps(replay(game), indent=2, ensure_ascii=False, allow_nan=False)
                                  + "\n", encoding="utf-8")
                notable.append(f"- [{game['id']}](replays/{filename}) · {game['reason']}")
    if notable:
        sections.extend(("## Team19-New losses and draws", "\n".join(notable)))
    result = dest / "report.md"
    result.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Name of the JSONL schedule in results/")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        parser.error("Expected a schedule name")
    print(write_report(RESULTS / f"{args.run_id}.jsonl"))
