"""Import audited 10-second records and format serial tournament progress."""

from .report import REASON_LABELS, load


V2 = "Team19-v2"
SNAPSHOT = "Team19-91ffcc8"
LIVE_PAIR = frozenset((V2, SNAPSHOT))


def _source_for(match):
    return "evolution-fast" if V2 in (match.attacker, match.defender) else "fast"


def reusable_games(root, matches, recorded):
    """Return all missing historical games, checked before the journal is written."""
    from .overview import audit_game

    expected = {match.id: match for match in matches
                if frozenset((match.attacker, match.defender)) != LIVE_PAIR}
    missing = expected.keys() - recorded.keys()
    imported = {}
    for source in ("fast", "evolution-fast"):
        pending = {(match.opening.name, match.attacker, match.defender): match
                   for match_id, match in expected.items()
                   if match_id in missing and _source_for(match) == source}
        if not pending:
            continue
        path = root / f"{source}.jsonl"
        if not path.is_file():
            raise ValueError(f"缺少复用所需的赛果 {path.name}")
        prior = {row["reusedFrom"]["schedule"] for row in recorded.values()
                 if row.get("reusedFrom", {}).get("file") == path.name}
        for raw in load(path):
            seats = {side: SNAPSHOT if source == "fast" and name == "Team19-New" else name
                     for side, name in raw["teams"].items()}
            match = pending.get((raw["opening"], seats["A"], seats["D"]))
            if match is None:
                continue
            expected_id = f"10-{raw['opening']}-{raw['teams']['A']}-A-{raw['teams']['D']}"
            if (raw["stage"] != source or raw["budget"] != 10
                    or raw["id"] != expected_id or (prior and prior != {raw["schedule"]})
                    or match.id in imported):
                raise ValueError(f"{path.name}: inconsistent reusable record {raw['id']}")
            if any(attempt["team"] != raw["teams"][attempt["side"]] for attempt in raw["attempts"]):
                raise ValueError(f"{path.name}: inconsistent player in {raw['id']}")
            attempts = [dict(attempt, team=seats[attempt["side"]]) for attempt in raw["attempts"]]
            row = dict(raw, id=match.id, teams=seats, attempts=attempts,
                       reusedFrom={"file": path.name, "id": raw["id"], "schedule": raw["schedule"]})
            audit_game(row, match)
            imported[match.id] = row
    if absent := missing - imported.keys():
        raise ValueError(f"缺少 {len(absent)} 局同开局、同预算的历史赛果，首个缺失场次 {min(absent)}")
    return [imported[match.id] for match in matches if match.id in imported]


def duration(seconds):
    minutes, seconds = divmod(max(0, int(round(seconds))), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def match_seconds(row):
    return row["elapsed"] + sum(row.get("initSeconds", {}).values())


def progress_text(completed, total, elapsed, live_times, next_match, recent=None):
    fraction = completed / total
    filled = int(fraction * 20)
    bar = "[" + "█" * filled + "░" * (20 - filled) + "]"
    remaining = total - completed
    eta = "待首局"
    if remaining == 0:
        eta = "0s"
    elif live_times:
        eta = duration(remaining * sum(live_times) / len(live_times))
    lines = [f"{bar} {completed}/{total} ({fraction:.1%}) | 已用 {duration(elapsed)} | 预计剩余 {eta}"]
    if recent is not None:
        seats = recent["teams"]
        winner = f"{seats[recent['winner']]} 胜" if recent["winner"] else "和棋"
        reason = REASON_LABELS.get(recent["reason"], recent["reason"])
        fallbacks = sum(bool(turn.get("fallback")) for turn in recent["attempts"])
        errors = sum("error" in turn or "exception" in turn for turn in recent["attempts"])
        lines.append(f"本局 攻 {seats['A']} vs 守 {seats['D']} · {winner}（{reason}）"
                     f" · {recent['elapsed']:.2f}s · {fallbacks} 兜底 / {errors} 异常")
    if next_match is None:
        lines.append("赛程已完成")
    else:
        lines.append(f"下一局 {next_match.opening.name} · 攻 {next_match.attacker} vs 守 {next_match.defender}")
    return "\n".join(lines)
