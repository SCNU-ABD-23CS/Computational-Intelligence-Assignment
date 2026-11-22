"""Build a self-contained HTML dashboard from the arena's stored results."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from fractions import Fraction
from html import escape
from pathlib import Path
import re

from . import HERE
from .dashboard_data import FORFEITS, HISTORICAL_VERSIONS, NATIVE_VERSION, STAGES
from .dashboard_data import Score, Stage, load_evolution, load_stages


RESULTS = HERE / "results"
NEW, PREFIX, OLD = "Team19-New", "Team19-PreFix", "Team19-Old"
GENERATIONS = (*HISTORICAL_VERSIONS, NATIVE_VERSION)
ROUND_LABELS = {"official": "60 秒官方赛", "fast": "10 秒快棋"}
STAGE_INTROS = {
    "official": "标准开局换边，按课程公式折算模拟分",
    "fast": "固定开局全循环，检验限时下的稳定性",
    "final": "新版与外部强敌逐队换边对照",
}
TEAM_LABELS = {
    NEW: ("New", "Numba 升级前版", "new"),
    NATIVE_VERSION: ("v2", "当前提交版", "v2"),
    PREFIX: ("PreFix", "估值修复前版", "prefix"),
    OLD: ("Old", "早期历史基线", "old"),
    "Grandmaster-Numba": ("Numba", "编译加速强敌", ""),
    "QSearch-PVS": ("QSearch", "延伸关键战术", ""),
    "BFS-Strategist": ("BFS", "侧重国王逃生", ""),
    "Guided-MCTS": ("MCTS", "模拟评估走法", ""),
    "NaiveAB-D2": ("AB-D2", "只看前两手", ""),
    "GreedyBot": ("Greedy", "优先眼前收益", ""),
}
REASONS = {
    "king escaped to the edge": ("国王逃生", "King escaped", "#137968"),
    "king captured": ("国王被捕", "King captured", "#345b9b"),
    "repeated position": ("重复局面和棋", "Repetition draw", "#9d6a33"),
    "fewer than two attackers remain": ("攻兵不足两枚", "Too few attackers", "#9564a4"),
    "no legal move": ("无合法步和棋", "No legal move", "#63737d"),
    "timeout": ("超时判负", "Timeout", "#b84343"),
    "illegal move": ("非法步判负", "Illegal move", "#d26a5e"),
    "exception": ("异常判负", "Exception", "#a43d67"),
}


def _label(name):
    return TEAM_LABELS.get(name, (name, "", ""))


def _percent(rate: float) -> str:
    return f"{rate:.1%}"


def _bar(rate: float, tip: str, variant="") -> str:
    return (
        f'<svg class="bar {variant}" viewBox="0 0 100 10" preserveAspectRatio="none" '
        f'role="img" aria-label="{escape(tip)}"><title>{escape(tip)}</title>'
        f'<rect class="rail" width="100" height="10" rx="2"/>'
        f'<rect class="fill" width="{rate * 100:.4f}" height="10" rx="2"/></svg>'
    )


def _section(title, english, chart, note="", extra=""):
    detail = f'<p class="chart-note">{note}</p>' if note else ""
    return (f'<section class="chart {extra}"><div class="chart-heading">'
            f'<h2>{title}<span lang="en">{english}</span></h2>{detail}</div>{chart}</section>')


def _metric(label, english, value, detail, extra=""):
    return (f'<div class="metric {extra}"><p>{label}<span lang="en">{english}</span></p>'
            f'<strong>{value}</strong><small>{detail}</small></div>')


def _team(stage: Stage, name: str):
    return next((team for team in stage.teams if team.name == name), None)


def _matchups(stage: Stage):
    return [(team, score) for team in stage.teams if (score := stage.matrix.get((NEW, team.name)))]


def _common_game_count(scores):
    counts = {score.games for score in scores}
    return counts.pop() if len(counts) == 1 else None


def _duel_groups(stage: Stage) -> str:
    scores = [score for _, score in _matchups(stage)]
    if each := _common_game_count(scores):
        return f"{len(scores)} 组 × {each} 局"
    return f"{len(scores)} 组换边对照"


def _games_metric(stages: list[Stage]):
    games = sum(stage.games for stage in stages)
    plies = sum(stage.plies for stage in stages)
    return _metric("已完成对局", "Games played", f"{games:,}", f"{plies:,} 手 · Recorded plies")


def _forfeits_metric(stages: list[Stage]):
    reasons = Counter()
    for stage in stages:
        reasons.update(stage.reasons)
    faults = sum(reasons[reason] for reason in FORFEITS)
    detail = f'{reasons["timeout"]} 超时 · {reasons["illegal move"]} 非法 · {reasons["exception"]} 异常'
    return _metric("裁判判负", "Forfeits", f"{faults:,}", detail,
                   "metric-good" if not faults else "metric-alert")


def _metrics(cards):
    return f'<div class="metrics">{"".join(cards)}</div>'


def _new_points_metric(score: Score):
    return _metric("新版积分", "Team19-New points", f"{score.points} / {2 * score.games}",
                   f"{_percent(score.point_rate)} 得分率 · {score.wins} 胜 {score.draws} 和 {score.losses} 负",
                   "metric-new")


def _round_version_metrics(rounds: list[Stage]):
    entries = [(stage, new) for stage in rounds if (new := _team(stage, NEW))]
    if not entries:
        return []
    stage, new = entries[0]
    if len(entries) == 1:
        rank = f"第 {new.rank} / {len(stage.teams)}"
        detail = f"{ROUND_LABELS[stage.name]} · {_percent(new.score.point_rate)} 得分率"
    else:
        fast = entries[1][1]
        rank = f"两档均第 {new.rank}" if new.rank == fast.rank else f"第 {new.rank} / 第 {fast.rank}"
        detail = "60 秒官方赛与 10 秒快棋"
    if stage.name == "official" and new.marks is not None:
        course_label = "官方赛 " if len(entries) > 1 else ""
        detail += f" · {course_label}{new.marks:g} / 30 模拟课程分"
    cards = [_metric("新版排名", "Team19-New rank", rank, detail, "metric-new")]

    gains = []
    for stage, new in entries:
        if old := _team(stage, OLD):
            gains.append((ROUND_LABELS[stage.name], new.score.points - old.score.points,
                          100 * (new.score.point_rate - old.score.point_rate)))
    if gains:
        label, points, change = gains[0]
        detail = f"{label} {points:+d} 积分"
        if len(gains) == 2:
            other, _, other_change = gains[1]
            detail += f" · {other} {other_change:+.1f} pp"
        cards.append(_metric("新版较旧版", "Gain vs Team19-Old", f"{change:+.1f} pp", detail))
    else:
        cards.append(_new_points_metric(entries[0][1].score))
    return cards


def _role_metric(stage: Stage):
    attack, defend = (stage.roles[NEW, side] for side in ("A", "D"))
    return _metric("新版攻守得分率", "Team19-New by side",
                   f"{_percent(attack.point_rate)} / {_percent(defend.point_rate)}", "执攻 / 执守")


def _final_metrics(stage: Stage):
    cards = [_metric("对照总场次", "Games played", f"{stage.games:,} 局", _duel_groups(stage))]
    if new := _team(stage, NEW):
        cards.extend((_new_points_metric(new.score), _role_metric(stage)))
    cards.append(_forfeits_metric([stage]))
    return _metrics(cards)


def _standings(stage: Stage):
    rows = []
    for team in stage.teams:
        name, score = escape(team.name), team.score
        _, description, variant = _label(team.name)
        tip = (f'{team.name} · {score.games} games · W/D/L {score.record} · {score.points} points · '
               f'{_percent(score.point_rate)} point rate · {_percent(score.win_rate)} win rate')
        marks = ""
        if stage.name == "official" and team.marks is not None:
            marks = f'<span><b>{team.marks:g}</b> 分 / 30</span>'
        rows.append(
            f'<div class="stand-row {variant}"><span class="rank">#{team.rank}</span>'
            f'<div class="team-name"><strong>{name}</strong><small>{escape(description)}</small></div>'
            f'<div class="rank-progress">{_bar(score.point_rate, tip)}'
            f'<b>{_percent(score.point_rate)}</b></div>'
            f'<div class="rank-facts"><span><b>{score.points}</b> 积分</span>'
            f'<span><b>{score.record}</b> W/D/L</span><span><b>{team.elo:.0f}</b> Elo</span>'
            f'{marks}</div></div>'
        )
    note = "胜 2 分、和 1 分、负 0 分。条形长度表示所得积分占满分的比例。"
    return _section("综合实力排行", "League standings", "".join(rows), note)


def _heat_color(rate: float) -> str:
    low, middle, high = (248, 190, 181), (240, 242, 244), (160, 220, 185)
    start, end = (low, middle) if rate <= 0.5 else (middle, high)
    blend = rate * 2 if rate <= 0.5 else rate * 2 - 1
    return "#" + "".join(f"{round(a + (b - a) * blend):02x}" for a, b in zip(start, end))


def _wide_svg(content, width, height, label, chart_class):
    return (f'<div class="wide-chart" tabindex="0" aria-label="{label}">'
            f'<svg class="{chart_class}" style="--chart-width:{width}px" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{label}"><title>{label}</title>{content}</svg></div>')


def _matrix_percent(score: Score) -> str:
    # Exact half-even rounding keeps complementary printed rates at 100.0%.
    tenths = round(Fraction(500 * score.points, score.games))
    whole, fraction = divmod(tenths, 10)
    return f"{whole}.{fraction}%"


def _heatmap(stage: Stage):
    names = [team.name for team in stage.teams]
    x_origin, y_origin, dx, dy = 169, 49, 86, 51
    pieces = ['<text class="axis-muted" x="12" y="27">行方 vs 列方</text>']
    for i, name in enumerate(names):
        short, _, variant = _label(name)
        x = x_origin + i * dx
        y = y_origin + i * dy
        pieces.append(f'<text class="axis-name {variant}" x="{x + dx / 2}" y="29" '
                      f'text-anchor="middle">{escape(short)}</text>')
        pieces.append(f'<text class="axis-name {variant}" x="12" y="{y + 29}">{escape(name)}</text>')
    for ri, left in enumerate(names):
        y = y_origin + ri * dy
        for ci, right in enumerate(names):
            x = x_origin + ci * dx
            score = stage.matrix.get((left, right))
            cell = f'<rect x="{x + 2}" y="{y + 2}" width="{dx - 4}" height="{dy - 4}" rx="6"'
            if score is None:
                pieces.append(f'{cell} fill="#e9ecee"/><text class="axis-muted" x="{x + dx / 2}" '
                              f'y="{y + 31}" text-anchor="middle">—</text>')
                continue
            rate = _matrix_percent(score)
            tip = (f"{left} vs {right} · {score.games} 局 · 胜/和/负 {score.record} · "
                   f"得分率 {rate} · 纯胜率 {_percent(score.win_rate)} · "
                   f"积分 {score.points} 比 {2 * score.games - score.points}")
            pieces.append(
                f'<g class="heat-cell" tabindex="0" aria-label="{escape(tip)}"><title>{escape(tip)}</title>'
                f'{cell} fill="{_heat_color(score.point_rate)}"/>'
                f'<text class="heat-value" x="{x + dx / 2}" y="{y + 23}" text-anchor="middle">'
                f'{rate}</text>'
                f'<text class="heat-record" x="{x + dx / 2}" y="{y + 39}" text-anchor="middle">'
                f'{score.record}</text></g>'
            )
    chart = _wide_svg("".join(pieces), x_origin + len(names) * dx + 4,
                      y_origin + len(names) * dy + 4, "对战得分率矩阵 · Point rate matrix", "heatmap")
    legend = ('<div class="heat-key"><span>0%</span><span class="heat-scale"></span>'
              '<span>100%</span><span class="heat-explainer">50% 为浅灰 · Cell label = W–D–L</span></div>')
    return _section("两两交手得分率", "Head-to-head point rate", legend + chart,
                    "从行方视角阅读。得分率 =（胜场 + 0.5 × 和棋）÷ 场次。每格第二行依次为胜、和、负。")


def _duels(stage: Stage, numba_nps: int | None):
    numba_speed = f"（赛前实测约 {numba_nps / 10_000:.0f} 万节点/秒）" if numba_nps else ""
    variations = {
        "QSearch-PVS": "沿用同一套 Python 引擎和估值，改用 PVS 加静搜，关键战术多看几手。",
        "BFS-Strategist": "沿用同一套 Python 引擎，估值改看国王逃生步数（BFS）和攻方包围网。",
        "Grandmaster-Numba": f"用 Numba 编译加速{numba_speed}，搜索和估值也一并重写。",
    }
    cards = []
    for team, score in _matchups(stage):
        opponent = escape(team.name)
        variation = variations.get(team.name, "")
        tip = f"{NEW} vs {team.name} · {score.games} games · W/D/L {score.record}"
        sides = []
        for side, text_label, variant in (("A", "执攻 Attack", "attack"), ("D", "执守 Defence", "defence")):
            split = stage.pair_roles.get((NEW, team.name, side), Score())
            sides.append(
                f'<div class="duel-role"><span>{text_label}</span>'
                f'<strong>{split.points} / {2 * split.games}</strong>'
                f'{_bar(split.point_rate, f"{text_label} · {split.games} games · W/D/L {split.record}", variant)}</div>'
            )
        cards.append(
            f'<article class="duel"><h3>{opponent}</h3><p class="duel-variable">{variation}</p>'
            f'<p class="duel-score">'
            f'<strong>{score.points}</strong><span> 比 </span><b>{2 * score.games - score.points}</b></p>'
            f'{_bar(score.point_rate, tip, "duel-bar")}<p class="duel-record">{score.games} 场 · '
            f'新版胜 / 和 / 负 {score.record}</p><div class="duel-splits">{"".join(sides)}</div></article>'
        )
    return _section("新版逐队对照", "Team19-New matchups", f'<div class="duels">{"".join(cards)}</div>',
                    "色条显示双方积分占比。左侧为 Team19-New，右侧为对手。")


def _roles(stage: Stage):
    final = stage.name == "final"
    teams = sorted(stage.teams, key=lambda team: team.name != NEW) if final else stage.teams
    count, x_start, dx = len(teams), 62, 112 if final else 101
    width, y_base, plot_height = x_start + count * dx + 12, 216, 172
    pieces = []
    for tick in (0, 25, 50, 75, 100):
        y = y_base - plot_height * tick / 100
        pieces.append(f'<line class="grid-line" x1="{x_start}" x2="{width - 13}" y1="{y}" y2="{y}"/>'
                      f'<text class="axis-muted" x="{x_start - 10}" y="{y + 4}" text-anchor="end">{tick}%</text>')
    for i, team in enumerate(teams):
        center = x_start + (i + 0.5) * dx
        short, _, variant = _label(team.name)
        if final and team.name == NEW:
            short = NEW
        pieces.append(f'<text class="axis-name {variant}" text-anchor="middle" x="{center}" y="244">'
                      f'{escape(short)}</text>')
        if final:
            suffix = " 局合计" if team.name == NEW else " 局"
            pieces.append(f'<text class="axis-muted" text-anchor="middle" x="{center}" y="263">'
                          f'{team.score.games}{suffix}</text>')
        for side, side_label, offset, color in (("A", "Attack", -39, "attack"),
                                                ("D", "Defence", 10, "defence")):
            score = stage.roles[team.name, side]
            height = score.point_rate * plot_height
            y, x = y_base - height, center + offset
            tip = (f"{team.name} · {side_label} · {score.points}/{2 * score.games} points · "
                   f"W/D/L {score.record} · {_percent(score.point_rate)}")
            pieces.append(
                f'<g class="{color}" tabindex="0" aria-label="{escape(tip)}"><title>{escape(tip)}</title>'
                f'<rect class="fill" x="{x}" y="{y:.2f}" width="29" height="{height:.2f}" rx="4"/>'
                f'<text class="bar-label" text-anchor="middle" x="{x + 14.5}" y="{y - 7:.2f}">'
                f'{_percent(score.point_rate) if score.games else "—"}</text></g>'
            )
    chart = _wide_svg("".join(pieces), width, 279 if final else 261,
                      "执攻与执守得分率 · Attack and defence", "role-chart")
    legend = ('<div class="legend"><span class="attack">执攻 · Attack</span>'
              '<span class="defence">执守 · Defence</span></div>')
    note = "按队伍和执棋方比较得分率。"
    if final and (new := _team(stage, NEW)):
        each = _common_game_count(score for _, score in _matchups(stage))
        subgroup = f"的 {each} 局" if each else "的对局"
        note = (f"各强敌的数据均来自与 Team19-New 对阵{subgroup}。"
                f"Team19-New 汇总全部 {new.score.games} 局。")
    return _section("执攻与执守", "Attack and defence", legend + chart, note)


def _evolution(stages: dict[str, Stage], comparison):
    series = []
    versions = GENERATIONS if comparison else HISTORICAL_VERSIONS
    title = "Team19 四代进化" if comparison else "Team19 三代进化"
    for name, label, color in (("official", "60 秒官方赛", "official"), ("fast", "10 秒快棋", "fast")):
        if comparison:
            scores = comparison.get(name, {})
        else:
            stage = stages.get(name)
            scores = {team.name: team.score for team in stage.teams if team.name in versions} if stage else {}
        if scores:
            series.append((label, color, scores))
    if not series:
        return _section(title, "Team19 progression",
                        '<p class="empty-figure">载入标准开局赛或快棋战报后可查看三代得分率。</p>')

    width = 656 if comparison else 544
    right = width - 18
    group_width = (right - 58) / len(versions)
    y_base, plot_height = 182, 128
    pieces = []
    for tick in (0, 25, 50, 75, 100):
        y = y_base - plot_height * tick / 100
        pieces.append(f'<line class="grid-line" x1="58" x2="{right}" y1="{y}" y2="{y}"/>'
                      f'<text class="axis-muted" x="48" y="{y + 4}" text-anchor="end">{tick}%</text>')
    for i, name in enumerate(versions):
        center = 58 + (i + 0.5) * group_width
        short, _, variant = _label(name)
        if name == NEW and comparison:
            short = "New@91ffcc8"
        pieces.append(f'<text class="axis-name {variant}" x="{center}" y="213" '
                      f'text-anchor="middle">{short}</text>')
        for j, (label, color, scores) in enumerate(series):
            if name not in scores:
                continue
            score = scores[name]
            rate = score.point_rate
            x = center + (j - (len(series) - 1) / 2) * 51
            height = rate * plot_height
            y = y_base - height
            tip = f"{name} · {label} · {score.points}/{2 * score.games} 分 · {_percent(rate)}"
            pieces.append(
                f'<g class="{color}" tabindex="0" aria-label="{tip}"><title>{tip}</title>'
                f'<rect class="fill" x="{x - 19}" y="{y:.2f}" width="38" height="{height:.2f}" rx="5"/>'
                f'<text class="bar-label" x="{x}" y="{y - 8:.2f}" text-anchor="middle">{_percent(rate)}</text></g>'
            )
    legend = (f'<div class="legend">'
              f'{"".join(f"<span class={color}>{label}</span>" for label, color, _ in series)}</div>')
    chart = (f'<svg class="evolution" viewBox="0 0 {width} 226" role="img" aria-label="Team19 progression">'
             f'<title>Team19 progression · 各赛段得分率</title>{"".join(pieces)}</svg>')
    note = "按历史版本先后排列，比较两个预算档的得分率。"
    if comparison:
        games = "、".join(f"{label}每版 {scores[NATIVE_VERSION].games} 局" for label, _, scores in series)
        note = f"四代均对阵原联赛同一批外部队伍，使用相同开局并交换攻守。{games}。"
    return _section(title, "Team19 progression", legend + chart, note)


def _ending_chart(reasons: Counter):
    games = reasons.total()
    bars, legend = [], []
    start = 0.0
    for reason, count in reasons.most_common():
        name, english, color = REASONS.get(reason, (reason, "", "#63737d"))
        share = count / games
        tip = f"{name} · {count} 局 · {_percent(share)}"
        bars.append(f'<rect x="{start:.4f}" width="{share * 100:.4f}" height="10" fill="{color}">'
                    f'<title>{escape(tip)}</title></rect>')
        legend.append(f'<li><i style="--key:{color}"></i><span>{escape(name)}'
                      f'<small lang="en">{escape(english)}</small></span><b>{count}</b>'
                      f'<span class="share">{_percent(share)}</span></li>')
        start += share * 100
    return ('<svg class="ending-bar" viewBox="0 0 100 10" preserveAspectRatio="none" role="img" '
            'aria-label="终局原因占比"><title>终局原因占比 · Game endings</title>'
            f'{"".join(bars)}</svg><ul class="ending-key">{"".join(legend)}</ul>')


def _stability(included: list[Stage]):
    reasons = Counter()
    statistics = defaultdict(Counter)
    for stage in included:
        reasons.update(stage.reasons)
        for team, counts in stage.stability.items():
            statistics[team].update(counts)
    max_fallbacks = max(statistics[team]["fallbacks"] for team in statistics)
    rows = []
    for team, counts in sorted(statistics.items(), key=lambda item: (-item[1]["fallbacks"], item[0])):
        fallback, errors = counts["fallbacks"], counts["internalErrors"]
        forfeits = sum(counts[reason] for reason in FORFEITS)
        tip = f"{team} · {fallback} 兜底 · {errors} 内部错误 · {forfeits} 裁判判负"
        rows.append(f'<div class="stability-row" title="{escape(tip)}"><b>{escape(team)}</b>'
                    f'<div class="fallback-count">{_bar(fallback / max_fallbacks if max_fallbacks else 0, tip)}'
                    f'<span>{fallback:,}</span></div><span>{errors:,}</span></div>')
    breakdown = ('<div class="stability-head"><span>队伍</span><span>兜底 · Fallbacks</span>'
                 '<span>内部错误</span></div>' + "".join(rows))
    return _section("终局与稳定性", "Endings and reliability", _ending_chart(reasons) + breakdown,
                    "每局只计一次终局。兜底按手计，选手可以用兜底走法继续对局。", "reliability")


def _stage_rules(stage: Stage):
    if stage.name == "final":
        matchups = _matchups(stage)
        each = _common_game_count(score for _, score in matchups)
        opening = f"{each // 2} 个固定开局 × 换边" if each and each % 2 == 0 else "固定开局换边"
        rules = ["每方整局 60 秒", opening,
                 f"1v{len(matchups)} 车轮对照 · Team19-New 逐一迎战 {len(matchups)} 支外部强敌",
                 f"{_duel_groups(stage)}，共 {stage.scheduled} 局"]
    else:
        official = stage.name == "official"
        each = _common_game_count(stage.matrix.values()) if stage.games == stage.scheduled else None
        per_team = _common_game_count(team.score for team in stage.teams) if each else None
        league = f"{len(stage.teams)} 队{'单循环' if official else '全循环'}，两两换边"
        if each and per_team:
            league += f"各 {each} 局，每队 {per_team} 局"
        opening = f"{each // 2} 个固定开局 × 换边" if each and each % 2 == 0 else "固定开局换边"
        rules = (["每方整局 60 秒", "标准初始棋盘", league, f"共 {stage.scheduled} 局",
                  "胜 2 / 和 1 / 负 0", "30 分制模拟课程分 = 30 × 本队积分 ÷ 榜首积分"] if official else
                 ["每方整局 10 秒", opening, league, f"共 {stage.scheduled} 局",
                  "用多开局大样本检验抗压能力与排名稳定性"])
    chips = "".join(f"<li>{rule}</li>" for rule in rules)
    return f'<ul class="stage-rules" aria-label="赛制说明">{chips}</ul>'


def _quick_links(stages: dict[str, Stage]):
    cards = []
    for key, stage in stages.items():
        result = ""
        if new := _team(stage, NEW):
            score = new.score
            result = (f"{score.points} / {2 * score.games} 分 · {_percent(score.point_rate)} 得分率" if key == "final"
                      else f"第 {new.rank} 名 · {score.points} / {2 * score.games} 分")
            result = f'<span class="stage-record">Team19-New · {result}</span>'
        cards.append(
            f'<button type="button" class="duel stage-link" data-stage-tab="tab-{key}" aria-controls="panel-{key}">'
            f'<span class="stage-name">{STAGES[key]}</span><span class="stage-intro">{STAGE_INTROS[key]}</span>'
            f'<strong>{stage.games:,} 局</strong>{result}</button>'
        )
    return _section("赛段速览", "Stage guide", f'<div class="duels">{"".join(cards)}</div>')


def _overview_panel(stages: dict[str, Stage], evolution):
    included = list(stages.values())
    rounds = [stages[key] for key in ROUND_LABELS if key in stages]
    cards = [_games_metric(included), *_round_version_metrics(rounds)]
    if not rounds and (final := stages.get("final")) and (new := _team(final, NEW)):
        cards.extend((_new_points_metric(new.score), _role_metric(final)))
    cards.append(_forfeits_metric(included))
    content = (_metrics(cards) + _quick_links(stages)
               + f'<div class="chart-pair">{_evolution(stages, evolution)}{_stability(included)}</div>')
    archive = "<p>九队联赛记录的是 New@91ffcc8 时的赛果。</p>" if evolution else ""
    return _panel("overview", "联赛总览", archive, content)


def _stage_panel(stage: Stage, numba_nps: int | None):
    if stage.name == "final":
        comparison = _final_metrics(stage) + _duels(stage, numba_nps)
    else:
        cards = [_games_metric([stage]), *_round_version_metrics([stage]), _forfeits_metric([stage])]
        comparison = _metrics(cards) + _standings(stage) + _heatmap(stage)
    detail = _stage_rules(stage) + f"<p>{stage.games:,} / {stage.scheduled:,} 场已完成</p>"
    return _panel(stage.name, STAGES[stage.name], detail, comparison + _roles(stage) + _stability([stage]))


def _panel(key, title, detail, content):
    hidden = "" if key == "overview" else " hidden"
    return (
        f'<section id="panel-{key}" class="report-panel" role="tabpanel" '
        f'aria-labelledby="tab-{key}" tabindex="0"{hidden}>'
        f'<div class="panel-heading"><h2>{title}</h2>{detail}</div>{content}</section>'
    )


def render(stages: dict[str, Stage], numba_nps: int | None = None, *, evolution=None) -> str:
    css = (HERE / "dashboard.css").read_text(encoding="utf-8")
    script = (HERE / "dashboard.js").read_text(encoding="utf-8")
    if stages:
        tabs = []
        for key, label in {"overview": "总览", **{key: STAGES[key] for key in stages}}.items():
            selected = key == "overview"
            tabs.append(f'<button type="button" id="tab-{key}" role="tab" aria-controls="panel-{key}" '
                        f'aria-selected="{str(selected).lower()}" tabindex="{0 if selected else -1}">'
                        f'{label}</button>')
        navigation = f'<nav class="tabs" role="tablist" aria-label="联赛赛段">{"".join(tabs)}</nav>'
        panels = _overview_panel(stages, evolution) + "".join(
            _stage_panel(stage, numba_nps) for stage in stages.values()
        )
    else:
        navigation = ""
        panels = ('<section class="chart"><h2>暂无可展示的赛果</h2><p>每个赛段需要已保存的 JSONL '
                  '以及 standings、roles、matrix、stability 四份 CSV。</p></section>')
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tablut 联赛战报 · JC4004</title>
<style>{css}</style>
</head>
<body>
<header class="page-header"><div class="shell header-layout">
<div><div class="course">JC4004 · Tablut</div><h1>联赛战报</h1></div>
<button type="button" class="print-button" id="print-report">打印视图<span lang="en">Print view</span></button>
</div></header>
<main class="shell">
{navigation}
{panels}
</main>
<script>{script}</script>
</body>
</html>
"""


def _numba_throughput(results_dir: Path) -> int | None:
    report = results_dir / "qualification.md"
    if report.is_file():
        match = re.search(r"中位吞吐 ([0-9]+(?:,[0-9]{3})*) 节点[/／]秒", report.read_text(encoding="utf-8"))
        if match:
            return int(match[1].replace(",", ""))
    return None


def write_dashboard(results_dir: Path = RESULTS, output: Path | None = None):
    stages = load_stages(results_dir)
    evolution = load_evolution(results_dir, stages)
    document = render(stages, _numba_throughput(results_dir) if "final" in stages else None,
                      evolution=evolution)
    output = output if output is not None else results_dir / "dashboard.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output, stages


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        path, stages = write_dashboard(args.results_dir, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    loaded = ", ".join(f"{name}: {stage.games}" for name, stage in stages.items()) or "none"
    print(f"{path.resolve()} ({path.stat().st_size:,} bytes)")
    print(f"{sum(stage.games for stage in stages.values()):,} games | {loaded}")


if __name__ == "__main__":
    main()
