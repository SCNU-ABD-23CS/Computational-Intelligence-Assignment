"""Build a standalone dashboard from saved tournament records."""

from argparse import ArgumentParser
from html import escape
from pathlib import Path

from . import HERE, RESULTS
from .dashboard_data import STAGES, Score, Stage, load_stages
from .report import FORFEITS, REASON_LABELS


V2 = "Team19-v2"
V3 = "Team19-v3"
LABELS = {
    V2: "v2", V3: "v3", "Grandmaster-Numba": "Numba", "Grandmaster-Plus": "GM Plus",
    "QSearch-PVS": "QSearch",
    "Team19-91ffcc8": "91ffcc8", "BFS-Strategist": "BFS", "Guided-MCTS": "MCTS",
    "Team19-New": "New", "Team19-PreFix": "PreFix", "Team19-Old": "Old",
    "GreedyBot": "Greedy", "NaiveAB-D2": "AB-D2", "RandomBot": "Random",
}
ENDING_COLORS = (
    "#167968", "#365e9d", "#986431", "#8468aa", "#64757d",
    "#a43d67", "#bd574d", "#b84343",
)


def _rate(score: Score):
    return f"{float(score.point_rate):.1%}" if score.games else "—"


def _matrix_percent(score: Score) -> str:
    # Rounding on the same thousandth scale keeps mirror labels adding to 100%.
    return f"{round(score.point_rate * 1000) / 10:.1f}%"


def _points(score: Score) -> str:
    return f"{score.points}/{2 * score.games}" if score.games else "—"


def _bar(score: Score, label: str, variant="") -> str:
    tip = escape(f"{label} · {_rate(score)} · {_points(score)} 分")
    return (
        f'<svg class="bar {variant}" viewBox="0 0 100 10" preserveAspectRatio="none" '
        f'role="img" aria-label="{tip}"><title>{tip}</title><rect class="rail" width="100" height="10"/>'
        f'<rect class="fill" width="{float(score.point_rate) * 100:.4f}" height="10"/></svg>'
    )


def _metric(label, value, detail, style=""):
    return (f'<article class="metric {style}"><p>{label}</p><strong>{value}</strong>'
            f'<small>{detail}</small></article>')


def _section(title, content, detail=""):
    note = f'<p class="note">{detail}</p>' if detail else ""
    return f'<section class="chart"><h2>{title}</h2>{note}{content}</section>'


def _incidents(counts):
    return (counts["timeout"], counts["illegal move"],
            counts["exception"] + counts["internalErrors"], counts["fallbacks"])


def _side_bar(label, variant, score):
    return (f'<div class="side-measure"><span>{label}</span><b>{_rate(score)}</b>'
            f'{_bar(score, label, variant)}<small>{_points(score)} 分</small></div>')


def _overview(stages: dict[str, Stage]):
    schedule = next(iter(stages.values())).schedule
    focus = schedule.anchor or next((name for name in (V3, V2) if name in schedule.teams), schedule.teams[0])
    short_name = LABELS.get(focus, focus)
    played = sum(stage.games for stage in stages.values())
    planned = sum(stage.schedule.games for stage in stages.values())
    forfeits = sum(stage.reasons[reason] for stage in stages.values() for reason in FORFEITS)
    breakdown = " · ".join(f"{stage.budget:g} 秒 {stage.games}/{stage.schedule.games}"
                           for stage in stages.values())
    metrics = [_metric("已完成对局", f"{played}/{planned}", breakdown)]
    links, roles = [], []
    for name, stage in stages.items():
        score = next(team.score for team in stage.teams if team.name == focus)
        total = stage.schedule.games
        metrics.append(_metric(f"{short_name} · {stage.budget:g} 秒得分率", _rate(score),
                               f"{_points(score)} 分 · {score.games} 局" if score.games else "等待赛果",
                               "featured"))
        phase = "已完成" if stage.complete else f"待跑 {total - stage.games} 局"
        reuse = f" · 已复用 {stage.reused} 局" if stage.reused else ""
        links.append(
            f'<button class="stage-link" type="button" data-stage-tab="tab-{name}" aria-controls="panel-{name}">'
            f'<b>{stage.title}</b><strong>{stage.games}/{total}</strong><span>{phase}{reuse}</span></button>'
        )
        attack = _side_bar("执攻", "attack", stage.roles.get((focus, "A"), Score()))
        defend = _side_bar("执守", "defence", stage.roles.get((focus, "D"), Score()))
        numbers = _incidents(stage.stability[focus])
        stats = " · ".join(f"{num} {label}" for num, label in zip(numbers, ("超时", "非法", "异常", "兜底")))
        info = f"出战 {score.games} 局 · {stats}" if score.games else "等待对局记录"
        roles.append(f'<article class="side-card"><h3>{stage.budget:g} 秒档</h3>'
                     f'{attack}{defend}<p class="note">{info}</p></article>')
    metrics.append(_metric("裁判判负", str(forfeits), "超时、非法步与裁判异常",
                           "warning" if forfeits else "good"))
    stage_grid = f'class="two-column" style="--stage-columns:{min(2, len(stages))}"'
    return (f'<div class="metrics" style="--metric-columns:{min(4, len(metrics))}">{"".join(metrics)}</div>'
            + _section("赛段速览", f'<div {stage_grid}>{"".join(links)}</div>')
            + _section(f"{focus} 攻守与稳定性", f'<div {stage_grid}>{"".join(roles)}</div>'))


def _standings(stage: Stage):
    anchor = stage.schedule.anchor
    teams = stage.teams
    if anchor:
        teams = sorted(teams, key=lambda team: (team.name != anchor, team.name))
    rows = []
    for team in teams:
        score = team.score
        rank = team.rank or "—"
        rating = f"{team.elo:.1f}" if team.elo is not None else "—"
        marks = f"{team.marks:.2f}" if team.marks is not None else "—"
        points = score.points if score.games else "—"
        record = score.record if score.games else "—"
        style = ' class="highlight"' if team.name in (V3, V2) else ""
        rank_cell = "" if anchor else f"<td>{rank}</td>"
        marks_cell = "" if anchor else f"<td>{marks}</td>"
        rows.append(f'<tr{style}>{rank_cell}<th scope="row">{team.name}</th><td>{score.games}</td>'
                    f'<td>{points}</td><td>{record}</td><td>{_rate(score)}</td><td>{rating}</td>{marks_cell}</tr>')
    rank_head = "" if anchor else "<th>排名</th>"
    marks_head = "" if anchor else "<th>课程模拟分 / 30</th>"
    table = (f'<table class="standings"><thead><tr>{rank_head}'
             '<th class="team-column">队伍</th><th>已赛</th><th>积分</th>'
             f'<th>胜–和–负</th><th>得分率</th><th>Elo</th>{marks_head}</tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table>')
    detail = "模拟课程分 = 30 × 本队积分 ÷ 榜首积分。Elo 同时估计执攻方偏差。"
    if anchor:
        detail = f"{anchor} 的数据合计所有挑战局，各挑战者的数据来自与擂主的对局。"
        title = "车轮战对阵成绩"
    else:
        title = f"{len(teams)} 队排行榜"
        if not stage.complete:
            detail = f"暂按已完成对局得分率排序。全部 {stage.schedule.games} 局完成后显示课程模拟分。"
    return _section(title, f'<div class="scroll-table">{table}</div>', detail)


def _heat_color(rate):
    neutral = (240, 243, 243)
    target = (157, 219, 186) if rate >= 0.5 else (248, 186, 176)
    strength = abs(rate * 2 - 1)
    return "#" + "".join(f"{round(a + strength * (b - a)):02x}" for a, b in zip(neutral, target))


def _heatmap(stage: Stage):
    teams = stage.schedule.teams
    heads = "".join(f'<th scope="col" title="{name}">{LABELS.get(name, name)}</th>' for name in teams)
    rows = []
    for name in teams:
        cells = []
        for opponent in teams:
            score = stage.matrix.get((name, opponent))
            if score is None:
                cells.append('<td class="matrix-cell empty-cell">—</td>')
                continue
            rate = _matrix_percent(score)
            reverse = stage.matrix[opponent, name]
            title = escape(f"{name} 对 {opponent} · {score.games} 局 · 得分率 {rate} · "
                           f"胜–和–负 {score.record} · 积分 {score.points}:{reverse.points}")
            cells.append(f'<td class="matrix-cell" title="{title}" style="background:{_heat_color(float(score.point_rate))}">'
                         f'<b>{rate}</b><small>{score.record}</small></td>')
        rows.append(f'<tr><th scope="row" title="{name}">{LABELS.get(name, name)}</th>{"".join(cells)}</tr>')
    width = max(300, 72 * (len(teams) + 1))
    table = (f'<table class="heatmap" style="--matrix-width:{width}px">'
             f'<thead><tr><th scope="col">行方</th>{heads}</tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table>')
    note = (f"从行方阅读。得分率为（胜 + 0.5 × 和）÷ 已赛场次。每组打满 {stage.schedule.games_per_pair} 局，"
            "第二行显示胜–和–负。")
    return _section(f"{len(teams)}×{len(teams)} 交手得分率", f'<div class="scroll-table">{table}</div>', note)


def _role_chart(stage: Stage):
    teams = stage.schedule.teams
    x0, dx, y0, height = 62, 118, 175, 143
    width = max(350, x0 + dx * len(teams) + 24)
    min_width = max(300, 75 * len(teams))
    parts = []
    for tick in (0, 25, 50, 75, 100):
        y = y0 - height * tick / 100
        parts.append(f'<line class="grid-line" x1="{x0}" x2="{width - 12}" y1="{y}" y2="{y}"/>'
                     f'<text class="axis-muted" x="{x0 - 9}" y="{y + 4}" text-anchor="end">{tick}%</text>')
    for index, team in enumerate(teams):
        center = x0 + (index + 0.5) * dx
        parts.append(f'<text class="axis-name" x="{center}" y="201" text-anchor="middle">{LABELS.get(team, team)}</text>')
        for side, label, style, offset in (("A", "执攻", "attack", -37), ("D", "执守", "defence", 8)):
            score = stage.roles.get((team, side), Score())
            bar_height = height * float(score.point_rate)
            x, y = center + offset, y0 - bar_height
            tip = escape(f"{team} · {label} · {_rate(score)} · {_points(score)} 分")
            parts.append(f'<g class="{style}" aria-label="{tip}"><title>{tip}</title>'
                         f'<rect class="fill" x="{x}" y="{y:.2f}" width="29" height="{bar_height:.2f}" rx="3"/>'
                         f'<text class="bar-label" x="{x + 14.5}" y="{y - 7:.2f}" text-anchor="middle">'
                         f'{_rate(score)}</text></g>')
    legend = '<div class="legend"><span class="attack">执攻</span><span class="defence">执守</span></div>'
    svg = (f'<svg class="role-chart" style="--plot-width:{width}px;--plot-min:{min_width}px" '
           f'viewBox="0 0 {width} 212" role="img" aria-label="执攻与执守">'
           f'<title>执攻与执守得分率</title>{"".join(parts)}</svg>')
    return _section("执攻与执守", legend + f'<div class="scroll-table role-scroll">{svg}</div>')


def _stability(stage: Stage):
    x, stripes, items = 0.0, [], []
    for index, (reason, count) in enumerate(stage.reasons.most_common()):
        label = escape(REASON_LABELS.get(reason, reason))
        color = ENDING_COLORS[index % len(ENDING_COLORS)]
        share = count / stage.games * 100
        stripes.append(f'<rect x="{x:.4f}" width="{share:.4f}" height="10" fill="{color}">'
                       f'<title>{label} · {count} 局</title></rect>')
        items.append(f'<li><i style="--key:{color}"></i><span>{label}</span><b>{count} 局</b></li>')
        x += share
    ending = ('<div><svg class="ending-bar" viewBox="0 0 100 10" preserveAspectRatio="none" '
              f'role="img" aria-label="终局原因占比">{"".join(stripes)}</svg>'
              f'<ul class="ending-key">{"".join(items)}</ul></div>')
    rows = []
    for team in stage.schedule.teams:
        stats = _incidents(stage.stability[team])
        cells = "".join(f"<td>{num}</td>" for num in stats)
        style = ' class="highlight"' if team in (V3, V2) else ""
        rows.append(f'<tr{style}><th scope="row">{team}</th>{cells}</tr>')
    table = ('<table class="stability"><thead><tr><th>队伍</th><th>超时</th><th>非法步</th>'
             f'<th>异常</th><th>兜底</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')
    return _section("终局与稳定性", f'<div class="reliability">{ending}<div class="scroll-table">{table}</div></div>',
                    "终局按局计。异常含裁判判负和搜索内部错误，兜底按手计。")


def _round(stage: Stage, resume_command: str | None):
    schedule = stage.schedule
    pairs = sum(1 for _ in schedule.pairs())
    rules = [f"每方整局 {stage.budget:g} 秒", f"前 {len(schedule.openings)} 个固定开局 × 换边",
             f"{schedule.format_label} · {pairs} 组各 {schedule.games_per_pair} 局", "胜 2 / 和 1 / 负 0"]
    if schedule.anchor:
        rules.append(f"擂主 {schedule.anchor}")
    if stage.reused:
        rules.append(f"已复用 {stage.reused} 局")
    rule_bar = f'<ul class="stage-rules">{"".join(f"<li>{rule}</li>" for rule in rules)}</ul>'
    remaining = schedule.games - stage.games
    status = f"已完成 {stage.games}/{schedule.games} 局"
    if remaining:
        status += f" · 待跑 {remaining} 局"
    content = f'{rule_bar}<p class="round-status">{status}</p>'
    if stage.games:
        content += _standings(stage) + _heatmap(stage) + _role_chart(stage) + _stability(stage)
    if remaining and (resume_command or stage.name in STAGES):
        command = resume_command or f"python -B -m tools.arena.runner --stage {stage.name} --workers 1 --resume"
        description = "运行后自动刷新这个看板。"
        if not stage.games and stage.name == "top6_10s":
            description = "启动时复用本地已有的同条件战绩，再跑剩余赛程。完成后自动更新看板。"
        content += _section("继续赛程", f'<code class="command">{escape(command)}</code>', description)
    return content


def _panel(key, title, content):
    hidden = "" if key == "overview" else " hidden"
    return (f'<section id="panel-{key}" class="report-panel" role="tabpanel" '
            f'aria-labelledby="tab-{key}" tabindex="0"{hidden}><h2 class="panel-title">{title}</h2>{content}</section>')


def render(stages: dict[str, Stage], *, resume_command: str | None = None) -> str:
    css = (HERE / "dashboard.css").read_text(encoding="utf-8")
    script = (HERE / "dashboard.js").read_text(encoding="utf-8")
    tabs = []
    for name, title in {"overview": "总览", **{name: stage.title for name, stage in stages.items()}}.items():
        selected = name == "overview"
        tabs.append(f'<button type="button" id="tab-{name}" role="tab" aria-controls="panel-{name}" '
                    f'aria-selected="{str(selected).lower()}" tabindex="{0 if selected else -1}">{title}</button>')
    panels = _panel("overview", "总览", _overview(stages)) + "".join(
        _panel(name, stage.title, _round(stage, resume_command)) for name, stage in stages.items()
    )
    schedule = next(iter(stages.values())).schedule
    title = "六强联赛" if tuple(stages) == tuple(STAGES) else f"{schedule.format_label}实验"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tablut {title} · JC4004</title>
<style>{css}</style>
</head>
<body>
<header class="page-header"><div class="shell header-layout">
<div><p class="course">JC4004 · Tablut</p><h1>{title}</h1></div>
<button type="button" class="print-button" id="print-report">打印视图</button>
</div></header>
<main class="shell">
<nav class="tabs" role="tablist" aria-label="联赛赛段">{"".join(tabs)}</nav>
{panels}
</main>
<script>{script}</script>
</body>
</html>
"""


def write_dashboard(results_dir: Path = RESULTS, output: Path | None = None, *,
                    schedules=None, resume_command: str | None = None):
    stages = load_stages(results_dir, schedules)
    document = render(stages, resume_command=resume_command)
    path = output if output is not None else results_dir / "dashboard.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path, stages


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        path, stages = write_dashboard(args.results_dir, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"{path.resolve()} ({path.stat().st_size:,} bytes)")
    print(" · ".join(f"{stage.title} {stage.games}/{stage.schedule.games}" for stage in stages.values()))


if __name__ == "__main__":
    main()
