import csv
from contextlib import redirect_stdout
from decimal import Decimal
from io import StringIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import dashboard, overview, report
from .dashboard import NEW, OLD
from .dashboard_data import Score


def game(stage, index, attack, defend, winner, reason, *, prefix=0, accepted=True, fallback=False):
    return {
        "id": f"{stage}-{index}", "schedule": stage, "stage": stage,
        "teams": {"A": attack, "D": defend}, "winner": winner, "reason": reason,
        "elapsed": 0.01, "remaining": {"A": 1, "D": 1}, "prefix": [{}] * prefix,
        "attempts": [{"side": "A", "team": attack, "fallback": fallback,
                      **({"event": {"ply": prefix + 1}} if accepted else {})}],
    }


def store_stage(root, stage, games):
    folder = root / stage
    folder.mkdir()
    path = root / f"{stage}.jsonl"
    path.write_text("".join(json.dumps({**game, "scheduledGames": len(games)}) + "\n" for game in games),
                    encoding="utf-8")
    for table, rows in report.tables(games).items():
        with (folder / f"{table}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return path


def panel(page, name):
    return page.split(f'<section id="panel-{name}"', 1)[1].split('<section id="panel-', 1)[0]


class DashboardTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_heatmap_uses_point_rate_and_keeps_pure_win_rate_in_tooltip(self):
        store_stage(self.root, "official", [
            game("official", 1, NEW, OLD, "A", "king captured"),
            game("official", 2, OLD, NEW, None, "repeated position"),
        ])
        output, stages = dashboard.write_dashboard(self.root)
        stage = stages["official"]
        matchup = stage.matrix[NEW, OLD]
        self.assertEqual((stage.games, stage.plies, matchup.record), (2, 2, "1–1–0"))
        self.assertEqual((matchup.win_rate, matchup.point_rate), (0.5, 0.75))
        page = output.read_text(encoding="utf-8")
        self.assertIn('id="tab-official"', page)
        self.assertNotIn('id="tab-fast"', page)
        self.assertIn("+50.0 pp", page)
        official = panel(page, "official")
        self.assertIn("得分率 75.0% · 纯胜率 50.0% · 积分 3 比 1", official)
        self.assertIn("得分率 25.0% · 纯胜率 0.0% · 积分 1 比 3", official)
        cells = re.findall(
            r'<g class="heat-cell".*?<rect[^>]+fill="([^"]+)"/><text class="heat-value"[^>]+>'
            r'([^<]+)</text><text class="heat-record"[^>]+>([^<]+)</text>',
            official,
        )
        self.assertCountEqual(cells, [("#c8e7d6", "75.0%", "1–1–0"), ("#f4d8d4", "25.0%", "0–1–1")])
        self.assertNotRegex(page, r'https?://|<script\s+src=|@import')

    def test_heatmap_pair_labels_sum_to_one_hundred_after_rounding(self):
        for games in (2, 12, 24, 80, 96):
            for points in range(2 * games + 1):
                wins, draws = divmod(points, 2)
                losses = games - wins - draws
                one = dashboard._matrix_percent(Score(wins, draws, losses))
                other = dashboard._matrix_percent(Score(losses, draws, wins))
                self.assertEqual(Decimal(one[:-1]) + Decimal(other[:-1]), Decimal("100.0"))

    def test_overview_summarizes_stages_and_evolution_without_round_robin_charts(self):
        for stage in ("official", "fast"):
            store_stage(self.root, stage, [
                game(stage, 1, NEW, OLD, "A", "king captured"),
            ])
        opponent = "QSearch-PVS"
        store_stage(self.root, "final", [
            game("final", 1, NEW, opponent, "A", "king captured", prefix=4),
            game("final", 2, opponent, NEW, "A", "king captured", prefix=4),
        ])
        output, stages = dashboard.write_dashboard(self.root)
        self.assertEqual(list(stages), ["official", "fast", "final"])
        page = output.read_text(encoding="utf-8")
        summary = panel(page, "overview")
        self.assertIn("12 手 · Recorded plies", summary)
        self.assertIn("两档均第 1", summary)
        self.assertIn("30 / 30 模拟课程分", summary)
        self.assertEqual(summary.count('class="metric '), 4)
        for chart in ('class="stand-row', 'class="heatmap"', 'class="role-chart"'):
            self.assertNotIn(chart, summary)
        self.assertIn('class="ending-bar"', summary)
        self.assertEqual(page.count('class="evolution"'), 1)

        for key in stages:
            self.assertIn(f'data-stage-tab="tab-{key}"', summary)
            segment = panel(page, key)
            self.assertIn('class="stage-rules"', segment)
            self.assertNotIn('class="evolution"', segment)
        self.assertIn("标准初始棋盘", panel(page, "official"))
        self.assertIn("每方整局 10 秒", panel(page, "fast"))
        self.assertIn("1v1 车轮对照", panel(page, "final"))

    def test_finals_show_matchups_and_side_rates_without_a_rank(self):
        pair_results = [
            ("Grandmaster-Numba", "DLLLLLLLLLLL", "DDDDLLLLLLLL"),
            ("QSearch-PVS", "WWLLLLLLLLLL", "WWWWWDLLLLLL"),
            ("BFS-Strategist", "WLLLLLLLLLLL", "WWWWWWDLLLLL"),
        ]
        reasons = {"A": "king captured", "D": "king escaped to the edge", None: "repeated position"}
        games = []
        for opponent, attack, defend in pair_results:
            for own_side, outcomes in (("A", attack), ("D", defend)):
                other_side = "A" if own_side == "D" else "D"
                seats = (NEW, opponent) if own_side == "A" else (opponent, NEW)
                for outcome in outcomes:
                    winner = {"W": own_side, "D": None, "L": other_side}[outcome]
                    games.append(game("final", len(games), *seats, winner, reasons[winner], prefix=4))
        store_stage(self.root, "final", games)
        (self.root / "qualification.md").write_text("中位吞吐 1,941,393 节点／秒。\n", encoding="utf-8")
        output, _ = dashboard.write_dashboard(self.root)
        final = panel(output.read_text(encoding="utf-8"), "final")
        self.assertEqual(final.count('class="metric '), 4)
        for label in ("1v3 车轮对照", "12 个固定开局 × 换边", "72 局", "3 组 × 24 局",
                      "35 / 144", "24.3% 得分率 · 14 胜 7 和 51 负", "9.7% / 38.9%",
                      "PVS 加静搜", "国王逃生步数（BFS）", "194 万节点/秒", "72 局合计",
                      "各强敌的数据均来自与 Team19-New 对阵的 24 局"):
            self.assertIn(label, final)
        self.assertEqual(final.count('class="duel"'), 3)
        self.assertLess(final.index("Team19-New matchups"), final.index('class="role-chart"'))
        for chart in ("League standings", "Team19-New rank", 'class="stand-row', 'class="heatmap"'):
            self.assertNotIn(chart, final)

    def test_missing_optional_stages_allow_an_empty_dashboard(self):
        (self.root / "fast.jsonl").touch()
        output, stages = dashboard.write_dashboard(self.root)
        self.assertEqual(stages, {})
        self.assertIn("暂无可展示的赛果", output.read_text(encoding="utf-8"))

    def test_forfeit_attempt_is_excluded_from_plies(self):
        store_stage(self.root, "final", [
            game("final", 1, NEW, "BFS-Strategist", "D", "timeout", prefix=4, accepted=False),
            game("final", 2, "BFS-Strategist", NEW, "A", "king captured", prefix=4, fallback=True),
        ])
        output, stages = dashboard.write_dashboard(self.root)
        stage = stages["final"]
        self.assertEqual((stage.plies, stage.reasons["timeout"]), (9, 1))
        self.assertEqual(stage.stability[NEW]["timeout"], 1)
        self.assertEqual(stage.stability["BFS-Strategist"]["fallbacks"], 1)
        self.assertIn('class="metric metric-alert"', output.read_text(encoding="utf-8"))

    def test_cli_custom_output_and_escaped_team_names(self):
        store_stage(self.root, "fast", [
            game("fast", 1, NEW, 'Bot <test> "&"', "A", "king captured"),
        ])
        output = self.root / "share" / "dashboard.html"
        overview_path = self.root / "overview.md"
        overview_path.write_text("existing report\n", encoding="utf-8")
        with patch("sys.argv", ["dashboard", "--results-dir", str(self.root), "--output", str(output)]):
            with redirect_stdout(StringIO()) as console:
                dashboard.main()
        page = output.read_text(encoding="utf-8")
        self.assertIn("1 games | fast: 1", console.getvalue())
        self.assertIn("Bot &lt;test&gt; &quot;&amp;&quot;", page)
        self.assertEqual(overview_path.read_text(encoding="utf-8"), "existing report\n")

    def test_mismatched_csv_and_jsonl_are_reported(self):
        path = store_stage(self.root, "official", [
            game("official", 1, NEW, OLD, "A", "king captured"),
        ])
        row = json.loads(path.read_text(encoding="utf-8"))
        row["winner"] = "D"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "official: CSV game or point totals differ"):
            dashboard.write_dashboard(self.root)

    def test_overview_can_skip_the_dashboard_refresh(self):
        with (patch.object(overview, "audit_stage", return_value=([{}], 1)),
              patch.object(overview, "write_report") as write_report,
              patch.object(overview, "write_overview", return_value=self.root / "overview.md") as write_md,
              patch.object(overview, "write_dashboard", return_value=(self.root / "dashboard.html", {})) as write_html,
              redirect_stdout(StringIO())):
            with patch("sys.argv", ["overview"]):
                overview.main()
            self.assertEqual(write_report.call_count, 3)
            write_md.assert_called_once()
            write_html.assert_called_once_with(overview.RESULTS)
            with patch("sys.argv", ["overview", "--stage", "official", "--no-dashboard"]):
                overview.main()
            self.assertEqual(write_report.call_count, 4)
            write_md.assert_called_once()
            write_html.assert_called_once()
