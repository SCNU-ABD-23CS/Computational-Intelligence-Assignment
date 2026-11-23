"""Read-only dashboard checks with small synthetic JSONL journals."""

from contextlib import redirect_stdout
from decimal import Decimal
from io import StringIO
from itertools import combinations
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import TOP6_BUDGETS, TOP6_GAMES, TOP6_TEAMS, dashboard, overview
from .dashboard_data import Score, load_stages


V2 = "Team19-v2"
GM = "Grandmaster-Numba"


def game(stage, opening, attack, defend, winner=None, *, reason=None, accepted=True, reused=False):
    budget = TOP6_BUDGETS[stage]
    return {
        "id": f"{budget}-{opening}-{attack}-A-{defend}",
        "stage": stage, "budget": budget, "opening": opening, "schedule": stage,
        "scheduledGames": TOP6_GAMES, "teams": {"A": attack, "D": defend}, "winner": winner,
        "reason": reason or {None: "repeated position", "A": "king captured", "D": "king escaped to the edge"}[winner],
        "prefix": [{}] * 4,
        "attempts": [{"side": "A", "team": attack, **({"event": {}} if accepted else {})}],
        **({"reusedFrom": {"file": "evolution-fast.jsonl"}} if reused else {}),
    }


def store(root, stage, rows, trailing=b""):
    path = root / f"{stage}.jsonl"
    path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode() + trailing)
    return path


def panel(html, name):
    return html.split(f'<section id="panel-{name}"', 1)[1].split('<section id="panel-', 1)[0]


class DashboardTests(unittest.TestCase):
    def setUp(self):
        workspace = TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)

    def test_empty_dashboard_keeps_exactly_three_tabs_and_excludes_legacy_reports(self):
        (self.root / "fast.jsonl").write_text("unneeded historical data", encoding="utf-8")
        output, stages = dashboard.write_dashboard(self.root)
        self.assertEqual(output, self.root / "dashboard.html")
        self.assertEqual([stage.games for stage in stages.values()], [0, 0])
        html = output.read_text(encoding="utf-8")
        self.assertEqual(re.findall(r'id="(tab-[^"]+)" role="tab"', html),
                         ["tab-overview", "tab-top6_60s", "tab-top6_10s"])
        summary = panel(html, "overview")
        self.assertEqual(summary.count('class="metric '), 4)
        self.assertIn("0/360", summary)
        self.assertIn("Team19-v2 攻守与稳定性", summary)
        for name in TOP6_BUDGETS:
            self.assertIn(f'data-stage-tab="tab-{name}"', summary)
            section = panel(html, name)
            self.assertIn("已完成 0/180 局 · 待跑 180 局", section)
            self.assertIn(f"--stage {name} --workers 1 --resume", section)
            self.assertNotIn('class="heatmap"', section)
        self.assertNotRegex(html, r'https?://|<script\s+src=|@import')

    def test_partial_dashboard_uses_draw_points_and_leaves_an_incomplete_line_untouched(self):
        stage = "top6_10s"
        rows = [
            game(stage, "o01", V2, GM, "A", reused=True),
            game(stage, "o01", GM, V2, reused=True),
        ]
        journal = store(self.root, stage, rows, trailing=b'{"id":"still writing')
        original = journal.read_bytes()
        output, stages = dashboard.write_dashboard(self.root)
        self.assertEqual(journal.read_bytes(), original)
        self.assertEqual(stages[stage].reused, 2)
        self.assertEqual(stages[stage].matrix[V2, GM], Score(1, 1, 0))
        html = panel(output.read_text(encoding="utf-8"), stage)
        self.assertIn("已完成 2/180 局 · 待跑 178 局", html)
        self.assertIn("已复用 2 局", html)
        self.assertIn("<b>75.0%</b><small>1–1–0</small>", html)
        self.assertIn("<b>25.0%</b><small>0–1–1</small>", html)
        self.assertIn("全部 180 局完成后显示课程模拟分", html)
        self.assertEqual(html.count('class="matrix-cell'), 36)

    def test_complete_dashboard_has_balanced_course_marks_and_role_chart(self):
        stage = "top6_60s"
        rows = []
        for first, second in combinations(TOP6_TEAMS, 2):
            for number in range(1, 7):
                for attack, defend in ((first, second), (second, first)):
                    winner = "A" if (attack, defend, number) == (V2, GM, 1) else None
                    rows.append(game(stage, f"o{number:02d}", attack, defend, winner))
        store(self.root, stage, rows)
        output, stages = dashboard.write_dashboard(self.root)
        report = stages[stage]
        self.assertTrue(report.complete)
        leader = report.teams[0]
        self.assertEqual((leader.name, leader.score.points, leader.marks), (V2, 61, 30.0))
        rival = next(team for team in report.teams if team.name == GM)
        self.assertGreater(leader.elo, rival.elo)
        self.assertEqual(report.roles[V2, "A"].games, 30)
        self.assertEqual(report.roles[V2, "D"].games, 30)
        html = panel(output.read_text(encoding="utf-8"), stage)
        self.assertIn("已完成 180/180 局", html)
        self.assertIn("课程模拟分 / 30", html)
        self.assertIn(">30.00<", html)
        self.assertIn('class="role-chart"', html)
        self.assertIn("终局与稳定性", html)
        self.assertNotIn("待跑", html)

    def test_forfeit_is_counted_against_the_losing_side_without_an_extra_ply(self):
        stage = "top6_60s"
        forfeited = game(stage, "o01", V2, GM, "D", reason="timeout", accepted=False)
        normal = game(stage, "o01", GM, V2, "A")
        store(self.root, stage, [forfeited, normal])
        stats = load_stages(self.root)[stage]
        self.assertEqual(stats.plies, 9)
        self.assertEqual(stats.stability[V2]["timeout"], 1)
        self.assertEqual(stats.stability[GM]["timeout"], 0)

    def test_journal_rejects_duplicate_fixture(self):
        stage = "top6_10s"
        row = game(stage, "o01", V2, GM)
        store(self.root, stage, [row, row])
        with self.assertRaisesRegex(ValueError, "top6_10s.jsonl: Invalid or duplicate fixture"):
            dashboard.write_dashboard(self.root)

    def test_complementary_heatmap_labels_sum_to_one_hundred_after_rounding(self):
        for games in range(1, 13):
            for points in range(2 * games + 1):
                wins, draws = divmod(points, 2)
                losses = games - wins - draws
                first = dashboard._matrix_percent(Score(wins, draws, losses))
                second = dashboard._matrix_percent(Score(losses, draws, wins))
                self.assertEqual(Decimal(first[:-1]) + Decimal(second[:-1]), Decimal("100.0"))

    def test_cli_supports_output_and_results_paths(self):
        output = self.root / "preview" / "dashboard.html"
        with (
            patch("sys.argv", ["dashboard", "--results-dir", str(self.root), "--output", str(output)]),
            redirect_stdout(StringIO()) as console,
        ):
            dashboard.main()
        self.assertTrue(output.is_file())
        self.assertIn("60 秒 6 强全循环 0/180", console.getvalue())

    def test_overview_can_skip_the_dashboard_refresh(self):
        with (
            patch.object(overview, "audit_stage", return_value=([{}], 1)),
            patch.object(overview, "write_report"),
            patch.object(overview, "write_overview", return_value=self.root / "overview.md"),
            patch.object(overview, "write_dashboard", return_value=(self.root / "dashboard.html", {})) as refresh,
            redirect_stdout(StringIO()),
        ):
            with patch("sys.argv", ["overview", "--stage", "official", "--no-dashboard"]):
                overview.main()
            refresh.assert_not_called()
            with patch("sys.argv", ["overview", "--stage", "official"]):
                overview.main()
            refresh.assert_called_once_with(overview.RESULTS)
