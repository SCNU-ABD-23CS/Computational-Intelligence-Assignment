"""Parameterised Arena CLI and its journals, using synthetic match results."""

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import HERE, TOP6_TEAMS, openings, report, runner


def forfeited_match(match):
    setup = openings.start(match.opening)
    return {
        "schema": runner.SCHEMA, "id": match.id, "budget": match.budget,
        "opening": match.opening.name, "teams": {"A": match.attacker, "D": match.defender},
        "initial": setup.positions[0], "prefix": setup.turns, "winner": "D", "reason": "timeout",
        "elapsed": match.budget + 1, "remaining": {"A": -1, "D": match.budget},
        "attempts": [{"side": "A", "team": match.attacker, "elapsed": match.budget + 1,
                      "refereeRemaining": -1, "move": []}],
    }


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        workspace = self.enterContext(TemporaryDirectory(dir=HERE))
        self.results = Path(workspace) / "arena-results"
        self.enterContext(patch.object(runner, "RESULTS", self.results))

    def cli(self, *args):
        with redirect_stdout(StringIO()) as output:
            runner.main(list(args))
        return output.getvalue()

    def test_team_names_and_presets(self):
        self.assertEqual(runner._select_teams(" Top6, GRANDMASTER-NUMBA "), TOP6_TEAMS)
        self.assertEqual(runner._select_teams("v2, QSearch-PVS, bFs"),
                         ("Team19-v2", "QSearch-PVS", "BFS-Strategist"))
        all_teams = tuple(name for name in runner.TEAMS if name != "Team19-New")
        self.assertEqual(runner.TEAMS["Team19-New"].module, runner.TEAMS["Team19-v4"].module)
        self.assertEqual(runner._select_teams("ALL"), all_teams)
        self.assertEqual(runner._select_teams("ALL,new"), all_teams)
        self.assertEqual(runner._select_teams("new,gm"), ("Team19-New", "Grandmaster-Numba"))
        self.assertEqual(runner._team_name("new", all_teams), "Team19-v4")
        with self.assertRaisesRegex(ValueError, "两个不同队伍"):
            runner._select_teams("new,v4")
        listed = self.cli("--list-teams")
        for full_name, team in runner.TEAMS.items():
            self.assertEqual(runner._team_name(team.alias.upper()), full_name)
            self.assertIn(full_name, listed)
            self.assertIn(team.description, listed)
        self.assertFalse(self.results.exists())

    def test_dry_run_schedules_360_games_per_budget_without_writing(self):
        with patch.object(runner, "run_match") as play:
            output = self.cli("--teams", "top6", "--budgets", "60,10", "--games-per-pair", "24", "--dry-run")
        self.assertLess(output.index("60 秒档 · 360 局"), output.index("10 秒档 · 360 局"))
        self.assertIn("12 个固定开局 × 换边", output)
        self.assertEqual(output.count("可复用 0 局 · 待跑 360 局"), 2)
        self.assertIn("合计 720 局", output)
        self.assertIn("14h00m", output)
        self.assertIn("dashboard.html", output)
        self.assertFalse(self.results.exists())
        play.assert_not_called()

    def test_standard_opening_dry_run_schedules_all_unique_teams_in_both_budgets(self):
        with (patch.object(runner, "load_suite") as suite,
              patch.object(runner, "run_match") as play):
            output = self.cli("--teams", "all", "--standard-opening", "--budgets", "60,10", "--dry-run")
        self.assertIn("14 队全循环 · 标准初始盘面 × 换边（预走 0 手） · 每组 2 局", output)
        self.assertIn("60 秒档 · 182 局", output)
        self.assertIn("10 秒档 · 182 局", output)
        self.assertIn("合计 364 局", output)
        self.assertNotIn("Team19-New", output)
        self.assertFalse(self.results.exists())
        suite.assert_not_called()
        play.assert_not_called()

    def test_standard_opening_logs_and_resume_stay_separate_from_fixed_openings(self):
        recipe = ("--teams", "greedy,random")
        with patch.object(runner, "run_match", side_effect=forfeited_match) as play:
            self.cli(*recipe, "--no-dashboard")
            fixed, = self.results.glob("*.jsonl")
            fixed_rows = report.load(fixed)
            prior = fixed.read_bytes()
            play.reset_mock()
            self.cli(*recipe, "--standard-opening", "--limit", "1")
            self.assertEqual(play.call_count, 1)
            opening = play.call_args.args[0].opening
            self.assertEqual(opening, runner.fixture_for(SimpleNamespace(stage="official"))[0].opening)
            self.assertEqual(opening.moves, [])

            standard, = self.results.glob("arena-standard-*-60s.jsonl")
            first, = report.load(standard)
            self.assertEqual((first["opening"], first["prefix"]), ("standard", []))
            self.assertEqual(first["initial"], openings.position(runner.Tablut.Tablut(), True))
            self.assertNotEqual(first["schedule"], fixed_rows[0]["schedule"])
            self.assertNotEqual(standard.stem.removeprefix("arena-standard-"), fixed.stem.removeprefix("arena-"))
            html = (self.results / "dashboard.html").read_text(encoding="utf-8")
            self.assertIn("<li>标准初始盘面 × 换边（预走 0 手）</li>", html)
            self.assertIn("--resume --standard-opening", html)

            play.reset_mock()
            self.cli(*recipe, "--standard-opening", "--resume")
            self.assertEqual(play.call_count, 1)
            rows = report.load(standard)
            self.assertEqual([row["id"] for row in rows], [
                "60-standard-GreedyBot-A-RandomBot",
                "60-standard-RandomBot-A-GreedyBot",
            ])
            self.assertEqual([row["prefix"] for row in rows], [[], []])
            self.assertEqual(fixed.read_bytes(), prior)

    def test_invalid_configuration_fails_before_running(self):
        invalid = (
            (("--games-per-pair", "5"), "正偶数"),
            (("--games-per-pair", "26"), "每组最多 24 局"),
            (("--standard-opening", "--games-per-pair", "4"), "--standard-opening 要求 --games-per-pair 为 2"),
            (("--budgets", "10,nan"), "有限的正数"),
            (("--budgets", "60,60.0"), "只能指定一次"),
            (("--format", "1vn", "--anchor", "random"), "必须在 --teams 中"),
            (("--anchor", "v2"), "只用于 --format 1vn"),
            (("--stage", "top6_60s"), "--teams 使用参数化赛程"),
        )
        for params, error in invalid:
            with self.subTest(params=params), redirect_stderr(StringIO()) as output:
                with self.assertRaises(SystemExit) as failure:
                    runner.main(["--teams", "v2,gm", *params])
                self.assertEqual(failure.exception.code, 2)
                self.assertIn(error, output.getvalue())
        self.assertFalse(self.results.exists())

    def test_standard_opening_requires_teams(self):
        with redirect_stderr(StringIO()) as output, self.assertRaises(SystemExit) as failure:
            runner.main(["--stage", "official", "--standard-opening"])
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("--standard-opening 需要搭配 --teams", output.getvalue())
        self.assertFalse(self.results.exists())

    def test_wheel_uses_the_first_team_or_explicit_anchor(self):
        args = ("--teams", "random,greedy,ab2", "--format", "1vn", "--budgets", "10")
        with patch.object(runner, "run_match", side_effect=forfeited_match) as play:
            self.cli(*args)
            seats = Counter((call.args[0].attacker, call.args[0].defender) for call in play.call_args_list)
            self.assertEqual(seats, {("RandomBot", team): 1 for team in ("GreedyBot", "NaiveAB-D2")}
                             | {(team, "RandomBot"): 1 for team in ("GreedyBot", "NaiveAB-D2")})
            play.reset_mock()
            self.cli(*args, "--anchor", "AB2")
            self.assertEqual(play.call_count, 4)
            self.assertTrue(all("NaiveAB-D2" in (call.args[0].attacker, call.args[0].defender)
                                for call in play.call_args_list))
        html = (self.results / "dashboard.html").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r'id="tab-[^"]+" role="tab"', html)), 2)
        self.assertIn("1v2 车轮战实验", html)
        self.assertIn("擂主 NaiveAB-D2", html)
        self.assertIn("已完成 4/4", html)
        self.assertIn("3×3 交手得分率", html)
        self.assertNotIn("课程模拟分", html)

    def test_fresh_resets_both_budgets_before_first_game_and_resume_repairs_tail(self):
        recipe = ("--teams", "greedy,random", "--budgets", "60,10", "--games-per-pair", "2")
        with patch.object(runner, "run_match", side_effect=forfeited_match):
            self.cli(*recipe, "--no-dashboard")
        originals = {path.name: path.read_bytes() for path in self.results.glob("*.jsonl")}
        self.assertEqual(len(originals), 2)
        planned = self.cli(*recipe, "--fresh", "--dry-run")
        self.assertEqual(planned.count("开跑前将归档旧赛果"), 2)
        self.assertEqual(len(list(self.results.glob("*.jsonl"))), 2)

        def interrupt_after_first(match):
            if match.attacker == "RandomBot":
                raise KeyboardInterrupt
            return forfeited_match(match)

        with patch.object(runner, "run_match", side_effect=interrupt_after_first):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as interrupted:
                self.cli(*recipe, "--fresh")
        self.assertEqual(interrupted.exception.code, 130)
        backups = {path.name.split(".before-")[0] + ".jsonl": path.read_bytes()
                   for path in self.results.glob("*.before-*.jsonl")}
        self.assertEqual(backups, originals)
        sixty = next(self.results.glob("*-60s.jsonl"))
        with sixty.open("ab") as stream:
            stream.write(b'{"id": "unfinished')
        partial = sixty.read_bytes()
        preview = self.cli(*recipe, "--resume", "--dry-run")
        self.assertIn("已落盘 1 局 · 可复用 1 局 · 待跑 1 局", preview)
        self.assertEqual(sixty.read_bytes(), partial)
        self.assertEqual(next(self.results.glob("*-10s.jsonl")).read_bytes(), b"")

        with patch.object(runner, "run_match", side_effect=forfeited_match) as play:
            output = self.cli(*recipe, "--resume")
        self.assertEqual([call.args[0].budget for call in play.call_args_list], [60, 10, 10])
        self.assertEqual(len(report.load(sixty)), 2)
        self.assertIn("2/2 (100.0%)", output)
        html = (self.results / "dashboard.html").read_text(encoding="utf-8")
        self.assertIn("2 队全循环实验", html)
        self.assertIn("2×2 交手得分率", html)
        self.assertIn("60 秒 2 队全循环", html)
        self.assertIn("10 秒 2 队全循环", html)
        self.assertNotRegex(html, r"https?://|<script\s+src=|@import")

    def test_selective_rerun_preserves_only_the_other_pair(self):
        recipe = ("--teams", "v2,greedy,random", "--budgets", "10", "--no-dashboard")
        with patch.object(runner, "_fingerprint", return_value="old-code"):
            with patch.object(runner, "run_match", side_effect=forfeited_match):
                self.cli(*recipe)
        journal = next(self.results.glob("*.jsonl"))
        before = journal.read_bytes()
        with patch.object(runner, "_fingerprint", return_value="updated-code"):
            with patch.object(runner, "run_match", side_effect=forfeited_match) as play:
                preview = self.cli(*recipe, "--rerun-team", "V2", "--dry-run")
                self.assertIn("可复用 2 局 · 待跑 4 局", preview)
                self.assertEqual(journal.read_bytes(), before)
                self.cli(*recipe, "--rerun-team", "V2")
            self.assertEqual(play.call_count, 4)
            self.assertTrue(all("Team19-v2" in (call.args[0].attacker, call.args[0].defender)
                                for call in play.call_args_list))
        rows = report.load(journal)
        backup = next(self.results.glob("*.before-*.jsonl"))
        self.assertEqual(backup.read_bytes(), before)
        self.assertEqual({row["schedule"] for row in rows}, {"updated-code"})
        reused = [row for row in rows if "reusedFrom" in row]
        self.assertEqual(len(reused), 2)
        for row in reused:
            self.assertEqual(row["reusedFrom"], {"file": backup.name, "schedule": "old-code", "id": row["id"]})
        self.assertFalse((self.results / "dashboard.html").exists())
