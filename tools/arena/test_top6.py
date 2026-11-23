"""Top-six schedule, historical import and terminal progress without live play."""

from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import HERE, TOP6_BUDGETS, TOP6_GAMES, TOP6_TEAMS, dashboard, overview, report, runner
from .top6 import LIVE_PAIR, SNAPSHOT, V2, progress_text


def fixture(stage):
    return runner.fixture_for(SimpleNamespace(stage=stage))


def fake_game(match, *, attacker=None, defender=None, stage=None):
    attack, defence = attacker or match.attacker, defender or match.defender
    return dict(id=f"{match.budget:g}-{match.opening.name}-{attack}-A-{defence}",
                opening=match.opening.name, teams={"A": attack, "D": defence},
                stage=stage, budget=match.budget, schedule=f"source-{stage}",
                schema=runner.SCHEMA, elapsed=5.0, winner="A", reason="king captured",
                attempts=[{"side": "A", "team": attack, "fallback": False}])


def write_sources(root, matches, *, omit_one=False):
    sources = {"fast": [], "evolution-fast": []}
    for match in matches:
        teams = {match.attacker, match.defender}
        if teams == LIVE_PAIR:
            continue
        if V2 in teams:
            source, attack, defence = "evolution-fast", match.attacker, match.defender
        else:
            source = "fast"
            attack, defence = (("Team19-New" if name == SNAPSHOT else name)
                               for name in (match.attacker, match.defender))
        sources[source].append(fake_game(match, attacker=attack, defender=defence, stage=source))
    if omit_one:
        sources["fast"].pop()
    for source, rows in sources.items():
        (root / f"{source}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )


class TopSixTests(unittest.TestCase):
    def test_every_pair_plays_first_six_openings_in_both_directions(self):
        for stage, budget in TOP6_BUDGETS.items():
            matches = fixture(stage)
            self.assertEqual(len(matches), TOP6_GAMES)
            self.assertEqual(len({match.id for match in matches}), TOP6_GAMES)
            self.assertEqual({match.budget for match in matches}, {budget})
            self.assertEqual({match.opening.name for match in matches}, {f"o{i:02d}" for i in range(1, 7)})
            self.assertEqual(Counter(match.attacker for match in matches), dict.fromkeys(TOP6_TEAMS, 30))
            self.assertEqual(Counter(match.defender for match in matches), dict.fromkeys(TOP6_TEAMS, 30))
            self.assertEqual(set(Counter(frozenset((m.attacker, m.defender)) for m in matches).values()), {12})

    def test_reuses_168_and_resumes_with_only_the_remaining_live_fixtures(self):
        matches = fixture("top6_10s")
        live = [match for match in matches if {match.attacker, match.defender} == LIVE_PAIR]
        with TemporaryDirectory(dir=HERE) as workspace:
            root = Path(workspace)
            write_sources(root, matches)
            fast = root / "fast.jsonl"
            original = fast.read_bytes()
            with (
                patch.object(runner, "RESULTS", root),
                patch.object(runner, "_fingerprint", return_value="top6-schedule"),
                patch.object(overview, "audit_game") as audit,
                patch.object(runner.os, "fsync") as sync,
                redirect_stdout(StringIO()) as console,
            ):
                with patch.object(runner, "run_match", side_effect=[fake_game(live[0]), KeyboardInterrupt]) as play:
                    with self.assertRaises(KeyboardInterrupt):
                        runner.run_schedule(matches, "top6_10s", "top6_10s", resume=True)
                    self.assertEqual([call.args[0] for call in play.call_args_list], live[:2])

                path = root / "top6_10s.jsonl"
                rows = report.load(path)
                self.assertEqual((len(rows), audit.call_count, sync.call_count), (169, 168, 169))
                imports = Counter(row["reusedFrom"]["file"] for row in rows[:168])
                self.assertEqual(imports, {"fast.jsonl": 120, "evolution-fast.jsonl": 48})
                self.assertTrue(any("Team19-New" in row["reusedFrom"]["id"] for row in rows[:168]))
                self.assertTrue(all("Team19-New" not in row["teams"].values() for row in rows))
                self.assertTrue(all(turn["team"] == row["teams"][turn["side"]]
                                    for row in rows for turn in row["attempts"]))
                self.assertIn("168/180 (93.3%)", console.getvalue())
                with path.open("ab") as journal:
                    journal.write(b'{"id":"interrupted')

                with (
                    patch("tools.arena.top6.load", side_effect=AssertionError("reuse is already recorded")),
                    patch.object(runner, "run_match", side_effect=fake_game) as play,
                ):
                    runner.run_schedule(matches, "top6_10s", "top6_10s", resume=True)
                    self.assertEqual([call.args[0] for call in play.call_args_list], live[1:])
                self.assertEqual({row["id"] for row in report.load(path)}, {m.id for m in matches})
                self.assertEqual(sync.call_count, TOP6_GAMES)
                self.assertEqual(fast.read_bytes(), original)
                self.assertIn("180/180 (100.0%)", console.getvalue())

    def test_missing_reusable_record_never_turns_into_an_extra_live_game(self):
        matches = fixture("top6_10s")
        with TemporaryDirectory(dir=HERE) as workspace:
            root = Path(workspace)
            write_sources(root, matches, omit_one=True)
            with (
                patch.object(runner, "RESULTS", root),
                patch.object(runner, "_fingerprint", return_value="top6-schedule"),
                patch.object(overview, "audit_game"),
                patch.object(runner, "run_match") as play,
            ):
                with self.assertRaisesRegex(ValueError, "缺少 1 局"):
                    runner.run_schedule(matches, "top6_10s", "top6_10s", resume=True)
                play.assert_not_called()
                self.assertEqual((root / "top6_10s.jsonl").read_bytes(), b"")

    def test_resume_during_import_only_appends_missing_history(self):
        matches = fixture("top6_10s")
        append = runner._append_result
        written = 0

        def stop_after_three(*args):
            nonlocal written
            if written == 3:
                raise KeyboardInterrupt
            append(*args)
            written += 1

        with TemporaryDirectory(dir=HERE) as workspace:
            root = Path(workspace)
            write_sources(root, matches)
            with (
                patch.object(runner, "RESULTS", root),
                patch.object(runner, "_fingerprint", return_value="top6-schedule"),
                patch.object(overview, "audit_game"),
                patch.object(runner, "run_match") as play,
                redirect_stdout(StringIO()),
            ):
                with patch.object(runner, "_append_result", side_effect=stop_after_three):
                    with self.assertRaises(KeyboardInterrupt):
                        runner.run_schedule(matches, "top6_10s", "top6_10s", resume=True, limit=0)
                path = root / "top6_10s.jsonl"
                before = path.read_bytes()
                runner.run_schedule(matches, "top6_10s", "top6_10s", resume=True, limit=0)
                play.assert_not_called()
                self.assertEqual(len(report.load(path)), 168)
                self.assertTrue(path.read_bytes().startswith(before))

    def test_sixty_second_schedule_starts_without_reuse(self):
        matches = fixture("top6_60s")
        with (
            TemporaryDirectory(dir=HERE) as workspace,
            patch.object(runner, "RESULTS", Path(workspace)),
            patch.object(runner, "_fingerprint", return_value="top6-schedule"),
            patch.object(runner, "reusable_games") as reuse,
            patch.object(runner, "run_match", side_effect=fake_game) as play,
            redirect_stdout(StringIO()),
        ):
            path = runner.run_schedule(matches, "top6_60s", "top6_60s", resume=True, limit=1)
            reuse.assert_not_called()
            play.assert_called_once_with(matches[0])
            self.assertNotIn("reusedFrom", report.load(path)[0])

    def test_named_stage_refreshes_the_canonical_dashboard(self):
        args = SimpleNamespace(stage="top6_60s", run_id=None, resume=True, limit=None)
        with (
            patch.object(runner, "run_schedule") as schedule,
            patch.object(dashboard, "write_dashboard", return_value=(Path("dashboard.html"), {})) as render,
            redirect_stdout(StringIO()),
        ):
            runner.run(args)
        self.assertEqual(schedule.call_args.args[1:], ("top6_60s", "top6_60s"))
        render.assert_called_once_with(runner.RESULTS)

    def test_progress_reports_current_run_time_eta_outcome_and_next_fixture(self):
        recent, upcoming = fixture("top6_60s")[:2]
        row = fake_game(recent)
        text = progress_text(42, 180, 48 * 60 + 12, [69], upcoming, row)
        self.assertIn("[████░░░░░░░░░░░░░░░░] 42/180 (23.3%)", text)
        self.assertIn("已用 48m12s | 预计剩余 2h38m", text)
        self.assertIn(f"{recent.attacker} 胜（国王被捕）", text)
        self.assertIn("0 兜底 / 0 异常", text)
        self.assertIn(f"下一局 {upcoming.opening.name} · 攻 {upcoming.attacker}", text)
        self.assertIn("预计剩余 待首局", progress_text(168, 180, 1, [], upcoming))
