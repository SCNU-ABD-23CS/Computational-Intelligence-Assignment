"""Serve the Team19 practice board from the local checkout."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
from unittest.mock import patch
import webbrowser

import play_human as practice
from game_log import MatchLog, ai_points, clocks, encode, summarize


WEB = Path(__file__).resolve().with_name("web")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def make_game(side: str, budget: float) -> practice.Game:
    if side not in ("A", "D"):
        raise ValueError("请选择进攻方或防守方。")
    budget = practice.seconds(budget)
    if budget <= 0:
        raise ValueError("请填写大于零的时间预算。")
    with patch.object(practice.rules, "TIME_BUDGET", budget):
        return practice.Game(side == "A")


class PracticeSession:
    def __init__(self):
        self.lock = Lock()
        self.game = make_game("D", practice.rules.TIME_BUDGET)
        self.reports: list[dict] = []
        self.record = MatchLog(self.game)
        self.revision = 0
        self.top3 = True

    def state(self) -> dict:
        game = self.game
        human_turn = game.attackers_turn == game.human_attacker and game.outcome is None
        legal = {}
        if human_turn:
            pieces = ("A",) if game.human_attacker else ("D", "K")
            for row, squares in enumerate(game.referee.board):
                for col, piece in enumerate(squares):
                    if piece in pieces:
                        destinations = practice.legal_destinations(game, (row, col))
                        if destinations:
                            legal[f"{row},{col}"] = destinations

        return {
            "revision": self.revision,
            "board": game.referee.board,
            "human": practice.side_code(game.human_attacker),
            "turn": practice.side_code(game.attackers_turn),
            "legal": legal,
            "moves": game.moves,
            "lastMove": asdict(game.last_move) if game.last_move else None,
            "budget": game.budget,
            "remaining": game.remaining,
            "playerRemaining": game.player.remaining,
            "canUndo": bool(game.checkpoints),
            "top3": self.top3,
            "report": self.reports[-1] if self.reports else None,
            "outcome": game.outcome,
            "winner": game.referee.winner,
            "points": ai_points(game),
            "repetition": self.record.repetition(game),
            "replayFile": self.record.name if (game.moves or game.outcome) else None,
            "summary": summarize(game),
        }

    def think(self) -> None:
        game = self.game
        if game.outcome or game.attackers_turn == game.human_attacker:
            raise ValueError("当前轮到你落子。")
        before = game.referee.board
        history = set(game.referee.past_positions)
        attacker = game.attackers_turn
        with patch.object(practice.rules, "TIME_BUDGET", game.budget):
            trace, stats = game.play_ai()

        event = game.last_move if not game.forfeit else None
        features, candidates, diagnostic_error = [], None, None
        if event:
            self.record.add_move(game)
            try:
                features = practice.feature_breakdown(before, game.referee.board, game.player.weights)
                if self.top3:
                    candidates = practice.analyze_candidates(
                        before, attacker, event.move, trace, history, game.player.weights,
                    )
            except Exception:
                logging.exception("Post-move diagnostic failed")
                diagnostic_error = "本步已结算，局面诊断未完成。"
        self.reports.append({
            "ply": len(game.moves),
            "boardBefore": before,
            "side": practice.side_code(attacker),
            "move": asdict(event) if event else None,
            "stats": asdict(stats),
            "allocation": trace.allocation,
            "depths": [asdict(record) for record in trace.depths],
            "interruptedDepth": trace.interrupted_depth,
            "interruptedNodes": trace.interrupted_nodes,
            "stop": trace.stop_reason(),
            "fallback": stats.fallback,
            "error": trace.error,
            "diagnosticError": diagnostic_error,
            "forfeit": game.forfeit,
            "features": features,
            "candidates": candidates,
            "clocksAfter": clocks(game),
            "weights": dict(zip(
                practice.evaluation.FEATURE_NAMES,
                game.player.weights if game.player.weights is not None else practice.evaluation.DEFAULT_WEIGHTS,
            )),
        })

    def load(self, name: str) -> None:
        if not practice.SNAPSHOT_NAME.fullmatch(name):
            raise ValueError("请选择有效的存档名。")
        path = practice.SAVED_POSITIONS / f"{name}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != practice.SNAPSHOT_VERSION:
            raise ValueError("不支持的存档版本。")
        loaded = make_game(data["human"], data["budget"])
        record = MatchLog(loaded, resumed_from=name)
        practice.restore_snapshot(loaded, data, on_move=lambda game: record.add_move(game, live=False))
        reports = data.get("web_reports", [])
        prior_ply = 0
        for report in reports:
            ply = report["ply"]
            if type(ply) is not int or not prior_ply < ply <= len(record.turns):
                raise ValueError("存档中的分析手数无效。")
            turn = record.turns[ply - 1]
            if turn["actor"] != "ai" or practice.move_pair(report["move"]["move"]) != turn["move"]:
                raise ValueError("存档中的分析与棋谱不一致。")
            turn["clocksAfter"] = report.get("clocksAfter")
            prior_ply = ply
        self.game = loaded
        self.record = record
        self.reports = reports

    def act(self, data: dict) -> dict:
        action = data.get("action")
        game = self.game
        notice = None
        if action == "new":
            self.game = make_game(data.get("side"), data.get("budget"))
            self.record = MatchLog(self.game)
            self.reports.clear()
        elif action == "move":
            if game.outcome or game.attackers_turn != game.human_attacker:
                raise ValueError("现在不能落子。")
            move = practice.move_pair(data.get("move"))
            if move is None or not all(map(practice.in_board, move)):
                raise ValueError("请选择棋盘内的起点和落点。")
            if not game.play_human(move):
                raise ValueError("这一步不合法。请选择标出的落点。")
            self.record.add_move(game)
        elif action == "think":
            self.think()
        elif action == "undo":
            if not game.undo():
                raise ValueError("当前没有可以撤回的走法。")
            self.record.undo(len(game.moves))
            self.reports = [report for report in self.reports if report["ply"] <= len(game.moves)]
            notice = "已恢复棋盘、历史和用时。"
        elif action == "save":
            if game.outcome:
                raise ValueError("请在对局结束前保存局面。")
            path = practice.save_snapshot(game, data.get("name", ""), web_reports=self.reports)
            notice = f"已保存 {path.stem}"
        elif action == "load":
            self.load(data.get("name", ""))
            notice = "已恢复存档。可以从续局后的走法开始悔棋。"
        elif action == "top3":
            if type(data.get("enabled")) is not bool:
                raise ValueError("候选分析设置无效。")
            self.top3 = data["enabled"]
            notice = "候选分析将在下一次 AI 落子后更新。"
        else:
            raise ValueError("未知操作。")
        if action in ("move", "think", "undo", "load"):
            try:
                self.record.save(self.game, self.reports)
            except (OSError, ValueError):
                logging.exception("Writing the local replay failed")
                notice = "复盘未能自动保存。可用「导出复盘」下载本局。"
        self.revision += 1
        return {"state": self.state(), "notice": notice}


class Handler(BaseHTTPRequestHandler):
    def send_body(self, status, mime, data, *, download=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if download is not None:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, data):
        self.send_body(status, "application/json; charset=utf-8",
                       json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"))

    def local_request(self):
        expected = f"127.0.0.1:{self.server.server_port}"
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        return host == expected and (origin is None or origin == f"http://{expected}")

    def do_GET(self):
        if not self.local_request():
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "请使用启动时显示的本地地址。"})
            return
        path = urlsplit(self.path).path
        if path in ASSETS:
            name, mime = ASSETS[path]
            self.send_body(HTTPStatus.OK, mime, (WEB / name).read_bytes())
        elif path == "/api/state":
            with self.server.session.lock:
                self.send_json(HTTPStatus.OK, {"state": self.server.session.state()})
        elif path == "/api/replay":
            session = self.server.session
            with session.lock:
                name = session.record.name
                data = encode(session.record.data(session.game, session.reports))
            self.send_body(HTTPStatus.OK, "application/json; charset=utf-8", data, download=name)
        elif path == "/api/saves":
            files = sorted(practice.SAVED_POSITIONS.glob("*.json"), key=lambda p: p.name, reverse=True)
            self.send_json(HTTPStatus.OK, {"files": [file.stem for file in files
                                                   if practice.SNAPSHOT_NAME.fullmatch(file.stem)]})
        else:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "页面不存在。"})

    def do_POST(self):
        if not self.local_request():
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "请使用启动时显示的本地地址。"})
            return
        if self.path != "/api/action":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "接口不存在。"})
            return
        session = self.server.session
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.headers.get_content_type() != "application/json" or not 0 < length <= 4096:
                raise ValueError("请求格式有误。")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("请求格式有误。")
            with session.lock:
                if data.get("revision") != session.revision:
                    self.send_json(HTTPStatus.CONFLICT, {
                        "error": "对局已更新。请使用当前棋盘。",
                        "state": session.state(),
                    })
                    return
                self.send_json(HTTPStatus.OK, session.act(data))
        except FileExistsError:
            self.send_json(HTTPStatus.CONFLICT, {"error": "存档名已存在。请换一个名字。"})
        except (ValueError, KeyError, TypeError, AttributeError, OSError, OverflowError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def log_message(self, format, *args):
        logging.debug(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8019, help="loopback HTTP port (default 8019)")
    parser.add_argument("--open", action="store_true", help="open the local board in your browser")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
            server.session = PracticeSession()
            url = f"http://127.0.0.1:{server.server_port}/"
            print(f"Tablut practice room  {url}", flush=True)
            if args.open:
                webbrowser.open(url)
            server.serve_forever()
    except KeyboardInterrupt:
        print()
    except OSError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
