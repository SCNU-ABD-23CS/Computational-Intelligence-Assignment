"""Review arena or practice v1 replay JSON on a read-only local board."""

from argparse import ArgumentParser
from functools import partial
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import Tablut


STYLES = Path(Tablut.__file__).resolve().parent / "tools" / "web" / "styles.css"
PIECES = {".": "空格", "A": "攻兵", "D": "护卫", "K": "国王", "X": "王座"}
REASONS = {
    "king captured": "国王被捕", "king escaped to the edge": "国王逃生",
    "fewer than two attackers remain": "攻兵不足两枚",
    "repeated position": "重复局面", "no legal move": "无合法步",
    "timeout": "超时判负", "illegal move": "非法步判负", "exception": "异常判负",
}


def load(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != "team19.tablut.replay" or record.get("version") != 1:
        raise ValueError("Expected a team19.tablut.replay v1 file")
    positions = [record["initial"], *(turn["positionAfter"] for turn in record["turns"])]
    for position in positions:
        board = position["board"]
        if len(board) != 9 or any(len(r) != 9 or any(c not in PIECES for c in r) for r in board):
            raise ValueError("Invalid board in replay")
    return record


def _move_text(move) -> str:
    return " → ".join(",".join(str(number) for number in square) for square in move)


def _board(board, last):
    squares = []
    for r, row in enumerate(board):
        for c, piece in enumerate(row):
            classes = ["square"]
            if (r + c) % 2:
                classes.append("dark")
            if r in (0, 8) or c in (0, 8):
                classes.append("edge")
            if last and [r, c] == last[0]:
                classes.append("last-from")
            if last and [r, c] == last[1]:
                classes.append("last-to")
            token = ""
            if piece in "ADK":
                token = f'<span class="piece piece-{piece.lower()}">{"♚" if piece == "K" else ""}</span>'
            elif piece == "X":
                token = '<span class="castle"></span>'
            squares.append(f'<div class="{" ".join(classes)}" role="cell" '
                           f'aria-label="{r},{c} {PIECES[piece]}">{token}</div>')
    labels = "".join(f"<span>{n}</span>" for n in range(9))
    return (f'<div class="board-wrap"><div class="col-labels">{labels}</div>'
            f'<div class="row-labels">{labels}</div><div class="board" role="grid" '
            f'aria-label="九乘九棋盘">{"".join(squares)}</div></div>')


def render(record: dict, ply: int) -> str:
    turns = record["turns"]
    ply = max(0, min(len(turns), ply))
    turn = turns[ply - 1] if ply else None
    board = turn["positionAfter"]["board"] if turn else record["initial"]["board"]
    teams = record.get("arena", {}).get("teams", {"A": "攻方", "D": "守方"})
    a, d = escape(teams["A"]), escape(teams["D"])
    final = record["final"]
    outcome = REASONS.get(final["reason"], final["reason"] or "未结束")
    winner = (escape(teams[final["winner"]]) + "获胜" if final["winner"] in teams else "和棋")
    heading = f"{winner} · {escape(outcome)}" if final["finished"] else escape(outcome)
    last_move = turn["move"] if turn else None
    summary = (escape(f'{teams[turn["side"]]} · {_move_text(last_move)}') if turn else "初始棋盘")
    details = []
    if turn:
        captures = "、".join(f"{PIECES[piece]} {r},{c}" for r, c, piece in turn["captures"])
        if captures:
            details.append(f"吃子 {captures}")
        if analysis := turn.get("analysis"):
            if analysis.get("depth") is not None:
                details.append(f'完成深度 D{analysis["depth"]}')
            if analysis.get("simulations") is not None:
                details.append(f'模拟 {analysis["simulations"]} 次')
            if analysis.get("elapsed") is not None:
                details.append(f'用时 {analysis["elapsed"]:.3f} 秒')
        clocks = turn.get("clocksAfter")
        if clocks and "bySide" in clocks:
            remaining = clocks["bySide"]
            details.append(f'剩余时间 · 攻 {remaining["A"]:.2f} 秒 · 守 {remaining["D"]:.2f} 秒')
    detail = "".join(f"<p class=muted>{escape(text)}</p>" for text in details)
    timeline = "".join(
        f'<li><a href="?ply={idx}" {"aria-current=step" if idx == ply else ""}>'
        f'{idx:02d} · {escape(teams[item["side"]])} · {escape(_move_text(item["move"]))}</a></li>'
        for idx, item in enumerate(turns, 1)
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tablut 联赛复盘</title><link rel="stylesheet" href="/styles.css">
<style>
nav a{{text-decoration:none}} .replay-nav{{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}}
.replay-list{{max-height:550px;overflow:auto;padding-left:1.6em;margin-top:12px}}
.replay-list li{{margin:4px 0}} .replay-list a{{color:var(--ink)}}
.replay-list a[aria-current]{{font-weight:700;color:var(--accent)}}
.replay-side{{padding:20px}} .replay-side p{{margin-top:6px}}
</style></head><body>
<header class="topbar"><div class="shell topbar-inner">
<strong>Tablut 联赛复盘</strong><nav><a class="button button-outline" href="/replay.json" download>下载棋谱</a></nav>
</div></header>
<main class="shell layout">
<section class="surface board-card">
<div class="board-header"><div><h1>{a} 对 {d}</h1>
<p class="muted">第 {ply} / {len(turns)} 手 · {summary}</p></div></div>
{_board(board, last_move)}
<nav class="replay-nav" aria-label="棋谱进度">
<a class="button button-outline" href="?ply=0">开局</a>
<a class="button button-outline" href="?ply={max(0, ply - 1)}">上一手</a>
<a class="button button-outline" href="?ply={min(len(turns), ply + 1)}">下一手</a>
<a class="button button-primary" href="?ply={len(turns)}">终局</a>
</nav>{detail}</section>
<aside class="surface replay-side"><h2>{heading}</h2>
<p class="muted">进攻方 {a} · 防守方 {d} · {record["budgetSeconds"]:g} 秒</p>
<ol class="replay-list">{timeline}</ol></aside>
</main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, record, **kwargs):
        self.record = record
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        request = urlsplit(self.path)
        if request.path == "/":
            try:
                ply = int(parse_qs(request.query).get("ply", ["0"])[0])
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            media, data = "text/html", render(self.record, ply).encode("utf-8")
        elif request.path == "/styles.css":
            media, data = "text/css", STYLES.read_bytes()
        elif request.path == "/replay.json":
            media, data = "application/json", json.dumps(self.record, ensure_ascii=False).encode("utf-8")
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{media}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    record = load(args.path)
    with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, record=record)) as server:
        print(f"http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
