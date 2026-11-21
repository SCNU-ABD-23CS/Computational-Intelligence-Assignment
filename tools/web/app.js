"use strict";

const byId = id => document.getElementById(id);
const pieceLabels = { A: "攻方棋子", D: "守方棋子", K: "国王", X: "空王座", ".": "空格" };
const sideLabels = { A: "进攻方", D: "防守方" };
const featureLabels = {
  material_balance: "子力差",
  low_attacker_penalty: "攻方子力不足",
  king_escape_routes: "国王逃生路线",
  king_edge_proximity: "国王距边缘",
  king_capture_pressure: "国王被围压力",
  king_guard_cover: "守方护卫",
  perimeter_blockade: "边缘封锁",
};
const outcomeLabels = {
  "king captured": "国王被俘",
  "king escaped to the edge": "国王抵达边缘",
  "fewer than two attackers remain": "攻方棋子少于两枚",
  "no legal move": "当前方无合法走法",
  "repeated position": "局面重复",
  "referee declared a draw": "裁判判和",
  "Team19 raised an exception": "Team19 运行异常判负",
  "Team19 exceeded the game clock": "Team19 整局超时判负",
  "Team19 returned an illegal move": "Team19 非法走步判负",
};

let game = null;
let busy = null;
let selected = null;
let preview = null;
let activeTab = "search";
let toastTimer;
const cells = [];
const board = byId("board");

function element(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text != null) item.textContent = text;
  return item;
}

function makePiece(piece) {
  if (piece === "X") return element("span", "castle");
  if (piece === ".") return null;
  return element("span", `piece piece-${piece.toLowerCase()}`, piece === "K" ? "♚" : null);
}

function moveLabel(move) {
  return move ? `${move[0].join(",")} → ${move[1].join(",")}` : "—";
}

const coord = square => square?.join(",");
const count = value => Number(value).toLocaleString("zh-CN");
const signed = (value, digits = 0) => `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;
const duration = value => value == null ? "—" : `${value.toFixed(3)} 秒`;
const humanTurn = () => game && game.turn === game.human && !game.outcome;
const afterPly = ply => ply === 0 ? "开局" : `第 ${ply} 手后`;

function populateBoard() {
  for (let index = 0; index < 9; index++) {
    byId("row-labels").append(element("span", null, index));
    byId("col-labels").append(element("span", null, index));
  }
  for (let row = 0; row < 9; row++) {
    for (let col = 0; col < 9; col++) {
      const key = `${row},${col}`;
      const cell = element("button", "square");
      const classes = ["square"];
      if ((row + col) % 2) classes.push("dark");
      if (row === 0 || row === 8 || col === 0 || col === 8) classes.push("edge");
      cell.type = "button";
      cell.dataset.square = key;
      cell.tabIndex = row === 0 && col === 0 ? 0 : -1;
      cell.addEventListener("click", () => {
        board.querySelector('[tabindex="0"]').tabIndex = -1;
        cell.tabIndex = 0;
        playAt(key);
      });
      cells.push({ cell, row, col, key, classes });
      board.append(cell);
    }
  }
  board.addEventListener("keydown", event => {
    const directions = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    const step = directions[event.key];
    if (!step || !event.target.dataset.square) return;
    event.preventDefault();
    const [row, col] = event.target.dataset.square.split(",").map(Number);
    const nextRow = Math.min(8, Math.max(0, row + step[0]));
    const nextCol = Math.min(8, Math.max(0, col + step[1]));
    const next = cells[nextRow * 9 + nextCol].cell;
    event.target.tabIndex = -1;
    next.tabIndex = 0;
    next.focus();
  });
  for (const item of cells) item.cell.className = item.classes.join(" ");
}

function playAt(key) {
  if (!humanTurn() || busy || preview) return;
  const destinations = game.legal[selected] || [];
  if (destinations.some(target => coord(target) === key)) {
    const move = [selected, key].map(square => square.split(",").map(Number));
    selected = null;
    act("move", { move }).then(ok => { if (ok) maybeThink(); });
  } else {
    selected = selected !== key && game.legal[key] ? key : null;
    renderBoard();
    renderStatus();
  }
}

function renderBoard() {
  const grid = preview ? game.report.boardBefore : game.board;
  const highlight = preview || game.lastMove?.move;
  const playable = humanTurn() && !busy && !preview;
  const targets = playable ? (game.legal[selected] || []).map(coord) : [];
  for (const { cell, row, col, key, classes } of cells) {
    const piece = grid[row][col];
    const marks = classes.slice();
    if (key === coord(highlight?.[0])) marks.push(preview ? "preview-from" : "last-from");
    if (key === coord(highlight?.[1])) marks.push(preview ? "preview-to" : "last-to");
    const legal = targets.includes(key);
    const own = playable && Boolean(game.legal[key]);
    const active = key === selected && playable;
    if (legal) marks.push("legal");
    if (own) marks.push("can-select");
    if (active) marks.push("selected");
    cell.className = marks.join(" ");
    cell.setAttribute("aria-label", `${row} 行 ${col} 列，${pieceLabels[piece]}${legal ? "，可落子" : ""}`);
    cell.setAttribute("aria-pressed", active);
    cell.setAttribute("aria-disabled", !(own || legal));
    if (cell.dataset.piece !== piece) {
      cell.dataset.piece = piece;
      const token = makePiece(piece);
      cell.replaceChildren(...(token ? [token] : []));
    }
  }
}

function renderStatus() {
  const status = byId("play-status");
  const rounds = game.outcome ? Math.max(1, Math.ceil(game.moves.length / 2)) : Math.floor(game.moves.length / 2) + 1;
  byId("game-meta").textContent = `第 ${rounds} 回合  ·  已走 ${game.moves.length} 手`;
  let message = "请点选你的棋子";
  let mode = "human";
  if (preview) {
    message = `候选预览  ·  ${moveLabel(preview)}`;
    mode = "preview";
  } else if (game.outcome) {
    const reason = outcomeLabels[game.outcome] || game.outcome;
    const result = game.winner ? `${sideLabels[game.winner]}获胜` : "和棋";
    message = game.repetition
      ? `和棋 · 回到${afterPly(game.repetition.firstPly)}的局面`
      : `${result} · ${reason}`;
    mode = "";
  } else if (!humanTurn()) {
    message = busy === "think" ? "Team19 正在思考…" : "轮到 Team19";
    mode = busy === "think" ? "thinking" : "";
  } else if (selected) {
    message = `已选 ${selected}，请点选绿色落点`;
  }
  status.className = `play-status ${mode}`;
  byId("play-message").textContent = message;
  byId("live-button").hidden = !preview;
}

function renderParticipants() {
  const opponent = game.human === "A" ? "D" : "A";
  for (const [id, side] of [["human", game.human], ["ai", opponent]]) {
    const current = game.turn === side && !game.outcome;
    byId(`${id}-card`).classList.toggle("active", current);
    byId(`${id}-card`).querySelector(".turn-indicator").hidden = !current;
    byId(`${id}-side`).textContent = sideLabels[side];
    byId(`${id}-piece`).replaceChildren(makePiece(side));
  }
}

function renderClock() {
  const ratio = Math.max(0, Math.min(1, game.remaining / game.budget)) * 100;
  byId("clock-left").textContent = Math.max(0, game.remaining).toFixed(2);
  byId("clock-total").textContent = `/ ${game.budget} 秒`;
  byId("clock-fill").style.width = `${ratio}%`;
  byId("clock-fill").parentElement.setAttribute("aria-valuenow", Math.round(ratio));
  byId("clock-mode").textContent = busy === "think" ? "思考中" : `内部 ${game.playerRemaining.toFixed(2)} 秒`;
  byId("allocation").textContent = duration(game.report?.allocation);
  byId("elapsed").textContent = duration(game.report?.stats.elapsed);
}

function empty(parent, heading, message) {
  const box = element("div", "empty");
  box.append(element("strong", null, heading));
  if (message) box.append(element("p", null, message));
  parent.append(box);
}

function renderSearch(parent, report) {
  const { stats, depths } = report;
  const metrics = element("div", "metrics");
  const rate = stats.elapsed > 0 ? Math.round(stats.nodes / stats.elapsed) : 0;
  for (const [label, value] of [
    ["完成深度", `D${stats.depth}`], ["完整层节点", count(stats.nodes)], ["每秒节点", count(rate)],
  ]) {
    const metric = element("div", "metric");
    metric.append(element("span", null, label), element("strong", null, value));
    metrics.append(metric);
  }
  parent.append(metrics);
  if (depths.length) {
    const wrap = element("div", "table-wrap");
    const table = element("table", "layers");
    const head = table.createTHead().insertRow();
    for (const title of ["深度", "最佳走法", "评分", "节点", "剪枝", "毫秒"]) {
      const th = element("th", null, title);
      th.scope = "col";
      head.append(th);
    }
    const body = table.createTBody();
    for (const row of depths) {
      const result = row.result;
      const tr = body.insertRow();
      const values = [`D${row.depth}`, moveLabel(result.move), signed(result.score),
        count(result.nodes_visited), count(result.cutoffs), (row.elapsed * 1000).toFixed(1)];
      for (const value of values) tr.insertCell().textContent = value;
    }
    wrap.append(table);
    parent.append(wrap);
  }
  let stop = report.stop;
  if (report.interruptedDepth != null) {
    stop = `D${report.interruptedDepth} 触及单步搜索截止时间，采用已完成的结果。`;
  } else if (depths.length && depths.at(-1).result.terminal_plies != null
      && depths.at(-1).result.score === (report.side === "A" ? 100000 : -100000)) {
    stop = `已证实能在 ${depths.at(-1).result.terminal_plies} 手内获胜，提前结束搜索。`;
  } else if (depths.length && !depths.at(-1).evaluated_horizon) {
    stop = "已在搜索范围内确定结果。";
  } else if (!depths.length) {
    stop = report.error ? "搜索出现异常。" : "本步没有完成搜索层。";
  } else if (report.stop.startsWith("Depth cap")) {
    stop = "已达到迭代加深的深度上限。";
  }
  if (report.fallback) stop += " 已返回备用走法。";
  if (report.forfeit) stop = outcomeLabels[game.outcome] || game.outcome;
  parent.append(element("p", `stop-note${report.error || report.forfeit ? " alert" : ""}`, stop));
  if (report.error) parent.append(element("p", "stop-note alert", report.error));
  if (report.diagnosticError) parent.append(element("p", "stop-note alert", report.diagnosticError));
  if (game.repetition?.repeatedPly === report.ply) {
    parent.append(element("p", "stop-note", "实战走法触发了局面重复，搜索给和棋记 0 分。"));
  }
}

function renderCandidates(parent, report) {
  const comparison = report?.candidates;
  const heading = element("div", "compare-heading");
  let scope = "最近一步的候选走法";
  if (comparison) {
    scope = `D${comparison.depth} · 已计算 ${comparison.scored}/${comparison.total} 候选`;
  }
  const toggle = element("input");
  toggle.type = "checkbox";
  toggle.checked = game.top3;
  toggle.disabled = Boolean(busy);
  toggle.addEventListener("change", () => act("top3", { enabled: toggle.checked }));
  const label = element("label", null, "候选分析");
  label.prepend(toggle);
  heading.append(element("p", "muted", scope), label);
  parent.append(heading);
  if (!comparison?.entries.length) {
    const hint = game.top3 ? "Team19 下次落子后更新" : "可开启候选分析";
    empty(parent, "暂无候选步", report?.diagnosticError || hint);
    return;
  }
  const list = element("ol", "candidates");
  comparison.entries.forEach((entry, index) => {
    const active = preview && moveLabel(preview) === moveLabel(entry.move);
    const li = element("li");
    const button = element("button", "candidate");
    button.type = "button";
    button.setAttribute("aria-pressed", Boolean(active));
    button.disabled = Boolean(busy);
    const label = element("span", "mono", moveLabel(entry.move));
    if (entry.played) label.append(element("span", "flag", "实战"));
    const score = element("span", "candidate-score mono", signed(entry.score, 1));
    score.append(element("small", null, `差 ${entry.gap.toFixed(1)}`));
    button.append(element("span", "rank", index + 1), label, score);
    button.addEventListener("click", () => {
      preview = active ? null : entry.move;
      render();
      byId("analysis-content").querySelectorAll(".candidate")[index].focus();
    });
    li.append(button);
    list.append(li);
  });
  parent.append(list);
  if (comparison.deep && comparison.depth === 1) {
    const deep = comparison.deep;
    parent.append(element("p", "stop-note", `D${deep.depth} 复算完成 ${deep.scored}/${deep.total}，上面显示 D1 结果。`));
  } else {
    parent.append(element("p", "stop-note", "点走法在左侧棋盘预览。"));
  }
}

function renderEvaluation(parent, report) {
  const features = report.features;
  if (!features.length) {
    empty(parent, "暂无局面估值");
    return;
  }
  const total = field => features.reduce((sum, entry) => sum + entry[field], 0);
  const heading = element("div", "feature-total");
  heading.append(element("span", null, "静态总分"), element("strong", "mono", `${signed(total("before"))} → ${signed(total("after"))}`));
  parent.append(heading, element("p", "muted", "正分利攻，负分利守。每行显示走后分数与变化量。"));
  const list = element("div", "features");
  const max = Math.max(1, ...features.map(entry => Math.abs(entry.delta)));
  for (const feature of features) {
    const row = element("div", "feature");
    const track = element("div", "delta-track");
    track.setAttribute("aria-hidden", "true");
    const bar = element("span", feature.delta >= 0 ? "positive" : "negative");
    bar.style.width = `${Math.abs(feature.delta) / max * 50}%`;
    track.append(bar);
    const color = feature.delta > 0 ? "positive-text" : feature.delta < 0 ? "negative-text" : "";
    row.append(element("span", null, featureLabels[feature.name] || feature.name), track,
      element("span", "mono", signed(feature.after)), element("span", `mono ${color}`, signed(feature.delta)));
    list.append(row);
  }
  parent.append(list);
}

function renderLedger(parent) {
  if (!game.moves.length) {
    empty(parent, "棋谱为空", "落子后会按顺序记录");
    return;
  }
  const list = element("ol", "ledger");
  game.moves.forEach((move, index) => {
    const ply = index + 1;
    const side = index % 2 ? "D" : "A";
    const item = element("li");
    const position = element("span", "mono", moveLabel(move));
    if (ply === game.repetition?.firstPly) position.append(element("span", "flag", "首次出现"));
    if (ply === game.repetition?.repeatedPly) position.append(element("span", "flag", "局面重复"));
    item.append(element("span", "muted mono", ply), position,
      element("span", "muted", `${side === game.human ? "你" : "Team19"} · ${sideLabels[side]}`));
    list.append(item);
  });
  parent.append(list);
  list.scrollTop = list.scrollHeight;
}

function renderAnalysis() {
  const panel = byId("analysis-content");
  const report = game.report;
  const tab = `tab-${activeTab}`;
  panel.replaceChildren();
  panel.setAttribute("aria-labelledby", tab);
  for (const button of document.querySelectorAll(".tab")) button.setAttribute("aria-selected", button.id === tab);
  byId("report-label").textContent = report?.move ? `第 ${report.ply} 手 · ${moveLabel(report.move.move)}` : "";
  if (activeTab === "history") renderLedger(panel);
  else if (activeTab === "candidates") renderCandidates(panel, report);
  else if (!report) empty(panel, "等待 Team19 落子", "搜索深度和用时将在这里显示");
  else if (activeTab === "evaluation") renderEvaluation(panel, report);
  else renderSearch(panel, report);
}

function renderResult() {
  const capture = byId("captures");
  const events = game.lastMove?.captures || [];
  capture.hidden = !events.length || Boolean(preview);
  capture.textContent = events.length ? `本步吃子  ${events.map(([r, c, piece]) => `${pieceLabels[piece]} (${r},${c})`).join("、")}` : "";
  const result = byId("result");
  result.hidden = !game.outcome;
  if (game.outcome) {
    const reason = outcomeLabels[game.outcome] || game.outcome;
    const title = game.winner ? `${sideLabels[game.winner]}获胜` : "和棋";
    const summary = game.summary;
    result.replaceChildren(element("strong", null, `${title} · ${reason}`),
      element("p", null, `本局积分  Team19 ${game.points} · 你 ${2 - game.points}`),
      element("p", "muted", `AI 落子 ${summary.aiTurns} 次 · 兜底 ${summary.fallbacks} 次 · 异常 ${summary.errors} 次`));
    if (game.repetition) {
      const { firstPly, repeatedPly, sideToMove } = game.repetition;
      const explanation = `${afterPly(repeatedPly)}和${afterPly(firstPly)}棋盘完全相同，且都轮到${sideLabels[sideToMove]}。同一局面再次出现，裁判立即判和。`;
      result.children[0].after(element("p", null, explanation));
    }
    if (summary.aiTurns) {
      const depths = Object.entries(summary.depths).map(([depth, turns]) => `D${depth} × ${turns}`);
      result.append(element("p", "muted",
        `总用时 ${duration(game.budget - game.remaining)} · 最慢 ${duration(summary.slowest)} · ${depths.join(" / ")}`));
    }
  }
}

function render() {
  if (!game) return;
  for (const id of ["new-button", "load-button"]) byId(id).disabled = Boolean(busy);
  byId("export-button").disabled = Boolean(busy || !game.replayFile);
  byId("save-button").disabled = Boolean(busy || game.outcome);
  byId("undo-button").disabled = Boolean(busy || !game.canUndo);
  renderParticipants();
  renderStatus();
  renderBoard();
  renderClock();
  renderAnalysis();
  renderResult();
}

function toast(message, error = false) {
  const feedback = byId("toast");
  feedback.textContent = message;
  feedback.classList.toggle("error", error);
  feedback.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { feedback.hidden = true; }, 4100);
}

async function request(path, data) {
  const response = await fetch(path, data ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
  } : {});
  const result = await response.json();
  if (!response.ok) {
    if (result.state) {
      game = result.state;
      preview = selected = null;
    }
    throw new Error(result.error || "无法完成操作。");
  }
  return result;
}

async function act(action, data = {}) {
  if (!game || busy) return false;
  busy = action;
  render();
  try {
    const response = await request("/api/action", { action, revision: game.revision, ...data });
    game = response.state;
    if (action !== "top3" && action !== "save") preview = selected = null;
    if (response.notice) toast(response.notice);
    return true;
  } catch (error) {
    toast(error.message, true);
    return false;
  } finally {
    busy = null;
    render();
  }
}

async function maybeThink() {
  if (game && !humanTurn() && !game.outcome) await act("think");
}

function selectTab(tab, focus = false) {
  activeTab = tab.dataset.tab;
  preview = null;
  render();
  if (focus) tab.focus();
}

for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    const tabs = [...tab.parentElement.querySelectorAll(".tab")];
    const offset = event.key === "ArrowLeft" ? -1 : 1;
    event.preventDefault();
    selectTab(tabs[(tabs.indexOf(tab) + offset + tabs.length) % tabs.length], true);
  });
}
byId("live-button").addEventListener("click", () => { preview = null; render(); });
byId("undo-button").addEventListener("click", () => act("undo"));
byId("export-button").addEventListener("click", () => {
  if (!game?.replayFile || busy) return;
  const link = element("a");
  link.href = "/api/replay";
  link.download = game.replayFile;
  document.body.append(link);
  link.click();
  link.remove();
});

byId("new-button").addEventListener("click", () => {
  byId("new-form").elements.side.value = game.human;
  byId("budget-input").value = game.budget;
  byId("new-dialog").showModal();
});
byId("new-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  byId("new-dialog").close();
  activeTab = "search";
  if (await act("new", { side: form.get("side"), budget: Number(form.get("budget")) })) maybeThink();
});
for (const preset of document.querySelectorAll("[data-budget]")) {
  preset.addEventListener("click", () => { byId("budget-input").value = preset.dataset.budget; });
}
for (const button of document.querySelectorAll("[data-close]")) {
  button.addEventListener("click", () => button.closest("dialog").close());
}
byId("save-button").addEventListener("click", () => {
  byId("save-name").value = "";
  byId("save-dialog").showModal();
});
byId("save-form").addEventListener("submit", event => {
  event.preventDefault();
  byId("save-dialog").close();
  act("save", { name: byId("save-name").value.trim() });
});
byId("load-button").addEventListener("click", async () => {
  try {
    const { files } = await request("/api/saves");
    const list = byId("save-list");
    list.replaceChildren(...(files.length ? files.map(file => element("option", null, file))
      : [element("option", null, "还没有存档")]));
    byId("load-submit").disabled = !files.length;
    list.disabled = !files.length;
    byId("load-dialog").showModal();
  } catch (error) {
    toast(error.message, true);
  }
});
byId("load-form").addEventListener("submit", async event => {
  event.preventDefault();
  const name = byId("save-list").value;
  byId("load-dialog").close();
  activeTab = "history";
  if (await act("load", { name })) maybeThink();
});

async function start() {
  populateBoard();
  try {
    game = (await request("/api/state")).state;
    render();
    await maybeThink();
  } catch (error) {
    byId("play-message").textContent = "无法连接本地棋手";
    toast(error.message, true);
  }
}
start();
