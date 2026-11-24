# JC4004 离线评测

激活已安装 `requirements.txt` 依赖的 Python 3.12 环境，从仓库根目录运行。每局由官方 `Tablut.Tablut.play()` 裁判，开局历史完整复放，双方各有独立的整局挂钟。所有赛程串行执行。

## 算法简称

运行 `python -X utf8 -B -m tools.arena.runner --list-teams` 可查看终端对照表。队名和简称大小写不敏感，逗号分隔，预设可与单队混用。

| 简称 | 全名 | 简介 |
| --- | --- | --- |
| `v4` | Team19-v4 | 当前提交版，逃线战术搜索与关键回合加时 |
| `v3` | Team19-v3 | 分值置换表版快照 be23011 |
| `v2` | Team19-v2 | 时钟及收官补丁版快照 3edc85c |
| `gmp` | Grandmaster-Plus | Grandmaster 原估值，增强搜索和时钟 |
| `gm` | Grandmaster-Numba | 原版手调权重和搜索 |
| `qs` | QSearch-PVS | 主变搜索与静态延伸 |
| `bfs` | BFS-Strategist | 国王逃生路径和包围网估值 |
| `91f` | Team19-91ffcc8 | Numba 重构前快照 |
| `mcts` | Guided-MCTS | 启发式引导蒙特卡洛树搜索 |
| `prefix` | Team19-PreFix | 估值修复前快照 e3d64c4 |
| `old` | Team19-Old | 三阶段优化前快照 99671ae |
| `greedy` | GreedyBot | 一步贪心基线 |
| `ab2` | NaiveAB-D2 | 固定两层 Alpha-Beta 基线 |
| `random` | RandomBot | 随机合法步冒烟基线 |
| `new` | Team19-New | 历史赛段入口，运行当前根目录版本 |

`top6` 选 v2、gm、qs、bfs、91f、mcts。`all` 选表中所有条目，包含与 v4 共用实现的历史入口 new。

## 常用命令

`--games-per-pair` 取 **2 至 24 的偶数**，默认 2。每组用前 N/2 个固定开局，每个开局换边各打一局。`--budgets` 默认 60，可按顺序指定多档，每个数是**每方整局秒数**。默认 `--format rr` 打全循环。`1vn` 只安排擂主对各挑战者，默认取队伍列表第一项。

```powershell
# v4 对 v3。每档 24 局，共 48 局
python -X utf8 -B -m tools.arena.runner --teams v4,v3 --format rr --games-per-pair 24 --budgets 60,10 --workers 1 --fresh

# 四队全循环。每档 6 组 × 24 局，共 288 局
python -X utf8 -B -m tools.arena.runner --teams v4,v3,gmp,gm --format rr --games-per-pair 24 --budgets 60,10 --workers 1 --fresh

# 六强双档封板。每档 15 组 × 24 局，共 720 局。已有同配置赛果会先归档
python -X utf8 -B -m tools.arena.runner --teams top6 --games-per-pair 24 --budgets 60,10 --fresh

# 单跑 60 秒档，共 360 局
python -X utf8 -B -m tools.arena.runner --teams top6 --games-per-pair 24 --budgets 60

# 快速 1v4 车轮战，共 16 局
python -X utf8 -B -m tools.arena.runner --teams v4,gm,qs,bfs,mcts --format 1vn --anchor v4 --games-per-pair 4 --budgets 10

# 预览赛程、已有记录、可复用局数和按双方用满预算估算的耗时
python -X utf8 -B -m tools.arena.runner --teams top6 --games-per-pair 24 --budgets 60,10 --dry-run

# Ctrl+C 中断后保留原赛程参数，只用 --resume 续跑
python -X utf8 -B -m tools.arena.runner --teams top6 --games-per-pair 24 --budgets 60,10 --resume

# v4 单独更新后，保留同配置下其他队伍之间已完成的局
python -X utf8 -B -m tools.arena.runner --teams v4,v3,gmp,gm --format rr --games-per-pair 24 --budgets 60,10 --rerun-team v4
```

`--fresh`、`--resume` 和 `--rerun-team` 三选一。双档 `--fresh` 在首局前归档两档旧赛果。局部重跑会验证复用棋谱，同一实现的多个参赛名会一并重跑。改动影响多队共用代码时，用 `--fresh` 重测。可用 `--run-id` 给独立实验加名称前缀。

## v4 战术与分时对照

`v4` 与 `v3` 共用 22 维进化权重。对照重点是静搜中的护卫开线与国王转线、封堵类走法排序及晚着保护，以及长预算下的关键回合加时。`v2` 保留作置换表升级前的对照。`gmp` 与 `gm` 共用原版手调估值。

v4 的逐手日志中，`live_replies` 是 D1 找到的可避开下一手落败的合法走法数，`used_hard_limit` 表示该手曾允许搜索使用加时上限。

v3 与 gmp 的分值表各有 524,288 项，约 8 MiB。相同局面可跨走法顺序复用深度足够、搜索模式一致的分值。搜索先判重复，再读分值。搜索中遇到重复的节点只写走法提示，中止分支不写表。根节点、静搜和杀棋认证不使用分值截断，使用过分值截断或晚着缩减的杀棋需完整复搜。每手日志的 `tt_cutoffs` 记录实际发生的分值截断。gmp 使用本机单调钟检查截止时间。

当前正式提交时带上根目录的全部 `Team19*.py`，包括置换表 `Team19_tt.py` 和战术判断 `Team19_tactics.py`。

## 赛果与看板

- `tools/arena/results/` 下每个预算一个 `arena-<配置标识>-<预算>s.jsonl`，记录逐局棋谱、裁判剩时和搜索统计。每局结束立即落盘，终端同步显示完成比例、对阵和 ETA。
- 自动归档带 `.before-<UTC 时间戳>.jsonl` 后缀。局部重跑的复用记录带原始赛果来源。
- 完成后覆盖同目录的 `dashboard.html`。双击离线查看对应预算、队伍、交手矩阵、攻守与稳定性。用 `--no-dashboard` 保留当前看板。用相同赛程参数加 `--resume --limit 0` 可按已落盘结果刷新看板。
- 如需 CSV 和 Markdown，运行 `python -B -m tools.arena.report <JSONL 文件名去掉扩展名>`。胜 2、和 1、负 0；完整循环赛显示 30 分制模拟课程分。

## 历史固定赛段

`--stage smoke|random|duel|official|fast|final|top6_60s|top6_10s` 沿用原有赛程和结果文件。例如 `python -X utf8 -B -m tools.arena.runner --stage top6_60s --resume` 跑前 6 个开局换边的 180 局赛程。`top6_10s` 仍从本地历史赛果复用 168 局。可运行 `python -B -m tools.arena.dashboard` 重新生成这两个固定赛段的看板，或用 `python -B -m tools.arena.overview` 审计原九队赛段。

## v2 权重与四代对照

根目录 README 提供训练、接力和验证命令。v2 的训练与验证使用封存的 v2 内核。训练使用固定开局及实战中局的 D3/D4 对局，适应度侧重攻方和对 Grandmaster 的得分。两项新增射线特征采用绝对变异步长，初始权重为零时也会探索。训练记录与候选 JSON 写入本地 `results/`。

验证冻结两份权重并生成另一组开局。候选与默认版、Grandmaster 的对比各包含 12 个开局换边，在 10 秒和 60 秒档串行执行。进化权重的更新依据独立对赛结果。
首轮热启动向量保存在 `results/team19-v2-hot-weights.json`。复跑首轮对照时可用 `--baseline` 指定它，并用独立的 `--run-id` 保存新赛果。

```powershell
python -B -m tools.arena.export_snapshots --version v_91ffcc8
python -B -m tools.arena.evolution --workers 1 --resume
python -B -m tools.arena.verify_submission
```

四代对照的赛果保存在 `results/evolution-official.jsonl` 和 `results/evolution-fast.jsonl`。只读冒烟在独立目录用官方裁判完成 10 秒和 60 秒对局。

## 回看棋谱

从对应赛段的 `results/<赛段>/replays/` 选择 `.json`，例如

```powershell
python -B -m tools.arena.replay tools/arena/results/official/replays/<棋谱文件>.json
```

将命令中的棋谱文件名换成实际文件名，在浏览器打开终端显示的地址。回放页可逐手查看落点、
吃子、耗时和两方剩时。人机对局页面的读档按钮用于继续未完成的存档。联赛棋谱使用只读回放页。

## 单独验证

```powershell
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s tools/arena -t . -v
python -B -m tools.arena.verify_numba rules
python -B -m tools.arena.verify_numba timing
```

Grandmaster-Numba 在创建选手时运行一次初始盘面的 D1 搜索完成编译。
`verify_numba` 对照 40 条引擎规则测试及 5,000 步官方裁判走法，再检查首次行棋和后续搜索的耗时。

## 开局和历史快照

`openings.json` 使用种子 20261121。12 个开局均为 4 手，排除了
开局吃子、末着立刻被吃或被一步赢棋的序列，以及执任一方能一步取胜的
最终盘面。对称等价局面只保留一个。

`python -B -m tools.arena.openings` 可重新生成开局库。
`python -B -m tools.arena.export_snapshots` 从五个固定 Git 提交导出历史版本，
仅调整模块间导入，并记录导出前的文件哈希。参赛时每个版本都用独立包名，
共用同一个官方裁判模块和时限。
