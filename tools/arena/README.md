# JC4004 离线评测

从仓库根目录用课程的 Python 3.12 环境运行。评测调用官方
`Tablut.Tablut.play()`，构造参赛者后才启动该局的两只挂钟。
四步开局由裁判逐步复放，双方从完整历史和原始预算开始。

## 五队冒烟

```powershell
python -B -m tools.arena.runner --stage smoke --openings 1
python -B -m tools.arena.runner --stage random
python -B -m tools.arena.report smoke
python -B -m tools.arena.report random
python -B -m unittest tools.arena.test_arena -v
```

中断后用相同参数加 `--resume`，已落盘的比赛自动跳过。
`--limit 1` 可先跑一局，`--run-id` 可为独立重复试验命名。
固定赛程严格串行。同一结果目录一次只允许一个写入进程。

`results/` 保存每局 JSONL、每赛段 Markdown 和 CSV、Team19-New 败局与和局的棋谱。
积分为胜 2、平 1、负 0。标准开局赛按 `30 × 本队积分 / 榜首积分` 折算课程模拟分。
BT Elo 同时估计队伍相对实力和执攻方偏差。Team19-Old 的历史合法兜底在各报告单列。

## 九队联赛

在安装了 `requirements.txt` 依赖的 Python 3.12 环境中，从仓库根目录依次运行。

```powershell
python -B -m tools.arena.runner --stage official --workers 1 --resume
python -B -m tools.arena.runner --stage fast --workers 1 --resume
python -B -m tools.arena.runner --stage final --workers 1 --resume
python -B -m tools.arena.overview
```

| 赛段 | 开局 | 时间 | 场次 |
| --- | --- | --- | --- |
| `official` | 标准开局，所有对阵均换边 | 60 秒 | 72 |
| `fast` | 前 6 个固定开局，所有对阵均换边 | 10 秒 | 432 |
| `final` | 全部 12 个固定开局，新版与每个强敌各换边 | 60 秒 | 72 |

强敌取外部六队中标准开局赛前三。同积分时比较快棋积分，再按队名排序。
`overview` 复放三个赛段、核对官方裁判判定与记录，并生成 `results/overview.md`。
用 `python -B -m tools.arena.report official` 可单独生成一个赛段的战报。

## 离线统计看板

在仓库根目录运行以下命令。看板读取已有的 JSONL 与 CSV，适用于单个赛段或完整联赛。

```powershell
python -B -m tools.arena.dashboard
```

双击 `tools/arena/results/dashboard.html` 即可离线打开，也可把这个文件单独发给队友。
用页顶标签或总览卡片切换赛段。点击「打印视图」可打印当前赛段或另存为 PDF。

总览汇总已加载赛段的对局、终局与稳定性。有四代对照数据时，按相同外部对手、开局和执棋方比较得分率。
循环赛交手热力图按得分率着色，和棋贡献 1 分。强敌赛按对手展示新版的换边结果。

可用 `--results-dir` 指定赛果目录，用 `--output` 指定生成位置。
`overview` 在输出原有战报后刷新看板，可用 `--no-dashboard` 跳过刷新。

## v2 权重与四代对照

根目录 README 提供训练、接力和验证命令。训练使用固定开局及实战中局的 D3/D4 对局，适应度侧重攻方和对 Grandmaster 的得分。两项新增射线特征采用绝对变异步长，初始权重为零时也会探索。训练记录与候选 JSON 写入本地 `results/`。

验证冻结两份权重并生成另一组开局。候选与默认版、Grandmaster 的对比各包含 12 个开局换边，在 10 秒和 60 秒档串行执行。进化权重的更新依据独立对赛结果。
首轮热启动向量保存在 `results/team19-v2-hot-weights.json`。复跑首轮对照时可用 `--baseline` 指定它，并用独立的 `--run-id` 保存新赛果。

```powershell
python -B -m tools.arena.export_snapshots --version v_91ffcc8
python -B -m tools.arena.evolution --workers 1 --resume
python -B -m tools.arena.dashboard
python -B -m tools.arena.verify_submission
```

四代图使用前三代在历史联赛对六支外部队伍的成绩，以及 v2 按相同赛制复跑的 12 局标准赛和 72 局快棋。只读冒烟在独立目录用官方裁判完成 10 秒和 60 秒对局。

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
`python -B -m tools.arena.export_snapshots` 从三个固定 Git 提交导出历史版本，
仅调整模块间导入，并记录导出前的文件哈希。参赛时每个版本都用独立包名，
共用同一个官方裁判模块和时限。
