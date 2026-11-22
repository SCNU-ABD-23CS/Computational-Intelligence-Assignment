# Computational Intelligence Assignment - Group 19

JC4004 Tablut AI 小组作业。

## 网页人机对弈

在浏览器里和 Team19 下棋，查看每手的搜索深度、用时和候选步，并保存复盘记录。以下命令从仓库根目录运行。

### 准备 Python 环境

使用 **Python 3.12**，并从 `requirements.txt` 安装课程依赖。下面的 `tablut` 是示例环境名，可以换成自己的名字。已有 Python 3.12 环境时，直接激活它并检查依赖即可。

```powershell
conda create -n tablut python=3.12 pip
conda activate tablut
python --version
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
python -m pip check
```

课程依赖清单中的 PyTorch 固定为 CUDA 12.1 版本，安装时需要对应平台的轮子。Team19-v2 对局使用依赖清单中的 NumPy 和 Numba。

### 启动并下棋

```powershell
python tools/play_web.py --open
```

浏览器会打开 [本地对弈室](http://127.0.0.1:8019/)。首次打开默认你执守方，Team19 先走，AI 的整局预算是 60 秒。保持终端运行，结束后在终端按 `Ctrl+C` 停止服务。

- 点击 **新对局** 选择攻方或守方，也可以切换到 10 秒预算。
- 点击自己的棋子，再点绿色落点完成走棋。**悔棋** 会恢复到你上一次落子前，包括棋盘、历史和用时。
- 在 **AI 分析** 查看 v2 的实战搜索轨迹和棋谱。候选步与七项局面估值使用兼容估值接口。点击候选步可预览走法，再点 **返回实战** 继续。
- 终局会显示胜负、积分和判定原因。局面重复时，棋谱会标记该局面首次出现和再次出现的手数。

### 复盘日志与存档

- 网页每局的 JSON 复盘日志自动写入 `tools/game_logs/`，每手落子后更新。日志包含棋谱、逐手棋盘、吃子、AI 用时和各手的搜索、候选与估值分析。点击 **导出复盘** 可下载当前记录，终局后也能导出。
- **存档** 将进行中的局面写入 `tools/saved_positions/`。用 **读档** 选择它，可以恢复棋盘、历史和预算，接着下棋。

## 离线联赛与看板

九队历史联赛共 **576 局**，包含 60 秒标准赛、10 秒快棋和 60 秒强敌对照。Team19-New@91ffcc8 在两项循环赛均列第 3。四代进化对照比较各版对同一批六支外部队伍、相同开局的成绩。

在仓库根目录用 Python 3.12 运行以下命令，根据本地已有的赛果生成看板。

```powershell
python -B -m tools.arena.dashboard
```

双击 `tools/arena/results/dashboard.html` 即可离线查看，也可把这个 HTML 文件发给队友。总览卡片可以切换赛段。循环赛用得分率热力图展示交手结果，强敌页逐队展示新版的攻守表现。点击「打印视图」可另存 PDF。

完整的评测和棋谱回放命令见 [Arena 使用说明](tools/arena/README.md)。

## AI 核心与人机工具

| 文件 | 当前改动 |
| --- | --- |
| `Team19.py` | 预热正式对局的编译内核，按整局余量滚动分配每手时间，并记录真实对局历史。 |
| `Team19_engine.py` | 提供 Python 规则接口和 Numba 扁平棋盘内核。编译轨在落子、吃子和回溯时更新双哈希、子力和王位。 |
| `Team19_search.py` | 编译轨使用迭代加深、PVS、期望窗口、静搜及晚着缩减。低余量完成 D1，跨步置换表保存走法提示，证实必胜时提前落子。Python 固定深度接口供对照实验。 |
| `Team19_eval.py`、`Team19_weights.py` | 提供七维兼容估值与 22 维编译估值。实战权重覆盖子力、BFS 逃生步数、包围网和护卫机动性。 |
| `EVALUATION.md` | 说明七维实验估值的特征与权重范围。 |
| `Tablut.py` | 本地演示入口将攻守两席都设为 Team19。 |
| `tools/play_human.py`、`tools/play_web.py`、`tools/web/app.js` | 终端记录中断层的节点数和已证实必胜的步数。网页后端传递中断层节点数，前端显示必胜早停原因。 |

## 新增离线评测与看板

以下路径以 `tools/arena/` 为起点。

| 文件 | 用途 |
| --- | --- |
| `runner.py` | 通过官方裁判串行运行比赛，逐局写入 JSONL，支持断点续跑。 |
| `openings.py`、`openings.json` | 生成并保存 12 个固定的四手开局，供双方在相同开局换边对照。 |
| `report.py`、`overview.py` | 前者导出积分、攻守数据、败局及和棋棋谱。后者复放记录核对判定，汇总赛段并刷新看板。 |
| `replay.py` | 在只读浏览器页面逐手查看棋谱、吃子和剩余用时。 |
| `dashboard.py`、`dashboard_data.py` | 读取并核对已有 CSV 和 JSONL，生成可离线分享的单文件看板。循环赛显示得分率矩阵，强敌赛按对手展示攻守结果。 |
| `dashboard.js`、`dashboard.css` | 实现赛段标签与总览卡片的切换，以及响应式图表和打印样式。 |
| `export_snapshots.py`、`snapshots/` | 以独立包保存 Old、PreFix 和 New@91ffcc8，记录源提交和原始文件哈希。 |
| `tune_ga.py`、`validate_v2.py` | 在 D3/D4 对局中进化编译权重，并用另一组开局进行 10 秒和 60 秒换边验证。 |
| `evolution.py`、`verify_submission.py` | 更新四代共同对手对照，在隔离的只读目录中通过官方裁判验证提交模块。 |
| `__init__.py`、`bots/__init__.py`、`snapshots/__init__.py` | 提供联赛、参赛程序及历史版本的独立模块入口。 |
| `README.md`、`.gitignore` | 记录联赛的运行与复盘方式，将本地生成的 `results/` 排除在 Git 跟踪之外。 |

参赛对照程序位于 `tools/arena/bots/`。

| 文件 | 对照内容 |
| --- | --- |
| `bots/random_bot.py`、`bots/greedy_bot.py`、`bots/naive_ab_d2.py` | 分别提供随机合法走法、一手贪心和两层 Alpha-Beta 基线。 |
| `bots/guided_mcts.py` | 蒙特卡洛树搜索，用估值和一步战术筛选引导探索。 |
| `bots/bfs_strategist.py` | 用 BFS 估计国王最少逃生步数，结合包围网评估局面。 |
| `bots/qsearch_pvs.py` | 沿用 Team19 的估值，使用 PVS 和静态延伸验证关键战术。 |
| `bots/grandmaster_numba.py`、`bots/numba_rules.py` | 前者实现 Numba 编译搜索和独立估值，后者提供扁平棋盘、增量哈希及落子规则。 |
| `bots/common.py`、`bots/tactics.py` | 分别提供 Python 参赛程序共用的计时与历史记录，以及一步制胜和必防威胁识别。 |

## 新增与扩充的测试

- `tests/test_eval.py` 增加护卫、封锁和受保护射线的估值回归。`tests/test_player.py` 增加长局和低预算分时测试。
- 新增 `tests/test_fast_search.py`，核对增量哈希、吃子、重复判和与必胜早停。
- `tools/arena/test_arena.py`、`tools/arena/test_specialists.py`、`tools/arena/test_dashboard.py` 分别覆盖赛程与裁判、对照程序战术、以及看板统计与结构。
- `tools/arena/verify_numba.py` 将编译版引擎与官方裁判对拍，并检查 JIT 预热、吞吐和完成深度。

## 成员 5 的实验接口

在同一棋盘上固定搜索深度，可以对照估值权重或走法排序。准备待测的 `board`、行棋方 `is_attacker`、权重 `weights` 和真实历史 `game_history`。

```python
from Team19_engine import apply_move, check_winner, compute_hash, get_legal_moves
from Team19_eval import evaluate_board
from Team19_search import SearchFunctions, search_fixed_depth

callbacks = SearchFunctions(
    get_legal_moves, apply_move, check_winner, evaluate_board, compute_hash
)
result = search_fixed_depth(
    board, is_attacker, depth=2, functions=callbacks,
    weights=weights, game_history=game_history,
)
```

`result` 包含 `move`、`score`、`nodes_visited` 和 `cutoffs`。传入 `deadline` 后，搜索超时会抛出 `SearchTimeout`。深度为 0 或根局面已终局时，`move` 为 `None`。

- `game_history` 是真实对局中已出现的局面哈希集合，可以包含根局面。用 `compute_hash(board, is_attacker)` 按该局面下一手的行棋方计算。单独测试一个局面时可传 `None`。
- 比较排序时，在相同局面、深度、历史和权重下，分别用引擎顺序与 `tactical_ordering=True` 运行。核对根节点评分，并比较访问节点数和剪枝次数。同分时可以选出不同走法。
- 比较权重时，固定局面、深度、历史和排序设置，只更换 `weights`。[特征顺序与权重范围](EVALUATION.md)

### 接力调试 v2 实战权重

在已激活的 Python 3.12 环境中运行。第一次探索可从当前默认权重开始。换电脑接力时，用已有候选 JSON 作为起点并选不同种子。

```powershell
python -B -m tools.arena.tune_ga train --depth 4 --time-limit 1560 --seed 20261122
python -B -m tools.arena.tune_ga train --resume tools/arena/results/team19-v2-candidate.checkpoint.json --time-limit 1560
python -B -m tools.arena.tune_ga train --init-weights tools/arena/results/team19-v2-candidate.json --seed 20261123 --depth 4 --output tools/arena/results/team19-v2-candidate-20261123.json
python -B -m tools.arena.tune_ga validate --candidate tools/arena/results/team19-v2-candidate-20261123.json --run-id team19-v2-validation-20261123 --resume
```

训练只输出候选、检查点和收敛记录。验证对比候选、当前默认权重和 Grandmaster，使用独立的 12 个开局。可用 `--depth 3` 做快速训练，用 `--init-weights DEFAULT_WEIGHTS` 明确指定当前默认权重。

## 验证

使用 Python 3.12，在仓库根目录运行：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 160 组随机树与穷举 Minimax 的对照、官方裁判下的一步捕王/逃生及阻止对手逃生、迭代加深的超时回退和重复状态处理。剪枝示例在结果不变时，将访问节点从 15 个降为 11 个。

`tests/test_player.py` 用官方裁判验证守方的开局误判、跨局重置、搜索结果的合法性、低预算兜底和缺少队友模块时的完整对局。

使用装好 `requirements.txt` 的环境运行正式模块和测试。

## 用官方裁判本地验证

把课程提供的 `Tablut.py` 与根目录的 `Team19*.py` 放在同一个本地测试目录。其中 `Team19_weights.py` 提供 v2 正式对局的权重。

仓库里的 `Tablut.py` 已将底部两个演示席位设为 `Team19`。如果使用课程原版，把两个 `__import__("Team00")` 换成 `__import__("Team19")`，模块名省略 `.py`。在测试目录运行以下命令，用官方裁判测试 Team19 的攻守接口。

```powershell
python Tablut.py
```

也可以在仓库根目录通过 Python 脚本调用同一个裁判。

```python
import Tablut
import Team19

game = Tablut.Tablut()
game.play(Team19.Player(), Team19.Player())
```

要测试每方整局 10 秒，在测试副本中将 `TIME_BUDGET` 设为 `10.0`。通过脚本调用时，可以在创建 `Player` 前设定 `Tablut.TIME_BUDGET = 10.0`。

## 分支与使用

- `main`：约定用于最终集成的分支。
- 其他：成员个人开发分支

```bash
git clone --branch kaaxi https://github.com/SCNU-ABD-23CS/Computational-Intelligence-Assignment.git
cd Computational-Intelligence-Assignment
```
