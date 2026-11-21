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

课程依赖清单中的 PyTorch 固定为 CUDA 12.1 版本，安装时需要对应平台的轮子。网页工具本身只用 Python 3.12 标准库，单独运行网页对弈时，有 Python 3.12 即可启动。

### 启动并下棋

```powershell
python tools/play_web.py --open
```

浏览器会打开 [本地对弈室](http://127.0.0.1:8019/)。首次打开默认你执守方，Team19 先走，AI 的整局预算是 60 秒。保持终端运行，结束后在终端按 `Ctrl+C` 停止服务。

- 点击 **新对局** 选择攻方或守方，也可以切换到 10 秒预算。
- 点击自己的棋子，再点绿色落点完成走棋。**悔棋** 会恢复到你上一次落子前，包括棋盘、历史和用时。
- 在 **AI 分析** 查看搜索轨迹、候选步、七项局面估值和棋谱。点击候选步可以预览它在落子前的棋盘上的起点和终点，再点 **返回实战** 继续。
- 终局会显示胜负、积分和判定原因。局面重复时，棋谱会标记该局面首次出现和再次出现的手数。

### 复盘日志与存档

- 网页每局的 JSON 复盘日志自动写入 `tools/game_logs/`，每手落子后更新。日志包含棋谱、逐手棋盘、吃子、AI 用时和各手的搜索、候选与估值分析。点击 **导出复盘** 可下载当前记录，终局后也能导出。
- **存档** 将进行中的局面写入 `tools/saved_positions/`。用 **读档** 选择它，可以恢复棋盘、历史和预算，接着下棋。

## 离线联赛与看板

九队评测包含 60 秒标准赛、10 秒快棋和 60 秒强敌对照。本次共完成 **576 局**，Team19-New 在两项循环赛均列第 3。

在仓库根目录用 Python 3.12 运行以下命令，根据本地已有的赛果生成看板。

```powershell
python -B -m tools.arena.dashboard
```

双击 `tools/arena/results/dashboard.html` 即可离线查看，也可把这个 HTML 文件发给队友。总览卡片可以切换赛段。循环赛用得分率热力图展示交手结果，强敌页逐队展示新版的攻守表现。点击「打印视图」可另存 PDF。

完整的评测和棋谱回放命令见 [Arena 使用说明](tools/arena/README.md)。

## AI 核心与人机工具

| 文件 | 当前改动 |
| --- | --- |
| `Team19.py` | 根据剩余时间、棋子数和已走手数分配搜索时间。长局预留后续手数，余量足够时允许短搜索，接近截止时返回合法步。 |
| `Team19_engine.py` | 增加快速落子和合法步存在判断。落子和吃子时同步更新攻守两种行棋方的哈希、子力和国王位置。 |
| `Team19_search.py` | 用快速落子和子力计数加速迭代加深的 Alpha-Beta 搜索。排序结合战术、缓存的走法提示和先前的剪枝记录。优先早胜，陷入必败时延后落败。证实必胜就停止加深，超时采用最后完成一层的走法。 |
| `Team19_eval.py` | 按国王可移动格、有效护卫和逃生射线重算护王与封锁分，使逼近受护卫射线得到相应的攻方分数。继续使用七维权重。 |
| `EVALUATION.md` | 说明当前特征、权重范围和调参接口。 |
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
| `export_snapshots.py`、`snapshots/v_99671ae/`（Old）、`snapshots/v_e3d64c4/`（PreFix） | 导出两版的 `Team19*.py`，记录源提交和文件哈希，并隔离模块导入。 |
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

## 验证

使用 Python 3.12，在仓库根目录运行：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 160 组随机树与穷举 Minimax 的对照、官方裁判下的一步捕王/逃生及阻止对手逃生、迭代加深的超时回退和重复状态处理。剪枝示例在结果不变时，将访问节点从 15 个降为 11 个。

`tests/test_player.py` 用官方裁判验证守方的开局误判、跨局重置、搜索结果的合法性、低预算兜底和缺少队友模块时的完整对局。

测试仅使用标准库和仓库内的官方框架，可在正式队友模块接入前运行。课程依赖清单见 `requirements.txt`。

## 用官方裁判本地验证

按作业说明，把 `Tablut.py`、`Team19.py`、`Team19_engine.py`、`Team19_eval.py` 和 `Team19_search.py` 放在同一个本地测试目录。若使用 `Team19_weights.py`，也放进去。

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
