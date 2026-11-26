# 成员 3 · 标准盘双边开局库

本轮基线为 `34c0a7c`。沿用课程裁判的零步标准盘、轮流攻守及胜 2 / 和 1 / 负 0 计分。对局数据中的坐标从零开始，四位走法依次为起点行、列和终点行、列。

## 在线入口

提交时将 `Team19_book.py`、`Team19_book_data.py` 与原有 Team19 模块放在同一目录。比赛仍调用 `Team19.Player()`。实验显式使用 `Player(use_book=True)` 或 `Player(use_book=False)`。每个条目是一条从标准盘开始的完整路径，只有最后一步是推荐着。索引键包含棋盘、行棋方及所有已出现的带行棋方局面哈希。每次命中再验证合法着和后继判重，离库后立即调用原 v4 搜索。竞技场在回放的 `attempts` 中用 `bookHit` 标记命中。

## 如何选择候选

1. 从 42 份 JSONL 中按对阵、预算、初始局面、前缀、走法和结果识别重复回放。3,406 条记录对应 3,045 个不同指纹。历史赛程、版本、对手和时限有差异。具体计数见 `book_corpus_report.json`。
2. 标准盘 80 种合法首着按旋转、镜像归成十类。对每类分别让 v4、v3、Grandmaster-Plus、Grandmaster-Numba、QSearch-PVS 执守，然后由 v4 对不同应手搜索至第四个半回合。按各家应手中最差的 v4 搜索分，领先三类为 0313、1412、0323。0313 与实际入库的 0515 对称。冻结 v4 在历史 60 秒标准盘的 13 盘执攻全部选 0515，取得 11 胜 2 负。另做六盘十秒执攻续局，每类各对 v3 和 QSearch 一次。0313 对 v3 负、对 QSearch 胜，1412 和 0323 均为两胜，详见 `book_root_finalists.json`。因此保留这两类作后续挑战者；当前库的 0515 已有六十秒正式对照，其他两类仅完成十秒筛选。
3. 在十类首着下强制走完前两手，交还无库 v4 执守，各对 v3、Grandmaster-Plus 下完一局，10 秒档合计 19 胜 1 和。前四手的历史条目取自固定 182 局的 60 秒标准盘赛程。生成器合并 D4 等价路径，并以筛选后的续局补充应手。
4. QSearch 第五手分别强制候选并让冻结 v4 搜对方反击。对方反击后再由己方应对，0407 的六秒搜索分为 −1062，3031 为 −2039，3032 为 −3001。十秒续局中 0407 分别战胜 QSearch 和 Grandmaster-Plus，3031 只战胜 QSearch。历史 60 秒原局走 0407 曾负，因此仅把 0407 放入待整盘验证的候选库。
5. 对 v3 执守第二手，4636 和 2423 在十秒续局均赢，4636 在历史六十秒原局曾负。该点采用首着家族筛选得到的通用应手。对 v3 执攻第十一手，5856 的历史败局与十秒续局胜局相冲突，1525 在十秒续局也输，此处交回搜索。

## 复现及交接

在仓库根目录运行以下命令。重建历史条目需要交接包内的 `results/arena-standard-b2c5e0df9948-60s.jsonl`，默认从仓库的上一级目录读取。线上只使用导出的静态数据。筛选和对局使用墙钟限时，复跑时可因搜索完成的深度不同而出现不同结局。

```powershell
python -m tools.arena.book_corpus --root ../results --output corpus.json
python -m tools.arena.book_screen --budget 10 --output screen.json
python -m tools.arena.book_screen --validate screen.json --budget 10 --output replies.jsonl
python -m tools.arena.book_trials --case qs-attacker-ply5 --deep-seconds 6 --run --budget 10
python -m tools.arena.build_book
python -m tools.arena.compare_book --opponents qs v3 gmp gm --budget 60 --output paired60.jsonl
```

成员 1 可在两种玩家上显式打开同一份库做盲评。成员 2 审查在线入口、索引和两套测试。成员 4 可按回放中的 `bookEnabled`、`bookHit` 区分实验组与命中步。对照方法采用相同对手、相同开局及种子、互换攻守的设计。外部参考为 [Stockfish 测试开局库](https://github.com/official-stockfish/books)，所有本项目走法均来自课程规则下的搜索或实战。

## 60 秒正式对照结果

四个对手分别为 QSearch-PVS、Team19-v3、Grandmaster-Plus 和 Grandmaster-Numba。每个对手均从标准盘出发，开库、关库各执攻和执守一次，16 盘串行执行。双方配置各取得 8 胜，积分均为 16/16。开库组共命中 13 手，执攻首手省时中位数为 2.985 秒。开库方无超时、异常、非法着或回退。开库执攻对 QSearch 的胜局由对方超时判胜，其余七胜均在棋盘上终结。每种配置每个对手每侧只有一盘，本轮积分已达到上限。

依据交接指南的净增分条件，`Player()` 默认沿用冻结 v4。候选库保留为显式开关，供成员 1 使用同一本库开展盲评。38 条源路径按 D4 展开得到 297 个局面、行棋方与历史组合，最长推荐到第五个半回合。

完整逐手回放为 `book_paired60.jsonl.gz`。复算积分、省时和默认开关判定：

```powershell
python -m tools.arena.book_acceptance tools/arena/book_paired60.jsonl.gz acceptance-local.json
python -m tools.arena.book_trials --root-finalists --run --budget 10 --output root-finalists-local.json
```
