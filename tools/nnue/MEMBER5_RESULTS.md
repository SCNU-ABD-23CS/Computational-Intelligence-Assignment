# 成员 5：NNUE 预校准、零中心残差与 Top-1 列表微调最终交接

日期：2026-11-27  
正式引擎基线：`main@34c0a7c5bba98af083fe1d6d1fe2756a47c710aa`  
工具代码基线：`QLxie@5e4f57a9cd6441773c2549e98564b66996f6cd5a`

## 1. 最终结论

本轮在不使用 400 题训练、选模或调参的前提下，为预校准模型增加了 **train-only 三候选列表式微调**。冻结第 20 轮后，只对该候选执行了一次 400 题最终评测，原双指标门槛均已严格通过：

| 主指标，排除 23 道 `tie_top2` | 静态基线 | 最终 NNUE | 变化 |
|---|---:|---:|---:|
| Top-1 | 132/377＝35.0133% | **133/377＝35.2785%** | **+1 题，+0.2653 个百分点** |
| Pairwise | 521.5/1065＝48.9671% | **537.0/1065＝50.4225%** | **+15.5 计分，+1.4554 个百分点** |
| 严格优于静态基线 | — | 是 | 通过 |

`gate.top1_strictly_better=true`、`gate.pairwise_strictly_better=true`、`gate.passed=true`。

**Top-1 只比静态基线多对 1 题，属于低余量通过。** 本结果证明的是当前封存题上的离线排名门槛通过，不等于已经证明在线对局增益，也不等于最终玩家已经可提交。根目录 Team19-v4 引擎和 22 维权重保持零改动；尚未创建 `Team19_nnue.py`，尚未执行在线零残差复现、增量一致性、NPS 和 60 秒实战。后续必须按原指南完成这些工作，不能仅凭本离线结果宣布上线完成。

最终权重：

```text
tools/nnue/results/volume1200/member5-c750-k6000-l0p5-tanh-precalib5-center1e3-listwise20-seed20261127/best_weights.npz
SHA-256: cf27439fdce79c34c05fdc831b755d9c210cd4019dbc41d12057c584150d3809
```

## 2. 数据、泄漏与评测边界

原始 NPZ 和 400 题文件未修改，最终 `verify_artifacts` 为 `passed=true`：

- train：17,862 条，SHA-256 `675ed46a1d1c2fa24fb699a289ecb4942fcf887a9fd4b5265dcf7588655883b4`；
- validation：3,166 条，SHA-256 `fbec32dff5b61e2178103eb0f885c218565039c4d5ce9a7e2eaf296336743af8`；
- questions：400 道，SHA-256 `2121c715dfbcb3d7e72b227d4d325b7b8d5b86703f4c1c847c1da88c772058eb`。

训练与验证的 `game_id` 隔离、D4 族隔离、400 题只含 validation 以及题号／D4 唯一性检查均通过。训练集 D4 后为 142,896 条；验证集、开发选择题和 400 题均不做 D4 增强。

保守标签策略不变：`abs(deep_eval)>=90000` 和 `abs(deep_eval)==3000` 的行只学习 outcome，不参与深搜回归。train 有 1,039 条 outcome-only，validation 有 202 条。NPZ 没有行级 depth，因此不能把所有 `abs==3000` 行都断言为那 8 条 `depth=64` 异常。

列表题只从 train NPZ 构造，文件明确记录 `train_npz_only=true`、`sealed_questions_used=false`。400 题没有参与候选生成、列表微调、早停、最佳轮次选择、残差缩放或参数选择。23 道 `tie_top2` 仍只作信息性单列，主指标使用其余 377 题及 1,065 个非同分标签候选对。

开发集仍有一个必须披露的限制：其 25% 游戏没有参与列表微调，但源模型曾在全部 train 游戏上做过点wise预训练，所以它不是“完全未见过的预训练游戏”。最终 400 题仍是真正放行门槛。

## 3. 最终代码范围

完整补丁从干净的 `QLxie@5e4f57a` 应用，只增加或修改以下 6 个离线源码文件：

| 文件 | 作用 |
|---|---|
| `tools/nnue/model.py` | 三平面编码、D4、标签掩码、NumPy 残差推理、隐藏诊断及开发用 tanh 等比缩放 |
| `tools/nnue/train_nnue.py` | 小随机输出头、两阶段 `side_calib` 预校准、点wise主训练、预 tanh 零中心约束与导出 |
| `tools/nnue/rank_questions.py` | 封存题静态／NNUE 排名、双指标严格门槛、D 头与校准隔离审计、残差诊断 |
| `tools/nnue/test_training.py` | 训练契约、预校准、中心约束、导出等价、缩放恒等式和列表 D4／游戏切分测试 |
| `tools/nnue/dev_top1.py` | 仅用 train NPZ 生成固定深度三候选题，并检查残差强度缩放是否双指标合格 |
| `tools/nnue/train_listwise.py` | 三候选列表交叉熵、点wise回放、联合零中心约束、按游戏开发切分与最佳轮次导出 |

未修改 `Team19.py`、`Team19_engine.py`、`Team19_eval.py`、`Team19_search.py`、`Team19_tactics.py`、`Team19_tt.py`、`Team19_weights.py`。完整补丁不包含原始数据、400 题、results、权重或任何正式玩家接入文件。

## 4. 训练契约

### 4.1 点wise源模型

网络为 243→32→2，隐藏层使用 `[0,1]` Clipped-ReLU；两个头都输出攻方视角残差，`sides` 只选择头。输出头采用 `std=1e-3` 的小随机非零初始化和零偏置，避免旧零初始化头的对称同步增长。

训练预测为：

```text
residual = C * tanh(selected_pre_tanh)
prediction = sigmoid((base_eval + residual) / K + side_calib[side])
```

其中 `C=750`、`K=6000`。主点wise目标为：

```text
y_deep = sigmoid(deep / K)
y_mix = 0.5 * outcome + 0.5 * y_deep
```

预校准前 5 轮冻结 hidden 和 heads，只用 AdamW `lr=0.05`、`weight_decay=0` 优化两个 `side_calib` logits。随后恢复全部参数，以 AdamW `lr=1e-3`、`weight_decay=1e-4`、batch 1024 联合训练，并加入：

```text
1e-3 * mean(selected_pre_tanh ** 2)
```

验证 NLL 不加入中心惩罚。`side_calib` 只用于离线训练／验证，不导出、不参与 400 题排名。导出仍为：

```text
head_weight_exported = C * head_weight
head_bias_exported = C * head_bias
residual = C * tanh(exported_score_domain_head / C)
```

源模型最佳为第 12 轮，验证 NLL `0.5738794516`，权重 SHA-256 为 `59d43552fb11c087e1bf38e5297b1d24325ebc943eada230dc38add6b4467ec7`。

### 4.2 先做的残差缩放开发实验

先用 train NPZ 构造 192 道固定深度题，root depth＝2、label depth＝4，非 tie 主题 183 道。对 `0.125, 0.25, …, 1.0` 八个强度只按开发集检查双指标。所有 scale 的 Top-1 都未严格超过静态的 54/183，虽然 Pairwise 有提高，但没有双指标合格项，因此没有导出缩放候选，也没有为该实验运行新的 400 题。

这一步的结论是：单纯缩小残差不能解决 Top-1，不能通过观察 400 题来反选 scale。

### 4.3 三候选列表式微调

重新从 train NPZ 生成 768 道固定深度三候选题，root depth＝2、label depth＝4；110 个失败盘面保留在 `failures` 中。题目 SHA-256 为：

```text
e86bf255656020423ee72fbd1a776a3c7c008119d683508e4c68514f1bd0787d
```

按 `game_id` 的 SHA-256 稳定顺序整体切分，75% 为列表训练、25% 为开发集，游戏不交叉：

- 列表训练 576 题，D4 后 4,608 题；
- 开发集 192 题，不增强，其中非 tie 主题 176 道；
- 原始点wise D4 数据同时用于随机回放，降低微调对原有点wise拟合的破坏。

微调目标为：

```text
listwise_ce = CE(
    softmax((static + residual) / T),
    softmax(deep / T)
)
objective = listwise_ce
            + replay_weight * pointwise_smooth_l1
            + center_weight * mean(concat(listwise, replay selected_pre_tanh) ** 2)
```

实际参数为 `T=1000`、`replay_weight=1`、`center_weight=1e-3`、AdamW `lr=1e-4`、`weight_decay=1e-4`、batch 128、20 轮、seed 20261127。所有候选子局面用 D 头评分，训练目标中的 `side_calib` 不进入列表排名分数。

每轮先要求开发集 Top-1 和 Pairwise 都严格优于静态，再按 Top-1 正确数、Pairwise 计分选择最佳轮次。第 20 轮被冻结：

| 开发集主指标 | 静态 | 列表微调第 20 轮 |
|---|---:|---:|
| Top-1 | 46/176＝26.1364% | **51/176＝28.9773%** |
| Pairwise | 229.0/512＝44.7266% | **241.0/512＝47.0703%** |

原始验证集 NLL 从源模型的 `0.5738794516` 变为 `0.5751187776`，略有变差；SmoothL1 为 `0.0314707922`，平均绝对残差为 `364.9334728`。这是为了改善候选排序而接受的点wise校准折衷，不能隐去或声称所有指标都改善。

## 5. 冻结后一次 400 题复验

结果文件：

```text
tools/nnue/results/volume1200/member5-c750-k6000-l0p5-tanh-precalib5-center1e3-listwise20-seed20261127/ranking_400.json
```

主集结果：

- Top-1：`133/377=35.2785%`，静态为 `132/377=35.0133%`；
- Pairwise：`537.0/1065=50.4225%`，静态为 `521.5/1065=48.9671%`；
- 静态基线有 31 个预测并列并获得半分，最终 NNUE 主集没有预测并列；
- 两个严格改善条件同时成立，`gate.passed=true`。

23 道 `tie_top2` 信息性题目不计入主门槛：Top-1 从静态 `10/23` 变为 `11/23`，Pairwise 从 `14.0/18` 变为 `15.0/18`。这些信息不参与选模，也不能替代主集结果。

最终残差诊断：

| 诊断 | 数值 |
|---|---:|
| 平均绝对残差 | 375.2644348 |
| 残差均值 | -5.0471344 |
| 最小／最大残差 | -738.4799805／+728.5647583 |
| 负／零／正候选数 | 609／0／591 |
| `abs(residual)>=735` 的近边界候选 | 4/1200 |
| 同题残差极差中位数 | 216.1038361 |
| 同题残差极差 p90 | 525.9649048 |
| 同题残差极差最大值 | 975.0133057 |
| 非零同题残差极差 | 400/400 |
| 相对静态实际改序题数 | 57 |
| 32 个隐藏单元全为 1 的候选局面 | 0 |
| 32 个隐藏单元全为 0 的候选局面 | 0 |

最终候选不再出现全部候选同号、贴同一边界或同题残差趋同的旧缺陷，但仍有 4 个近边界候选，不能描述为“完全没有饱和”。这些诊断只用于解释，不参与选模或 gate。

## 6. 全部真实实验对照

静态基线固定为 Top-1 `132/377`、Pairwise `521.5/1065`。以下每一行都是对应候选当时冻结后得到的真实结果，不表示对最终候选反复试参：

| 候选 | 选中轮 | 验证 NLL | Top-1 | Pairwise | 非零残差极差题 | 改序题 | 双指标 gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 旧零初始化，30 轮 | 30 | 0.6102532 | 131/377 | 521.0/1065 | 9/400 | 2 | 否 |
| 旧零初始化，60 轮 | 60 | 0.6076778 | 131/377 | 521.0/1065 | 9/400 | 2 | 否 |
| 线性缩放＋hard clip C750 | 17 | 0.5984247 | 132/377 | 521.5/1065 | 0/400 | 0 | 否 |
| `C=750 tanh` | 17 | 0.5983087 | 132/377 | 523.0/1065 | 389/400 | 14 | 否 |
| `C=1500 tanh` | 17 | 0.5863801 | 131/377 | 525.0/1065 | 399/400 | 8 | 否 |
| `C=750 tanh + side_calib` | 27 | 0.5755986 | 129/377 | 520.0/1065 | 400/400 | 18 | 否 |
| `C=750 tanh + 预校准5轮 + 零中心1e-3` | 12 | 0.5738795 | 130/377 | 531.0/1065 | 400/400 | 52 | 否 |
| 上述源模型＋train-only 列表微调 | 20 | 0.5751188 | **133/377** | **537.0/1065** | 400/400 | 57 | **是** |

最终候选的 Pairwise 比预校准源模型增加 6.0 计分，Top-1 增加 3 题；相对静态基线则分别增加 15.5 计分和 1 题。Top-1 余量很小，报告中应原样保留，不应写成“大幅提升”。

## 7. 最终自检与证据

执行：

```cmd
python -B -m tools.nnue.verify_artifacts --tag volume1200 --questions questions_400.json
python -B -W error -m unittest tools.nnue.test_pipeline tools.nnue.test_training -v
python -B -W error -m unittest discover -s tests -v
```

结果：数据核验 `passed=true`，20 项 NNUE 测试通过，71 项冻结引擎测试通过；两组测试均使用 `-W error`，无被忽略的警告。

最终真实权重额外检查：

- 在全部 3,166 条 validation 盘面上复算 NumPy／重建 PyTorch 残差，最大绝对差 `0.000274658203125`，平均绝对差约 `0.0000508693`；
- 在全部 1,200 个400题候选子盘面上复算，最大绝对差同为 `0.000274658203125`，平均绝对差约 `0.0000517971`；
- 两组均小于预先采用的 `0.0005` float32 容差，没有为通过检查修改模型结果；
- 400 题静态分数逐候选由冻结原生评估器复算，最大差为 `0.0`；
- NPZ 只含网络参数、`feature_layout`、`output_activation`、C 和 K，不含 `side_calib`、中心权重或训练损失；
- `ranking_400.json` 明确记录 `training_only_side_calibration_applied=false`、所有子局面选择 D 头、题目未用于训练或选模；
- 最终权重哈希与 `validation_metrics.json`、`ranking_400.json` 和 `final_equivalence.json` 一致；
- 根目录 Team19 正式引擎文件相对基线无差异。

对应日志和机器可读结果均收入真实实验产物压缩包。最终候选未因这次复验再训练或改变任何权重。

## 8. 复现命令

在已具备原始数据、`questions_400.json`、NumPy、Numba 和 PyTorch 的仓库根目录运行。训练全部使用 CPU 单线程。

### 8.1 点wise源模型

```cmd
python -B -m tools.nnue.train_nnue --tag volume1200
```

### 8.2 生成 768 道 train-only 列表题

```cmd
python -B -m tools.nnue.dev_top1 --tag volume1200 --count 768 --root-depth 2 --label-depth 4 --generate-only --weights "tools/nnue/results/volume1200/member5-c750-k6000-l0p5-tanh-precalib5-center1e3-seed20261126/best_weights.npz" --output "tools/nnue/results/volume1200/member5-listwise-train768-seed20261127"
```

`--generate-only` 不读取权重做选择，但当前命令行仍要求提供 `--weights` 参数。输出目录必须为空；需要复现实验时应换一个新的 `--output`，不要覆盖已封存证据。

### 8.3 列表微调

```cmd
python -B -m tools.nnue.train_listwise --tag volume1200 --questions "tools/nnue/results/volume1200/member5-listwise-train768-seed20261127/development_questions.json" --weights "tools/nnue/results/volume1200/member5-c750-k6000-l0p5-tanh-precalib5-center1e3-seed20261126/best_weights.npz" --source-validation "tools/nnue/results/volume1200/member5-c750-k6000-l0p5-tanh-precalib5-center1e3-seed20261126/validation_metrics.json" --output "tools/nnue/results/volume1200/member5-listwise-reproduction"
```

### 8.4 对冻结权重独立复验

```cmd
python -B -m tools.nnue.rank_questions --tag volume1200 --weights "tools/nnue/results/volume1200/member5-c750-k6000-l0p5-tanh-precalib5-center1e3-listwise20-seed20261127/best_weights.npz"
```

此命令只用于独立复验，不得根据输出继续调整同一候选后把结果仍称为一次最终评测。gate 失败时程序返回退出码 2；本次最终候选返回 0。

## 9. 补丁选择与 GitHub 更新

交付补丁有两种互斥用法：

1. **本机已应用上一版“预校准与零中心”代码**：只应用 `成员5_NNUE_Top1列表微调增量.patch.txt`；
2. **全新的干净 `QLxie@5e4f57a` 克隆**：只应用 `成员5_NNUE训练与排序代码.patch.txt`，该完整补丁已包含全部 6 个源码文件。

不要把完整补丁和增量补丁连续应用，也不要重复应用已经存在的补丁。实验 ZIP 用于传递真实日志、开发题和最终权重，不属于源码补丁，不要强行加入 Git。

### 9.1 从上一版预校准代码更新

在 CMD 中执行：

```cmd
cd /d "你的仓库路径"
git switch member5-nnue
git status --short
git apply --check "完整路径\成员5_NNUE_Top1列表微调增量.patch.txt"
git apply "完整路径\成员5_NNUE_Top1列表微调增量.patch.txt"
copy /Y "完整路径\成员5_NNUE实施与实验交接.md" "tools\nnue\MEMBER5_RESULTS.md"
```

若本机还停留在更早的 `side_calib` 版本，先按上一版交接应用 `成员5_NNUE预校准与零中心增量.patch.txt`，再应用本次 Top-1 增量；若无法确认本地补丁状态，优先备份工作区后使用干净克隆和完整补丁，不要强行叠加。

### 9.2 测试、暂存和提交

```cmd
python -B -m tools.nnue.verify_artifacts --tag volume1200 --questions questions_400.json
python -B -W error -m unittest tools.nnue.test_pipeline tools.nnue.test_training -v
python -B -W error -m unittest discover -s tests -v

git add -- "tools/nnue/model.py" "tools/nnue/train_nnue.py" "tools/nnue/rank_questions.py" "tools/nnue/test_training.py" "tools/nnue/dev_top1.py" "tools/nnue/train_listwise.py" "tools/nnue/MEMBER5_RESULTS.md"
git diff --cached --check
git diff --cached --name-only
git status --short
```

应只看到上述 6 个源码文件和结果报告。确认没有原始 NPZ、400 题、results、权重、缓存或临时文件后：

```cmd
git commit -m "feat(nnue): add train-only listwise top-1 fine-tuning"
git -c http.version=HTTP/1.1 push origin member5-nnue
```

不要改写已经推送的历史。Token 相当于密码，不要发到聊天中、写入代码或直接放进远程 URL。

## 10. 后续协作与上线边界

- 成员 2：审查列表标签是否确为 train-only、按游戏切分、回放与中心约束、导出隔离，以及最终一次评测证据。
- 成员 4：仍建议提供行级 depth 或异常行标识；不要用 NPZ 中不存在的 depth 信息做额外断言。
- 成员 1：可以开始审查该**离线合格候选**的上线接入，但必须继续遵守冻结 v4、NumPy＋Numba、纯 CPU 单进程、无后台线程、运行期零磁盘写入和 Numba 缓存等约束。
- 在线接入后仍须分别证明：零残差固定深度 100% 复现原版、增量与全量一致、NPS 至少为 v4 的 85% 且绝不触碰 65% 硬淘汰线，再完成 60 秒实战和独立盲评。
- 严禁开启任何后台进程在对手回合偷算，违者直接取消比赛资格（DQ）。
- 不再围绕同一 400 题调参。Top-1 只多 1 题，如在线结果不稳定，应如实保留负结果，而不是重新刷题寻找另一个“通过”。

## 11. 英文报告替换段落

**Train-only listwise fine-tuning and final ranking evaluation.** The pre-calibrated bounded-residual model still failed Top-1, although its pairwise discrimination had improved. We therefore avoided any further tuning on the sealed questions and constructed 768 three-candidate development questions exclusively from the training NPZ, using fixed-depth frozen-v4 searches at root depth 2 and label depth 4. A preceding residual-strength study on 192 train-only questions found no scale that strictly improved both development metrics, so no scaled candidate was evaluated on the sealed set. For listwise fine-tuning, whole games were split into 576 training questions and 192 development questions using a stable SHA-256 ordering. The objective combined a three-candidate cross-entropy between softmax-normalised static-plus-residual scores and deep-search labels with random replay of the original D4 pointwise SmoothL1 loss and the same pre-tanh centring penalty. We used temperature 1000, replay weight 1, AdamW at learning rate 0.0001, batch size 128 and 20 epochs. An epoch was eligible only if both development Top-1 and Pairwise strictly exceeded their static baselines; selection then maximised Top-1 followed by Pairwise. Epoch 20 was frozen before any sealed evaluation. On its development set, Top-1 improved from 46/176 to 51/176 and Pairwise from 229.0/512 to 241.0/512, while original validation NLL slightly worsened from 0.5738795 to 0.5751188.

**Final sealed result.** The frozen candidate was evaluated once on the 400-question set. Excluding the 23 separately reported top-two ties, Top-1 increased from the static baseline's 132/377 (35.0133%) to 133/377 (35.2785%), and Pairwise increased from 521.5/1065 (48.9671%) to 537.0/1065 (50.4225%). Both strict-improvement conditions therefore passed, although the Top-1 margin was only one question. On the informational tie subset, Top-1 was 11/23 and Pairwise was 15.0/18. Across all 1,200 candidate positions, residuals ranged from -738.48 to +728.56, with 609 negative and 591 positive values; every question had a non-zero within-question residual range, 57 orderings changed, and four candidates were within two percent of a bound. Side calibration remained absent from both the exported weights and ranking calculation. Full NumPy-versus-reconstructed-PyTorch checks over 3,166 validation positions and all 1,200 question candidates gave a maximum absolute residual discrepancy of 0.00027466. Artifact verification, all 20 NNUE tests and all 71 frozen-engine regressions passed. These results establish a narrow offline ranking-gate pass, not yet an online playing-strength improvement: zero-residual reproduction, incremental consistency, NPS and full-match checks remain mandatory before release.
