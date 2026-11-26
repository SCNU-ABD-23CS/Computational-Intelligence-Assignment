# 成员 5：NNUE 尺度修复、攻守校准、真实复验与交接结论

日期：2026-11-26  
代码基线：`main@34c0a7c5bba98af083fe1d6d1fe2756a47c710aa`  
工具基线：`QLxie@5e4f57a9cd6441773c2549e98564b66996f6cd5a`

## 1. 当前结论

本轮已按组长反馈完成两项修正及其实验复核：

1. 修复零初始化输出头直接学习分数域残差所造成的对称增长、隐藏饱和和输出尺度问题；
2. 在 `tanh` 残差之外加入攻守行棋方的训练期 logit 校准，使全局攻守偏移不必全部占用有界残差额度。

最终自检结果：

- 16 项 NNUE 数据／训练测试全部通过；
- 71 项冻结引擎测试全部通过；
- 官方 `verify_artifacts` 数据核验通过；
- side calibration 的公式、D／A 索引、梯度、最佳轮次保存和导出隔离均有回归测试；
- `C=750 tanh + side_calib` 真实训练完成 30 轮，按验证 NLL 选择第 27 轮；
- 校准项学到了明显数值，但 400 题 Top-1 和 Pairwise **都低于静态基线**；
- 残差仍贴近 `±750`，同题残差极差中位数仅 `0.010376`，没有出现预期的明显拉开。

**因此当前候选仍不得上线。** 不创建 `Team19_nnue.py`，不修改正式搜索引擎，不开展 NPS、60 秒实战、离库分支或成员 1 上线盲评。本轮不再围绕同一 400 题继续调参，保留代码、训练日志和真实负结果供组内审查。

## 2. 数据与分支复核

重新上传的 train、validation、questions、manifest、verification、collection summary、HANDOFF 和成员 5 说明文件，与上一轮文件逐字节一致。原始 NPZ 未修改。

数据指纹：

- train：`675ed46a1d1c2fa24fb699a289ecb4942fcf887a9fd4b5265dcf7588655883b4`
- validation：`fbec32dff5b61e2178103eb0f885c218565039c4d5ce9a7e2eaf296336743af8`
- questions：`2121c715dfbcb3d7e72b227d4d325b7b8d5b86703f4c1c847c1da88c772058eb`

官方 `verify_artifacts` 得到 `passed=true`。训练集 17,862 条、验证集 3,166 条、400 道题，跨集 `game_id` 和 D4 隔离保持有效。

仍采用原保守标签策略：`abs(deep_eval)>=90000` 和 `abs(deep_eval)==3000` 的行保留 outcome，但不参与深搜回归。train 有 785＋254＝1,039 条 outcome-only，validation 有 172＋30＝202 条。NPZ 没有行级 depth，仍不能声称所有 `abs==3000` 行都是那 8 条 `depth=64` 异常。

400 题继续只用于训练完成后的封存评测，不参与训练、D4、早停、选轮次或参数搜索。23 道 `tie_top2` 单列，主指标仍为其余 377 题和 1,065 个非同分标签候选对。

## 3. 代码改动与数学契约

只修改成员 5 的 4 个离线文件；根目录 Team19 正式引擎文件不变。

| 文件 | 当前内容 |
|---|---|
| `tools/nnue/train_nnue.py` | 小随机输出头、`C*tanh(head)`、攻守 logit 校准、导出缩放、校准日志和配置记录 |
| `tools/nnue/model.py` | 三平面编码、D4、标签掩码、导出权重的 tanh NumPy 推理和旧 clip 权重兼容 |
| `tools/nnue/rank_questions.py` | 静态基线对照、Top-1／Pairwise 门槛、残差极差和隐藏饱和诊断；明确不应用校准项 |
| `tools/nnue/test_training.py` | 初始化、标签、校准公式与梯度、导出隔离、PyTorch／NumPy 一致性等回归测试 |

### 3.1 有界残差

训练态计算：

```text
hidden = ClippedReLU(Linear1(features), 0, 1)
raw = Linear2(hidden)
residual = C * tanh(raw[side])
```

两个输出头都由 `side` 选择，但都输出**攻方视角**残差。输出头采用 `Normal(0, 1e-3)` 小随机权重和零偏置，避免全零输出头造成的对称同步增长。

导出时，两个输出头的权重和偏置都乘以 `C`。部署态使用 score-domain 权重：

```text
raw_exported = hidden @ (C * head_weight).T + C * head_bias
residual = C * tanh(raw_exported / C)
```

这与训练态等价。没有 `output_activation` 字段的旧权重仍按原 hard-clip 格式读取；新权重明确记录 `output_activation=tanh`。

### 3.2 攻守 logit 校准

新增参数：

```python
self.side_calib = nn.Parameter(torch.zeros(2))
```

训练和验证时的概率预测严格为：

```text
prediction = sigmoid((base_eval + residual) / K + side_calib[side])
```

索引固定为 `side_calib[0]` 对应 D，`side_calib[1]` 对应 A。校准项位于除以 `K` 后的 logit 空间，不是直接加到分数域残差上。

隔离约束：

- 参与离线训练和验证指标；
- 包含在完整 `state_dict` 中，因此最佳验证轮次恢复时也会恢复对应校准值；
- 不写入 `best_weights.npz`；
- 不进入 NumPy 残差推理；
- 不进入 `rank_questions` 的 400 题打分。

每轮日志和最终配置同时记录 logit 及 `logit*K` 的尺度换算。后者仅用于观察校准量级，不应解释为一个实际导出并叠加到引擎分数上的残差。

## 4. 旧问题复核

旧 60 轮模型实际检查结果：

- 守方头权重约为 -8.48～-8.36，攻方头约为 +8.32～+8.43；
- 400 题中 391 题的三个候选残差完全相同；
- 只有 2 题改变了静态排序；
- 旧 30／60 轮 Top-1 都是 131/377，Pairwise 都是 521.0/1065。

因此旧模型确实主要学习了整体校准，不能有效区分同题候选。旧结果不能用于判断修复后残差学习的真实效果。

## 5. 全部真实实验

固定配置：CPU 单线程、seed 20261126、32 隐藏单元、SmoothL1、AdamW（lr `1e-3`、weight decay `1e-4`）、batch 1024、30 轮上限、验证 NLL 早停、`K=6000`、`lambda=0.5`。训练集 D4 后 142,896 条；验证集和 400 题不增强；不使用类别权重或重采样。旧零初始化 60 轮实验是唯一的 60 轮历史对照。

静态基线在 377 道非 `tie_top2` 主指标题上为：

- Top-1：`132/377 = 35.0133%`
- Pairwise：`521.5/1065 = 48.9671%`

| 候选 | 最佳轮 | 验证 NLL | NNUE Top-1 | NNUE Pairwise | 非零残差极差题 | 改序题 | 放行 |
|---|---:|---:|---:|---:|---:|---:|---|
| 旧零初始化，30 轮 | 30 | 0.6102532 | 131/377 | 521.0/1065 | 9/400 | 2 | 否 |
| 旧零初始化，60 轮 | 60 | 0.6076778 | 131/377 | 521.0/1065 | 9/400 | 2 | 否 |
| `C=750`，线性缩放＋hard clip | 17 | 0.5984247 | 132/377 | 521.5/1065 | 0/400 | 0 | 否 |
| `C=750 tanh` | 17 | 0.5983087 | 132/377 | 523.0/1065 | 389/400 | 14 | 否 |
| `C=1500 tanh` | 17 | 0.5863801 | 131/377 | 525.0/1065 | 399/400 | 8 | 否 |
| `C=750 tanh + side_calib` | 27 | 0.5755986 | 129/377 | 520.0/1065 | 400/400 | 18 | 否 |

指标解释：

1. 线性缩放消除了原先隐藏层全单元同步锁死，但所有候选残差都硬截断到 `-750`，Top-1 和 Pairwise 与静态基线完全相同。
2. `C=750 tanh` 消除了硬截断零梯度区。Top-1 与静态基线持平，Pairwise 增加 1.5 个计分对，达到 `49.1080%`，但“两个指标都严格提高”的门槛仍未通过。
3. `C=1500 tanh` 的 Pairwise 达到 `49.2958%`，比静态基线多 3.5 个计分对，但 Top-1 比静态基线少 1 题，因此仍未通过。
4. 最新 side-calibrated 候选的验证 NLL 降至 `0.5755986149`，但主指标 Top-1 比静态少 3 题，Pairwise 少 1.5 个计分对。验证损失改善不能替代排序放行门槛。
5. 23 道 `tie_top2` 题继续单列。最新候选在这组信息性题目上的 Top-1 为 9/23、Pairwise 为 14/18，不纳入主指标，也不能用于抵消主指标下降。

权重指纹：

- 线性缩放 C750：`b84ca91d27aec7c253984c7a1bdd6159ba9c185f3610487d9125f73830762b5c`
- tanh C750：`de71e8cb8c7cd2d8afec5c19ed8e92a7b26452d7d88d1d69ddce838ec89736c6`
- tanh C1500：`04b8c4f355d5d85c8de3b8127fe754fa2b8a4dd2619ce62f190bdf10b9f56de9`
- tanh C750＋side_calib：`f7c2ef3d2371e7d695120b685fb4bcc3f1c86c4a209c1981d5cd098d993a3a31`

## 6. 最新 side_calib 实验详录

运行目录：

```text
tools/nnue/results/volume1200/member5-c750-k6000-l0p5-tanh-sidecalib-seed20261126/
```

训练完成 30 轮，最佳第 27 轮。最佳验证指标：

| 指标 | 数值 |
|---|---:|
| SmoothL1 | 0.0316389859 |
| NLL | 0.5755986149 |
| 平均绝对残差 | 703.7999052 |

最佳轮次的校准参数：

| 行棋方 | logit | logit×K 尺度换算 |
|---|---:|---:|
| D | -0.4852930009 | -2911.7581 |
| A | +0.3683262169 | +2209.9573 |

这证明校准项不是零初始化后停留在零点，而是实际学到了组长预期的攻守全局偏移。不过，400 题诊断并未显示候选区分度得到预期改善：

| 诊断 | 数值 |
|---|---:|
| 400 题平均绝对残差 | 749.8917847 |
| 非零同题残差极差 | 400/400 |
| 同题残差极差中位数 | 0.01037598 |
| 同题残差极差 p90 | 0.13937378 |
| 同题残差极差最大值 | 5.23480225 |
| 实际改序题数 | 18 |
| 32 个隐藏单元全部为 1 的候选局面 | 0 |
| 32 个隐藏单元全部为 0 的候选局面 | 0 |

“400/400 非零”只表示浮点输出不再完全相同，不表示差异具有实际排序价值。中位极差约 0.0104，且平均绝对残差仍接近上界 750，说明本轮没有解决残差贴边和同题区分度过小的问题。

主指标结果：

- Top-1：`129/377 = 34.2175%`，静态为 132/377，减少 3 题；
- Pairwise：`520.0/1065 = 48.8263%`，静态为 521.5/1065，减少 1.5 个计分对；
- `top1_strictly_better=false`；
- `pairwise_strictly_better=false`；
- `gate.passed=false`。

因此本轮结论是：**校准实现和学习行为正确，但该候选的真实排序效果变差，不得上线。**

## 7. 自检范围

最终回归命令：

```cmd
python -B -m tools.nnue.verify_artifacts --tag volume1200 --questions questions_400.json
python -B -m unittest tools.nnue.test_pipeline tools.nnue.test_training -v
python -B -m unittest discover -s tests -v
```

结果分别为数据核验通过、16 项通过、71 项通过。

额外检查：

- side_calib 的 D／A 索引和 logit 位置由专门测试验证；
- 两个校准参数都能收到非零梯度；
- 导出文件不包含 `side_calib`；
- 即使训练态设置非零校准，导出的 NumPy 残差仍与 PyTorch 残差一致；
- 400 题报告明确记录 `training_only_side_calibration_applied=false`；
- 完整补丁只包含上述 4 个离线源码文件，不包含正式引擎、原始数据或实验权重；
- 增量补丁和完整补丁均已在干净临时副本中执行 `git apply --check`、实际应用和逐文件哈希比对。

## 8. 复现命令

在 `member5-nnue` 分支仓库根目录运行。以下是 CMD 命令。

```cmd
python -B -m tools.nnue.train_nnue --tag volume1200
python -B -m tools.nnue.rank_questions --tag volume1200 --weights "tools/nnue/results/volume1200/member5-c750-k6000-l0p5-tanh-sidecalib-seed20261126/best_weights.npz"
```

默认 run id 已改为 `c750-k6000-l0p5-tanh-sidecalib-seed20261126`。若该输出目录已有文件，不要直接加 `--overwrite` 覆盖证据；应换一个新的 `--run-id`。

`rank_questions` 在门槛失败时返回退出码 2，这是预期的效果拒绝，不是程序异常。每个训练目录保存训练日志、配置、验证指标、权重和 400 题逐题报告。

## 9. 更新现有 GitHub 分支

本地已经提交 `9c21809` 的工作区，使用 `成员5_NNUE侧别校准增量.patch.txt`。全新的 `QLxie@5e4f57a` 克隆才使用完整 `成员5_NNUE训练与排序代码.patch.txt`，不要把两个补丁连续应用。

已有 `9c21809` 工作区在 CMD 中执行：

```cmd
cd /d "你的仓库路径"
git switch member5-nnue
git status
git log --oneline -3
git apply --check "完整路径\成员5_NNUE侧别校准增量.patch.txt"
git apply "完整路径\成员5_NNUE侧别校准增量.patch.txt"
copy /Y "完整路径\成员5_NNUE实施与实验交接.md" "tools\nnue\MEMBER5_RESULTS.md"
```

然后运行测试并检查：

```cmd
python -B -m tools.nnue.verify_artifacts --tag volume1200 --questions questions_400.json
python -B -m unittest tools.nnue.test_pipeline tools.nnue.test_training -v
python -B -m unittest discover -s tests -v
git add -- "tools/nnue/train_nnue.py" "tools/nnue/rank_questions.py" "tools/nnue/test_training.py" "tools/nnue/MEMBER5_RESULTS.md"
git diff --cached --check
git diff --cached --name-only
```

本次 `model.py` 无需变化，所以增量补丁只修改 3 个源码文件；第 4 个待提交文件是结果报告。确认暂存内容没有原始 NPZ、questions、results 或权重后，创建新提交：

```cmd
git commit -m "fix(nnue): separate side calibration from bounded residual"
git -c http.version=HTTP/1.1 push origin member5-nnue
```

不要改写已有提交。如果 `9c21809` 因先前网络故障尚未推送，这次推送会把它和新提交一起发送；无需重复提交旧改动。Token 不要发到聊天中、写入代码或直接放进远程 URL。

PR 中应明确：本次是训练期校准隔离、真实实验和负结果交接，不是上线候选。原始数据、results 目录、权重包和实验 ZIP 不提交到 Git 仓库。

## 10. 后续协作

- 成员 2：审查训练／导出数学等价性、校准隔离、诊断和真实结果。
- 成员 4：仍建议提供行级 depth 或 8 条异常行标识，并修正文档中的攻守比例。
- 成员 1：当前无合格上线模型，不启动 NPS、60 秒或离库盲评。
- 成员 5：停止围绕同一 400 题反复调参。若继续，应优先改善标签、分析饱和成因或新增独立最终验证题，再考虑新的模型实验。

## 11. 英文报告替换段落

**Side-calibrated residual NNUE and sealed evaluation.** The initial residual NNUE was numerically ill-conditioned: zero-initialised output heads learned score-domain corrections directly, producing nearly symmetric head growth and saturated hidden units. We therefore introduced small asymmetric head initialisation and a dimensionless output transformed by a smooth bounded residual, with export-time scaling and a direct PyTorch-versus-NumPy equivalence test. Following further review, we added a two-parameter side calibration in logit space, using `sigmoid((base_eval + residual)/K + side_calib[side])` for offline training and validation only. The calibration is restored with the best validation checkpoint but is neither exported nor used by the sealed move-ranking evaluation. In the C=750 tanh experiment, the best epoch was 27 and validation NLL decreased to 0.5755986. The learned defender and attacker calibrations were -0.485293 and +0.368326 logits respectively, confirming that the model learned substantial side-specific global offsets. Nevertheless, the mean absolute residual on the 400 sealed questions remained 749.89, and the median within-question residual range was only 0.010376. Top-1 decreased from the static baseline's 132/377 to 129/377, while Pairwise decreased from 521.5 to 520.0 out of 1065. Thus, neither release criterion was met. All 16 NNUE tests and all 71 frozen-engine regressions passed, but online Numba integration and match testing remained disabled. The result separates a correct calibration implementation from model effectiveness: the global offset was learned successfully, yet the present residual model did not improve sealed move ordering and is not releasable.
