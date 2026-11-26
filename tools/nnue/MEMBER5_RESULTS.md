# 成员 5：NNUE 尺度修复、真实复验与交接结论

日期：2026-11-26
代码基线：`main@34c0a7c5bba98af083fe1d6d1fe2756a47c710aa`
工具基线：`QLxie@5e4f57a9cd6441773c2549e98564b66996f6cd5a`

## 1. 当前结论

组长指出的数值尺度问题成立。旧训练器以全零输出头直接学习分数域残差，在 `lr=1e-3` 下造成输出头对称同步增长和隐藏单元饱和，旧 30／60 轮结果不能用来判断 NNUE 的真实效果。

本轮已完成：

- 小随机非对称输出头初始化；
- 输出尺度解耦和平滑 `tanh` 有界残差；
- 导出权重与训练态 PyTorch 输出的 NumPy 一致性检查；
- 隐藏单元饱和、同题残差极差、实际改序题数诊断；
- `C=750` 线性缩放、`C=750 tanh`、`C=1500 tanh` 三组真实复验；
- 15 项 NNUE 测试和 71 项冻结引擎测试。

**最终仍没有任何候选同时严格优于静态基线的 Top-1 和 Pairwise，因此当前 NNUE 不得上线。** 不创建 `Team19_nnue.py`，不修改正式搜索引擎，不开展 NPS、60 秒实战或成员 1 上线盲评。此次结论不再归因于“完全没训练”，而是：尺度修复后模型开始学习并改善 Pairwise，但 Top-1 仍未过门。

## 2. 数据与分支复核

重新上传的 train、validation、questions、manifest、verification、collection summary、HANDOFF 和成员 5 说明文件，与上一轮文件逐字节一致。原始 NPZ 未修改。

数据指纹：

- train：`675ed46a1d1c2fa24fb699a289ecb4942fcf887a9fd4b5265dcf7588655883b4`
- validation：`fbec32dff5b61e2178103eb0f885c218565039c4d5ce9a7e2eaf296336743af8`
- questions：`2121c715dfbcb3d7e72b227d4d325b7b8d5b86703f4c1c847c1da88c772058eb`

官方 `verify_artifacts` 再次得到 `passed=true`。训练集 17,862 条、验证集 3,166 条、400 道题，跨集 game_id 和 D4 隔离保持有效。

仍采用原保守标签策略：`abs(deep_eval)>=90000` 和 `abs(deep_eval)==3000` 的行保留 outcome，但不参与深搜回归。train 有 785＋254＝1,039 条 outcome-only，validation 有 172＋30＝202 条。NPZ 没有行级 depth，仍不能声称所有 `abs==3000` 行都是那 8 条 `depth=64` 异常。

## 3. 代码改动

只修改成员 5 的 4 个离线文件：

| 文件 | 本轮修正 |
|---|---|
| `tools/nnue/train_nnue.py` | 小随机输出头、无量纲输出、`C*tanh(head)`、导出缩放、参数化记录 |
| `tools/nnue/model.py` | 导出权重的 tanh 推理、旧 clip 权重兼容、隐藏激活诊断接口 |
| `tools/nnue/rank_questions.py` | 饱和率、残差极差、非零差异题数、实际改序题数 |
| `tools/nnue/test_training.py` | 初始化、导出缩放和 PyTorch／NumPy 一致性回归 |

训练态计算：

```text
raw = Linear2(ClippedReLU(Linear1(features)))
residual = C * tanh(raw)
```

导出时将两个输出头的权重和偏置都乘以 `C`。部署态使用导出后的 score-domain 权重：

```text
raw_exported = hidden @ (C * head_weight).T + C * head_bias
residual = C * tanh(raw_exported / C)
```

这与训练态等价，且保持原有纯 NumPy／Numba 的小矩阵推理形式。没有 `output_activation` 字段的旧权重按原 hard-clip 格式读取；新权重明确记录 `output_activation=tanh`。

400 题不参与训练、D4、早停或模型选择。所有候选仍按验证 NLL 选轮次；新增诊断只用于解释训练行为，不是放行指标。

## 4. 旧问题复核

旧 60 轮模型实际检查结果：

- 守方头权重约为 -8.48～-8.36，攻方头约为 +8.32～+8.43；
- 400 题中 391 题的三个候选残差完全相同；
- 只有 2 题改变了静态排序；
- 旧 30／60 轮 Top-1 都是 131/377，Pairwise 都是 521.0/1065。

因此旧模型确实主要学习了整体校准，不能区分同题候选。

## 5. 修复后的真实实验

固定配置：CPU 单线程、seed 20261126、32 隐藏单元、SmoothL1、AdamW（lr `1e-3`、weight decay `1e-4`）、batch 1024、30 轮上限、验证 NLL 早停、`K=6000`、`lambda=0.5`。训练集 D4 后 142,896 条；验证集和 400 题不增强；不使用类别权重或重采样。

静态基线在 377 道非 `tie_top2` 主指标题上为 Top-1 `132/377=35.0133%`，Pairwise `521.5/1065=48.9671%`。

| 候选 | 最佳轮 | 验证 NLL | NNUE Top-1 | NNUE Pairwise | 非零残差极差题 | 改序题 | 放行 |
|---|---:|---:|---:|---:|---:|---:|---|
| `C=750`，线性缩放＋hard clip | 17 | 0.5984247 | 132/377 | 521.5/1065 | 0/400 | 0 | 否 |
| `C=750 tanh` | 17 | 0.5983087 | 132/377 | 523.0/1065 | 389/400 | 14 | 否 |
| `C=1500 tanh` | 17 | 0.5863801 | 131/377 | 525.0/1065 | 399/400 | 8 | 否 |

更准确的指标解释：

1. 线性缩放消除了原先隐藏层全单元同步锁死，但所有候选残差都硬截断到 `-750`，Top-1 和 Pairwise 与静态基线完全相同。
2. `C=750 tanh` 消除了硬截断零梯度区。Top-1 与静态基线持平，Pairwise 增加 1.5 个计分对，达到 `49.1080%`，但“两个指标都严格提高”的门槛仍未通过。
3. `C=1500 tanh` 的 Pairwise 达到 `49.2958%`，比静态基线多 3.5 个计分对，但 Top-1 比静态基线少 1 题，因此仍未通过。
4. 23 道 `tie_top2` 题继续单列，不纳入主指标。不能因为 Pairwise 改善或验证 NLL 降低而放宽 Top-1 门槛。

权重指纹：

- 线性缩放 C750：`b84ca91d27aec7c253984c7a1bdd6159ba9c185f3610487d9125f73830762b5c`
- tanh C750：`de71e8cb8c7cd2d8afec5c19ed8e92a7b26452d7d88d1d69ddce838ec89736c6`
- tanh C1500：`04b8c4f355d5d85c8de3b8127fe754fa2b8a4dd2619ce62f190bdf10b9f56de9`

## 6. 复现命令

在 `member5-nnue` 分支仓库根目录运行。CMD 与 PowerShell 都可使用以下命令。

```text
python -B -m tools.nnue.verify_artifacts --tag volume1200 --questions questions_400.json
python -B -m unittest tools.nnue.test_pipeline tools.nnue.test_training -v

python -B -m tools.nnue.train_nnue --tag volume1200

python -B -m tools.nnue.rank_questions --tag volume1200 --weights tools/nnue/results/volume1200/member5-c750-k6000-l0p5-tanh-seed20261126/best_weights.npz

python -B -m tools.nnue.train_nnue --tag volume1200 --residual-clip 1500 --run-id c1500-k6000-l0p5-tanh-seed20261126

python -B -m tools.nnue.rank_questions --tag volume1200 --weights tools/nnue/results/volume1200/member5-c1500-k6000-l0p5-tanh-seed20261126/best_weights.npz

python -B -m unittest discover -s tests -v
```

`rank_questions` 在门槛失败时返回退出码 2，这是预期的效果拒绝，不是程序异常。每个训练目录保存训练日志、配置、验证指标、权重和 400 题逐题报告。

## 7. 更新现有 GitHub 分支

如果本地已经应用过上一版补丁，使用 `成员5_NNUE尺度修复增量.patch.txt`；全新 `QLxie` 克隆则使用完整 `成员5_NNUE训练与排序代码.patch.txt`。

已有旧工作区在 CMD 中执行：

```cmd
cd /d "你的仓库路径"
git switch member5-nnue
git apply --check "完整路径\成员5_NNUE尺度修复增量.patch.txt"
git apply "完整路径\成员5_NNUE尺度修复增量.patch.txt"
copy /Y "完整路径\成员5_NNUE实施与实验交接.md" "tools\nnue\MEMBER5_RESULTS.md"
```

然后运行测试并检查：

```cmd
python -B -m unittest tools.nnue.test_pipeline tools.nnue.test_training -v
python -B -m unittest discover -s tests -v
git add -- "tools/nnue/model.py" "tools/nnue/train_nnue.py" "tools/nnue/rank_questions.py" "tools/nnue/test_training.py" "tools/nnue/MEMBER5_RESULTS.md"
git diff --cached --check
git diff --cached --name-only
```

如果旧改动尚未提交，直接完成首次提交：

```cmd
git commit -m "feat(nnue): add residual training and sealed ranking gate"
git push -u origin member5-nnue
```

如果旧改动已经提交，则创建新的修复提交，不要改写已推送历史：

```cmd
git commit -m "fix(nnue): decouple residual scale and diagnose saturation"
git push
```

PR 中应明确：本次是训练器尺度修复和真实负结果复验，不是上线候选。原始数据、results 目录、权重包和实验 ZIP 不提交到 Git 仓库。

## 8. 后续协作

- 成员 2：审查训练／导出数学等价性、旧权重兼容、诊断和真实结果。
- 成员 4：仍建议提供行级 depth 或 8 条异常行标识，并修正文档中的攻守比例。
- 成员 1：当前无合格上线模型，不启动 NPS、60 秒或离库盲评。
- 成员 5：停止围绕同一 400 题反复调参。若继续，应优先改善标签或新增独立最终验证题，再考虑新的模型实验。

## 9. 英文报告替换段落

**Residual NNUE scale correction and evaluation.** The first residual NNUE experiment was found to be numerically ill-conditioned. Its zero-initialised output heads learned score-domain corrections directly, causing nearly symmetric head growth and saturated hidden units; 391 of the 400 sealed questions therefore received identical residuals for all three candidate moves. We corrected this implementation rather than treating the original failure as evidence against residual learning. The revised network uses small asymmetric output-head initialisation and a dimensionless output transformed by a smooth residual bound. During export, output-head weights and biases are converted back to the score domain, and NumPy inference is checked directly against the training-time PyTorch output. Data splits, D4 policy, outlier masks, optimisation settings and validation-NLL model selection were kept unchanged. A linear-scale hard-clipped control removed the original hidden-unit symmetry but clipped every sealed candidate residual to -750. The smooth C=750 model instead produced non-constant residuals on 389 questions and changed 14 orderings; its Pairwise score improved from 521.5 to 523.0 out of 1065, while Top-1 remained tied with the static baseline at 132/377. A guideline-sanctioned C=1500 ablation reached 525.0/1065 Pairwise but decreased Top-1 to 131/377. Consequently, none of the corrected candidates passed the requirement to improve both metrics strictly. Online Numba integration and match testing remained disabled. These results show that the scale defect was repaired and that residual learning can affect move ordering, but the present data and architecture have not yet demonstrated a releasable improvement.
