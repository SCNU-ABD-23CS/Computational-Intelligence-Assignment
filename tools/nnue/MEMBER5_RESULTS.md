# 成员 5：NNUE 实施、真实实验与交接结论

日期：2026-11-26
代码基线：`main@34c0a7c5bba98af083fe1d6d1fe2756a47c710aa`
工具基线：`QLxie@5e4f57a9cd6441773c2549e98564b66996f6cd5a`

## 1. 当前结论

已经完成新版成员 5 的分支核对、数据核验、训练器、纯 NumPy 导出复算、400 题评测和两轮真实训练。

**30 轮与 60 轮候选均未通过 400 题硬门槛，因此当前 NNUE 不得合入正式对局版本。** 按交接指南，本轮不创建 `Team19_nnue.py`，不修改搜索引擎，不进行 NPS 或 60 秒实战测试，也不交成员 1 做上线盲评。

这不是代码运行失败：数据核验、训练、权重导出和排序评测均已实际跑通；失败的是模型效果门槛。

## 2. 分支进展与协作关系

- 最新 `main` 是正式引擎基线。
- `QLxie` 相对该 `main` 只新增 `tools/nnue/` 的 9 个数据脚本，没有改动根目录正式引擎。
- `QLxie` 当前没有训练器、400 题评测器或在线推理模块。
- 本轮在 `QLxie` 上创建 `member5-nnue`，只新增 4 个离线文件：

| 文件 | 用途 |
|---|---|
| `tools/nnue/model.py` | 三平面编码、训练集 D4、标签掩码、NPZ 权重读取、NumPy 推理 |
| `tools/nnue/train_nnue.py` | 243→32→2 双头残差训练、验证 NLL 早停、配置与指纹、权重导出 |
| `tools/nnue/rank_questions.py` | 400 题静态基线与 NNUE 的 Top-1／Pairwise 对比及硬门槛 |
| `tools/nnue/test_training.py` | 编码、D4、标签、双头视角和导出一致性回归测试 |

根目录 `Team19*.py` 和 `Team19_weights.py` 均未修改。旧 `member5-ga` 仅保留为历史实验，不要把它覆盖到新版基线上。

## 3. 数据核验回执

已运行：

```powershell
python -B -m tools.nnue.verify_artifacts --tag volume1200 --questions questions_400.json
```

结果为 `passed=true`：17,862 条训练样本、3,166 条验证样本、400 道题全部通过字段、形状、dtype、编码、跨集隔离和合法性检查。原 9 项数据管线测试也全部通过。

数据指纹：

- train：`675ed46a1d1c2fa24fb699a289ecb4942fcf887a9fd4b5265dcf7588655883b4`
- validation：`fbec32dff5b61e2178103eb0f885c218565039c4d5ce9a7e2eaf296336743af8`
- questions：`2121c715dfbcb3d7e72b227d4d325b7b8d5b86703f4c1c847c1da88c772058eb`

需要反馈给成员 4 的三点：

1. 文档所写攻守约 66%／34% 与实际不符。实际 train 为 9,127／8,735（约 51.1%／48.9%），validation 为 1,705／1,461（约 53.9%／46.1%）。
2. NPZ 没有 `depth` 字段，无法精确识别 manifest 所述 8 条 `depth=64` 行。实际 `abs(deep_eval)==3000` 有 train 254 条、validation 30 条，远大于 8 条，不能声称这 284 条全是已知异常。
3. `collection_summary.json` 的整体采集 `depth_ok=false`、平均深度约 4.2696，与训练／验证子集平均深度约 5 并不矛盾，但建议交接文档明确二者统计范围。

原始 NPZ 没有被修改。本轮采用 HANDOFF 允许的保守办法：所有绝杀分和 `abs(deep_eval)==3000` 行仍参与 outcome 学习，但深搜回归项被屏蔽。由此 train 有 785＋254＝1,039 条 outcome-only，validation 有 172＋30＝202 条 outcome-only。若成员 4 后续提供行级 `depth` 或异常掩码，再改成精确的 8 行过滤并重训。

## 4. 本地更新与应用补丁

以下命令用于已有 Git 仓库的 Windows PowerShell。先确认旧分支改动已经提交；不要直接在 `main` 上应用。

```powershell
git fetch origin main QLxie
git switch -c member5-nnue origin/QLxie

git apply --check "完整路径\成员5_NNUE训练与排序代码.patch.txt"
git apply "完整路径\成员5_NNUE训练与排序代码.patch.txt"
git status --short
```

如果本地已经存在 `member5-nnue`，使用 `git switch member5-nnue`，不要重复执行 `switch -c`。

把六个数据文件放到以下位置，`results` 被 Git 忽略是正常现象：

```text
tools/nnue/results/volume1200/
├── team19_nnue_train.npz
├── team19_nnue_val.npz
├── dataset_manifest.json
├── collection_summary.json
├── verification.json
└── questions_400.json
```

## 5. 完整复现命令

在仓库根目录运行。

```powershell
# 数据契约与泄漏检查
python -B -m tools.nnue.verify_artifacts --tag volume1200 --questions questions_400.json

# 快速回归：数据管线 9 项 + 本轮训练契约 5 项
python -B -m unittest tools.nnue.test_pipeline tools.nnue.test_training -v

# 首版 30 轮
python -B -m tools.nnue.train_nnue --tag volume1200

# 首版 400 题；门槛失败时退出码为 2，表示效果拒绝而非程序崩溃
python -B -m tools.nnue.rank_questions --tag volume1200 `
  --weights tools/nnue/results/volume1200/member5-c750-k6000-l0p5-seed20261126/best_weights.npz

# 因第 30 轮验证 NLL 仍在下降而进行的 60 轮复验
python -B -m tools.nnue.train_nnue --tag volume1200 --epochs 60 `
  --run-id c750-k6000-l0p5-e60-seed20261126

python -B -m tools.nnue.rank_questions --tag volume1200 `
  --weights tools/nnue/results/volume1200/member5-c750-k6000-l0p5-e60-seed20261126/best_weights.npz

# 确认没有破坏冻结引擎
python -B -m unittest discover -s tests -v
```

每个训练目录包含 `history.jsonl`、`training_config.json`、`validation_metrics.json`、`best_weights.npz` 和运行排序后的 `ranking_400.json`。400 题文件从不传给训练器，模型选择只看验证集 NLL。

## 6. 真实训练与排序结果

固定配置：CPU 单线程、seed 20261126、Clipped-ReLU `[0,1]`、两个攻方视角输出头、`C=750`、`K=6000`、`lambda=0.5`、SmoothL1、AdamW（lr `1e-3`、weight decay `1e-4`）、batch 1024。训练集 D4 后为 142,896 条；验证集和 400 题均未增强；未使用类别权重或重采样。

| 指标 | 30 轮 | 60 轮 |
|---|---:|---:|
| 最佳轮 | 30 | 60 |
| 验证 SmoothL1 | 0.0388399 | 0.0382896 |
| 验证 NLL | 0.6102532 | 0.6076778 |
| 验证平均绝对残差 | 134.787 | 263.817 |
| 权重 SHA-256 前缀 | `1922460b0791` | `560424298de5` |

主指标排除 23 道 `tie_top2` 题，使用其余 377 道；三对候选中深搜标签完全相同的 pair 也排除，最终比较 1,065 对。预测分完全相同的 pair 记半分，Top-1 使用固定候选顺序破同分。

| 主指标 | 静态基线 | 30 轮 NNUE | 60 轮 NNUE |
|---|---:|---:|---:|
| Top-1 | 132/377＝35.0133% | 131/377＝34.7480% | 131/377＝34.7480% |
| Pairwise | 521.5/1065＝48.9671% | 521.0/1065＝48.9202% | 521.0/1065＝48.9202% |
| 相对静态 Top-1 | — | -1 题 | -1 题 |
| 相对静态 Pairwise | — | -0.5 计分对 | -0.5 计分对 |

23 道 tie 题只作单列信息：两种模型的 Top-1 均为 10/23，Pairwise 均为 14/18，不计入主门槛。

60 轮虽然降低了验证 NLL，却没有改变任何一题的 NNUE 候选顺序。两轮在 1,200 个候选局面上的残差几乎全部为负；30 轮范围约 `[-140.68, -4.14]`，60 轮约 `[-276.86, -8.27]`。这表明当前学习主要体现为整体向下校准，而不是改善同局候选之间的相对排序。

**放行判定：两轮均为 `gate.passed=false`。不得用验证 NLL 的改善替代 400 题排序改善。**

## 7. 下一步分工

### 成员 5

保留代码、训练日志和负结果。不要继续针对这 400 题反复调参直到“碰巧通过”。若继续研究，应优先取得更精确的标签／行级异常标识，或新增独立题目作最终确认；任何新候选仍须先过同一排序门。

### 成员 4

回复上述字段文档不一致，并优先补充每条样本的 `depth` 或 8 条异常行的稳定行标识。若更新数据，应生成新 tag 和新指纹，不覆盖旧 NPZ。

### 成员 1

当前没有合格模型，不启动同库 60 秒盲评、离库分支测试或 NPS 接收。避免浪费比赛机时。

### 成员 2

可审查离线训练与评测 PR，但不应批准 `Team19_nnue.py` 上线，因为本轮没有创建该文件，且排序门槛未通过。`QLxie` 未合入 `main` 前，本分支 PR 可先以 `QLxie` 为目标；数据工具合入后再 rebase 到最新 `main`。

## 8. GitHub 提交命令

```powershell
git add tools/nnue/model.py tools/nnue/train_nnue.py tools/nnue/rank_questions.py tools/nnue/test_training.py
git diff --cached --check
git diff --cached --stat
git commit -m "feat(nnue): add residual training and sealed ranking gate"
git push -u origin member5-nnue
```

提交前可用下面的命令确认没有改到根目录正式模块：

```powershell
git diff --cached --name-only
```

预期只列出本节命令中的 4 个 `tools/nnue/` 文件。实验结果目录被 `.gitignore` 排除，不要把大型 NPZ 数据或本地权重直接提交到仓库；通过团队数据包或 release 附件交接。

## 9. 可直接放入英文报告的段落

**Residual NNUE experiment and rejection.** Member 5 implemented the complete offline pipeline for a 243–32–2 residual NNUE on the frozen v4 evaluator. The input used attacker, defender and king planes, with empty squares and the empty throne mapped to zero in all three planes. D4 augmentation was applied only to the 17,862 training positions, producing 142,896 examples; neither the 3,166 validation positions nor the 400 sealed questions were augmented. No class weighting or resampling was used. Mate scores and scores with an absolute value of 3,000 retained their game-outcome target but were excluded from the deep-search regression term. This conservative policy was necessary because the supplied arrays did not contain row-level depth values, so the eight documented depth-64 anomalies could not be isolated reliably. Two attacker-perspective output heads were trained with SmoothL1 loss and AdamW, and model selection used validation negative log-likelihood only. The initial 30-epoch model reached a validation NLL of 0.610253. Since validation was still improving, training was extended to 60 epochs, reaching 0.607678. Nevertheless, both models scored 131/377 Top-1 and 521.0/1065 Pairwise on the non-tied sealed questions, compared with 132/377 and 521.5/1065 for the static baseline. The 23 top-two-tied questions were reported separately. Because neither candidate strictly improved both ranking metrics, the NNUE was rejected before online Numba integration, NPS testing or 60-second matches. This prevented a calibration-only improvement from being misreported as a playing-strength gain.
