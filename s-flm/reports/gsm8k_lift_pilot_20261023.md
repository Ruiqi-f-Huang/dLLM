# MDLM 代码区域 LIFT 初筛（2026-10-23）

状态：A/B/C 训练、14 次完整 validation 生成和 checkpoint 审计均已完成。代码区域 LIFT-3 在本次 200 步预算下没有显示稳健解题收益。

## 假设与方法边界

检验模型相关的 token mask 选择，是否能比原始随机 MASK 或候选扩展后的随机恢复，改善 GSM8K 程序生成。此版本限制在非 docstring 代码＋EOS 区域，并保留 MDLM 原有损失，是 LIFT 的 MDLM 适配，不是原版全区域复现。具体来源与改动见 [实现说明](../docs/mdlm_lift.md)。

## 协议

- 独立服务器目录：`/root/autodl-tmp/s-flm/lift-pilot-20261023-01`，源码位于其 `source/`；未覆盖原 `/root/s-flm`。
- 硬件：单张 A800-SXM4-80GB；复用服务器现有 Python 3.12 / PyTorch 2.7 环境。
- 原始 PoT 数据：5,682 train / 632 validation；Base EMA 初始化，optimizer/scheduler/EMA 历史重新建立。
- LoRA rank 8、alpha 16、dropout .05；LR `3e-6`，warmup 20，全局 batch 8，BF16，训练 seed 1，workers 0。
- A：原始 MDLM 随机 MASK；B：扩展候选后随机恢复；C：LIFT-3 时间策略。三组均 200 步，保留 100/200 步 checkpoint。
- 全部生成对照在本台 A800 重跑：Base 和六个 checkpoint，632 题 validation、EMA、ancestral、T=.1、32 NFE、生成 seeds 1/2。未使用 Test 选择方法。
- 训练指标保存本地 CSV/JSONL 和离线 W&B；随后按用户要求将 C 组同步到线上，并追加其生成评估，A/B 保持离线。

## 预检查结果

服务器 11 项 LIFT 测试通过。最初测试把装有 CUDA FlashAttention 的环境用于 CPU 小模型前向，导致 Triton 指针错误；将该测试调整为在有 CUDA 时使用 GPU 后通过。此失败发生在正式训练前，没有 checkpoint 参与对照。

B 组首轮正式训练在第一次 backward 时失败：Lightning 外层 AMP 的权重 cast cache 被无梯度评分污染，使随后训练前向不带梯度。初版独立 smoke 没有包裹外层 AMP，未覆盖该条件。修正为评分阶段保留原 dtype、仅关闭 cast cache，并给真实小模型测试和完整模型 smoke 都增加外层 BF16 AMP 后，服务器 11 项测试重新通过。失败 B 组没有完成更新或产生有效 checkpoint，保留于 `failed-train-b-autocast/`；最终 B 从 Base EMA 重新开始。A 不调用评分分支，其完成的结果保持有效。

完整数据区域检查：训练 5,682 条、validation 632 条均成功，回退数为 0。

区域范围审计：训练集有 430,315 个可筛选代码/EOS token，占 2,553,270 个监督位置的 16.85%；validation 为 47,605 / 284,135（16.75%）。这是位置占比，不等于梯度贡献占比。该版本的结论不覆盖原 LIFT 对全部目标位置的筛选。

| batch-8 检查 | B：随机恢复 | C：LIFT-3 |
| --- | ---: | ---: |
| 单 batch 损失 | 0.58603 | 0.61709 |
| 峰值 allocated 显存（MiB） | 6,432.9 | 6,439.6 |
| 更新的 LoRA 张量 | 96 | 96 |
| 冻结参数逐张量一致 | 是 | 是 |

表中为补充外层 AMP 后重跑的检查结果。上述损失只用于有限性和更新检查，不能作优劣排名；首轮包含内核预热，也不使用其耗时比较训练速度。该检查没有保存权重，正式训练重新从 Base EMA 初始化。修正前 smoke、初次源码清单和失败日志亦保留。

## 证据与复现

- 驱动：[run_lift_pilot.py](../scripts/run_lift_pilot.py)。
- 统计：[analyze_lift_pilot.py](../scripts/analyze_lift_pilot.py)。
- 远端证据：`environment.json`、`source_manifest.json`、`region_audit.json`、`smoke.json`、`preflight-tests-gpu.log`、各组训练日志与 checkpoint、`evaluation/`。
- 本地证据：[comparison.json](lift-pilot-20261023-01/comparison.json)、[逐题配对](lift-pilot-20261023-01/paired_by_question.jsonl)、[区域范围](lift-pilot-20261023-01/region_scope_audit.json)、[mask 对照](lift-pilot-20261023-01/mask_control_audit.json)。完整评估记录及执行审计保留于该目录的 `evaluation/`。
- 六个约 693 MB 的 checkpoint 保留在远端各组 `checkpoints/`，没有重复下载权重。源码快照与逐文件 SHA-256 清单随证据保留。
- 完整证据包已拉回并通过服务器 SHA-256 校验；本地确认 14 次评估共 8,848 条生成记录、完成标记和关键源码 SHA-256 一致，传输记录见 [evidence_transport.json](lift-pilot-20261023-01/evidence_transport.json)。

仅一个训练 seed；两个生成 seed 重复使用相同问题，不能视为独立的 1,264 道题。LIFT 筛选后的训练损失不再具有原始随机 MASK ELBO 的解释；validation NLL 仍采用原随机 MASK。

## 已完成训练

| 组别 | 成功更新 | 样本呈现 | 非有限梯度 / AMP 跳过 | 训练 batch 时间（秒） |
| --- | ---: | ---: | ---: | ---: |
| A：原随机 MASK | 200 | 1,600 | 0 / 0 | 19.69 |
| B：随机恢复 | 200 | 1,600 | 0 / 0 | 27.25 |
| C：LIFT-3 | 200 | 1,600 | 0 / 0 | 26.41 |

三组累计监督 token 均为 717,763，非 PAD 监督 token 均为 221,914。B/C 的 20 个已记录节点中，候选数、保留数和目标代码 mask 比例完全一致，记录在 `mask_control_audit.json`。该核验覆盖日志节点，不声称保存了全部训练步的完整 mask。

100/200 步随机 MASK validation code/EOS NLL：A 为 1.954875 / 1.935694，B 为 1.955131 / 1.935386，C 为 1.954962 / 1.936082。这些值不能替代生成正确率。

## 完整低温生成结果

全部为同台 A800、632 题 validation、EMA、ancestral、T=.1、32 NFE、batch 32。100/200 步是预先指定节点；不使用 Test 选择节点。

| 模型 | seed 1 正确数 | seed 2 正确数 | 平均正确率 | 相对 Base 平均题数变化 |
| --- | ---: | ---: | ---: | ---: |
| Base EMA | 177 | 165 | 27.06% | — |
| A：原随机 MASK，100 步 | 180 | 165 | 27.29% | +1.5 |
| A：原随机 MASK，200 步 | 176 | 164 | 26.90% | -1.0 |
| B：随机恢复，100 步 | 176 | 166 | 27.06% | 0.0 |
| B：随机恢复，200 步 | 180 | 161 | 26.98% | -0.5 |
| C：LIFT-3，100 步 | 175 | 162 | 26.66% | -2.5 |
| C：LIFT-3，200 步 | 180 | 163 | 27.14% | +0.5 |

本轮 Base 与此前 Blackwell 的 179/161 不完全一致，所有比较均使用本轮 A800 Base 的 177/165，不混用旧分数。

### C 组逐题配对

下表为“参考错误→C 正确 / 参考正确→C 错误”；p 值是各 seed 单独的双侧精确配对二项检验，未进行多重比较校正。

| C 节点 | 参考 | seed 1 错转对 / 对转错（p） | seed 2 错转对 / 对转错（p） |
| --- | --- | --- | --- |
| 100 | Base | 0 / 2（.50） | 0 / 3（.25） |
| 100 | A100 | 0 / 5（.0625） | 2 / 5（.4531） |
| 100 | B100 | 2 / 3（1.00） | 1 / 5（.2188） |
| 200 | Base | 4 / 1（.375） | 0 / 2（.50） |
| 200 | A200 | 6 / 2（.2891） | 1 / 2（1.00） |
| 200 | B200 | 2 / 2（1.00） | 4 / 2（.6875） |

C200 相对 Base 平均仅多 0.5 题（约 +0.08 个百分点），且 seed 1 净增 3 题、seed 2 净减 2 题。相对 A200 的净变化也是一升一降；相对 B200 为持平和小幅上升。均未形成显著或跨 seed 稳定的优势。点估计最高的是 A100，不能据此声称原随机 MASK 已有稳健收益。

C200 相对 Base 的输出变化为 51/56 题，双方可解析条件下非 docstring AST 变化为 41/47 题。训练确实影响生成，但行为变化不等于数学正确率提高。

原始程序质量也没有一致改善：Base 两个 seed 的原始可解析数为 552/555，C200 为 549/552；严格可解析后执行异常数为 Base 146/152、C200 141/158。未闭合三引号为 Base 0/1、C200 0/1。完整程序的这些统计与评估器截断恢复后的结果分别保留，不能用恢复后解析率代替程序可靠性，也没有据此宣称数量关系或变量绑定能力已改善。

## Checkpoint 审计与结论

六个 checkpoint 的冻结参数逐张量与重新导入的 Base EMA 一致，48 个 LoRA B 张量均非零；100/200 步 EMA 更新次数分别为 100/200。全部三组最终成功更新 200 次，无非有限梯度或 AMP 跳过。不能把本轮低收益解释为适配器没有更新或底座被误改。

本轮支持该适配方法在 A800 上可训练、成本可控，也验证了候选 mask 对照；没有证据说明当前代码区域 LIFT-3 能突破已有 MDLM SFT 的生成瓶颈。不以 C200 的 +0.5 题为理由延长至 711/3,000 步或扩量，也未追加训练 seed、128 NFE 或 RL。

结论限于 rank-8 LoRA、LR `3e-6`、原 PoT、代码/EOS 范围、一个训练 seed 和 200 步预算。原 LIFT 的全区域筛选、rank-128、不同学习率/预算及其他适配尚未验证，不能据此断言 LIFT 对 MDLM 一般无效。若继续探索，需要与数量关系、必要步骤或变量数据流有关的明确新假设，而非仅凭 token loss 或单个 seed 的几题变化。

## C 组 W&B 上传

后续使用这些 checkpoint 完成 [固定 MASK 关键 token 诊断](gsm8k_lift_fixed_mask_diagnostic_20261023.md)：完整 validation 上仅见极小 CE 改善，关键 top-1 区间跨零，且未优于 A/B；本次无追加训练或解答生成，不改变本报告对续训的判断。该后续诊断保留在本地/服务器独立证据目录，以下 W&B 上传记录指本轮训练和生成评估。

已上传到 [lift-c-200](https://forge.coreweave.com/wandb/ruiqi-f-huang-nus/s-flm/runs/zog10k73)，保留原 run ID `zog10k73`。API 核验到 20 个训练 loss 日志点、200 次成功更新，以及最终 code/EOS validation NLL 1.9360818。

在同一 run 追加 C100/C200 的双 seed 执行正确率、Base 对照、错转对/对转错、配对 p 值和六行汇总表；曲线使用 `evaluation/checkpoint_step` 为横轴，没有追加模型训练。聚合 JSON 保存为 evaluation artifact `lift-c-validation-zog10k73`，本地副本见 [wandb_c_evaluation.json](lift-pilot-20261023-01/wandb_c_evaluation.json)，核验记录见 [wandb_c_upload.json](lift-pilot-20261023-01/wandb_c_upload.json)。

同步使用服务器现有 W&B 0.31.0 客户端；旧 0.19.9 客户端返回完成但 API 未查到 run，未将该返回值作为上传成功证据。最终已通过 API 验证 C200 正确数 180/163 与本地比较记录一致。没有上传 checkpoint 权重。

## 用户指定的 711 步续训补充

随后按用户要求将 C200 续训至总计 711 步，新增 511 次有效更新，无非有限梯度或 AMP 跳步。最终 EMA 在相同双生成种子的完整 validation 上为 177/162 题，平均 26.82%，低于 Base 的 27.06% 和 C200 的 27.14%；配对差异未达显著。此前“未追加训练”描述对应当时诊断范围，现以本补充为最新状态。详见 [711 步续训测评](gsm8k_lift_711_validation_20261023.md)。
