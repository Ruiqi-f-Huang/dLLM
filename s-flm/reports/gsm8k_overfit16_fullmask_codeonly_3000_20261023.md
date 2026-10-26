# 全 MASK + code-only：续训到 3,000 步（2026-10-23）

结论：全 MASK 的非 docstring 代码 token 预测改善，但完整程序生成后期退化，没有达到 16 题记忆目标。代码 CE 累计下降约 40.7%，top-1 从 26.60% 升至 31.38%；同题生成正确数从 3/16 降至 0/16，可解析数从 12/16 降至 4/16。仅看 loss 会误判本轮收益。

[W&B 同一训练曲线](https://wandb.ai/ruiqi-f-huang-nus/s-flm/runs/x4989caj)，已同步至 optimizer_step=3000。

## 完整结果

所有节点使用同一 FP32 设置、同一 16 题、seed 1、128 NFE，评估 raw 参数。代码 token 指标按固定参考目标位置评分，不含 docstring/PAD，含函数头与 EOS。

| 步数 | 全 MASK 代码 CE | 代码 top-1 | 执行正确 /16 | 可解析 /16 | 未闭合三引号 /16 |
|---|---:|---:|---:|---:|---:|
| 0 | 4.2704187 | 250/940（26.60%） | 3 | 12 | 0 |
| 50 | 4.2652805 | 250/940（26.60%） | 3 | 12 | 0 |
| 200 | 4.2361533 | 250/940（26.60%） | 3 | 12 | 0 |
| 500 | 4.1160030 | 254/940（27.02%） | 3 | 12 | 0 |
| 1000 | 3.7020879 | 258/940（27.45%） | 3 | 14 | 0 |
| 2000 | 2.7605902 | 273/940（29.04%） | 2 | 12 | 3 |
| 3000 | 2.5318180 | 295/940（31.38%） | 0 | 4 | 9 |

2,000 步时，相对本轮 Base 有 2 题错转对、原本正确的 3 题全部对转错；3,000 步时没有错转对，原本正确的 3 题仍全部答错。逐题记录见 `paired_changes.json`。

## 训练与恢复核对

- 从该实验的 step200 checkpoint 继续到总计 3,000 步，未从 Base 重开，未恢复旧的随机遮盖实验。保持 rank 8、LR 3e-6、batch 8、FP32、TF32 关闭、LoRA dropout 0.05，其他优化设置不变。
- 固定原 16 题，原程序与 docstring 保留。题目之后的 code/docstring/EOS/PAD 全部 MASK，仅非 docstring 代码与 EOS 计损失；不随机采样训练噪声时间。
- 恢复模型、Adam、scheduler、Python/NumPy/Torch/CUDA RNG 及剩余数据顺序。恢复后 CE 精确复现 step200 的 4.2361533307，代码 top-1 复现 250/940；Adam 与 scheduler 计数均为 200，LR=3e-6，未重新 warmup。
- 新增 2,800 次更新全部完成，共呈现 22,400 个样本；连同前 200 步，累计 24,000 个样本呈现，每题 1,500 次。模型冻结参数前后哈希完全相同，1,179,648 个 LoRA 参数元素发生变化。
- 最终 checkpoint 的 global_step、scheduler 与所有 Adam step 计数均为 3,000。没有非有限 loss/梯度，也没有 AMP 跳过（本轮为 FP32）。
- 本段运行约 423 秒（包含四次评估及 checkpoint 写入），峰值 allocated 显存约 7,297 MiB。

## 生成退化与 EOS 核查

3,000 步的 12 个不可解析输出中，9 个是未闭合三引号；本轮 Base、step200/500/1000 均没有这一错误，step2000 为 3 个。实际保留了待生成的 docstring，却未对它计损失，这是本设计需要明确考虑的限制。其余输出还存在非法语法、未绑定变量和计算错误；不能把全部失败都归于 docstring。

对已有 token 记录做首次 EOS 截断、重新评分，不重采样，不改变主指标：

| 节点 | 完整响应正确 /16 | EOS 截断正确 /16 | EOS 截断可解析 /16 |
|---|---:|---:|---:|
| 0 | 3 | 3 | 13 |
| 2000 | 2 | 2 | 12 |
| 3000 | 0 | 1 | 4 |

截断仅挽回最终一个样本，可解析数仍只有 4/16，因此主要退化不只是 EOS 之后的尾部垃圾。该核查是事后诊断，W&B 的主曲线仍保留完整响应正确数 0/16。

## 解释边界

此实验展示了代码 token 概率与 top-1 的学习信号，但尚未学会生成可靠的完整程序。训练只见全 MASK 状态，而 128 NFE 生成会经过部分填充的状态；同时 docstring 是生成任务的一部分却没有受到直接监督。这些是设计上的差异，可能与退化有关，但本轮没有控制实验来确定各因素的因果贡献。

不能据此说模型完全不能学、LoRA 容量一定不足，或把低正确率仅归因于采样器。CE 到 3,000 步仍低于 2,000 步，不能仅据当前几个节点宣称重建已达到平台。生成只有一个 seed，全部指标来自训练题，不是泛化结果。

## 证据与产物

- [合并的全部节点指标](overfit16-fullmask-codeonly-20261023-01/resume-to3000/summary.json)
- [完整 1–3,000 步训练指标](overfit16-fullmask-codeonly-20261023-01/resume-to3000/training_metrics.json)
- [续训日志](overfit16-fullmask-codeonly-20261023-01/resume-to3000/training.log)、[完整性记录](overfit16-fullmask-codeonly-20261023-01/resume-to3000/integrity.json)
- [逐题变化](overfit16-fullmask-codeonly-20261023-01/resume-to3000/paired_changes.json)、[EOS 核查](overfit16-fullmask-codeonly-20261023-01/resume-to3000/eos_audit.json)
- 0/50/200 步逐题输出保留于原 `probe/`；500/1000/2000/3000 步逐题输出位于 `resume-to3000/probe/`。
- 新 checkpoint 保存在服务器 `/root/autodl-tmp/s-flm/outputs/overfit16-fullmask-codeonly-20261023-01/resume-to3000/checkpoints/`；未下载大权重。
- 自定义训练循环支持本实验 checkpoint 恢复，不是 Lightning loop resume 格式。续训入口与全部覆盖参数保留在 Hydra 快照及训练日志中，脚本为 `scripts/train_overfit16_fullmask_codeonly.py`。
- W&B 已核对 finished、optimizer_step=3000、代码 CE=2.5318179841、主指标正确数=0。EOS 核查位于 Summary 的 `posthoc_eos_audit` 表及 `overfit16-codeonly-eos-audit` artifact。

## 固定格式前缀的后续生成检查

对 Base、step1000、step3000 提供正确函数头与 docstring，只生成后续代码。参考原格式与多行格式两组的正确数均为 1/16；Base 也从仅题目时 3/16 下降到 1/16，说明前缀干预自身改变了条件兼容性，不能据此单独归因代码能力或 docstring。详见 [固定前缀诊断报告](gsm8k_overfit16_fixed_prefix_20261023.md)。
