# 结构保持的语义加权 SFT（2026-10-23）

## 实验问题与预算

在完整原始 PoT 训练集上，相同 711 步预算内，提高计算相关 token 的相对监督权重能否提高程序执行正确率。复用历史 all-ones Base-aligned LoRA 对照；不沿用 16 题过拟合的 3,000 步预算。

## 设置

继承 `configs/sft-a800-lora-control.yaml`：TinyGSM MDLM Base EMA 初始化，原始 5,682/632 train/validation，完整 docstring，随机 MASK，code/EOS/PAD 都保留监督，LoRA r8/alpha16/dropout0.05，LR 3e-6，warmup20，batch8，BF16，seed1，一轮711步，保存100/200/711。新配置 `configs/sft-semantic-weighted.yaml`。

仅改变优化权重：计算代码中的数字、算术/比较运算符、赋值/计算/return 中的变量及整个 return 表达式置2，其余目标位置仍为1。AST位置与 Python tokenize 定位字符，再与原始 tokenizer offset 对齐；同一 BPE 多次命中取2，不累加。函数类型注解、docstring数字不因名称相同而加权。该标注是启发式，不代表已识别真正的数学因果重要性。

优化目标为 `sum(w_i * valid_i * original_mdlm_loss_i) / sum(w_i * valid_i)`，分母包含全部原有监督位置，与本步被 MASK 的数量无关。MDLM 原时间权重和未 MASK 位置的零 loss 保持原样。`train/weighted_loss` 为优化指标；train/val NLL、PPL 和分区 NLL 均不使用语义权重，validation 优化返回值同样未加权。

## 对照与评估

旧对照 validation 0/100/200/711：54/54/55/53 正确（632题，EMA，ancestral32，seed1，长度512，batch32）。全部预测文件在本地保留，只有step200 checkpoint仍在两台服务器上。先沿用该生成协议比较，不再为“建立权重1对照”重复完整训练。

新实验在 RTX PRO6000 Blackwell 上执行，旧实验为A800。额外评估当前服务器上的 Base 与保留的旧step200，用于检查跨服务器可比性；若明显不一致，历史100/711只能作为参考，不能把差异全部归因于加权。后续是否补充同机器对照和多seed，取决于本次结果。

历史证据索引：`reports/semantic-weighted-20261023-01/historical_control.json`；远端输出 `/root/autodl-tmp/s-flm/outputs/semantic-weighted-20261023-01`。

## 验证与状态

基础SFT测试16项通过，权重测试4项通过，包含权重1严格退化为原目标、validation不加权、NLL指标不变、docstring/Unicode定位。完整train与validation BPE对齐审计通过；train关键BPE273,160/2,553,270监督token（10.70%），validation30,133/284,135（10.61%）。审计保存每split32个固定样例。两步真实训练短测通过，峰值显存约7,098MiB。

正式711步训练已完成：5,682样本，711次成功更新，0次非有限梯度或AMP跳过；训练batch耗时47.67秒，含validation约50.34秒。最终96个LoRA张量均有限，131个冻结状态张量与旧对照逐元素相同，优化器所有状态step均为711。100/200/711全量validation生成评估已完成，旧step200跨服务器复评亦已完成。W&B： https://wandb.ai/ruiqi-f-huang-nus/s-flm/runs/sw2o23a1

当前服务器Base复评55/632，相对历史Base54/632保留全部54题并新增1题；520/632条输出完全相同。此差异要求对一两题的变化保持谨慎。

## 初筛结果

| 节点 | 历史权重1对照（A800） | 语义权重2（Blackwell） |
|---|---:|---:|
| Base | 54/632 | 55/632 |
| 100 | 54/632 | 53/632 |
| 200 | 55/632 | 54/632 |
| 711 | 53/632 | 53/632 |

本次加权三个节点均未超过同机器Base；711步与历史权重1对照正确数相同。当前没有支持扩大训练预算或自动增加权重的生成证据。单seed、32NFE和跨机器历史对照限制了结论强度，不能据此排除所有语义加权方式。

711步相对当前Base保留51题、丢失4题、新增2题；Python可解析612/632。100步丢失2题、无新增；200步丢失2题、新增1题。

W&B保留训练曲线与汇总生成指标。完整题目/生成程序附件的上传被自动审批拒绝（训练曲线授权不包含导出完整评估记录），因此完整逐题记录仅保存在服务器和本地。

## 最终核对与结论

保留的旧LoRA200在当前Blackwell服务器为53/632，相对A800的55/632保留53题、丢失2题；525/632条输出完全相同。故同机器step200比较为均匀权重53、加权54，仅增加1题，不构成稳健收益。711步加权与历史均匀权重对照不仅都是53/632，答对的53题也完全相同；523/632条完整输出相同。当前Base与711步加权的可解析数均为612/632，没有出现小样本code-only全MASK实验那样的明显语法坍塌。

本轮未观察到明显生成语法退化，但没有支持“关键位置2倍权重提高了数学正确率”。不直接延长到3000步、不自动提高权重或扩展消融。此处是单seed历史协议初筛，不能证明所有加权方法无效，也不能把与历史对照的一两题差异归因为加权。若继续研究，应提出比本启发式位置标注更明确的新假设。

完整机器可读比较：`reports/semantic-weighted-20261023-01/paired_comparison.json`；逐题原始输出、实际训练配置、CSV曲线、审计样例、源代码快照均已归档。Checkpoint100/200/711保留在远端`train/checkpoints`，本地仅归档轻量证据。全部任务已完成，没有继续训练进程。

## 后续校准对本结论的限制

同日Base完整Test复评：T=0.1、32NFE、EMA达到303/1319（22.97%），而旧T=1为76/1319。因而本报告“未见收益、不直接延长”的判断仅适用于本轮T=1初筛。不能据此排除低温下可能存在的SFT收益；下一步应优先在同一validation、T=0.1下复评Base与保留checkpoint。参见[Base低温校准](gsm8k_base_temperature_retest_20261023.md)。

## 完成后的低温比较

T=0.1、32NFE、EMA、seed1、完整632题validation：Base179，原LoRA200为177，语义加权200为175，语义加权711为177。加权711的非docstring AST变化99题、错转对5题、对转错7题；按行为改变程度与SFT正确数并列最高的综合要求，它优先于加权200，但仍未超过Base。见[低温比较报告](gsm8k_sft_lowtemp_comparison_20261023.md)。
