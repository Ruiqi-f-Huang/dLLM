# Question Conditioning Test（2026-10-23）

结论：Base 和 step3000 都有效利用题目前缀，不能将全 MASK 低准确率解释成模型忽略题目。在全 MASK 的 16/16 道题上，正确题目的非 docstring 代码 CE 均低于长度相同的错配题目；交换题目前缀改变约 58.5% 的代码 token 最高概率预测。不过题目条件的存在不等于数量绑定、运算计划或程序执行正确。

## 设计与控制

- 复用原 16 份 PoT 程序，仅评估现有 Base EMA 和 step3000 raw；无训练、无梯度更新、无新生成。
- 每份参考程序分别配正确题目与一份无关数学题。错配题从原始 SFT train 中选取，排除这 16 题，各错配题不重复；没有使用 validation/Test。
- 要求正确与错配题的固定 SmolLM tokenizer token 数完全一致，再选择去常用词后的单词 Jaccard 最低者，以 sample_id 决定并列顺序。本次 16 对的该 Jaccard 都为 0，已逐题检查为不同题意。长度范围为 32–79 个题目 token。
- 只替换 BOS 后的题目 token，BOS、separator、目标代码、目标起点、EOS、PAD 和 MASK 位置逐项一致。噪声时间、模型参数和评估模式相同，长度 512、batch 8、BF16、dropout 关闭。
- 50% 与全 MASK 使用既有固定 corruption 规则；代码区按所有被遮盖的非 docstring 代码 token（含函数头、EOS）计分。PAD 不计分。
- 另测“代码 50% MASK、docstring 全 MASK”：保持原 50% 条件的代码/PAD MASK 与噪声时间不变，仅把剩余 docstring 也遮盖，以排查 docstring 泄漏原题。这是一个受控干预，并非全序列均匀 50% corruption。

## 非 docstring 代码与 EOS 指标

CE 越低越好；CE 差值为错配−正确，正值表示正确题目有利。准确率为参考 token 的 top-1 匹配。

| 模型 | 条件 | 正确题 CE | 错配题 CE | CE 差值 | 正确题准确率 | 错配题准确率 | 正确题 CE 更低的题数 |
|---|---|---:|---:|---:|---:|---:|---:|
| Base EMA | 目标 50% MASK | 0.7750 | 1.2923 | +0.5172 | 88.35% | 80.30% | 14/16 |
| Base EMA | 代码 50%，docstring 全 MASK | 0.8489 | 1.5598 | +0.7110 | 89.62% | 78.60% | 14/16 |
| Base EMA | 目标全 MASK | 4.2708 | 6.8598 | +2.5890 | 26.91% | 26.28% | 16/16 |
| step3000 raw | 目标 50% MASK | 0.4052 | 0.9085 | +0.5033 | 92.16% | 83.26% | 16/16 |
| step3000 raw | 代码 50%，docstring 全 MASK | 0.4184 | 1.0076 | +0.5892 | 92.16% | 81.36% | 15/16 |
| step3000 raw | 目标全 MASK | 3.6248 | 6.2455 | +2.6207 | 26.60% | 25.21% | 16/16 |

## 解释

1. **题目前缀没有被忽略。** 全 MASK 时正确题目的代码 CE 优势约为 2.59（Base）、2.62（step3000），每道题都同方向。更换题目使两模型各有 58.51% 的代码 argmax 改变，显示的不只是极小概率扰动。
2. **不依赖可见 docstring 才有条件效应。** docstring 全遮盖、代码 50% 遮盖时，正确题目仍使代码 CE 降低约 0.71/0.59，准确率提高约 11.02/10.81 个百分点。
3. **SFT 的条件优势没有明显扩大。** 全 MASK 代码 CE 的正确−错配差距在 Base 和 step3000 相近；训练后正确与错配输入上的 CE 都下降。因此不能把整体损失下降直接归结为更强的题意到程序映射。该比较只有每题一个错配，不构成稳定效应大小估计。
4. **条件敏感与解题正确是不同问题。** 题目替换改变许多预测，但全 MASK 的正确题 top-1 仍只有约 27%，既有执行正确数仅从 2/16 到 3/16。主要未解决的是利用题目信息形成一致、正确程序，而不是完全不读取题目。

## 边界

- 用整道无关题替换会同时改变实体、数字、主题、句法与关系；本测试不能区分词汇匹配、数量提取和数学关系理解，也不能证明模型真正理解了题意。
- token 指标固定在一份参考程序的位置上，不覆盖其他等价解法；与可执行生成正确率不能互相替代。
- 只有 16 题、每题一个固定错配，没有进行随机错配多次重复或数字反事实测试。由于已出现明显 CE 优势，本轮没有自动扩大到更多诊断。
- CE 是指定噪声条件的普通交叉熵，不是原训练跨随机时间的加权 MDLM NLL。

## 文件

- [16 对题目及长度清单](overfit16-20261023-01/question-conditioning/pairs.json)
- [逐题与汇总指标](overfit16-20261023-01/question-conditioning/results.json)
- [运行日志](overfit16-20261023-01/question-conditioning/run.log)
- 复现脚本：`scripts/diagnose_overfit16_question_conditioning.py`。
- [W&B](https://wandb.ai/ruiqi-f-huang-nus/s-flm/runs/yi6e6u7w)：Summary 中 `question_conditioning_table`；原始指标、配对表与脚本位于 `overfit16-question-conditioning` artifact。
