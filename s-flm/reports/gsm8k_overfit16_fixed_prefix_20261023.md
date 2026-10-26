# 固定格式前缀生成诊断（2026-10-23）

结论：固定函数头与正确 docstring 没有恢复程序生成能力。参考格式与多行格式两组中，Base、step1000、step3000 均只有 1/16 正确。由于 Base 自身也由仅题目时的 3/16 降至 1/16，这项前缀干预明显改变了生成条件，不能用负结果直接断言代码主体学不会，也不能把生成退化单独归因于 docstring。

## 正确数与可解析数

每格为执行正确数 / 可解析数，分母均为 16。所有权重来自同一 FP32 全 MASK + code-only 实验，step0 为 Base EMA 初始化，后续为 raw。

| 模型 | 仅题目：生成完整程序 | 固定参考格式前缀 | 固定多行格式前缀 |
|---|---:|---:|---:|
| Base EMA | 3 / 12 | 1 / 1 | 1 / 4 |
| step1000 | 3 / 14 | 1 / 2 | 1 / 5 |
| step3000 | 0 / 4 | 1 / 7 | 1 / 1 |

首次 EOS 截断后，两种固定前缀在三个节点的正确数仍全部为 1/16。严格原文 AST 解析及严格解析后执行也没有比主口径更多的正确样本；不是尾部截断就能恢复的结果。

## 设计与输入校验

- 不训练，仅对现有 Base EMA、full-mask/code-only step1000 与 step3000 权重做生成。不是此前随机 MASK 实验的 checkpoint。
- 同一 16 题、FP32、TF32 关闭、seed 1、128 NFE、batch 8、总长度 512；sampler 和评分器沿用原设置。
- 输入为 `[BOS] + question + separator + 已知函数头及完整 docstring`，其余位置由 sampler 生成。提供的 docstring 经空白归一化后与题目完全一致，没有参考计算、赋值或答案。
- 每份前缀通过 AST 检查：只有一个函数，函数体只有一条字符串表达式；没有计算语句。生成后逐 token 验证固定前缀未被 sampler 改动。
- 参考格式：截取原目标 token 序列中首条计算语句之前的最大完整 token 前缀，保留原训练布局与 token 位置。截断可能保留部分行首缩进（本批常见三个空格），剩余缩进仍需生成。每题剩余 337–431 个生成位置。
- 多行格式：函数头不变，题面放入独立开闭行的多行 docstring，前缀在闭合引号后的换行处结束，由模型生成完整正文缩进。重新编码前缀，改变了目标位置；每题剩余 335–429 个生成位置。它是格式敏感性对照，不是只改变一个 token 的单因素比较。
- 仅题目结果复用此前同精度、同 seed/NFE 的生成记录；固定前缀结果为本次新采样。同 seed 不保证两种不同前缀条件下每个随机转移使用完全相同的随机变量。

## 输出中发生了什么

参考格式已提供闭合 docstring，但模型仍可能在正文开头再次输出三引号。以生成后缀去除起始空白后是否直接以三引号开始计数：

| 模型 | 参考格式 | 多行格式 |
|---|---:|---:|
| Base EMA | 13/16 | 3/16 |
| step1000 | 13/16 | 1/16 |
| step3000 | 9/16 | 0/16 |

多行格式减少了正文开头重复引号，但没有增加正确题数；step3000 多行格式中仍有大量代码语法或变量/计算错误。固定前缀只是已知条件，不是语法约束解码，不能强制模型只输出合法代码语句。

## 可支持的结论

1. 本次两种固定前缀都没有明显救回正确率，不能把“提供正确格式壳”当作已经验证有效的修复。
2. 参考格式在 Base 上也显著降低可解析数，说明该干预本身存在明显的格式/条件兼容性问题。多行对照改变了错误分布，但未恢复正确率。
3. 因此前缀控制没有成功形成一个对 Base 无害的代码主体测试，不能用它把瓶颈干净地定位到数学计算、docstring 或采样器之一。也不构成重新监督 docstring 会有效的证据。
4. 所有数字只有 16 道训练题、一个 seed；两种格式完整披露，没有选择其中较好的格式作为性能提升结论。

## 证据与复现

- [参考格式汇总](overfit16-fullmask-codeonly-20261023-01/fixed-prefix-diagnostic/summary.json)、[参考格式前缀清单](overfit16-fullmask-codeonly-20261023-01/fixed-prefix-diagnostic/prefix_manifest.json)
- [多行格式汇总](overfit16-fullmask-codeonly-20261023-01/fixed-prefix-diagnostic-multiline/summary.json)、[多行格式前缀清单](overfit16-fullmask-codeonly-20261023-01/fixed-prefix-diagnostic-multiline/prefix_manifest.json)
- 两目录的 `step-000000.json`、`step-001000.json`、`step-003000.json` 保留完整程序、生成后缀 token、EOS 截断程序与逐题评分，共 96 个新生成样本。
- 复现脚本：`scripts/diagnose_overfit16_fixed_prefix.py`。默认运行参考格式；`+fixed_prefix_style=multiline` 运行多行格式。
- 服务器产物保存在原实验目录下的 `fixed-prefix-diagnostic/` 与 `fixed-prefix-diagnostic-multiline/`；未保存新权重。
- [W&B](https://wandb.ai/ruiqi-f-huang-nus/s-flm/runs/x4989caj)：Summary 的 `fixed_prefix_generation_table` 和 `fixed_prefix_generation_table_multiline`；对应 artifact 为 `overfit16-codeonly-fixed-prefix` 与 `overfit16-codeonly-fixed-prefix-multiline`。主训练曲线未改动。
