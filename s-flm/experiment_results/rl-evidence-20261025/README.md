# RL及成功程序正向学习：本地证据归档

首次归档2026-10-25；2026-10-26从旧服务器18707补齐原始记录。
来源为`connect.bjb1.seetacloud.com`的53432与18707端口。本次只复制、整理和校验证据，没有重新训练、评估或修改模型。**原缺失清单已清空：合并后1032个证据文件全部通过SHA256校验，覆盖21个正式/诊断/短测目录。** 大模型checkpoint、adapter权重、optimizer状态和大梯度向量按计划不下载。

## 从这里开始

- [实验索引](evidence_index.csv)：每轮的范围及训练曲线、评分、采样文件是否存在。
- [当前完整性清单](_archive/coverage-current.json)：以此为准，已将旧服务器完整非权重记录与新服务器产物合并。
- [统一哈希清单](_archive/manifest.json)：每文件SHA256、字节数、源路径和原mtime。
- [校验结果](verification.json)：1032/1032；重新校验：`python3 verify_archive.py`。
- [补录与合并记录](supplement_verification.json)：712个补充文件，其中620个新增、92个与本地已有文件一致，无覆盖冲突。
- [缺失清单](missing_evidence.json)：原缺失/部分归档的运行均已解决；保留原清单以追溯补录范围。

`_archive/manifest-53432.json`和`manifest-18707.json`保留分来源清单；`_archive/coverage.json`是第一次53432归档的历史清单，其partial标记已经由`coverage-current.json`更新，不能将历史清单视为当前状态。

## 正式实验与关键证据

所有路径相对`outputs/`。

| 运行目录 | 关键内容 |
| --- | --- |
| `grpo-pilot-20261024-01/` | Base候选审计、100步在线训练、验证和逐题结果 |
| `grpo-500-20261024-01/` | 500步在线训练、节点评估、逐样本输出和源码配置 |
| `grpo-controlled-20261024-01/` | BF16/FP32×两学习率四组；每组updates、scores、gradient_geometry和summary |
| `grpo-fp32-online-20261024-01/` | 两学习率FP32在线训练；metrics、monitor、gradients、rollouts及重复生成 |
| `grpo-fp32-resample-20261024-01/` | 独立采样批与配对复评 |
| `grpo-partial-mask-20261024-01/` | 随机Partial-MASK逐输入/逐题评分、MASK计划和比较 |
| `grpo-trajectory-20261024-01/` | 真实Base轨迹张量、逐步评分及配对比较 |
| `grpo-partial75-20261024-01/` | Partial75在线GRPO训练、监测、生成与轨迹复评 |
| `grpo-partial75-objective-20261024-01/` | 固定程序对评分、MASK计划、梯度方向摘要 |
| `grpo-partial75-overfit-20261024-01/` | 有符号Partial75拟合；完整更新/评分曲线及独立MASK诊断 |
| `grpo-partial75-positive-20261024-01/` | 仅正向8题训练；完整updates、scores、MASK和参数核验 |
| `grpo-partial75-positive-eval-20261024-01/` | 初次checkpoint转换失败记录；不是有效生成评估 |
| `grpo-partial75-positive-eval-20261024-02/` | 正式8题模型生成、逐题输出及真实轨迹评分 |
| `grpo-positive32-20261024-01/` | 32题正向训练完整曲线、MASK计划、128题validation及训练题比较 |
| `grpo-positive-coverage-20261024-01/` | Base/8/32模型原批、新批逐样本输出、合并统计和配对区间 |
| `grpo-remaining504-20261025-01/` | A800上的504题8/32模型比较，全部32256条生成输出 |
| `grpo-positive128-20261025-01/` | 候选筛选、128题独立核验、训练曲线、完整632题40448条生成输出 |

另有4个工程短测目录。完整fixed32在线过拟合及其前置失败已在[独立本地归档](../grpo-fixed32-20261024/)保存，不重复复制。

## 绘图、样例和开销入口

- 在线GRPO：`train/metrics.jsonl`绘制loss、reward、KL和裁剪；FP32/Partial75另有`monitor.jsonl`、`gradients.jsonl`。
- 固定拟合/正向学习：`updates.jsonl`绘制loss和梯度，`scores.jsonl`绘制成功/失败评分、训练及独立MASK评分。32/128题这些文件位于`train/`下；128题还含原32题/新增96题分组。
- 生成样例：各运行的`rank_0000.json`、`samples-*.jsonl`或`*-t*.jsonl`；均配合对应运行清单解释题池和采样协议。
- 数据准确性：128题`data_verification.json`和`DATA_VERIFIED`记录128条独立重跑均正确；`candidate_rollouts.jsonl`保留1208个候选。
- 运行时间线：`server/`及`server/port18707/`中的日志；原文件mtime另保存在统一清单。mtime不等同于开始时间，缺少直接计时的轮次不能写成精确端到端耗时。
- 代码与环境：各运行的`source/`保存当轮快照，`project_snapshot/`是首次本地归档时的新服务器代码；`_archive/packages.json`、`gpu.txt`是53432环境，不能误标为旧服务器环境。

## 证据边界

“完整非权重记录”表示复制了源目录现存的日志、配置、评估输出及选定数据/MASK张量，不表示源运行曾记录所有希望报告的指标。模型权重、adapter/optimizer和大梯度向量仍留在服务器；源实验未记录的峰值显存、端到端耗时或训练种子不能据本归档补造。

本归档没有包含尚未完成的三模型统一最终Test评估。实验结论见[RL及正向学习报告](../../reports/gsm8k_mdlm_d1_rl_report_20261024.md)，SFT证据位于`../../reports/`下。
