# 16 题过拟合：梯度来源与单步代码监督诊断（2026-10-23）

结论：被检查的 LoRA 梯度与参数更新通路有效，代码监督能够降低同题全 MASK 的代码 CE。全 MASK 下 docstring 对原目标梯度的贡献范数约为代码区的 9–10 倍，PAD 没有表现为最大的梯度来源；但 docstring 与代码梯度正向对齐，尚不能据此认定它在阻碍代码学习。一次小更新没有提高 token top-1，也不能据此证明模型无法学习。

## 实验边界

- 使用现有 Base EMA 初始化与 step3000 raw 的两个内存副本，各自做诊断，不恢复旧 optimizer 动量，不保存更新后权重。原 checkpoint 未覆盖。
- 复用原 16 题；固定 50% 与 100% MASK，题目保持可见。docstring、其余代码（含函数头及 EOS）、PAD 分开统计。
- micro batch 4，累计覆盖 16 题。计算时 model.eval()，关闭 dropout，冻结主模型，仅对 96 个 FP32 LoRA 参数张量求导；原 BF16 backbone 计算设置保持。
- 梯度贡献对应 sum(区域时间加权 MDLM token loss) / 全部监督 token 数。分母为 7,406，docstring / 非 docstring 代码 / PAD 分别 770 / 940 / 5,696。另保存区域单独平均 CE 的梯度范数，避免将 token 数量与梯度强度混淆。
- 受控更新只监督全 MASK 的非 docstring 代码与 EOS，按 940 token 平均，fresh AdamW，LR 3e-6，其余 betas/eps/weight_decay/clip 沿用配置；full MASK 的时间权重为 1。与正式训练的旧动量、dropout 和随机时间混合有所不同，不能视为原训练的精确重放。

## 按原目标分母计算的梯度贡献 L2 范数

| 模型 | MASK | docstring | 非 docstring 代码+EOS | PAD | cos(总梯度,代码梯度) |
|---|---:|---:|---:|---:|---:|
| Base EMA | 50% | 0.041607 | 0.021508 | 0.008248 | 0.5227 |
| Base EMA | 100% | 0.176611 | 0.017134 | 0.014546 | 0.4899 |
| step3000 raw | 50% | 0.046850 | 0.040056 | 0.019321 | 0.6458 |
| step3000 raw | 100% | 0.928308 | 0.101300 | 0.050897 | 0.8446 |

step3000 全 MASK 下，docstring 与代码梯度余弦约 0.8100，PAD 与代码约 0.1099。docstring 虽然贡献大，但没有出现明显反向抵消。50% MASK 时 docstring/代码梯度范数比只有约 1.17，故不能把端点诊断推广为所有训练时间的梯度都由 docstring 主导；更不能单凭 PAD 占比大就认定它主导优化。

## 一次代码区更新：精度对照

| 模型 | backbone 精度 | 更新前代码 CE | 更新后代码 CE | 正确 token 数（前→后） |
|---|---|---:|---:|---:|
| Base EMA | BF16 | 4.2707625 | 4.2710165 | 253 → 252 / 940 |
| Base EMA | FP32（TF32 关闭） | 4.2704187 | 4.2702510 | 250 → 250 / 940 |
| step3000 raw | BF16 | 3.6246746 | 3.6238803 | 252 → 251 / 940 |
| step3000 raw | FP32（TF32 关闭） | 3.6226422 | 3.6214167 | 252 → 252 / 940 |

FP32 下，Base 代码 CE 下降约 0.000168，step3000 下降约 0.001225，平均 gold token 概率均略升，top-1 不变。因此代码区可以被梯度直接推动，但一次 3e-6 更新仍只带来很小的概率变化。

BF16 下，Base 的 CE 微升约 0.000254，step3000 的 CE 微降约 0.000794；其幅度与数值精度扰动接近，不能据此断言优化方向错误。两种精度各自的无更新重复测量完全一致；跨精度或跨 batch size 时会有少量 token argmax 变化，故只做各自设置内的更新前后比较。FP32 对照仅在诊断实例上关闭 backbone autocast 与 TF32，未修改正式模型源码或训练配置。

## 更新路径检查

- 96 个 LoRA 参数张量均为 FP32；观察到非零、有限梯度，参数确实变化。
- Base 初始化时 LoRA B=0，所以第一次更新 A 梯度为 0、B 梯度非零，这是该初始化的正常行为；step3000 的 A/B 都有非零梯度。
- BF16 受控更新中，Base 改变 663,439 个参数元素，step3000 改变 1,179,551 个。冻结参数在每次更新前后按内容哈希核对，完全不变；FP32 对照也通过冻结参数检查。
- 两种精度均未保存新权重；未启动多步训练、学习率消融或 rank 消融。

## 可支持与不能支持的判断

当前没有发现梯度断开、adapter 没有更新或冻结参数误更新的迹象。代码监督能降低 CE，但 top-1/程序正确率仍存在转换差距；这与此前全 MASK token 准确率不变的结果一致。

本诊断不能证明 BF16 是 3,000 步低收益的主要原因，不能证明 rank 8 足够，也不能证明移除 docstring 就会提高生成。全 MASK 梯度同向，且以前的格式和高噪声实验存在负面证据，不应将本结果简单转换成重复训练配方。

## 文件与复现

- [梯度与 BF16 单步更新原始指标](overfit16-20261023-01/gradient-diagnostic/results.json)
- [FP32 对照原始指标](overfit16-20261023-01/gradient-diagnostic/fp32-control.json)
- [梯度运行日志](overfit16-20261023-01/gradient-diagnostic/run.log)、[FP32 日志](overfit16-20261023-01/gradient-diagnostic/fp32.log)。
- 脚本：`scripts/diagnose_overfit16_gradients.py`、`scripts/diagnose_overfit16_precision.py`。
- [W&B](https://wandb.ai/ruiqi-f-huang-nus/s-flm/runs/yi6e6u7w)：Summary 中 `gradient_diagnostic_table` 与 `controlled_code_update_table`；原始数据及脚本在 `overfit16-gradient-diagnostic` artifact。
