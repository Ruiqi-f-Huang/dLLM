# 全 MASK + code-only：200 步短程记忆实验（2026-10-23）

结论：在本次 rank-8 LoRA、LR 3e-6、FP32、200 步预算下，全 MASK 的代码 CE 仅小幅下降，代码 top-1 与同题生成正确数均没有提高，尚未达到记忆目标。不能据此认定模型无法学习，亦不能认为该目标已收敛。

[W&B 训练曲线](https://wandb.ai/ruiqi-f-huang-nus/s-flm/runs/x4989caj)，主指标为 `train/code_ce`、`full_mask/code_ce`、`full_mask/code_top1` 和 `memorization/correct`。

## 结果

| 步数 | 全 MASK 代码 CE | gold token 平均概率 | 代码 top-1 | 执行正确 /16 | 可解析 /16 |
|---|---:|---:|---:|---:|---:|
| 0 | 4.2704187 | 22.6511% | 250/940（26.60%） | 3 | 12 |
| 50 | 4.2652805 | 22.6559% | 250/940（26.60%） | 3 | 12 |
| 200 | 4.2361533 | 22.6848% | 250/940（26.60%） | 3 | 12 |

代码 CE 从 4.2704187 降到 4.2361533，降低约 0.80%；gold token 平均概率略升，但 top-1 的正确 token 数固定在 250/940。

逐题正确集合三个节点均为 `gsm8k_train_5649`、`gsm8k_train_1515`、`gsm8k_train_7383`，没有错转对或对转错。50 步的 16 条输出全部与本轮 Base 逐字符相同；200 步仍有 15/16 条相同。唯一改变为 `gsm8k_train_2910`，计算逻辑仍不正确。

## 实际执行设置

- 从 TinyGSM MDLM Base EMA 重新初始化，重置 LoRA/optimizer/scheduler 历史；不使用此前 step3000 权重。
- 固定原始 PoT 的同一 16 题，保留原 docstring，不改目标源代码；数据及 ID 见 `data/gsm8k_pot_overfit16/manifest.json`。
- 题目前缀保持可见；题目之后 code、docstring、EOS、PAD 的全部位置均设为 MASK。固定 t=1、eps=0.001；不采样随机噪声时间。
- 损失只覆盖非 docstring 代码（包含函数头）与 EOS；排除 docstring、PAD 和 prompt，分母为每个 batch 的代码/EOS token 数。此端点的 MDLM 时间权重为 1，使用 token 平均 CE 等价。
- LoRA rank 8、alpha 16、dropout 0.05、目标模块及其余模型配置沿用已有对照；训练时启用原 train-mode dropout，评估时关闭。
- AdamW：LR 3e-6、warmup 20、beta=(0.9,0.999)、eps=1e-8、weight_decay=0、clip=1。batch 8，每轮打乱 16 题，200 次 optimizer 调用，共呈现 1,600 条样本，每题 100 次。warmup 初始 LR 为 0，之后正常升至 3e-6。
- FP32，TF32 关闭；仅在本实验实例上禁用 backbone 内部 autocast，不修改原模型实现。保存的 resolved config 额外包含 `diagnostic` 字段记录实际精度覆盖。
- 0/50/200 步评估 raw 参数，无 EMA 交换。全 MASK 指标使用相同 16 题、micro batch 4；生成使用 seed 1、128 NFE、batch 8、512 长度、FP32，其余 sampler 设置不变。评估后恢复所有 RNG 与模型训练模式。

## 基线与比较边界

本轮重新测得的 FP32 Base 为 3/16，而早先 BF16 Base 是 2/16；生成数值精度变化会改变离散采样结果。因此本轮训练收益以自己的 FP32 step0 为对照，不能直接把两个实验的正确数作严格归因比较。与原实验同时改变了 corruption、loss 区域与计算精度，本实验回答简化条件下能否产生记忆，不是单因素泛化消融。

只使用一个生成 seed 和 16 道训练题；不评估 validation/Test。结果没有显示生成收益，但 200 步、3e-6 下的 loss 仅微降，尚不足以判断更长训练的上限，也没有据此追加学习率或 rank 消融。

## 完整性与产物

LoRA 的 1,179,648 个参数元素均发生变化；所有反向传播梯度与 loss 均有限，冻结参数前后内容哈希完全一致。200 次 optimizer 调用正常完成，运行时间约 86.5 秒（含本轮三次评估、不含模型和在线日志初始化），峰值 allocated 显存约 7,296 MiB。

- 本地证据：[逐题评估、配置、日志与汇总](overfit16-fullmask-codeonly-20261023-01/)。
- 配置：`configs/sft-overfit16-fullmask-codeonly.yaml`。脚本：`scripts/train_overfit16_fullmask_codeonly.py`。
- 服务器目录：`/root/autodl-tmp/s-flm/outputs/overfit16-fullmask-codeonly-20261023-01/`。
- 50/200 步 checkpoint 保留在服务器的 `checkpoints/`。这是自定义诊断训练循环的 checkpoint，包含 state_dict、optimizer、scheduler、RNG、剩余数据顺序和配置；不是 Lightning loop resume 格式，不能直接交给原 main.py 的 Lightning resume。
- W&B 已完成在线同步并经 API 确认 step=200，指标与本地一致；artifact `overfit16-fullmask-codeonly-200-logs` 保留训练指标、配置与完整性记录。

## 续训结果

本实验已从 step200 恢复至总计 3,000 步，配置不变。代码 CE 最终为 2.5318，top-1 为 295/940，但生成正确数为 0/16、可解析 4/16；EOS 截断后正确为 1/16。完整结果与退化边界见 [3,000 步报告](gsm8k_overfit16_fullmask_codeonly_3000_20261023.md)。
