# SFT 审查记录（2026-10-21）

结论：当前数据与条件 MDLM 训练目标兼容；两步 GPU 训练和离线 W&B/CSV 双写通过。
本次修复日志横轴回退、AMP 跳过更新时错误推进 EMA，并增加程序/EOS 与 PAD 分区损失。
未启动正式 SFT；短测 loss 不构成模型性能结果。

## 数据检查

| 项目 | train | validation |
|---|---:|---:|
| 题目数 | 5,682 | 632 |
| 最短 token 数（含 BOS/题目/分隔符/程序/EOS） | 71 | 83 |
| 最长 token 数 | 504 | 449 |
| 平均 token 数 | 199.98 | 199.13 |
| PAD 占监督 token 比例 | 69.44% | 69.59% |

- 数据 SHA256 与 split_manifest 一致；全部选中记录的 question/code/reference_answer 与已有审计候选一致，执行审计均为 verified，数字答案一致。原审计文件自身的 SHA256 也已核对。
- 题目 ID 和规范化题面的 train/validation 交叉均为 0；逐题与缓存的原始 GSM8K Train 对齐，参考答案与原始 Train 一致；与原始 GSM8K Test 规范化题面交叉为 0。
- 使用固定 revision 的 SmolLM tokenizer 全量重编码，与 loader 缓存的每个 token 和 mask 逐项一致；未截断程序，token 长度与准备阶段记录一致。
- 实际模型输入仅包含 question/code 及特殊 token；answer/reference_answer 不作为输入。生成验证入口只输入题目前缀。
- 此次核对既有执行审计及其哈希，没有重新执行全部 6,314 个程序；机械执行正确不保证每份程序都具有正确的通用推理。也未证明 TinyGSM 预训练数据不存在历史污染。

## 算法检查

- 原始 checkpoint SHA256：`251b4c935424622c13d75712cfefef0cae6829ad7bab6b712e009fc372bf1583`；预训练进度与 EMA 更新数均为 250,000。
- separator 为字面量反斜杠+n，train_on_prompt=false、train_on_pad=true、模型结构和 512 长度与 Base 匹配。
- 新 SFT 从 Base EMA 参数初始化，optimizer/scheduler/EMA 历史重新开始；SFT resume 恢复自身状态。已有初始化和 resume 验证记录仍适用。
- 题目不加噪；程序、EOS、PAD 参与吸收态 MASK 加噪，双向 DiT 预测原 token；保持原 MDLM 时间加权重建目标。
- 数学回归测试核对条件加噪边界、未遮盖位置零损失与零梯度、MASK 类排除、加权 NLL 及监督 token 分母。
- 修复 AMP 溢出跳过 optimizer.step 时仍更新 EMA 的问题。根据 GradScaler scale 回退识别跳过，并用实际 CUDA 强制溢出测试确认：2 次优化尝试、1 次成功更新、1 次 EMA 更新。

## 监控检查

- 原诊断日志以 completed steps 写入，而 Lightning 使用零起始索引，导致混写时 W&B/CSV 横轴回退。现已统一 logger step；JSONL step 和 monitor/optimizer_step 保留已完成尝试数。
- 同时写入 CSV 与离线 W&B；包含学习率、unscale 后 clip 前的梯度范数、AMP scale、非有限梯度次数、跳过/成功更新数、吞吐、显存和 session 计数。
- 新增 train/val 的 code_eos_nll、pad_nll，按各自 token 数加权，不改变优化损失。总 NLL 与分区 NLL 的加权关系已用真实训练输出验证。
- trainer/loss 是 micro batch loss，不能当作整次梯度累积的平均；epoch train/nll 按监督 token 加权。训练 loss 使用当前参数，验证 NLL 默认使用 EMA。
- W&B 在线上传与账号认证未在此次审查中验证；默认在线配置已保留，短测使用 offline=true。

## 验证结果与限制

15 项回归检查全部通过，无跳过项；包括真实 GPU AMP 溢出、离线 W&B 双写和单调日志步数检查。
真实 Base checkpoint 额外完成 2 次优化尝试、16 个训练样本及 2 个验证 batch；成功更新 2 次，无溢出。
峰值 allocated 显存约 4,927.5 MiB。

短测 train/nll=0.421259、train/code_eos_nll=1.349778、train/pad_nll=0.027702；
val/nll=0.640693、val/code_eos_nll=2.126580、val/pad_nll=0.000888。
仅用于验证数值和日志口径；不是全量 validation，更不是 GSM8K Test 性能。

当前 best.ckpt 仍按随机加噪的 val/nll 选择，并非按解题准确率选择；生成评估独立执行。
原 DataLoader 在 epoch 中途恢复不保证数据顺序/RNG 的精确延续。未做完整一轮训练稳定性验证。

证据位于 `reports/sft-review/`：测试日志、全量数据核对 JSON、短测 CSV/JSONL/summary、
离线 W&B run 文件、配置快照与本地/服务器代码 SHA256。
