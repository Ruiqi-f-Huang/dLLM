# FP32 在线 GRPO 学习率对照与监控

两组都从 Base EMA 独立初始化，使用 fixed32 正式运行冻结的 32 道题。仅比较 LR=3e-6/3e-5；其余保持 beta=.04、epsilon=.2、rank-8 LoRA、group=8、每组两次更新、rollout T=.3、32 NFE、250 次更新。训练仍按有效 token 汇总平均，不改成受控拟合的逐程序目标。

入口：

```bash
source /root/s-flm/activate.sh
python -u scripts/run_grpo_fp32_online.py \
  --base checkpoints/tinygsm/mdlm.ckpt \
  --fixed-run /root/autodl-tmp/s-flm/outputs/grpo-fixed32-20261024-02 \
  --output /root/autodl-tmp/s-flm/outputs/grpo-fp32-online-new
```

执行顺序：FP32 Base 重新评估 → 第一组训练及 100/250 步评估 → 第二组训练及 100/250 步评估。每个模型每题每温度 16 次采样，T=.3 为主、T=.1 为辅助。旧 BF16 Base 不作为 FP32 的直接配对基线。本流水线不运行 validation/Test，不自动选择最佳节点或追加采样。

## 每步和每组记录

- `train/metrics.jsonl`：奖励、优势、全对/混合/全错可据此推导、题目累计访问/混合访问、有效长度、AST 唯一比例、执行详情、策略和 KL 损失、裁剪前梯度。
- `train/monitor.jsonl`：内循环第 1/2 次更新；正、负、零优势各自的 ratio 分位数（5/50/95/99%）、上下界外比例；损失贡献以整组有效 token 为共同分母；候选 current/old/ref 平均 logp；参数更新范数、相对更新量和实际梯度裁剪系数。
- 在梯度拆分节点，记录真实 AdamW 更新与负策略梯度的余弦。概率观测为更新前，参数变化为更新后，日志显式标记时点。
- `train/rollouts.jsonl`：完整 token、首个 EOS 截断 token/文本、奖励、EOS 标记及执行详情。执行详情区分 EOS 截断原始程序与评估器恢复后程序的语法、执行、数值返回、答案正确性；不是 EOS 后完整尾部的执行评估。

## 定期梯度拆分

在状态步 0/1/10/50/100 和最终 250 上，额外测量当前 rollout 与固定成功/失败程序的策略梯度、含 beta 的参考惩罚梯度、范数比、余弦和合成梯度范数。固定程序诊断把 old 设为当前策略的 detached 概率，保留固定 Base 参考与奖励，避免旧 ratio 饱和。

在线状态步 0/1/10/50/100 在下一次优化之前测量，`group` 表明样本所属的 rollout 组。最终状态用最后一个 rollout 组及其旧策略重测，日志标记 `online-final-reused-group`。若 KL>.01 或 clipping>.05，在每 10 步边界触发额外更新后诊断。它们只观测，不改变 optimizer、奖励或约束。

梯度向量不直接保存，只记录几何统计；拆分完成后清空梯度，再正常执行训练 backward。固定程序分数不是实际多步生成 likelihood。

## 节点评估与检查

- 单次成功概率、mean reward（两者相同）、16 候选覆盖率、逐题成功数和按题配对 bootstrap。
- 执行/数值正确分型；逐题文本与去 docstring AST 唯一比例、正确程序 AST 唯一比例。没有正确程序时后者记 null。
- 固定成功/失败程序分别的平均/总 logp、有效长度、偏好差；新增 code/format 与 docstring/comment 区域平均分数。区域划分要求源程序可解析，且 decode→encode 精确复现原 token；否则明确记 unavailable，跨区域 token 不强行归类。
- 三个检查包括：raw backbone 输出确为 FP32；冻结权重与 Base EMA 一致；48 个 LoRA B 非零。

checkpoint 额外保存 Python/NumPy/Torch/CUDA RNG、专用调度 RNG、题目调度、组内更新进度、当前候选、优势、mask seeds 与 old/ref 缓存。**这用于追溯和未来实现 resume，当前没有实现或验证完整断点恢复。** 不应直接用 Lightning fit 恢复该训练。

`comparison.json` 保存跨学习率与节点的主要结果；源码快照和哈希用于追溯，`COMPLETE`/`FAILED` 标记后台结果。
