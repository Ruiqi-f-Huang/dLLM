# 已有 RL checkpoint 的 Partial-MASK 配对诊断（第一阶段）

该阶段只评分，不训练、不生成新程序、不修改权重。比较 FP32 Base EMA 和在线 LR=3e-5 的 250 步 raw checkpoint（最小两模型配置），使用此前独立生成的同一组 32 题、64 个成功/失败程序。

```bash
source /root/s-flm/activate.sh
python -u scripts/diagnose_grpo_partial_mask.py \
  --base checkpoints/tinygsm/mdlm.ckpt \
  --online-run /root/autodl-tmp/s-flm/outputs/grpo-fp32-online-20261024-01 \
  --output /root/autodl-tmp/s-flm/outputs/grpo-partial-mask-new
```

## 固定协议

- 问题始终可见。回答有效区为前缀之后、首个 EOS 之前及 EOS 自身，排除 PAD/MASK。
- q=1/.75/.25；每个部分 MASK 比例用四个固定排列，按题 ID、成功/失败标签和 replicate 确定。不同 q 使用同一排列取前 k 个位置，形成嵌套 mask；跨模型完全一致。
- 每个程序 q=1 仅评分一次，不复制四份当作独立观测。MASK 数量为四舍五入后的 q×有效长度，至少一个；记录实际比例。
- 先将整个回答后缀替换为 MASK，再恢复未遮盖的有效回答 token。EOS 后原始生成尾部、PAD 和其他无效位置始终 MASK，避免长度/尾部条件变化混入 q 对照。这是为匹配原全 MASK 代理的控制协议，不声称复现实际轨迹。
- 只对被 mask 的 token 计算 logp、CE、top-1。可见位置的 MDLM 确定分布不参与评分，不加 ELBO 权重。
- 同时统计全部有效回答、代码/格式＋EOS、docstring/comment。源程序无法解析或 token 对齐不可靠时，代码分区标为不可用，不将其混为零损失。
- 先对每题每个程序的 mask replicate 求平均，再做模型间按题配对 bootstrap。成功/失败分开，不把多 mask、token 或两类程序当作更多独立题目。
- 当前 Base 不使用时间条件；仍按 log-linear 日程传入对应 sigma。实际遮盖数量和 schedule 的 eps 使 q=1 是人为全遮盖条件，报告中保留该区别。

每个模型评分 64×(1+2×4)=576 个固定输入；默认推理 batch=8。全部使用 FP32，关闭 dropout 和参数梯度。q=1 会额外与现有 `proxy_logps` 在相同程序上比对，均值差阈值 2e-5；失败则停止，保留证据。

## 产物

`mask_plan.pt` 与哈希固定输入/遮盖位置；`program_metadata.json` 报告区域覆盖。各模型 JSONL 记录逐程序、q、mask、区域分数；`per_question.json` 保存 replicate 平均后的值；`comparison.json` 给出相对 Base 的 logp/CE/top-1 差与按题区间。`*-q1-check.json` 记录全 MASK 复现误差。状态用 `STAGE`、`COMPLETE`、`FAILED` 标记。

该诊断属于 teacher-forced 条件恢复：可见 token 来自固定程序，失败程序也重建其自身 token。它不能单独代表真实采样遇到错误 token 的状态，也不验证泛化或直接确定目标错位。真实轨迹检查属于尚未执行的第二阶段。

按用户要求优先最小实验：默认仅 Base/高学习率节点和三个 q，不自动增加实验组。若结果不明确，应报告不确定性；不能保证有限评分一定给出显著结论。低学习率节点与其他 q 保留为显式 CLI 可选项，当前不运行。
