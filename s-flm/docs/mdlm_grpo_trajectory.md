# 固定 Base 真实去噪轨迹评分

只比较两个模型：FP32 Base EMA 与在线 LR=3e-5 的 250 步 raw checkpoint。无新训练，不运行 validation/Test。

```bash
source /root/s-flm/activate.sh
python -u scripts/diagnose_grpo_trajectory.py \
  --base checkpoints/tinygsm/mdlm.ckpt \
  --online-run /root/autodl-tmp/s-flm/outputs/grpo-fp32-online-20261024-01 \
  --output /root/autodl-tmp/s-flm/outputs/grpo-trajectory-new
```

## 协议

- 同一固定 32 题，每题 2 条 Base 轨迹；T=.3、ancestral、32 NFE，固定独立种子，不按成功/失败筛选或重抽。
- 直接调用原 sampler 的 `init_state`/`step`，保存 32 次真实转移的 before/after、时间、揭示概率和最终程序。不改采样器、不插入教师 token、不删除 EOS 后的真实上下文。
- Base 与 RL 在完全相同状态上评分，只选步骤 0/8/16/24/31；两种打分温度 1/.3 用于分别对照原未缩温代理与真实采样分布，不增加采样实验组。
- 只评估原状态仍为 MASK 的位置。主要区域是最终 Base 程序首个 EOS 以内的剩余 MASK，以及该次转移中新揭示的有效程序 token；另记录所有后缀新揭示 token，包含 PAD/尾部，避免假装这些因素不参与采样。
- 监督目标是 Base 最终实际生成的 token。已揭示 token 必须保持不变并与最终结果一致，否则停止。成功/失败标签来自该 Base 最终程序的执行正确性。
- T=.3 时额外保存吸收态条件转移公式的完整后缀 log probability：揭示位置贡献 log h + log p(token)，保持 MASK 的位置贡献 log(1-h)，已可见位置贡献 0。它是单次转移评分，不是最终程序的边缘 likelihood。

## 汇总与限制

每个模型每个步骤的分数先在同题同奖励类别的轨迹中平均，再按题配对 bootstrap。没有剩余目标的区域不记作零损失；后期有效题数可能减少。成功/失败类别的问题覆盖可能不同，不能将两类总体平均当作严格匹配的偏好差。

这是真实 Base 访问状态上的重放，不能代表 RL 自身状态分布，也不能证明相同输入下 RL 生成的后续程序会更正确。失败轨迹的未来 token 是失败程序自身内容，不是数学正确的 gold 程序。完整转移评分可能受 PAD/尾部影响，因此必须同时查看有效程序位置。

输出：`base_trajectories.pt`、`trajectory_metadata.json`、轨迹哈希、两模型逐步 JSONL、`per_question.json`、`comparison.json`。完整转移分数保留在逐步 JSONL；比较表汇总相同状态上仍隐藏目标和实际揭示 token 的平均 logp、CE 和 top-1。`STAGE/COMPLETE/FAILED` 表示后台状态。
