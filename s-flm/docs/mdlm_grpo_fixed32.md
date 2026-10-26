# 固定 32 题的 MDLM RL 过拟合诊断

本实验只检验当前配方能否提高固定训练题的采样成功概率。它不是泛化评估，也不自动启动 validation 或 Test。

## 运行

在 s-flm 的 CUDA 训练环境运行，必须使用新的输出目录：

```bash
source /root/s-flm/activate.sh
python -u scripts/run_grpo_fixed32.py \
  --base checkpoints/tinygsm/mdlm.ckpt \
  --audit /root/autodl-tmp/s-flm/outputs/grpo-pilot-20261024-01/audit/rollouts.jsonl \
  --output /root/autodl-tmp/s-flm/outputs/grpo-fixed32-new
```

审计文件需要同目录的 `run.json`。默认训练/validation 分区来自 `data/gsm8k_pot_result/`。首次运行前需确保本地新增脚本与更新的 `diffu_grpo.py`、`train_mdlm_grpo.py` 已部署。当前提交仅完成本地代码和测试，未启动远端实验。

## 固定协议

1. 从已有 Base、T=.3、每题 8 候选的审计中找出奖励同时有 0/1 的题目。按固定 seed 的题目 ID 哈希排序，选择 32 题，不按成功率或题目难度挑选；验证题不参与。
2. 在 Base 上用独立随机数命名空间，为每题生成一份成功、一份失败程序，冻结成代理诊断集。每题最多生成 64 个候选，若无法取得完整 pair 就停止，不静默换题。独立采样仍可能生成相同文本，不保证候选内容互异。
3. 对 Base 在 T=.3/.1 下分别每题独立采样 16 次，记录完整 token、首个 EOS 截断程序、执行奖励、输出长度及 EOS 出现比例。
4. 从 Base EMA 重新初始化 rank-8 LoRA，固定这 32 个问题，在线重新生成回答。每周期打乱题目但完整遍历一次；每题 8 候选，每组复用 2 次。LR=3e-6、epsilon=.2、beta=.04、prompt mask=0、dropout=0、32 NFE。训练 500 次更新，保存 100/250/500 步。
5. 对三个节点重复相同随机种子、温度、batch 与采样次数的测量；固定程序集只用于评分，不进入优化。检查底座未变和 adapter 已更新。

500 步对应 250 个在线题组、2,000 个候选；每题出现 7 或 8 组（14 或 16 次更新参与）。不是每题连续更新 500 次，也不是固定一批程序做离线优化。

## 指标与解释

- `success_at_1`：每题正确次数/16 后的题目平均，即单次成功概率估计；不是仅取第一个回答。
- `mean_reward`：二元奖励下与上述指标相同，不作为独立证据。
- `empirical_pass_at_k`：本次 16 候选中至少一次正确的题目比例。所选题已有成功候选，覆盖率可能饱和；不称为模型能力上限。
- 每题成功/失败程序分别记录平均 token log probability、token log probability 总和与有效长度。报告成功减失败的差距及相对 Base 的变化。长度不同可能影响两种分数，不把它解释为精确程序概率。
- 配对 bootstrap 以题目为重采样单位，保留同题全部 16 次采样；区间仅描述这组选择后的训练题，不表示全训练集或未见题泛化。
- 训练日志增加 `policy_loss`、`kl_penalty`，两者之和为总 loss；另有 `kl`、ratio clipping 和梯度范数。跨不同 rollout 的 GRPO loss 不直接当作固定目标学习曲线。

若固定题成功概率提高，只能说明所选可解题被强化；若代理偏好扩大但重采样不提高，支持进一步检查目标错位，不单独确定根因；若两者均未改善，应检查信号、更新幅度和约束，而不是直接判定实现错误。

## 产物

- `fixed/questions.jsonl`、`selection.json`：冻结题池、选择规则、来源哈希与审计参数。
- `base/pair_candidates.jsonl`、`pairs.jsonl`：独立生成的候选和固定诊断程序。
- `base/`、`step-000100/`、`step-000250/`、`step-000500/`：`sampling.json`、`samples-t*.jsonl`、`proxy.json`。
- `train/group_schedule.json`：逐组题目 ID，可核查覆盖和重复次数。
- `parameter_checks.json`、`comparison.json`：参数检查、逐节点配对成功率差与代理差距。
- `source/`、`source_sha256.json`：源码快照。`STAGE`、`COMPLETE` 或 `FAILED` 标记后台状态。

失败或输出不全时不会输出成功标记。脚本不支持训练恢复；修复后使用新的运行目录，保留失败证据。

## CPU 测试

从 s-flm 目录运行：

```bash
python -m unittest discover -s tests -p test_grpo_overfit.py -v
python -m unittest discover -s tests -p test_diffu_grpo.py -v
```

CPU 检查覆盖题池筛选、平衡调度、随机种子隔离、重复采样统计、题目层配对区间，以及策略损失与梯度。真实 checkpoint 生成与完整流水线尚需 CUDA 环境短测。
