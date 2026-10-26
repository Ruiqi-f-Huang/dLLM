# MDLM GSM8K PoT SFT

本适配复用原有 MDLM 加噪、加权重建目标、Lightning 训练和 EMA。默认从 Base EMA 参数开始新的 SFT，重新建立 optimizer、scheduler 和 EMA 历史，不恢复预训练的 250k 步进度。

## 数据与环境

- 数据：`data/gsm8k_pot/sft_train.jsonl`（5,682 题）、`sft_validation.jsonl`（632 题）。
- Base：`checkpoints/tinygsm/mdlm.ckpt`，或设置 `CKPT_PATH`。
- tokenizer 固定为已审计的 SmolLM-135M revision。
- RTX 2080 Ti 使用 fp16 mixed precision、标准 MDLM SDPA 与原生 RoPE fallback，概率归一化为 float32。其他算法的 FlashAttention 专用路径不属于此次验证范围。
- 本地 Mac 未安装 PyTorch 依赖；GPU 验证在服务器的 `/root/venvs/s-flm` 进行。

JSONL 只将 question 编码为 prompt、code 编码为目标，不将 answer/reference_answer 拼入输入。保留原分隔符（反斜杠和 n 两个字符）、BOS/EOS、512 总长度、answer-only mask 和 padding 监督；超长程序直接报错，不截断。训练前拒绝 train/validation 题目 ID 重叠。缓存包含数据 hash、tokenizer 名称/revision、特殊 token、长度和 mask 设置。

`data/gsm8k_pot_result/` 是保留原题、docstring、数学计算、答案和划分的格式规范化变体。
它把每个函数的最后返回统一为 `result = <原表达式>` 和 `return result`；6,310 条被
改写，原本符合的 4 条不改代码。转换前后 6,314 条均经过受限执行并与 gold 核对。
固定 SmolLM tokenizer 复核后的最大长度为 508，没有样本超过 512。训练时通过
`data=gsm8k-pot-result` 选择该变体；原始 `data/gsm8k_pot/` 保持不变。

## 运行

在已准备环境的服务器上先激活环境并进入项目：

```bash
source /root/s-flm/activate.sh
```

短程 pilot 示例（20 次 optimizer 更新，160 个 micro batches，最多 8 个验证 batch）：

```bash
OUTPUT_DIR=/root/autodl-tmp/s-flm/outputs/sft-pilot \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
bash scripts/train/gsm8k/mdlm_sft.sh \
  trainer.max_steps=20 \
  trainer.limit_train_batches=160 \
  trainer.limit_val_batches=8
```

离线选项仅用于 tokenizer 已缓存的服务器；新机器需先下载固定 revision。也可通过 `data.tokenizer_name_or_path=/path/to/snapshot` 指定本地 tokenizer。

一轮完整 SFT（不会自动执行，需自行启动）：

```bash
bash scripts/train/gsm8k/mdlm_sft.sh
```

默认 LR=1e-5、micro batch=1、梯度累积=8、warmup=20 optimizer steps、1 epoch，约 711 次更新；训练与验证生成样本默认关闭。`best.ckpt` 按随机噪声验证的 val/nll 选取，不等于按程序解题准确率选取。启动脚本默认创建带时间戳的输出目录；可用 OUTPUT_DIR 显式指定。

验证集程序生成准确率另行评估。以下入口只把验证题目送入采样器，不会把参考代码作为 prefix：

```bash
python -m main --config-name sft \
  mode=gsm8k_eval \
  eval.checkpoint_path=/path/to/sft/checkpoints/best.ckpt \
  sampler.steps=64 sampler.temperature=1.0 \
  gsm8k.output_dir=/path/to/validation-results
```

这里使用 PoT validation 文件，不是 GSM8K Test。Base、SFT、RL 应使用相同验证题目与采样配置作比较。此命令沿用仓库原有评分器，不代表已完成操作系统级程序隔离。

## 恢复与验证

首次 SFT：`checkpointing.resume_from_ckpt=false`。从已保存 SFT 继续时设置：

```text
checkpointing.resume_from_ckpt=true
checkpointing.resume_ckpt_path=/absolute/path/to/sft/checkpoints/last.ckpt
```

并增大训练目标 steps/epochs。恢复会保留 SFT 参数、optimizer、scheduler、scaler 和 EMA 更新次数，不再次导入 Base EMA。

限制：原项目 DataLoader 的 epoch 中途恢复不能保证精确延续数据顺序/RNG；Lightning 会提示这一点。需要可复现的训练比较时优先使用完整 epoch 边界的 checkpoint，不声称中途续训与不中断训练逐位一致。

单元检查：

```bash
python -m unittest discover -s tests -p 'test_sft*.py' -v
```

服务器短程验证已完成：真实 Base EMA 初始化，2 次更新和 validation、checkpoint 保存，然后恢复至第 3 次更新。130 个参数张量相对 Base EMA 发生变化，权重与 Adam 状态均有限；optimizer step、global step、EMA 更新次数一致。恢复测试峰值分配显存约 4.20GiB。小批验证 loss 不是 GSM8K 性能结果。

服务器记录：`/root/setup-logs/sft-tests.log`、`sft-smoke.log`、`sft-resume-smoke.log`、`sft-checkpoint-verification.log`。验证 checkpoint 位于 `/root/autodl-tmp/s-flm/outputs/sft-smoke/` 和 `sft-resume-smoke/`，不要将它们当作正式实验模型。

## SFT 训练监控

SFT 默认启用 `sft_monitoring: true`，同时写入 W&B 与本地 CSV。每次启动创建新的
`<输出目录>/metrics/version_N/`，恢复训练也会新建日志版本，避免覆盖旧记录：

- `metrics.csv`：原有 `trainer/loss`，修正后的 epoch `train/nll`、`train/ppl`、`train/bpd`，以及 `val/nll`、`val/ppl`、`val/bpd`；默认每 10 次更新记录 step 指标，epoch 指标在对应结束时记录。
- `monitor.jsonl`：每 10 次 optimizer 更新记录一次诊断，正常训练结束补写不足 10 步的尾段。
- `summary.json`：本次运行起止 optimizer step、累计样本/token 数、训练 batch 耗时与包含验证的总耗时；正常结束生成。

诊断字段：`monitor/lr` 是最近一次 optimizer 更新实际使用的第一参数组学习率；
`monitor/grad_norm_pre_clip` 是该次更新 AMP unscale 后、clip 前的全参数梯度 L2 范数。
非有限梯度单独累计计数，范数记为 null，保留 AMP 原有的跳过更新行为；同时记录 AMP scale、实际跳过次数和成功更新次数。溢出跳过时 EMA 不更新。梯度范数不是窗口平均值。
`monitor/train_seconds_per_update`、`examples_per_second`、`padded_tokens_per_second`
按记录窗口内训练 batch 的总时间计算，包括梯度累积，排除数据加载间隙、验证和 checkpoint。
CUDA 在 batch 边界同步，以保证计时可靠，会带来少量监控开销。
`allocated_mib`、`reserved_mib` 为采样时刻显存，`peak_allocated_mib` 是本次训练开始后峰值；不代表整张卡的 nvidia-smi 占用。

累计字段带 `session`，恢复时从零开始，并在 summary 中标明 initial_step：
`examples_seen_session` 包括重复遍历的题目；`supervised_tokens_seen_session` 包括程序、EOS 和 PAD；
`code_eos_tokens_seen_session` 排除 PAD；吞吐量的 padded tokens 包括题目和 padding。
这些计数是已处理 batch 数量，AMP 跳过更新时也包含对应 batch。
当前诊断只支持单设备，避免把单卡统计误当作全局统计。

验证 NLL 默认使用 EMA 参数；训练 loss 使用当前训练参数。NLL 随机加噪，且包含 PAD 监督，
不能代替程序解题正确率。此次新增监控不会自动运行 GSM8K Test，也不在每轮训练执行生成评估。

## W&B

默认 project 为 `s-flm`，group 为 `mdlm-gsm8k-pot`，job_type 为 `sft`；run 名包含启动时间。
配置、训练/验证损失及 `monitor/*` 诊断同时记录到 W&B，本地 CSV/JSONL 保留。
`log_model: false` 表示不自动上传 checkpoint；模型仍保存在原 checkpoints 目录。

首次在线运行前，在服务器激活环境后执行 `wandb login`，按终端提示登录。
不要将 API key 写进 YAML、脚本或 Git。然后使用原 SFT 启动命令即可。
可覆盖 `wandb.project=项目名`、`+wandb.entity=团队或用户名`、`wandb.name=实验名`。
当前未替用户配置在线凭据。

离线测试：在启动命令后加 `wandb.offline=true`，记录保存在输出目录的 `wandb/offline-run-*`；
之后登录并执行 `wandb sync /absolute/path/to/offline-run-*` 上传指定 run。
仅需本地日志时加 `wandb=null`，CSV 监控仍启用。
默认每次启动创建新 W&B run；若希望续接原在线 run，可在恢复 checkpoint 时设置
`wandb.id=原runID`（同时保持相同 project/entity）。checkpoint 恢复与 W&B run 续接分别配置。

日志横轴约定：W&B/CSV 的 step 与 Lightning 一致，从 0 开始；
`monitor/optimizer_step` 和 JSONL 的 step 是已完成的优化尝试次数。
Lightning global_step 在 AMP 溢出跳过时仍递增，成功更新数应查看
`monitor/successful_updates_session`，跳过数查看 `monitor/amp_skipped_updates_session`。
吞吐和 seconds_per_update 的 update 分母也是优化尝试次数，而非保证成功的参数更新次数。

SFT 审查后增加 `train/code_eos_nll`、`train/pad_nll`、`val/code_eos_nll`、`val/pad_nll`：
按对应区域的监督 token 数加权计算 epoch 均值，不改变反向传播的原始总损失。
当前训练集监督 token 中 PAD 占约 69.44%，验证集约 69.59%；应同时观察程序/EOS NLL。
PAD 监督保持与 Base 相同。程序/EOS NLL 仍是扩散重建指标，不是程序执行准确率。

## A800 环境与配置

A800 使用 `--config-name sft-a800`：BF16 mixed precision、backbone BF16、单卡 micro batch=8、
全局 batch=8（无梯度累积）、验证 batch=8、4 个数据加载 worker。
训练目标、数据、Base EMA 初始化、学习率和 1 epoch 预算与原 SFT 配置相同。
2080 Ti 的 FP16 配置仍保留为 `sft`，不要在 A800 上默认沿用它。

启动脚本可覆盖配置名：

```bash
bash scripts/train/gsm8k/mdlm_sft.sh --config-name sft-a800
```

新环境按照 README 的非 NGC 方案，优先 Python 3.12、torch 2.7.0、NumPy 1.26.4，
其余采用 requirements.txt 的固定版本。FlashAttention 在原 requirements 中未指定版本，
需要按此环境独立安装并验证 CUDA 前后向。普通 MDLM 的完整注意力本身使用 PyTorch SDPA；
安装 FlashAttention 后会使用其 RoPE 实现，不能把安装成功等同于所有注意力调用都走外部 FlashAttention。

A800 服务器的 `activate.sh` 会设置 `SFT_CONFIG_NAME=sft-a800`，因此激活后直接运行
`bash scripts/train/gsm8k/mdlm_sft.sh` 也会选择 A800 配置。其他环境未设置此变量时仍默认使用 `sft`。

### Base-aligned 里程碑实验

后续诊断使用 `sft-a800-base-aligned`。它沿用 V5 的双 mask 实现，但设置
`train_on_pad=true`，使 code、EOS、PAD 同时参与加噪和 loss，与 Base 的 MDLM
数据目标一致；保留 GSM8K PoT 原始 docstring。全参数 SFT 使用 `lr=3e-6`、
warmup 20、全局 batch 8 和一轮 711 次更新。这里的 batch、学习率和训练预算是
SFT 设置，并非复刻 Base 的 25 万步预训练优化过程。

```bash
SFT_CONFIG_NAME=sft-a800-base-aligned \
OUTPUT_DIR=/root/autodl-tmp/s-flm/outputs/sft/$(date +%Y%m%d)-base-aligned-01 \
bash scripts/train/gsm8k/mdlm_sft.sh
```

训练只保存 `step-000100.ckpt`、`step-000200.ckpt` 和 `step-000711.ckpt`。
第 0 步直接使用 Base EMA checkpoint，
不复制一份相同的大文件。默认的周期、last 和 val-NLL best 保存已关闭，避免终点重复文件。

训练结束后，在 632 道固定 SFT validation 上统一使用 EMA、seed 1 和 32 步采样：

```bash
CKPT_DIR=/root/autodl-tmp/s-flm/outputs/sft/DATE-base-aligned-01/checkpoints \
TOKENIZER_PATH=/root/autodl-tmp/s-flm/cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M/snapshots/1d461723eec654e65efdc40cf49301c89c0c92f4 \
OUTPUT_ROOT=/root/autodl-tmp/s-flm/outputs/eval/DATE-base-aligned-validation \
bash scripts/sample/tinygsm/mdlm_sft_milestones.sh
```

脚本依次评估第 0、100、200、711 步，并生成 `milestone_summary.json`，
记录数值准确率、Python 可解析率和未配对三引号比例。使用 validation 选择训练步数；
选定一次方案后再运行 GSM8K Test。

### LoRA 严格对照实验

`sft-a800-lora-control` 继承 Base-aligned 数据和训练配置，只把全参数更新改为
LoRA。默认 rank 8、alpha 16、dropout 0.05，目标是每个 DiT block 的 attention
`attn_qkv`、`attn_out` 和 MLP 的两个线性层。Base 主参数被冻结；optimizer 和新建的
EMA 只跟踪 LoRA A/B 参数，冻结权重保持从 Base EMA 导入的数值。checkpoint 仍保存
完整模型状态，因此当前每个文件约 661 MiB。

```bash
SFT_CONFIG_NAME=sft-a800-lora-control \
OUTPUT_DIR=/root/autodl-tmp/s-flm/outputs/sft/DATE-lora-control-01 \
bash scripts/train/gsm8k/mdlm_sft.sh
```

训练后统一评估 Base 与 100、200、711 步：

```bash
CKPT_DIR=/root/autodl-tmp/s-flm/outputs/sft/DATE-lora-control-01/checkpoints \
TOKENIZER_PATH=/path/to/fixed/SmolLM-135M/snapshot \
OUTPUT_ROOT=/root/autodl-tmp/s-flm/outputs/eval/DATE-lora-control-validation \
bash scripts/sample/tinygsm/mdlm_lora_milestones.sh
```

评估脚本在 Base 节点关闭 LoRA，在三个 SFT 节点启用相同的 LoRA 结构，并统一使用
EMA、seed 1、32 步采样。2026-10-22 的实际验证结果为 Base 54/632、step 100
54/632、step 200 55/632、step 711 53/632；各节点 Python 可解析率均高于 96.5%。
完整结论见[总体迭代报告](../reports/gsm8k_sft_overall_iteration_report.md)和
[逐题评估结果](../eval_runs/gsm8k-sft-validation/20261022-lora-control-validation/)。
按 validation 选择 step 200 后，其 GSM8K Test 32 步结果为 76/1,319（5.76%），
与 Base 同分；[正式 Test 结果](../eval_runs/gsm8k-test/20261022-lora-control-step200-ema-steps32-seed1/)
包含全部生成和逐题配对统计。

已验证的 A800 环境额外固定 `pytorch-lightning==2.5.1` 与 `pyarrow==19.0.1`：前者与 Lightning 主包
保持同版，后者兼容 NumPy 1.26.4。全量依赖锁定和验证记录保存在 `reports/a800-environment/`。
此环境未安装不参与 SFT 的 Jupyter、mauve-text、OCI 存储及 lm-evaluation-harness。
服务器默认使用迁移好的 Hugging Face 离线缓存；下载新模型时需要显式取消离线变量并配置可达的数据源。
