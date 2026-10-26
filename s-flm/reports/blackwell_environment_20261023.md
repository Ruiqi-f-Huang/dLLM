# Blackwell 服务器运行检查（2026-10-23）

目标：`connect.bjb1.seetacloud.com:54592`，项目目录 `/root/s-flm`。
本次检查针对当前 MDLM / GSM8K PoT SFT 工作流，不代表其他算法和数据集均已验证。

## 环境与结论

- Ubuntu 22.04.5，Python 3.12.3。
- NVIDIA RTX PRO 6000 Blackwell Server Edition，约 96 GiB 显存，计算能力 12.0。
- 驱动 590.44.01；PyTorch 2.7.0+cu128（CUDA runtime 12.8），包含 sm_120 支持。驱动显示的 CUDA 13.1 是驱动支持版本。
- 复用 `/root/venvs/s-flm`，没有重新安装或替换依赖。Lightning 2.5.1、Transformers 4.45.0、Datasets 3.5.0、FlashAttention 2.7.4.post1；`pip check` 通过。
- 使用 `source /root/s-flm/activate.sh` 激活，设置离线 tokenizer 缓存；现有 `sft-a800*` 配置名可以继续用于该 MDLM BF16 工作流。

## 缺失项与补齐

初始服务器代码停留在 10 月 21 日，数据盘缺少 Base checkpoint 和 Hugging Face 缓存；代码中的 checkpoint 符号链接目标不存在，且没有较新的数据变体。

- 从本地上传当前运行源码、配置、脚本、测试、文档和整个 `data/`。压缩包 SHA256：`5ae0c8887c8ae9e8581529ec0cb46a6fa78239dd28881201b8e94dea20ec8697`。
- 逐文件核验 212 个源码及数据文件，缺失 0、哈希差异 0。包含原始 PoT、result-normalized、完整 CoT 注释及 512-token 注释变体，以及 GSM8K Test。完整注释变体不保证满足 512-token 限制，训练应使用对应的 512-token 版本。
- 从 A800（端口 53432）直接 rsync Base checkpoint、固定 revision 的 SmolLM tokenizer 缓存和 Base-aligned LoRA step 200 参照。大文件直传约 25–30 MB/s，比本地上传高效；本地上传在重建连接后约 1.5 MB/s。
- Base：`/root/s-flm/checkpoints/tinygsm/mdlm.ckpt`，约 2.6 GiB，SHA256 `251b4c935424622c13d75712cfefef0cae6829ad7bab6b712e009fc372bf1583`，与历史审计记录一致。
- LoRA 参照：`/root/autodl-tmp/s-flm/checkpoints/lora-control-step200.ckpt`，约 661 MiB，SHA256 `ed8e54e38a63abb561324a85aea424851728935141e895ad9c562efcca4f331d`。同时建立历史路径 `/root/autodl-tmp/s-flm/outputs/sft/20261022-lora-control-01/checkpoints/step-000200.ckpt` 的链接。
- tokenizer 缓存：`/root/autodl-tmp/s-flm/cache/huggingface`，revision `1d461723eec654e65efdc40cf49301c89c0c92f4`。
- 未迁移全部历史实验模型；重跑依赖其他 checkpoint 的历史脚本时仍须补齐其权重。

## 已完成验证

- 基础 CUDA 运算通过。
- `python -m unittest discover -s tests -v`：21 项通过。
- 真实 Base EMA 初始化、rank-8 LoRA、BF16、batch 8，运行两次训练更新和两个 validation batch，通过；峰值 allocated 显存 7,098 MiB。
- Base EMA 两题、seed 1、32 步去噪的生成及评分流程通过。此检查不用于估计模型准确率。
- 迁移后的 LoRA step 200 EMA 使用同样两题生成设置也通过；环境快照在 `/root/setup-logs/blackwell-environment.json`，对应日志为 `blackwell-lora-generation.log`。
- 检查结束时系统盘可用约 18 GiB，数据盘可用约 47 GiB。
- 上传前服务器顶层 Python 源码备份在 `/root/autodl-tmp/s-flm/backups/pre-workspace-sync/`。

服务器证据：`/root/setup-logs/blackwell-{tests,smoke,generation}.log`、`blackwell-files-verification.json`。训练输出在 `/root/autodl-tmp/s-flm/outputs/blackwell-env-smoke/`；Base 生成输出在 `/root/autodl-tmp/s-flm/outputs/blackwell-generation-smoke/`。短测权重不作为正式实验模型。

上述日志、LoRA 生成日志、环境快照及逐文件校验结果已下载到本地 [`blackwell-environment/`](blackwell-environment/)。

复现训练短测：

```bash
source /root/s-flm/activate.sh
SFT_CONFIG_NAME=sft-a800-lora-control \
OUTPUT_DIR=/root/autodl-tmp/s-flm/outputs/blackwell-env-smoke-new \
bash scripts/train/gsm8k/mdlm_sft.sh \
  wandb.offline=true trainer.max_steps=2 \
  trainer.limit_train_batches=2 trainer.limit_val_batches=2 \
  trainer.log_every_n_steps=1 +trainer.enable_progress_bar=false
```

本次未启动正式训练、全量 validation 或 GSM8K Test 方案筛选，也未验证其他算法的 FlashAttention 专用路径。
