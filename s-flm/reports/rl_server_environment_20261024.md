# MDLM RL 服务器环境检查（2026-10-24）

目标：connect.bjb1.seetacloud.com:18707。仅检查环境，没有安装依赖、同步代码、下载权重或启动正式训练。

## 结果

硬件和已有虚拟环境支持 MDLM GPU 运算，但当前缺少运行本工作区 RL 所需文件，尚不能直接训练。

- Ubuntu 22.04.5，Python 3.12.3；22 核 CPU、约 110 GB 内存。
- NVIDIA RTX PRO 6000 Blackwell Server Edition，97,887 MiB 显存；计算能力 12.0。
- NVIDIA 驱动 590.44.01；驱动显示 CUDA 13.1。
- 项目环境 `/root/venvs/s-flm`：PyTorch 2.7.0+cu128，包含 sm_120；Lightning 2.5.1、Transformers 4.45.0、Datasets 3.5.0、Hydra 1.3.2、OmegaConf 2.3.0、FlashAttention 2.7.4.post1、NumPy 1.26.4。`pip check` 通过。
- 默认 Python 是另一个环境（PyTorch 2.8.0+cu128），缺少多数项目依赖，应使用 `source /root/s-flm/activate.sh`。
- 系统盘剩余约 18 GB，数据盘 `/root/autodl-tmp` 剩余约 50 GB。

## 实测通过

- 默认环境 CUDA BF16 矩阵乘法及反向传播。
- 项目环境 `algo`、`dataloader`、`main`、`samplers` 导入。
- 项目环境 BF16 SDPA attention 及反向传播，数值有限。
- 当前服务器 DiT small，长度 512、词表 49153、batch 1：随机参数前向与反向通过，输出 `(1, 512, 49153)`，loss 有限，测试进程峰值 allocated 显存约 1739 MiB。

上述随机模型测试不等于真实 Base checkpoint 的生成或 RL 端到端验证，也未覆盖 FlashAttention 专用路径。

## 缺失项

- `/root/s-flm/checkpoints/tinygsm/mdlm.ckpt` 不存在；`/root` 下深度 4 的检查未发现 checkpoint。
- 新增 `diffu_grpo.py`、`scripts/train_mdlm_grpo.py` 未同步。
- `configs/sft-a800-lora-control.yaml` 等当前 LoRA 配置未同步；服务器仅见早期 `sft.yaml` 和 `sft-a800.yaml`。
- `data/gsm8k_pot_result/sft_train.jsonl` 不存在；服务器有旧 `gsm8k_pot` 数据及 GSM8K Test。
- 固定 revision 的 SmolLM tokenizer 本地缓存加载失败（OSError）；需补齐缓存。激活脚本当前启用了离线模式。

后续应同步当前源码、配置、测试、训练数据与 tokenizer 缓存，补齐 Base 权重，再执行 RL 单元测试、一组候选审计和少量更新短测。此前端口 54592 的历史检查不能代替本端口当前文件状态。

## 同日补齐与真实 RL 短测

按用户指定来源补齐：Base 从公开 Hugging Face `jdeschena/s-flm` 下载，主站连接失败后使用 `hf-mirror.com`。SHA256 为 `251b4c935424622c13d75712cfefef0cae6829ad7bab6b712e009fc372bf1583`，与既有 Base 审计一致。文件保存在 `/root/autodl-tmp/s-flm/checkpoints/tinygsm/mdlm.ckpt`，项目路径的原有符号链接现已可用。

从本地同步 220 个源码、配置、脚本、测试与数据文件，部署时逐文件 SHA256 无差异。从旧服务器端口 54592 取得固定 revision 的 SmolLM tokenizer 缓存，现已通过新服务器离线加载。原服务器源码备份保存在 `/root/autodl-tmp/s-flm/backups/before-rl-sync-20261024.tar.gz`。

新服务器 RL 六项单元测试通过。真实 Base EMA 初始化、rank-8 LoRA、ancestral 32 NFE、T=.3、每题四候选、microbatch 1，完成同一组回答的两次更新：

| 更新 | 奖励均值 | 梯度范数（裁剪前） | loss | 代理 KL |
| --- | --- | --- | --- | --- |
| 1 | .5 | .02623760 | .00659631 | 0 |
| 2 | .5 | .02613102 | .00676090 | .00002243 |

该组题目 `gsm8k_train_2960` 的四候选中两对两错。冻结状态与 Base EMA 比较无变化，48 个 LoRA B 张量非零。独立审计模式也完成两题、每题四候选，均出现混合奖励组。这些样本仅验证工程通路，不估计成功率或证明 RL 收益。

训练输出：`/root/autodl-tmp/s-flm/outputs/grpo-env-smoke-20261024/`；审计输出：`/root/autodl-tmp/s-flm/outputs/grpo-audit-smoke-20261024/`。训练 checkpoint 为 raw 参数，没有 EMA，不支持直接恢复 Lightning fit 训练。

重载验证首次发现 `on_load_checkpoint` 强制读取 Lightning loops 的兼容问题；已在本地及新服务器修复，GRPO raw 导出加载时初始化数据循环计数，若误用 EMA 配置则明确报错。修改后的 trainer_base.py 两端 SHA256 均为 `22aaa504b7048d3ec19d7ae28700e805259e7b2d92bbdf763c87a8f4064f9973`。

修复后 `MDLM.load_from_checkpoint` 成功重载，继续采样输出形状 `[1, 512]`。原有 `test_sft_algorithm.py` 的六项回归测试全部通过。结论：当前服务器已具备候选组审计及短程 MDLM diffu-GRPO 训练的运行条件；未启动正式训练或全量 validation。

短测日志、配置、rollout 与逐步指标已下载至本地 `reports/rl-server-20261024/`；大模型权重仅保留在服务器。
