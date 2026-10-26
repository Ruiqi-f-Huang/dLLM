# A800 SFT 环境（2026-10-21）

已按 README 的非 NGC 安装方案配置独立环境 `/root/venvs/s-flm`，项目在 `/root/s-flm`。
原系统 Python 环境保留；没有升级/替换服务器驱动或 CUDA 工具链。

| 组件 | 实际版本 |
|---|---|
| GPU | NVIDIA A800-SXM4-80GB，80 GiB |
| OS / Python | Ubuntu 22.04.5 / 3.12.3 |
| NVIDIA driver | 580.126.09 |
| PyTorch / CUDA runtime | 2.7.0+cu128 / 12.8 |
| NumPy | 1.26.4 |
| Lightning / pytorch-lightning | 2.5.1 / 2.5.1 |
| Transformers / tokenizers | 4.45.0 / 0.20.3 |
| datasets / huggingface-hub | 3.5.0 / 0.30.2 |
| torchmetrics / W&B | 1.7.1 / 0.19.9 |
| FlashAttention | 2.7.4.post1，官方 torch2.7、cp312、CXX11 ABI TRUE wheel |
| PyArrow | 19.0.1 |

原 requirements 固定的 SFT 依赖逐项验证版本一致。README 未固定 torch/NumPy，因此采用其明确列出的
2.7.0/1.26.4；CUDA runtime 选择与此服务器 toolkit 匹配的 12.8。这是独立 venv，不是原作者的完整 NGC 镜像。
额外固定 pytorch-lightning=2.5.1 保持同版；PyArrow 26 实际导入要求 NumPy 2，故固定为兼容的 19.0.1。
FlashAttention 在原 requirements 中没有版本声明，此处固定经验证的 2.7.4.post1。
不参与 SFT 的 Jupyter、mauve-text、OCI 存储、lm-evaluation-harness 未安装。

## 使用

```bash
source /root/s-flm/activate.sh
wandb login
bash scripts/train/gsm8k/mdlm_sft.sh
```

activate.sh 自动选择 `SFT_CONFIG_NAME=sft-a800`。配置位于 `configs/sft-a800.yaml`：
BF16 mixed precision、backbone BF16、单卡 micro/global batch=8、梯度累积=1、验证 batch=8、4 个 worker。
学习率、数据、EMA 初始化及其余 SFT 设置继承 sft.yaml。
上面的最后一行会启动完整一轮 SFT，此次环境配置没有自动执行正式训练。
W&B API 地址网络可达；在线账号登录/上传尚未验证，短测使用离线 W&B 并保留 CSV。

服务器无法直连 Hugging Face，镜像 API 返回 403。已迁移固定 revision 的 tokenizer 缓存和 Base checkpoint，
activate.sh 默认开启 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE；这不影响 W&B 在线模式。
需要下载新模型时必须显式取消这些离线变量并使用可达数据源。

- 数据：`/root/s-flm/data/gsm8k_pot/`，与本地 split_manifest 的 SHA256 全部一致。
- Base：`/root/s-flm/checkpoints/tinygsm/mdlm.ckpt`，指向数据盘文件，SHA256 为
  `251b4c935424622c13d75712cfefef0cae6829ad7bab6b712e009fc372bf1583`。
- tokenizer revision：`1d461723eec654e65efdc40cf49301c89c0c92f4`。
- FlashAttention wheel 上游 SHA256：`74e5f217430b3014db70fe085b0b0e3f4b4309cd3a26c5b3d91f21f58aee5405`，已校验。

## 验证

- pip check 通过；项目模块正常导入。
- 外部 FlashAttention BF16 CUDA 前后向通过，与 SDPA 对照最大绝对误差 0.00048828125，梯度有限。
- 15 项 SFT 回归检查通过，无跳过项。
- 真实 Base EMA 完成 2 次优化尝试/成功更新，16 条训练样本，EMA 更新数为 2，无非有限梯度或跳过更新。
- CSV 的训练/验证总损失、程序/EOS 损失、显存及步数均已检查；W&B 使用 offline 模式完成同次训练。
- micro batch=8 短测峰值 allocated 显存约 9,627 MiB（9.40 GiB）。这是两步验证结果，不代表完整训练性能。

短测目录：`/root/autodl-tmp/s-flm/outputs/a800-env-smoke/`。
本地 `reports/a800-environment/` 保存环境 freeze、兼容约束、激活脚本、代码哈希、配置快照及验证日志。
普通 MDLM 的完整注意力沿用项目 SDPA；安装 FlashAttention 后 RoPE 使用其实现，不等于所有注意力调用都改成外部 FlashAttention。
