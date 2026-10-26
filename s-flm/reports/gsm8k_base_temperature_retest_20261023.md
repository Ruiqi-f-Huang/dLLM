# MDLM Base GSM8K Test 低温复评（2026-10-23）

## 目的与协议

核对此前 Base 仅76/1319（5.76%）与其他来源约22%成绩的差异。原始保存配置确认旧运行温度为1.0、32 NFE、EMA。此次仅将温度改为0.1，保持祖先采样、32 NFE、seed1、eval batch32、长度512、全量1319题、相同checkpoint与tokenizer；硬件由旧A800变为Blackwell，已有validation复评显示可能有1–2题数值波动。

温度 `T` 与去噪步数不同。本次不把NFE默认为1024，也不进行温度/步数搜索，以便直接检验降温效果。使用公开原始TinyGSM MDLM Base，EMA启用，LoRA关闭；没有新训练。输出`/root/autodl-tmp/s-flm/outputs/base-test-temp01-32nfe-20261023-01`。

## 评估路径核对

`samplers.get_sampler`将配置温度传给`AncestralSampler`，每步通过`AncestralContext`传入`TrainerBase.forward`；在log-softmax前执行`model_output / temperature`。独立前向检查验证T=0.1时logits为T=1的10倍。`_sample_gsm8k`调用`_eval_mode()`复制EMA参数。

Tokenizer固定为已有离线缓存revision `1d461723eec654e65efdc40cf49301c89c0c92f4`。所有输入逐条核对为`BOS + question + 原separator`，无参考答案或程序前缀。执行评分沿用项目原始`sandbox_gsm8k.evaluate_samples`。

## 外部参考

原论文在GSM8K比较T=1与T=0.1，明确指出低温显著改善MDLM；表2中的MDLM18.0%是1024NFE、T=1，低温高NFE约33%。这些数值不能直接与32NFE、T=1的5.76%相比。本次32NFE结果以实际复评为准，不预先认定“约22%”的精确条件。

来源：[Language Modeling with Hyperspherical Flows，第4.2节与图3](https://arxiv.org/html/2605.11125v1#S4.SS2)。

## 结果与影响

| 配置 | 正确数 | 正确率 |
|---|---:|---:|
| Base EMA，ancestral，32 NFE，T=1（旧A800） | 76/1319 | 5.76% |
| Base EMA，ancestral，32 NFE，T=0.1（本次Blackwell） | 303/1319 | 22.97% |

本次Python可解析1318/1319，未闭合三引号0条；第二次独立执行评分确认303题正确。温度前向缩放、完整题目输入、EMA开关、checkpoint与tokenizer配置均已核查。两次硬件不同，因此不是严格同机器温度消融；但该大幅差异说明此前5.76%不能代表Base在低温解码下的能力。

相对旧结果增加227题、约17.21个百分点。本次已复现用户所述约22%的量级，未进行NFE或温度搜索，没有重新训练。

此前SFT“没有提升”的结论需要明确限定在原T=1评估条件下。小样本关于LoRA参数可学习、模型使用题目条件的诊断不因此作废；但是否改善低温完整生成，需要在固定T=0.1协议下重评Base与保留的SFT checkpoint。当前不应仅凭T=1下的负面结果排除语义加权等方案，也不应立即用新温度分数宣称SFT有效。本次Test只用于用户明确要求的Base校准；后续方案选择应回到validation。

## 证据

`reports/base-test-temp01-32nfe-20261023-01/`保存运行脚本、实际Hydra配置、日志、1319题生成记录、温度路径检查、输入配置检查、独立评分与正确题索引。远端原路径保留。本次仅评估，无训练，任务已完成。

