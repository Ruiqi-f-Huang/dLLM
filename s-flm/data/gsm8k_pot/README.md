# GSM8K PoT：本地 SFT / RL 数据

已从服务器下载审计候选并整理；未启动训练。按合格题池划分，与旧审计中“原始 Train 留出 200 题”的容量测算不同。

| 文件 | 题数 | 内容 |
|---|---:|---|
| sft_train.jsonl | 5,682 | 原题、验证后的 Python 函数、标准答案、来源 ID |
| sft_validation.jsonl | 632 | 同上，留出验证，不更新参数 |
| rl_train.jsonl | 5,682 | 同一训练题及数值答案，不含参考程序 |
| rl_validation.jsonl | 632 | 同一验证题及数值答案，不含参考程序 |
| split_manifest.json | — | seed、题目 ID 划分、数据来源、SHA256 |

来源：MathInstruct 的 `data/PoT/gsm_gpt4.json`，匹配到原始 GSM8K Train。保留服务器已验证、总长度 ≤512 的候选，排除简单常数答案，每题选最短已验证程序，原始行号用于确定性排序。得到 6,314 道不同题，固定 seed=42 抽取 632 道验证题，其余用于训练。所有同题变体属于同一 split；首版仅保留一条。

SFT 的模型输入只使用 `question`；目标使用 `code`。`answer`、`reference_answer` 和来源元数据不得拼入模型输入。RL rollout 只使用 `question`，标准数值答案供外部 reward 计算，不能泄漏给生成模型或执行程序。

采用 checkpoint 分词器、512 长度及原始分隔符（反斜杠与 n 两字符，并非换行），保持 prompt 不加噪/不监督，程序、EOS、padding 参与监督。程序超过长度应过滤，不能截断。`unpadded_total_tokens` 是服务器使用固定分词器计算的完整输入加输出长度。

最终 Test 使用项目已有的 `data/gsm8k_test.json`，1,319 题，不参与训练和选参。本地已验证两个训练/验证集合之间无题目重叠，与 Test 无规范化精确题目重叠；未声称完成语义近重复或预训练污染检查。

本地原始审计备份在 `data/mathinstruct-audit/`。其中 verified-audit-candidates.jsonl 含 14,309 条候选，覆盖 6,316 题；本目录是其按首版规则筛查和划分后的派生数据。

重建命令（项目根目录）：

```bash
python3 scripts/data/prepare_gsm8k_pot.py
```

完整性检查已通过：8 项原审计哈希、4 项派生数据哈希、记录数、SFT/RL ID 对齐、函数/docstring 结构、长度和题目划分。尚待正式训练前的人工语义质量抽查；本次未在 Mac 上重新执行全部程序，执行正确性沿用哈希核验的服务器审计。

原项目尚无本地 PoT 的训练数据入口，需要后续接入；这些文件存在不代表当前 `data=tiny-gsm` 配置会自动使用它们。
