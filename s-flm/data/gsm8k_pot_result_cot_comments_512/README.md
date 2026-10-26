# GSM8K PoT：参考解答注释版（512-token 安全版）

该目录由 `scripts/data/trim_gsm8k_pot_cot_comments.py` 从完整注释版生成。每条样本使用 GSM8K 原有参考解答作为 Python 注释；为满足 512-token 上限，只从超长样本末尾删除完整注释行。题目、可执行 Python AST、答案以及 train/validation 划分保持不变。

`length_trim_manifest.json` 记录 tokenizer、长度上限、删减数量和文件哈希；`tokenization_report.json` 记录最终长度统计。
