# GSM8K PoT：参考解答注释版

该目录由 `scripts/data/prepare_gsm8k_pot_cot_comments.py` 从 `data/gsm8k_pot_result/` 确定性生成。原题、docstring、可执行 PoT 代码、答案及 train/validation 划分均保留；GSM8K `reference_answer` 中的解释步骤经过去除 `<<...>>` 标记和最终 `####` 答案行后，作为 Python 注释放在题目 docstring 之后。原始数据不会被覆盖。

每条改造后的程序均通过 AST 解析与受限执行，执行结果与改造前相同，且与 gold 答案一致。
