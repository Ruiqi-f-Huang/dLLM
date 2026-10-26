# GSM8K PoT SFT：无重复题目 docstring 变体

该目录由 `scripts/data/prepare_gsm8k_pot_nodoc.py` 从 `data/gsm8k_pot/` 确定性生成。题目、答案、训练/验证划分及程序计算逻辑不变；只删除 `simple_math_problem()` 开头重复题目的函数 docstring。

生成过程逐条解析并执行转换前后的程序，要求二者返回相同数值且与标准答案一致。原始数据不会被修改。
