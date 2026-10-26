# GSM8K PoT：统一 result-return 的 SFT 变体

该目录由 `scripts/data/prepare_gsm8k_pot_result.py` 从 `data/gsm8k_pot/` 确定性生成。原题、docstring、数学计算、答案及 train/validation 划分不变；函数结尾统一为：

```python
result = <原 return 表达式>
return result
```

转换前后逐条受限执行，要求返回相同数值并与 GSM8K gold 一致。原始数据不会被修改。转换后的 token 长度由训练数据加载器使用固定 tokenizer 重新计算。

固定 SmolLM tokenizer 的复核结果见 `tokenization_report.json`：全部 6,314 条均不超过 512 tokens，最大长度为 508。
