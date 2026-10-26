# 完整 SFT checkpoint 低温比较（2026-10-23）

## 协议

固定原始PoT validation 632题，Base EMA、原均匀权重LoRA step200 EMA、语义加权LoRA step200 EMA与step711 EMA。全部在同一Blackwell服务器，ancestral、温度0.1、32NFE、seed1、batch32、长度512。只做生成评估，不训练、不修改checkpoint，不使用GSM8K Test选择模型。

Base正确率由本次validation实际生成给出，不用先前Test的22.97%替代。四组保持相同题目顺序、batch划分、随机seed、tokenizer和采样设置。

## 指标

主指标为执行正确数及相对Base的错转对、对转错。辅助记录Python可解析数、三引号闭合、原始输出变化、去除docstring和格式后的AST变化。AST变化不代表数学正确性改善。逐题配对精确二项检验仅用于当前单seed探索性比较，不作多重比较校正；不足以把一两题优势称为确定收益。

脚本：`scripts/sample/tinygsm/mdlm_lowtemp_comparison.sh`、`scripts/compare_lowtemp_sft.py`。实际配置、生成记录和比较结果归档到`reports/lowtemp-sft-comparison-20261023-01/`；远端路径`/root/autodl-tmp/s-flm/outputs/lowtemp-sft-comparison-20261023-01`。

## 最终结果

| 模型 | 正确数/632 | 正确率 | 相对Base错转对 | 相对Base对转错 | 可解析 | 输出变化 | 非docstring AST变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base EMA | 179 | 28.32% | 0 | 0 | 631 | 0 | 0 |
| 原均匀LoRA200 EMA | 177 | 28.01% | 3 | 5 | 631 | 62 | 55 |
| 语义加权200 EMA | 175 | 27.69% | 1 | 5 | 632 | 64 | 53 |
| 语义加权711 EMA | 177 | 28.01% | 5 | 7 | 632 | 120 | 99 |

四组均无未闭合三引号。AST变化仅在Base与候选均可解析的631题上统计；它排除docstring和格式差异，但仍可能包含变量命名变化，不能直接当成数学改善。

## 排名与结论

若只按本次正确数：Base最高；SFT中原LoRA200与语义加权711并列177，语义加权200为175。此前T=1下暂选语义加权200的排序不适用于本次低温结果。

若同时要求生成行为确实改变，当前优先候选调整为**语义加权711 EMA**：它与原LoRA200正确数持平，但输出变化120/632（18.99%），非docstring AST变化99/631（15.69%），有5题错转对、7题对转错。该候选不是“已经超过Base”的模型，仍没有净解题收益。

与Base的配对精确检验：原LoRA200 p=0.7265625，语义加权200 p=0.21875，加权711 p=0.7744140625。均未显示显著差异；这里只是单seed探索性结果，不证明收益或退化在更多seed上稳定。当前不以一两题差异宣布某个SFT配方有效，也不据此直接增加训练步数。

## 核查与归档

四组实际采样协议完全相同，632个前缀逐条核对仅包含题目；实际平均NFE为32。Checkpoint SHA256、训练step与EMA更新数见`checkpoint_manifest.json`，Base EMA130个张量、LoRA EMA96个张量。逐题重新执行评分与各运行主程序结果一致。完整机器可读汇总为`comparison.json`，配对题目正确性为`paired_by_question.jsonl`，所有输出与配置在各模型目录保存。所有任务完成，无继续训练。

推荐候选远端路径：`/root/autodl-tmp/s-flm/outputs/semantic-weighted-20261023-01/train/checkpoints/step-000711.ckpt`，评估必须启用EMA。
