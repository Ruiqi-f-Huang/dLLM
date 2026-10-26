# Base 自生成程序 SFT 配对实验（2026-10-23）

## 问题与实验控制

检验在同一批GSM8K Train题目上，Base自己生成并经执行验证的程序，是否比现有原PoT目标更适合SFT。两组只改变目标程序来源；从Base EMA分别初始化，LoRA r8/alpha16/dropout0.05，LR3e-6、warmup20、batch8、BF16、训练seed1，随机MASK、完整code/docstring/EOS/PAD等权监督，各200步。由于题集不同，原先全量训练checkpoint不能代替本次同题对照。两组max_epochs=-1以保证实际达到200步，训练中按原配置做validation NLL。

## 数据采集

从固定5682题train按selection_seed20261023无放回抽512题，保持原顺序；与632题validation无ID交叉。不使用Test采集、筛选或选择checkpoint。

- Base EMA、32NFE、seed1、T=0.1在该题池评估一次，识别本次低温答对/答错题（不把单次答对称为跨seed稳定答对）。
- 同一Base EMA、32NFE、T=0.7，seeds11/22/33/44，每题生成4个候选，共2048。温度为预先指定的采集设置，不用validation选择。
- 对候选重新执行验证，要求项目标准函数、完整题目docstring（仅忽略空白差异）、长度不超过512且不截断，并以基本算术检查排除直接硬编码答案的明显情况。抽查32份接受程序的数学语义；执行正确与简单AST检查不能完全排除碰巧算对。
- 每题选择seed顺序下第一个通过条件的候选；每题最多一个训练目标，不按验证集表现、最短程序或候选数量选择。
- 同一接受题集分别构造原PoT与自生成目标，除code外字段完全相同。记录低温答错但采样成功的覆盖数、目标代码相似度与长度。

题池初次写出遇到Unicode换行导致JSONL加载失败，已统一转义并逐条验证512条记录；该失败没有产生候选或训练更新。原512份PoT docstring全部匹配题目。

## 评估与复现

固定完整632题validation，EMA、ancestral、T=0.1、32NFE、batch32、长度512、生成seeds1/2。Base seed1复用此前相同协议的179/632，不重复生成；补充Base seed2。两组SFT各做两个seed。比较原PoT与自生成的直接配对变化，并分别与Base配对；逐seed报告，不把两次同题评估当作1264个独立问题。

源脚本：`scripts/sample/tinygsm/mdlm_selftrain_acquire.sh`、`scripts/prepare_selftrain_pilot.py`、`scripts/train/gsm8k/selftrain_pilot_compare.sh`、`scripts/analyze_selftrain_pilot.py`；配置`configs/sft-selftrain-pilot.yaml`。数据在`data/gsm8k_pot_selftrain_pilot/`，远端输出`/root/autodl-tmp/s-flm/outputs/selftrain-pilot-20261023-01`，轻量证据将归档到`reports/selftrain-pilot-20261023-01/`。

## 数据与训练核查

512题低温基准145/512。四个T=0.7批次正确数80/79/77/75，总计311/2048（15.19%），覆盖201题。完整docstring筛选后296个候选覆盖192题。人工抽查发现2道题面歧义（鱼的五种颜色只给三种数量；绳子切成五份时未明确选择哪段），从两组同步剔除。

最终294个候选覆盖190道训练题（题池37.11%）；每题选第一个通过条件的seed，共190个目标。其中86题在本次低温基准答对，104题低温答错但采样成功。最终32份人工抽查为16题低温答错、16题低温答对，未发现直接硬编码gold；该抽查不等于全部190题的形式化语义证明。

两个训练集的190题均执行得到gold，除code外字段完全一致；实际编码的题目前缀、loss mask、noise mask逐项一致，无语义加权、无程序截断。原目标平均117.88个程序token，自生成124.27；没有逐字相同或去docstring后AST完全相同的目标，但AST差异不能等同于数学解法不同。

数据已冻结，两组各200步训练及双seed全量validation评估已完成。W&B训练曲线：[原PoT组](https://wandb.ai/ruiqi-f-huang-nus/s-flm/runs/stp23w0original)、[自生成组](https://wandb.ai/ruiqi-f-huang-nus/s-flm/runs/stp23w0generated)。

首轮原PoT训练在跨epoch数据加载时出现worker SIGABRT，进度监控最后记录到20步，未得到200步checkpoint；该尝试日志保存在`failed-original-workers4/`与同名log。两组统一改用num_workers=0并从Base EMA重新初始化，W&B使用新的run ID，失败的部分更新不参与最终比较。此调整仅改变数据加载方式，两组仍保持相同的训练方法和预算。

两组单进程重跑均完成200步、200次成功更新，各1584次样本呈现；实际训练batch约13秒，含validation约27秒/组，无非有限梯度或AMP跳过。完整validation双seed生成评估已完成，结果见下表。

补充checkpoint核查：两组优化器step、global_step和EMA更新数均为200，LoRA B的48个张量均非零，全部LoRA权重有限；96个LoRA张量在两组间均不同，说明目标来源确实改变更新。131个冻结状态张量在两组间及既有Base EMA底座参照间逐元素一致。影响训练的公共配置逐项相同。



## 最终 validation 结果

| 模型 | seed1正确数/632 | seed2正确数/632 | 平均正确数 | 平均正确率 |
|---|---:|---:|---:|---:|
| Base EMA | 179 | 161 | 170 | 26.90% |
| 同题原PoT目标，200步EMA | 180 | 162 | 171 | 27.06% |
| 自生成验证目标，200步EMA | 177 | 161 | 169 | 26.74% |

两组均为190题、200次成功更新、1584次样本呈现。自生成相对同题原PoT在两seed分别少3题和1题，平均少2/632（约0.32个百分点）；相对Base分别少2题和持平。原PoT相对Base两seed均多1题（平均约0.16个百分点），不足以宣称稳健收益。

逐题变化：

| 模型/seed | 相对Base错转对 | 相对Base对转错 | 相对原PoT错转对 | 相对原PoT对转错 |
|---|---:|---:|---:|---:|
| 原PoT/1 | 3 | 2 | 0 | 0 |
| 原PoT/2 | 2 | 1 | 0 | 0 |
| 自生成/1 | 2 | 4 | 1 | 4 |
| 自生成/2 | 4 | 4 | 3 | 4 |

自生成相对原PoT的配对精确检验p值为0.375、1.0；相对Base为0.6875、1.0。所有比较差异均不显著。由于只有一个训练seed、两个生成seed，本轮只是一轮数据采集加短程SFT的配对初筛，不能据此证明所有自训练方法无效。

两组SFT在两个生成seed均为632/632可解析，未闭合三引号0。自生成相对Base有56/62条完整输出变化、47/55题非docstring AST变化；相对原PoT有57/58条输出变化、45/50题非docstring AST变化。因此不是完全没有更新或没有影响生成，但这些变化没有形成数学正确率收益。

## 结论与状态

本轮证实“在train上采样、执行验证后构造有新覆盖的正确程序训练集”可行：最终190题中有104题低温答错但高温采样能答对。但在当前200步、原随机MASK与完整等权监督下，**更换为自生成正确程序目标没有优于同题原PoT对照**，也未超过Base；没有支持直接扩大这套配方的生成证据。该结论限于本次数据、筛选和预算，不把训练loss下降作为解题收益。

全部采集、审核、配对训练和双seed生成评估已完成，没有继续训练或扩大题池。实际配置、六组validation输出、逐题配对、数据快照、训练CSV/监控、checkpoint与模型更新核查均按原路径保留；两个200步checkpoint保留在远端`train-original/checkpoints`和`train-generated/checkpoints`，轻量证据归档到`reports/selftrain-pilot-20261023-01/`。W&B仅有训练曲线与汇总指标，未上传题目/程序附件。
