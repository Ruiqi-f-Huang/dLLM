TinyGSM MDLM: GSM8K SFT / Online GRPO / Positive Learning
项目提交入口

1. 建议阅读顺序
   Result/report/TinyGSM_MDLM_SFT_RL_最终实验报告.md
     最终报告：实验动机、诊断逻辑、结果、案例、限制及后续方向。
   Result/report/TinyGSM_MDLM_方法示意图.md
     Mermaid方法图、实际GRPO公式及代码依据；最终报告已嵌入三张图。
   Result/preview.png
     7张核心图、2张附录图；各图另有PNG/PDF/SVG。
   Result/tables/results_tables.md
     Markdown结果表。该文件的历史编号与正文重排编号有区别，报告附录已说明。

2. 最主要结果与边界
   Positive32在128题validation子集、两批采样合并后：
     T=0.3: Base 26.86% -> Positive32 27.98%, +1.12 pp, 配对95% CI [+0.17,+2.08] pp。
     T=0.1: Base 27.71% -> Positive32 29.35%, +1.64 pp, 配对95% CI [+0.59,+2.69] pp。
   完整1319题Test，T=0.1、32 NFE、FP32、两个生成seed：
     Base EMA 22.18%, SFT711 EMA 22.14%, Positive32 raw 22.67%。
     Positive32 - Base +0.493 pp，配对95% CI [-0.493,+1.516] pp，尚未确认稳定提升。
   Positive Learning是固定成功程序正向学习，不等价于在线GRPO成功。
   validation和Test都有开发/探索使用历史；具体边界以最终报告为准。

3. 提交内容
   s-flm/              修改后的MDLM数据处理、训练、采样、评估、测试与分支运行说明。
   s-flm/data/         项目实际使用的PoT划分、变体及数据处理/审计产物。
   s-flm/reports/      历史SFT及RL记录、训练日志、逐题结果和诊断证据。
   s-flm/checkpoints/  保留历史训练元数据。大模型权重不随提交包附带。
   s-flm/eval_runs/    历史评估输出和配置。
   s-flm/experiment_results/  RL非权重证据归档、哈希清单与完整性验证脚本。
   d1/, LIFT/         方法来源代码、README及LICENSE；不作为本项目的运行入口。
                      未包含其无关的Sudoku/Countdown/AIME数据集。
   Result/            最终报告、图表、绘图脚本及最终Test完整证据。
   SUBMISSION_MANIFEST.json   提交文件逐一SHA256与大小，ZIP自身不列入。
   Result/submission_cleanup.json  本次清理及上游来源记录；未改写原始实验清单。

4. 复现训练与评估
   CUDA训练需要单独配置环境，项目曾使用Python 3.12、PyTorch 2.7.0+cu128、
   Lightning 2.5.1、Transformers 4.45.0、FlashAttention 2.7.4系列。
   参见 s-flm/requirements.txt、s-flm/docs/sft.md、s-flm/docs/mdlm_grpo.md，
   以及s-flm/docs/mdlm_grpo_positive32.md和mdlm_grpo_positive128.md。
   各分支依赖、题池及运行参数不同，请按对应文档使用，不混用历史默认配置。
   最终Test评估命令、EMA核验及断点续评规则见最终报告附录C。

   权重准备：
   - Base来自上游发布的jdeschena/s-flm仓库 tinygsm/mdlm.ckpt。
   - 最终SFT711和Positive32权重的SHA256及原始服务器路径在最终报告和
     Result/final-test-20261026-01/protocol.json中。
   - 这两份训练后权重没有附在本地提交包中；全量模型生成需先取得匹配权重。
     无需权重即可审阅全部逐题结果、复现统计绘图并重新执行选定生成程序。
   - 被移出的2.5GiB文件是旧Full SFT checkpoint，不是上述最终模型。

5. 本地复现图表和案例（不需要GPU或模型权重）
   使用Python 3.14及独立环境，避免污染CUDA训练依赖：
     python3 -m venv /tmp/dllm-figures
     /tmp/dllm-figures/bin/pip install -r Result/requirements-lock.txt
     /tmp/dllm-figures/bin/python Result/generate_figures.py
     /tmp/dllm-figures/bin/python Result/report/verify_qualitative_cases.py
   绘图锁定依赖不用于训练环境；其他Python版本需检查包兼容性。

6. 无需第三方Python包的提交完整性检查
   在项目根目录运行：
     python3 verify_submission.py
   RL归档另有来源哈希验证：
     python3 s-flm/experiment_results/rl-evidence-20261025/verify_archive.py
   最终Test本地核验在Result/data/final_test_verification.json，共7914条结果。

7. 整理原则
   保留可追溯的数据、配置、结果、源码快照、LICENSE与历史失败记录。
   历史报告描述的“尚未执行”等状态是当时记录，不替代最终报告。
   旧实施计划、报告大纲、缓存、无关上游任务数据及旧checkpoint已移出工作区，
   原位置与备份位置见清理记录。受原始归档哈希保护的少量macOS元数据予以保留。
   本地.git保留以便后续开发；提交ZIP排除所有.git历史和运行缓存。
