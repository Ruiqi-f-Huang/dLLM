# GSM8K MDLM diffu-GRPO 小规模验证（2026-10-24）

状态：运行中。服务器 `connect.bjb1.seetacloud.com:18707`，运行目录 `/root/autodl-tmp/s-flm/outputs/grpo-pilot-20261024-01/`。

## 预定协议

- 来源：公开 TinyGSM MDLM Base 的 EMA 参数，SHA256 `251b4c935424622c13d75712cfefef0cae6829ad7bab6b712e009fc372bf1583`。
- 分区：现有 `gsm8k_pot_result` 的 5,682 train / 632 validation；不使用 GSM8K Test。
- 候选信号审计：训练分区 seed 42 洗牌后取 256 题，每题 8 个候选，ancestral、32 NFE、T=.3、microbatch 8。总计 2,048 个程序。
- 初筛条件：审计完整结束，且至少 10% 的题组同时包含正确和错误候选。这是工程初筛门槛，不是算法有效性检验。
- RL：从 Base EMA 重新初始化 rank-8 LoRA，seed 42、LR 3e-6、beta .04、epsilon .2、prompt mask=0、dropout=0、每题 8 候选、microbatch 8、每组复用 2 次，共 100 次 optimizer 更新，保存 50/100 步。顺序使用同一 256 题池的前 50 组，即 50 个独立题目、400 个在线候选、800 次样本更新呈现，不等于训练遍历全部 256 题。
- 奖励：首个 EOS 截断后，用原评估器程序恢复及数值比较，答对 1，否则 0；进程隔离与超时。不添加格式奖励。
- 损失：d1 式全回答 MASK 代理概率，组内奖励只减均值，token 级 clipping，加固定 Base 参考的代理 KL，按有效回答 token 汇总平均。
- 验证：固定完整 632 题，同机 Base EMA / RL100 raw，ancestral、32 NFE、T=.1、batch 32、生成 seeds 1/2。模型加载后重新设 seed，控制初始化消耗随机数的差异。
- 同时报告完整生成输出（历史主口径）和首个 EOS 截断（训练奖励口径）；逐 seed 报告正确数、错转对/对转错与配对精确检验，不将两次生成视为 1,264 道独立题目。

## 工程检查

本轮之前已完成两次更新短测、checkpoint 重载、冻结参数检查和 RL/SFT 回归测试。实验不是从短测 checkpoint 续训；审计、RL 初筛均从 Base EMA 开始。

本轮源代码入口：`scripts/train_mdlm_grpo.py`、`scripts/eval_grpo_pilot.py`；流水线脚本 `scripts/run_grpo_pilot_20261024.sh` 在服务器后台等待审计完成后依次训练和评估，日志和完成标记可独立查看。

## 待填写结果

审计、训练和 validation 尚未完成，不据部分题组推断提升。仅一个训练 seed 的 100 步初筛不能证明稳定泛化收益，也没有同预算正确样本 SFT 对照。

## 审计完成与后台训练交接

256 题、2,048 候选审计已完成：候选平均正确率 26.2695%，至少一次正确的题目比例 69.9219%（179/256），混合正确/错误题组 68.3594%（175/256），平均文本唯一率 97.5098%。审计达到预定信号门槛，分区交集检查通过，已从 Base 独立开始 RL。

交接检查时训练已完成至少 12/100 次更新，loss、梯度与代理 KL 均有限。按用户要求，确认训练正常启动后停止主动监管，后台流水线继续完成 100 步训练与两个 seed 的 Base/RL validation。后续结果尚未读取，不能视为已完成或已有收益。

可查看服务器运行目录中的 `STAGE`、`train.log`、`validation.log`、`COMPLETE` 或 `FAILED`；最后两项分别表示流水线成功结束或命令失败。源码和运行参数见本报告前述路径。验证脚本将保留逐题生成和两种评分协议的配对结果。
