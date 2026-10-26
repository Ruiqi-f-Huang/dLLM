# TinyGSM MDLM：方法示意图

本文件使用 Mermaid，可直接嵌入实验报告。三张图分别对应大纲的研究流程、MDLM 去噪过程和 Online GRPO 适配方法。方法图与 Fig. 1–7 的实验数据图独立编号，不改变已有图表编号。

## 示意图 A：整体研究流程

```mermaid
flowchart TD
    Base["TinyGSM 预训练 MDLM<br/>导入 Base EMA 权重"]
    Data["GSM8K PoT 数据处理<br/>题目匹配、AST 清理、执行核验、长度过滤和去重<br/>5682 train / 632 validation"]
    SFT["PoT SFT<br/>全参数 / LoRA / 语义加权<br/>自生成程序 / LIFT 适配"]
    SD["SFT 诊断<br/>16 题过拟合、MASK 比例<br/>题目条件、梯度、错误分型"]
    RL["d1 风格 Online GRPO<br/>在线生成候选 + 执行奖励<br/>全 MASK 概率代理 + LoRA 更新"]
    RD["GRPO 诊断<br/>固定题重复训练、精度和学习率<br/>Partial-MASK、真实轨迹、固定程序对"]
    Signed["成功 / 失败程序受控对照<br/>有符号目标与仅正向目标"]
    PL["成功 rollout 正向学习<br/>固定 75% Partial-MASK、FP32、200 updates<br/>8 → 32 → 128 道训练题"]
    Dev["开发验证<br/>128 题重复采样及完整 632 题比较<br/>按题配对 Bootstrap"]
    Freeze["冻结最终比较对象<br/>Base EMA / SFT711 EMA / Positive32 raw"]
    Test["统一 GSM8K Test 比较<br/>1319 题、T=0.1、32 NFE、FP32<br/>同一 A800、两个生成 seed"]
    Base --> SFT
    Base --> RL
    Data --> SFT
    Data --> RL
    SFT --> SD
    RL --> RD
    RD --> Signed
    Signed --> PL
    SD -. "重建与生成差距：研究背景" .-> PL
    PL --> Dev
    Dev --> Freeze
    Base --> Freeze
    SFT --> Freeze
    Freeze --> Test
```

**图注：**实验从同一预训练 Base EMA 出发，先后研究 SFT、在线执行奖励优化及固定成功程序正向学习。箭头表示实验推进或比较关系；Positive Learning 从 Base EMA 重新初始化，不从失败的 SFT 或在线 GRPO checkpoint 续训。32题模型进入最终比较；128题扩量未显示额外可确认收益。632题为多轮使用的开发验证集，Test也有历史探索性使用。

## 示意图 B：MDLM 条件去噪与训练—生成区别

```mermaid
flowchart TB
    subgraph Train["训练：参考程序的随机 MASK 去噪"]
        T0["训练输入<br/>BOS + question + separator + program + EOS + PAD"]
        T1["采样噪声时间 t<br/>根据噪声日程遮盖目标 token"]
        T2["题目前缀可见<br/>程序区域为可见 token 与 MASK 的混合"]
        T3["DiT MDLM 前向<br/>预测原始 token 分布"]
        T4["按实际配置计算 MDLM 去噪目标<br/>区别 docstring / code / EOS / PAD<br/>更新全参数或 LoRA"]
        T0 --> T1 --> T2 --> T3 --> T4
    end
    subgraph Generate["生成：无参考程序条件下的 ancestral 采样"]
        G0["输入题目前缀<br/>BOS + question + separator"]
        G1["回答区域初始化为全 MASK<br/>总上下文长度 512"]
        G2["当前 token 状态 + 噪声水平<br/>DiT 预测 token 分布"]
        G3["按反向后验随机采样<br/>恢复部分 MASK token"]
        G4["重新固定原题目前缀<br/>进入下一去噪时间步"]
        G5{"完成预设去噪步数？"}
        G6["取生成区域、按首个 EOS 截断<br/>解码并执行评分"]
        G0 --> G1 --> G2 --> G3 --> G4 --> G5
        G5 -- "否" --> G2
        G5 -- "是" --> G6
    end
```

**图注：**MDLM 使用吸收态 MASK corruption；随机 MASK 训练时可能看到参考程序的一部分，而完整生成从未知回答区域开始。生成中题目前缀始终固定，已揭示token在本项目吸收态后验中保持不变。NFE表示模型前向次数，最终比较采用32 NFE；图中不意味着每步恢复相同数量token，也不按首次生成EOS提前停止整条去噪链。SFT通常保留EMA用于生成，在线GRPO使用当前raw策略。

**输入格式说明：**separator为配置中实际编码的分隔字符串，并非另设一个名为`[SEP]`的特殊token。512是本实验使用的总上下文长度，可生成空间等于512减去题目前缀长度。参考答案不进入策略输入。PAD是否参与SFT损失取决于实验配置，不能将所有SFT分支画成同一loss mask。

## 示意图 C：d1 风格 Online GRPO 的 MDLM 适配

```mermaid
flowchart TD
    Q["从训练题池取一道题 x"]
    Policy["当前 raw MDLM 策略<br/>冻结 Base EMA 底座 + rank-8 LoRA"]
    Rollout["生成 G=8 个候选程序<br/>ancestral、32 NFE、T=0.3<br/>不反向传播采样过程"]
    Eval["按首个 EOS 截断并解码<br/>受限进程执行与数值答案核验"]
    Gold["Gold answer<br/>仅用于评分"]
    Reward["二元执行奖励 R_i ∈ {0,1}"]
    Adv["组内中心化优势<br/>A_i = R_i − mean(R)<br/>不除以组内标准差"]
    Mask["把已生成回答重新全 MASK<br/>保留可见题目前缀<br/>t=1，代理前向不作 rollout 温度缩放"]
    Old["更新前缓存 old token logp<br/>同一候选组复用两次更新"]
    Ref["参考 token logp<br/>临时归零 LoRA B，随后恢复<br/>参考为冻结 Base EMA"]
    Current["当前策略前向<br/>gather 候选程序 token logp"]
    Obj["逐 token 代理概率比<br/>PPO-style clipped surrogate<br/>加参考代理惩罚项"]
    Valid["有效位置<br/>回答区域至首个 EOS，包含 EOS<br/>排除 PAD、MASK 与 EOS 后位置"]
    Update["按整组有效 token 求均值<br/>AdamW + 梯度裁剪<br/>仅更新 LoRA"]
    Next["两次更新完成后<br/>用更新后的策略生成下一组候选"]
    Q --> Rollout
    Policy --> Rollout
    Rollout --> Eval
    Gold --> Eval
    Eval --> Reward --> Adv
    Rollout --> Mask
    Mask --> Old
    Mask --> Ref
    Mask --> Current
    Old --> Obj
    Ref --> Obj
    Current --> Obj
    Adv --> Obj
    Valid --> Obj
    Obj --> Update --> Next
    Next -. "在线刷新" .-> Rollout
```

**图注：**本图对应最初的全MASK在线GRPO实现。old/ref分数在组内更新前缓存；current分数随LoRA更新变化。计算策略比率和参考项时使用生成token的局部条件概率代理，没有计算完整多步生成轨迹likelihood。训练关闭dropout但启用梯度。后续Partial75在线消融修改代理输入及评分位置，不改变上述执行奖励来源。

实际目标为：

$$
\ell_{\theta,i,j}=\log p_\theta(y_{i,j}\mid x_i,[\mathrm{MASK}]^L),\qquad
r_{i,j}=\exp(\ell_{\theta,i,j}-\ell_{\mathrm{old},i,j}),
$$

$$
\delta_{i,j}=\ell_{\mathrm{ref},i,j}-\ell_{\theta,i,j},\qquad
K_{i,j}=\exp(\delta_{i,j})-\delta_{i,j}-1,
$$

$$
L=\operatorname{mean}_{(i,j)\in V}\left[
-\min\!\left(r_{i,j}A_i,\operatorname{clip}(r_{i,j},1-\epsilon,1+\epsilon)A_i\right)
+\beta K_{i,j}\right].
$$

主在线实验使用$\epsilon=0.2$、$\beta=0.04$。这里的$K$是候选token上的参考代理项，不是完整序列或完整词表KL；$r$也不能解释为完整采样策略的精确重要性比率。组内全对或全错时策略优势为零，但参考项仍可能产生梯度。

**与Positive Learning的区别：**正向学习使用预先筛选的固定成功程序、固定75%训练MASK，以及4套独立MASK进行评分；每次更新不刷新在线rollout，不使用负优势、clipping或参考KL。独立MASK仍作用于相同训练程序，不表示独立测试题。该方法应单独介绍，不能直接标为“成功的Online GRPO”。

## 实现依据

- [MDLM与采样实现](../../s-flm/samplers.py)：ancestral状态更新及题目前缀固定。
- [GRPO目标实现](../../s-flm/diffu_grpo.py)：有效位置、组内优势、概率代理、参考适配器和clipped loss。
- [在线训练入口](../../s-flm/scripts/train_mdlm_grpo.py)：候选生成、执行奖励及LoRA更新。
- [Positive32运行说明](../../s-flm/docs/mdlm_grpo_positive32.md)：固定成功程序正向学习的目标与评估边界。
