# 最终Test评估：20035服务器检查（2026-10-26）

本次只检查环境、依赖、文件、tokenizer加载和一个小型FlashAttention内核，不启动模型生成、训练或benchmark，也未迁移文件。

- GPU：A800-SXM4-80GB，检查前空闲；CPU16核，RAM120GB。
- 数据盘50GB，检查时基本空闲；系统盘可用约17GB。
- 环境：`source /root/s-flm/activate.sh`；Python3.12.3，PyTorch2.7.0+cu128，CUDA可用，Lightning2.5.1，Transformers4.45.0，Hydra1.3.2，OmegaConf2.3.0，datasets3.5.0，einops0.8.1，FlashAttention2.7.4.post1。
- 小型BF16 FlashAttention实际CUDA调用通过，输出有限。
- Test：`/root/s-flm/data/gsm8k_test.json`，1319题，SHA256为`c93390d9990e8232ef01ec74153f8164f8a277ceffec30523b08a843861b001c`，与此前记录一致。

## 未满足项

1. 未找到任何模型checkpoint；`/root/s-flm/checkpoints`指向尚不存在的`/root/autodl-tmp/s-flm/checkpoints`。
2. 缺少已审计SmolLM tokenizer缓存。HF_HOME指向数据盘，离线加载失败。
3. 项目是较早版本；缺少`grpo_fixed32.py`及`evaluate_final_gsm8k_test.py`等后续代码，需同步完整依赖而不是只复制单个入口。

## 已只读核对的53432来源

- Base checkpoint存在，2688828376字节。
- 正向32题模型存在，678275260字节。
- 最终Test评估脚本存在；固定revision tokenizer缓存存在。
- 指定语义加权SFT711 checkpoint仍不在53432。历史来源为54592，路径`/root/autodl-tmp/s-flm/outputs/semantic-weighted-20261023-01/train/checkpoints/step-000711.ckpt`，本次未连接该来源；取得后应匹配历史SHA256 `5b2f034c078a7557814cb9d243387ca525f9cb418568556740a569a23427029a`，并核对EMA更新711、96个张量。

结论：硬件和已安装依赖满足运行要求，但模型、tokenizer和代码不齐，尚不能直接启动三模型评估。未执行任何benchmark。

## 后续迁移更新（2026-10-26）

按用户后续指令，从53432补齐可取得的文件；没有启动任何生成评估或训练。

- 已迁移Base EMA checkpoint、正向32题step200 checkpoint、固定revision tokenizer缓存、完整Test和PoT train/validation、代码/配置/测试/运行说明，共281个文件、约3.38GB。
- 全部281个文件逐一通过源SHA256和字节数校验，原代码/同名数据备份位于`/root/autodl-tmp/before-test-migration-20261026.tar.gz`。
- 迁移清单：`/root/autodl-tmp/test-eval-migration-20261026.json`；核验记录：`/root/autodl-tmp/test-eval-migration-verified-20261026.json`。
- 激活环境后离线tokenizer加载通过，词表49153；模型模块导入通过，29项GRPO相关单元测试通过，最终评估入口编译通过。
- checkpoint软链接现已指向实际存在的Base文件。最终模型路径保持与53432一致，评估脚本为`/root/s-flm/scripts/evaluate_final_gsm8k_test.py`。
- 仍缺指定语义加权SFT711 EMA权重；尚不能启动完整三方对比，不能以现有其他SFT代替。

## SFT711补齐与最终就绪检查（2026-10-26）

再次连接54592成功，找到历史指定checkpoint并直接迁移至20035的同名路径，文件约660MiB。源和目标SHA256均为`5b2f034c078a7557814cb9d243387ca525f9cb418568556740a569a23427029a`。

最终三模型评估入口以`--preflight-only`通过：Base、SFT711、正向32题和1319题Test哈希均与历史清单一致。另在CPU实际加载SFT模型，确认EMA更新数711、96个参数张量形状匹配，复制EMA后逐张量相等，避免误用raw权重。

三个模型、tokenizer、数据和代码现已齐备。就绪记录：`/root/autodl-tmp/final-test-ready-20261026.json`。本次仅迁移、加载及核验；没有训练、模型生成或benchmark评估，SSH已断开。
