#!/usr/bin/env bash
set -euo pipefail
source /root/s-flm/activate.sh
RUN=/root/autodl-tmp/s-flm/outputs/base-test-temp01-32nfe-20261023-01
python -u -m main mode=gsm8k_eval eval.checkpoint_path=/root/s-flm/checkpoints/tinygsm/mdlm.ckpt eval.disable_ema=false lora.enabled=false data=gsm8k-test data.data_path=/root/s-flm/data/gsm8k_test.json data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M +data.tokenizer_revision=1d461723eec654e65efdc40cf49301c89c0c92f4 data.cache_dir=/root/s-flm/data_cache model=small model.length=512 algo=mdlm sampler=ancestral sampler.temperature=0.1 sampler.steps=32 loader.eval_batch_size=32 loader.num_workers=4 trainer.num_nodes=1 trainer.devices=1 seed=1 gsm8k.output_dir="$RUN/eval" +wandb.offline=true hydra.run.dir="$RUN/eval" > "$RUN/eval.log" 2>&1
touch "$RUN/COMPLETE"
