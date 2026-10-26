import json, subprocess, sys, time
from pathlib import Path
run=Path('/root/autodl-tmp/s-flm/lift-pilot-20261023-01')
root=run/'source'
for seed in (1,2):
    out=run/'evaluation'/f'c711-seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    args=['main.py','mode=gsm8k_eval',f'eval.checkpoint_path={run}/train-c-711/checkpoints/step-000711.ckpt','eval.disable_ema=false','lora.enabled=true','data=gsm8k-pot','model=small','model.length=512','algo=mdlm','sampler=ancestral','sampler.steps=32','sampler.temperature=0.1','backbone_compute_dtype=bfloat16',f'data.validation_file={root}/data/gsm8k_pot/sft_validation.jsonl','loader.eval_batch_size=32','loader.num_workers=0','trainer.num_nodes=1','trainer.devices=1',f'seed={seed}',f'gsm8k.output_dir={out}',f'hydra.run.dir={out}']
    (out/'command.json').write_text(json.dumps(args,indent=2))
    with (out/'eval.log').open('w') as log:
        subprocess.run([sys.executable,'-u',*args],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
(run/'evaluation'/'C711_COMPLETE').write_text(str(time.time()))
