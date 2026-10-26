"""Run the authorized single-A800 LIFT pilot, preserving isolated evidence."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def smoke(run):
    import lightning as L
    import torch
    from hydra import compose, initialize_config_dir
    import main
    import algo
    import dataloader

    stats = []
    for arm in ('b', 'c'):
        L.seed_everything(1)
        overrides = [f'training.finetune_path={ROOT}/checkpoints/tinygsm/mdlm.ckpt',
                     f'data.train_file={ROOT}/data/gsm8k_pot/sft_train.jsonl',
                     f'data.validation_file={ROOT}/data/gsm8k_pot/sft_validation.jsonl',
                     f'data.cache_dir={run}/data_cache']
        with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
            cfg = compose(config_name=f'sft-lift-{arm}', overrides=overrides)
        tokenizer = dataloader.get_tokenizer(cfg)
        dataset = dataloader.get_gsm8k_pot_dataset(cfg, tokenizer, 'train')
        validation = dataloader.get_gsm8k_pot_dataset(cfg, tokenizer, 'valid')
        counts = {'train': len(dataset), 'validation': len(validation),
                  'train_region_fallback': sum(not bool(r['lift_region_ok']) for r in dataset),
                  'validation_region_fallback': sum(not bool(r['lift_region_ok']) for r in validation)}
        assert counts['train'] == 5682 and counts['validation'] == 632
        write_json(run / 'region_audit.json', counts)
        model = main._initialize_finetune(algo.MDLM, cfg, tokenizer)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        adapters = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        model = model.cuda().train()
        model.log = lambda *args, **kwargs: None
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=8)))
        batch = {k: v.cuda() for k, v in batch.items()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.optim.lr)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model._loss(batch['input_ids'], batch['attention_mask'],
                               noise_tokens=batch['noise_mask'], lift_region=batch['lift_region'],
                               train_mode=True)
        loss.loss.backward()
        assert torch.isfinite(loss.loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        current = dict(model.named_parameters())
        assert all(torch.equal(current[n].detach().cpu(), p) for n, p in frozen.items())
        changed = sum(not torch.equal(current[n].detach().cpu(), p) for n, p in adapters.items())
        assert changed > 0
        torch.cuda.synchronize()
        stats.append({'arm': arm, 'batch_size': 8, 'loss': float(loss.loss),
                      'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                      'seconds': time.monotonic() - start, 'changed_adapter_tensors': changed,
                      'frozen_parameters_unchanged': True, **counts})
        del loss, model, optimizer, current, frozen, adapters, batch
        gc.collect()
        torch.cuda.empty_cache()
    write_json(run / 'smoke.json', stats)


def command(run, label, args):
    write_json(run / 'status.json', {'phase': label, 'started': time.time(), 'pid': os.getpid()})
    with (run / f'{label}.log').open('w') as log:
        subprocess.run([sys.executable, '-u', *args], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve(); run.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    write_json(run / 'source_manifest.json', {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink()
        and p.suffix in ('.py', '.yaml', '.sh')
        and not any(x in p.parts for x in ('data', 'checkpoints', 'data_cache', '__pycache__'))})
    if not (run / 'smoke.json').exists():
        write_json(run / 'status.json', {'phase': 'smoke', 'pid': os.getpid()})
        smoke(run)
    if args.smoke_only:
        return
    for arm in 'abc':
        directory = run / f'train-{arm}'
        if (directory / 'checkpoints/step-000200.ckpt').exists():
            continue
        command(run, f'train-{arm}', ['main.py', '--config-name', f'sft-lift-{arm}',
            f'training.finetune_path={ROOT}/checkpoints/tinygsm/mdlm.ckpt',
            f'data.train_file={ROOT}/data/gsm8k_pot/sft_train.jsonl',
            f'data.validation_file={ROOT}/data/gsm8k_pot/sft_validation.jsonl',
            f'data.cache_dir={run}/data_cache', f'hydra.run.dir={directory}',
            'wandb.offline=true', f'wandb.name=lift-{arm}-200',
            '+trainer.enable_progress_bar=false'])
    for label in ['base', 'a100', 'a200', 'b100', 'b200', 'c100', 'c200']:
        ckpt = (ROOT / 'checkpoints/tinygsm/mdlm.ckpt' if label == 'base' else
                run / f'train-{label[0]}/checkpoints/step-{int(label[1:]):06d}.ckpt')
        for seed in (1, 2):
            directory = run / 'evaluation' / f'{label}-seed{seed}'
            if (directory / 'results.json').exists():
                continue
            command(run, f'eval-{label}-seed{seed}', ['main.py', 'mode=gsm8k_eval',
                f'eval.checkpoint_path={ckpt}', 'eval.disable_ema=false',
                f'lora.enabled={str(label != "base").lower()}',
                'data=gsm8k-pot', 'model=small', 'model.length=512', 'algo=mdlm',
                'sampler=ancestral', 'sampler.steps=32', 'sampler.temperature=0.1',
                'backbone_compute_dtype=bfloat16',
                f'data.validation_file={ROOT}/data/gsm8k_pot/sft_validation.jsonl',
                'loader.eval_batch_size=32', 'loader.num_workers=0',
                'trainer.num_nodes=1', 'trainer.devices=1', f'seed={seed}',
                f'gsm8k.output_dir={directory}', f'hydra.run.dir={directory}'])
    command(run, 'analysis', ['scripts/analyze_lift_pilot.py', str(run)])
    (run / 'COMPLETE').touch()
    write_json(run / 'status.json', {'phase': 'complete', 'finished': time.time()})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
