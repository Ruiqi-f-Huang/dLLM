"""Full-MASK/code-only memorization diagnostic with resumable raw weights."""
import inspect
import json
import random
import sys
import textwrap
import time
import types
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as entry
import algo
import dataloader
from callbacks.overfit_probe import _generate
from scripts.diagnose_overfit16_gradients import tensors, forward, measure, frozen_hash


def fp32_backbone(model):
    original = type(model.backbone).forward
    source = textwrap.dedent(inspect.getsource(original))
    assert source.count('enabled=x.is_cuda') == 1
    local = {}
    exec(source.replace('enabled=x.is_cuda', 'enabled=False'), original.__globals__, local)
    model.backbone.forward = types.MethodType(local['forward'], model.backbone)


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


@torch.no_grad()
def evaluate(model, data, rows, out, step):
    state = rng_state()
    was_training = model.training
    model.eval()
    try:
        stats = measure(model, data, rows)
        dataset = [{'input_ids': data[0][i, :int(data[1][i])].tolist(),
                    'response_ground_truth': row['reference_answer']} for i, row in enumerate(rows)]
        records = _generate(model, model.tokenizer, dataset)
        for row, record in zip(rows, records):
            record['sample_id'] = row['sample_id']
        result = {'step': step, 'weights': 'Base EMA initialization' if step == 0 else 'raw',
                  'precision': 'FP32; TF32 disabled', 'nfe': 128, 'seed': 1,
                  'correct': sum(r['full_correct'] for r in records),
                  'parseable': sum(r['full_strict_parseable'] for r in records),
                  'n': len(rows), 'full_mask_code': stats, 'records': records}
        (out / f'probe/step-{step:06d}.json').write_text(json.dumps(result, indent=2))
        print('CODEONLY_PROBE ' + json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
        return result
    finally:
        model.train(was_training)
        restore_rng(state)


@hydra.main(version_base=None, config_path='../configs', config_name='sft-overfit16-fullmask-codeonly')
def run(config):
    import os
    import wandb
    os.environ.pop('WANDB_MODE', None)
    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    out = Path(config.checkpointing.save_dir)
    resume = bool(config.checkpointing.resume_from_ckpt)
    max_steps = int(config.trainer.max_steps)
    if (out / 'probe/step-000000.json').exists():
        raise FileExistsError('Refusing to overwrite an existing experiment')
    for sub in ['probe', 'checkpoints']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_container(config, resolve=True)
    resolved['diagnostic'] = {'backbone_autocast': 'disabled on experiment instance',
                              'effective_precision': 'FP32; TF32 disabled',
                              'loss_region': 'non-docstring code and EOS',
                              'corruption': 'all suffix positions including docstring and PAD',
                              'generation_precision': 'FP32'}
    (out / 'config.json').write_text(json.dumps(resolved, indent=2))
    rows = [json.loads(s) for s in Path(config.data.train_file).read_text().splitlines()]
    assert len(rows) == 16 and max_steps > 0
    tokenizer = dataloader.get_tokenizer(config)
    data = tensors(rows, tokenizer, str(config.data.separator))
    checkpoint = None
    initial_step = 0
    source = None
    if resume:
        source = Path(config.checkpointing.resume_ckpt_path)
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        assert checkpoint['checkpoint_format'] == 'custom-fullmask-codeonly-v1'
        for key in ['algo', 'model', 'lora', 'optim', 'lr_scheduler', 'noise', 'seed']:
            assert checkpoint['config'][key] == resolved[key], f'Resume setting changed: {key}'
        assert checkpoint['config']['loader']['batch_size'] == resolved['loader']['batch_size']
        initial_step = int(checkpoint['global_step'])
        assert initial_step < max_steps
        model = algo.MDLM(config, tokenizer=tokenizer)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model = entry._initialize_finetune(algo.MDLM, config, tokenizer)
    model.ema = None
    model.to('cuda')
    fp32_backbone(model)
    frozen_before = frozen_hash(model)
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    assert all(name.endswith(('lora_A', 'lora_B')) for name, _ in parameters)
    assert all(value.dtype == torch.float32 for _, value in parameters)
    params = [value for _, value in parameters]
    initial = [value.detach().clone() for value in params]
    optimizer = torch.optim.AdamW(params, lr=float(config.optim.lr),
                                 betas=(float(config.optim.beta1), float(config.optim.beta2)),
                                 eps=float(config.optim.eps), weight_decay=float(config.optim.weight_decay))
    scheduler = hydra.utils.instantiate(config.lr_scheduler, optimizer=optimizer)
    order = []
    restored_rng = None
    summary = []
    losses = []
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Adam step counters belong on CPU when capturable=False.
        for parameter, state in optimizer.state.items():
            for name, value in state.items():
                if torch.is_tensor(value) and name != 'step':
                    state[name] = value.to(parameter.device)
        order = list(checkpoint['pending_order'])
        restored_rng = checkpoint['rng_state']
        previous = source.parent.parent
        summary = json.loads((previous / 'summary.json').read_text())
        losses = json.loads((previous / 'training_metrics.json').read_text())
        summary = [r for r in summary if r['step'] <= initial_step]
        losses = [r for r in losses if r['optimizer_step'] <= initial_step]
        model.eval()
        restored_measure = measure(model, data, rows)
        expected_measure = next(r['full_mask_code'] for r in summary if r['step'] == initial_step)
        assert abs(restored_measure['code_ce'] - expected_measure['code_ce']) < 1e-5
        assert restored_measure['top1_correct_tokens'] == expected_measure['top1_correct_tokens']
        print('RESUME ' + json.dumps({'step': initial_step, 'lr': optimizer.param_groups[0]['lr'],
              'scheduler_last_epoch': scheduler.last_epoch, 'pending_order': order,
              'restored_full_mask_code': restored_measure,
              'optimizer_steps': sorted({int(s['step']) for s in optimizer.state.values()})}), flush=True)
        del checkpoint
    wb = None
    if config.wandb is not None:
        resume_wandb = ({'id': config.wandb.id, 'entity': config.wandb.entity, 'resume': 'must'}
                        if resume else {})
        wb = wandb.init(project=config.wandb.project, group=config.wandb.group,
                        job_type=config.wandb.job_type, name=config.wandb.name,
                        mode='online', dir=str(out), config=None if resume else {**resolved,
                        'experiment_note': 'all suffix positions MASK; loss only non-docstring code+EOS; FP32 train and generation; raw weights'},
                        settings=wandb.Settings(init_timeout=60), **resume_wandb)
        if resume:
            wb.config.update({'trainer': resolved['trainer'], 'continuation_from_step': initial_step},
                             allow_val_change=True)
        wb.define_metric('optimizer_step')
        wb.define_metric('*', step_metric='optimizer_step')
        (out / 'wandb_run.json').write_text(json.dumps({'url': wb.url, 'id': wb.id, 'entity': wb.entity}, indent=2))
        print('WANDB_RUN_URL ' + wb.url, flush=True)
    started = time.monotonic()

    def probe(step):
        result = evaluate(model, data, rows, out, step)
        summary.append({k: v for k, v in result.items() if k != 'records'})
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        return {'memorization/correct': result['correct'], 'memorization/parseable': result['parseable'],
                'full_mask/code_ce': result['full_mask_code']['code_ce'],
                'full_mask/code_top1': result['full_mask_code']['top1_accuracy'],
                'full_mask/gold_probability': result['full_mask_code']['mean_gold_probability']}

    if not resume:
        baseline = probe(0)
        if wb:
            wb.log({'optimizer_step': 0, **baseline}, step=0)
    model.train()
    batch_size = int(config.loader.batch_size)
    assert batch_size == 8
    if restored_rng is not None:
        restore_rng(restored_rng)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    milestones = [500, 1000, 2000, 3000] if resume else [50, 200]
    for step in range(initial_step + 1, max_steps + 1):
        if not order:
            order = torch.randperm(len(rows)).tolist()
        indices, order = order[:batch_size], order[batch_size:]
        batch = tuple(x[indices] for x in data)
        optimizer.zero_grad(set_to_none=True)
        ce, _, masked, _, _, _ = forward(model, batch, 1., [rows[i] for i in indices])
        labels = batch[3].to('cuda')
        code_mask = labels == 2
        assert bool(masked[labels > 0].all())
        loss = ce[code_mask].mean()
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in params)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float(config.trainer.gradient_clip_val)))
        lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        metrics = {'optimizer_step': step, 'train/code_ce': float(loss.detach()),
                   'train/grad_norm_pre_clip': grad_norm, 'train/lr': lr}
        losses.append(metrics)
        if step in milestones or step == max_steps:
            # Custom-loop checkpoints retain weights, optimizer, scheduler and RNG;
            # these are inference checkpoints, not Lightning loop checkpoints.
            torch.save({'state_dict': model.state_dict(), 'global_step': step,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'rng_state': rng_state(), 'pending_order': order,
                        'config': resolved, 'checkpoint_format': 'custom-fullmask-codeonly-v1'},
                       out / f'checkpoints/step-{step:06d}.ckpt')
            metrics.update(probe(step))
        if wb:
            wb.log(metrics, step=step)
        if step % 10 == 0:
            print('TRAIN ' + json.dumps(metrics), flush=True)
    (out / 'training_metrics.json').write_text(json.dumps(losses, indent=2))
    integrity = {'initial_step': initial_step, 'final_step': max_steps,
                 'successful_updates': max_steps - initial_step,
                 'presentations': (max_steps - initial_step) * batch_size,
                 'frozen_parameters_unchanged': frozen_before == frozen_hash(model),
                 'changed_lora_elements': sum(int(torch.count_nonzero(v.detach() - old)) for v, old in zip(params, initial)),
                 'wall_seconds_including_evaluation': time.monotonic() - started,
                 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2}
    assert integrity['frozen_parameters_unchanged'] and integrity['changed_lora_elements'] > 0
    (out / 'integrity.json').write_text(json.dumps(integrity, indent=2))
    if wb:
        artifact = wandb.Artifact('overfit16-fullmask-codeonly-logs', type='experiment-logs')
        for name in ['summary.json', 'training_metrics.json', 'integrity.json', 'config.json']:
            artifact.add_file(str(out / name), name=name)
        wb.log_artifact(artifact)
        wb.finish()
    (out / 'COMPLETE').write_text(f'{max_steps} steps and all evaluations completed\n')
    print('FULLMASK_CODEONLY_COMPLETE ' + json.dumps(integrity), flush=True)


if __name__ == '__main__':
    run()
