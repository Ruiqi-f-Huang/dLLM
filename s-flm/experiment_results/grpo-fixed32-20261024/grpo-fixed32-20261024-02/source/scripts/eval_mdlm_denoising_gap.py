#!/usr/bin/env python3
"""Combined GSM8K PoT denoising-condition and sampling diagnostic.

Loads EMA checkpoints sequentially, measures masked target-token CE at fixed
corruption ratios, then samples a deterministic validation subset at selected
ancestral step counts. Sampling records preserve token IDs so EOS-truncated
and untruncated outputs can be compared without another generation pass.
"""
from __future__ import annotations

import ast
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from tqdm import trange

# When invoked as ``python scripts/eval_mdlm_denoising_gap.py``, Python puts
# the scripts directory (not the repository root) on sys.path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import algo
import dataloader
import main as main_module
import sandbox_gsm8k


MASK_RATES = (0.25, 0.50, 0.75, 0.90, 1.00)
SAMPLE_STEPS = (32, 128)


def _load_rows(path: Path):
  return [json.loads(line) for line in path.read_text().splitlines()
          if line.strip()]


def _make_target_tensors(rows, tokenizer, length, separator):
  input_ids, prefix_lengths, target_ends = [], [], []
  sep_ids = tokenizer(separator, add_special_tokens=False).input_ids
  for row in rows:
    prefix = ([tokenizer.bos_token_id]
              + tokenizer(row['question'].strip(),
                           add_special_tokens=False).input_ids
              + sep_ids)
    target = (tokenizer(row['code'].strip(),
                        add_special_tokens=False).input_ids
              + [tokenizer.eos_token_id])
    ids = prefix + target
    if len(ids) > length:
      raise ValueError(f"{row['sample_id']} has {len(ids)} tokens > {length}")
    input_ids.append(ids + [tokenizer.pad_token_id] * (length - len(ids)))
    prefix_lengths.append(len(prefix))
    target_ends.append(len(prefix) + len(target))
  return (torch.tensor(input_ids, dtype=torch.long),
          torch.tensor(prefix_lengths, dtype=torch.long),
          torch.tensor(target_ends, dtype=torch.long))


def _fixed_corruption(x0, prefix_lengths, target_ends, rate, pad_id,
                      mask_id, sample_ids, device):
  """Mask an exact fraction within code/EOS and PAD regions per example."""
  xt = x0.clone()
  batch_mask = torch.zeros_like(x0, dtype=torch.bool)
  for i, sample_id in enumerate(sample_ids):
    start = int(prefix_lengths[i])
    end = int(target_ends[i])
    regions = (list(range(start, end)), list(range(end, x0.shape[1])))
    for region_idx, positions in enumerate(regions):
      if not positions or rate <= 0:
        continue
      count = len(positions) if rate >= 1 else int(round(rate * len(positions)))
      if count == 0:
        continue
      seed_text = f'{sample_id}|{rate:.4f}|{region_idx}|20261022'
      rng = random.Random(seed_text)
      chosen = sorted(rng.sample(positions, count))
      batch_mask[i, chosen] = True
  xt = xt.to(device)
  x0_device = x0.to(device)
  mask = batch_mask.to(device)
  xt[mask] = mask_id
  return x0_device, xt, mask


@torch.no_grad()
def _mask_sweep(model, rows, tokenizer, batch_size, device):
  length = int(model.config.model.length)
  all_ids, all_prefix, all_ends = _make_target_tensors(
    rows, tokenizer, length, str(model.config.data.separator))
  code_mask = (all_ids != tokenizer.pad_token_id)
  code_mask &= (torch.arange(length)[None, :] >= all_prefix[:, None])
  totals = {rate: {'nll_sum': 0.0, 'correct': 0, 'tokens': 0,
                   'masked': 0, 'suffix_tokens': 0}
            for rate in MASK_RATES}
  eps = float(model.config.noise.eps)

  for start in trange(0, len(rows), batch_size, desc='mask sweep', leave=False):
    stop = min(start + batch_size, len(rows))
    batch_rows = rows[start:stop]
    x0 = all_ids[start:stop]
    prefix = all_prefix[start:stop]
    ends = all_ends[start:stop]
    ids = [r['sample_id'] for r in batch_rows]
    for rate in MASK_RATES:
      x0_dev, xt, masked = _fixed_corruption(
        x0, prefix, ends, rate, tokenizer.pad_token_id,
        model.mask_index, ids, device)
      # For the configured log-linear schedule, P(mask) = 1 - alpha(t).
      t_value = min(1.0, rate / max(1.0 - eps, 1e-8))
      t = torch.full((len(batch_rows),), t_value, device=device,
                     dtype=torch.float32)
      _, alpha = model.noise(t)
      sigma = model._sigma_from_alphat(alpha.unsqueeze(-1))
      log_probs = model.forward(xt=xt, sigma=sigma)
      targets = x0_dev
      scored = masked & code_mask[start:stop].to(device)
      if scored.any():
        selected = log_probs[scored]
        target = targets[scored]
        loss_sum = float(-selected.gather(1, target[:, None]).sum().item())
        correct = int((selected.argmax(-1) == target).sum().item())
        count = int(target.numel())
      else:
        loss_sum, correct, count = 0.0, 0, 0
      stats = totals[rate]
      stats['nll_sum'] += loss_sum
      stats['correct'] += correct
      stats['tokens'] += count
      stats['masked'] += int((masked & code_mask[start:stop].to(device)).sum())
      stats['suffix_tokens'] += int(code_mask[start:stop].sum())

  result = []
  for rate in MASK_RATES:
    x = totals[rate]
    result.append({
      'requested_mask_rate': rate,
      'actual_code_eos_mask_rate': (
        x['masked'] / x['suffix_tokens'] if x['suffix_tokens'] else 0.0),
      'masked_code_eos_tokens': x['tokens'],
      'masked_token_ce': x['nll_sum'] / max(x['tokens'], 1),
      'masked_token_accuracy': x['correct'] / max(x['tokens'], 1),
    })
  return result


def _extract_code(text):
  fence = re.search(r'```(?:python)?\s*(.*?)```', text,
                    flags=re.I | re.S)
  if fence:
    text = fence.group(1)
  start = text.find('def ')
  return text[start:].strip() if start >= 0 else text.strip()


def _is_strictly_parseable(text):
  try:
    ast.parse(_extract_code(text))
    return True
  except (SyntaxError, ValueError, TypeError):
    return False


def _sample_subset(eval_dataset, count):
  n = min(count, len(eval_dataset))
  rng = random.Random(20261022)
  return sorted(rng.sample(range(len(eval_dataset)), n))


@torch.no_grad()
def _generate_subset(model, config, tokenizer, eval_dataset, indices,
                     steps, seed, batch_size, device):
  L.seed_everything(seed, workers=True)
  records = []
  eos_id = tokenizer.eos_token_id
  for start in trange(0, len(indices), batch_size,
                      desc=f'generate-{steps}-seed{seed}', leave=False):
    batch_indices = indices[start:start + batch_size]
    id_lists = [torch.as_tensor(eval_dataset[i]['input_ids'],
                                dtype=torch.long)
                for i in batch_indices]
    prefix_tokens, prefix_lengths = main_module._pad_prefix_batch(
      id_lists, device)
    samples, metadata = model.generate_samples(
      num_samples=len(batch_indices), num_steps=steps,
      prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths)
    for j, dataset_index in enumerate(batch_indices):
      plen = int(prefix_lengths[j].item())
      token_ids = samples[j, plen:].detach().cpu().tolist()
      eos_pos = token_ids.index(eos_id) if eos_id in token_ids else None
      trimmed_ids = token_ids[:eos_pos] if eos_pos is not None else token_ids
      full_text = tokenizer.decode(token_ids, skip_special_tokens=True)
      trimmed_text = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
      target = eval_dataset[dataset_index]['response_ground_truth']
      records.append({
        'dataset_index': int(dataset_index),
        'sample_id': None,
        'eos_position': eos_pos,
        'token_ids': token_ids,
        'full_response': full_text,
        'eos_truncated_response': trimmed_text,
        'full_strict_parseable': _is_strictly_parseable(full_text),
        'eos_strict_parseable': _is_strictly_parseable(trimmed_text),
        'full_correct': bool(sandbox_gsm8k.evaluate_samples(
          full_text, target, timeout_s=2)),
        'eos_correct': bool(sandbox_gsm8k.evaluate_samples(
          trimmed_text, target, timeout_s=2)),
      })
  return records


def _summarize_generation(records):
  n = len(records)
  eos = [r for r in records if r['eos_position'] is not None]
  return {
    'n': n,
    'eos_observed': len(eos),
    'eos_rate': len(eos) / max(n, 1),
    'full_parseable': sum(r['full_strict_parseable'] for r in records),
    'eos_parseable': sum(r['eos_strict_parseable'] for r in records),
    'full_correct': sum(r['full_correct'] for r in records),
    'eos_correct': sum(r['eos_correct'] for r in records),
    'eos_to_correct_changes': sum(
      (not r['full_correct']) and r['eos_correct'] for r in records),
    'eos_to_wrong_changes': sum(
      r['full_correct'] and (not r['eos_correct']) for r in records),
    'median_eos_position': (
      sorted(r['eos_position'] for r in eos)[len(eos) // 2]
      if eos else None),
  }


@hydra.main(version_base=None, config_path='../configs',
            config_name='sft-a800-lora-result-lr1e5')
def main(config: DictConfig):
  os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
  os.environ.setdefault('HF_HUB_OFFLINE', '1')
  torch.backends.cuda.matmul.allow_tf32 = True
  if not torch.cuda.is_available():
    raise RuntimeError('This diagnostic must run on the rented CUDA GPU.')
  device = torch.device('cuda:0')
  config.model.length = 512
  config.data.train_on_prompt = False
  config.data.train_on_pad = True
  config.eval.disable_ema = False
  config.algo.name = 'mdlm'
  config.sampler.predictor = 'ancestral'
  config.sampler.noise_removal = 'ancestral'

  out_root = Path(os.environ.get(
    'DIAG_OUT', '/root/autodl-tmp/s-flm/outputs/eval/'
    '20261022-denoising-gap'))
  out_root.mkdir(parents=True, exist_ok=True)
  valid_path = Path(config.data.validation_file)
  rows = _load_rows(valid_path)
  tokenizer = dataloader.get_tokenizer(config)
  eval_dataset = dataloader.get_gsm8k_pot_eval_dataset(config, tokenizer)
  subset_size = int(os.environ.get('DIAG_GENERATION_SUBSET', '128'))
  subset_indices = _sample_subset(eval_dataset, subset_size)
  sample_ids = [rows[i]['sample_id'] for i in subset_indices]
  device_batch = int(os.environ.get('DIAG_BATCH_SIZE', '8'))
  seeds = [int(s) for s in os.environ.get('DIAG_SEEDS', '1').split(',')
           if s.strip()]
  sample_steps = [int(s) for s in os.environ.get(
    'DIAG_SAMPLE_STEPS', '32,128').split(',') if s.strip()]
  selected_labels = {
    s.strip() for s in os.environ.get(
      'DIAG_CHECKPOINTS', 'base,step200,step711').split(',') if s.strip()}
  skip_mask_sweep = os.environ.get('DIAG_SKIP_MASK_SWEEP', '0') == '1'

  all_checkpoints = [
    ('base', os.environ.get('BASE_CKPT',
      str(ROOT / 'checkpoints/tinygsm/mdlm.ckpt')), False),
    ('step200', os.environ.get('STEP200_CKPT', ''), True),
    ('step711', os.environ.get('STEP711_CKPT', ''), True),
    ('highnoise200', os.environ.get('HIGHNOISE_CKPT', ''), True),
    ('cotcomments200', os.environ.get('COTCOMMENTS_CKPT', ''), True),
  ]
  unknown_labels = selected_labels - {x[0] for x in all_checkpoints}
  if unknown_labels:
    raise ValueError(f'Unknown DIAG_CHECKPOINTS labels: {sorted(unknown_labels)}')
  checkpoints = [x for x in all_checkpoints if x[0] in selected_labels]
  if any(not path or not Path(path).is_file() for _, path, _ in checkpoints):
    raise FileNotFoundError('Set BASE_CKPT, STEP200_CKPT, STEP711_CKPT to existing files.')

  summary = {
    'validation_file': str(valid_path),
    'validation_rows': len(rows),
    'generation_subset_size': len(subset_indices),
    'generation_subset_seed': 20261022,
    'generation_sample_ids': sample_ids,
    'mask_rates': list(MASK_RATES),
    'generation_steps': sample_steps,
    'generation_seeds': seeds,
    'mask_sweep_skipped': skip_mask_sweep,
    'models': {},
  }
  all_generation = {}
  for label, checkpoint, lora_enabled in checkpoints:
    print(f'Loading {label}: {checkpoint}', flush=True)
    config.lora.enabled = lora_enabled
    config.eval.checkpoint_path = checkpoint
    model = algo.MDLM.load_from_checkpoint(
      checkpoint, tokenizer=tokenizer, config=config,
      strict=config.eval.strict_loading, map_location='cpu')
    model.to(device)
    model._eval_mode()  # evaluate EMA weights for consistent comparisons
    start_time = time.monotonic()
    mask_results = ([] if skip_mask_sweep else _mask_sweep(
      model, rows, tokenizer, device_batch, device))
    model_summary = {
      'checkpoint': checkpoint,
      'weights': 'EMA',
      'mask_sweep': mask_results,
      'generation': {},
      'elapsed_seconds': None,
    }
    for seed in seeds:
      for steps in sample_steps:
        records = _generate_subset(
          model, config, tokenizer, eval_dataset, subset_indices,
          steps, seed=seed, batch_size=device_batch, device=device)
        for item, dataset_index in zip(records, subset_indices):
          item['sample_id'] = rows[dataset_index]['sample_id']
        key = f'{label}/steps{steps}/seed{seed}'
        all_generation[key] = records
        model_summary['generation'][f'{steps}/seed{seed}'] = (
          _summarize_generation(records))
        del records
    model_summary['elapsed_seconds'] = time.monotonic() - start_time
    summary['models'][label] = model_summary
    print(json.dumps(model_summary, indent=2), flush=True)
    del model
    torch.cuda.empty_cache()

  (out_root / 'denoising_gap_summary.json').write_text(
    json.dumps(summary, indent=2) + '\n')
  (out_root / 'generation_raw_tokens.json').write_text(
    json.dumps(all_generation, indent=2) + '\n')
  (out_root / 'COMPLETE').write_text(
    f'diagnostic_complete={time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n')
  print(f"Saved combined diagnostic to {out_root}", flush=True)


if __name__ == '__main__':
  main()
