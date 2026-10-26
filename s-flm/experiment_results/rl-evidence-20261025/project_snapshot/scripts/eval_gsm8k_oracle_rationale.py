#!/usr/bin/env python3
"""Test GSM8K PoT continuation with and without gold reasoning as fixed context.

For each validation problem, the model generates a complete Python program.
The oracle condition appends the cleaned GSM8K reference reasoning to the
question before the original separator. This is a diagnostic, not a fair
held-out accuracy benchmark, because it uses gold reasoning as input.
"""
from __future__ import annotations

import ast
import json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import algo
import dataloader
import main as main_module
import sandbox_gsm8k


def _load_rows(path: Path):
  return [json.loads(line) for line in path.read_text().splitlines()
          if line.strip()]


def _reference_steps(reference_answer):
  """Match the cleanup used when creating the reference-comment targets."""
  cleaned = re.sub(r'<<.*?>>', '', reference_answer)
  cleaned = re.sub(r'^\s*####\s*.*$', '', cleaned, flags=re.MULTILINE)
  return '\n'.join(line.strip() for line in cleaned.splitlines()
                   if line.strip())


def _prefix_ids(row, tokenizer, separator, condition):
  question = tokenizer(row['question'].strip(), add_special_tokens=False).input_ids
  sep = tokenizer(separator, add_special_tokens=False).input_ids
  reasoning_ids = []
  if condition == 'oracle_reasoning':
    reasoning = _reference_steps(row['reference_answer'])
    reasoning_ids = tokenizer(
      '\n\nReference reasoning:\n' + reasoning,
      add_special_tokens=False).input_ids
  return [tokenizer.bos_token_id] + question + reasoning_ids + sep


def _sample_indices(size, count):
  rng = random.Random(20261022)
  return sorted(rng.sample(range(size), min(count, size)))


def _strict_parseable(text):
  try:
    ast.parse(text)
    return True
  except (SyntaxError, ValueError, TypeError):
    return False


@torch.no_grad()
def _generate(model, rows, indices, tokenizer, separator, condition,
              steps, seed, batch_size, device, max_length):
  L.seed_everything(seed, workers=True)
  eos_id = tokenizer.eos_token_id
  records = []
  for start in trange(0, len(indices), batch_size,
                      desc=f'{condition}-{steps}-seed{seed}', leave=False):
    batch = indices[start:start + batch_size]
    prefixes = []
    for idx in batch:
      ids = _prefix_ids(rows[idx], tokenizer, separator, condition)
      if len(ids) >= max_length - 8:
        raise ValueError(f"Oracle prefix leaves too little generation room: {rows[idx]['sample_id']}")
      prefixes.append(torch.tensor(ids, dtype=torch.long))
    prefix_tokens, prefix_lengths = main_module._pad_prefix_batch(prefixes, device)
    samples, _ = model.generate_samples(
      num_samples=len(batch), num_steps=steps,
      prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths)
    for j, idx in enumerate(batch):
      plen = int(prefix_lengths[j].item())
      token_ids = samples[j, plen:].detach().cpu().tolist()
      eos_pos = token_ids.index(eos_id) if eos_id in token_ids else None
      tail_ids = token_ids[:eos_pos] if eos_pos is not None else token_ids
      full_program = tokenizer.decode(tail_ids, skip_special_tokens=True)
      record = {
        'dataset_index': idx,
        'sample_id': rows[idx]['sample_id'],
        'condition': condition,
        'eos_position': eos_pos,
        'program': full_program,
        'strict_parseable': _strict_parseable(full_program),
        'correct': bool(sandbox_gsm8k.evaluate_samples(
          full_program, rows[idx]['reference_answer'], timeout_s=2)),
      }
      records.append(record)
  return records


def _summary(records):
  return {
    'n': len(records),
    'eos_observed': sum(r['eos_position'] is not None for r in records),
    'strict_parseable': sum(r['strict_parseable'] for r in records),
    'correct': sum(r['correct'] for r in records),
  }


@hydra.main(version_base=None, config_path='../configs',
            config_name='sft-a800-lora-result-cot-comments-pilot')
def main(config: DictConfig):
  os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
  os.environ.setdefault('HF_HUB_OFFLINE', '1')
  torch.backends.cuda.matmul.allow_tf32 = True
  if not torch.cuda.is_available():
    raise RuntimeError('This diagnostic requires the rented CUDA GPU.')
  device = torch.device('cuda:0')
  config.model.length = 512
  config.data.train_on_prompt = False
  config.data.train_on_pad = True
  config.eval.disable_ema = False
  config.algo.name = 'mdlm'
  config.sampler.predictor = 'ancestral'
  config.sampler.noise_removal = 'ancestral'

  rows = _load_rows(Path(config.data.validation_file))
  tokenizer = dataloader.get_tokenizer(config)
  indices = _sample_indices(len(rows), int(os.environ.get('ORACLE_SUBSET', '64')))
  seeds = [int(s) for s in os.environ.get('ORACLE_SEEDS', '1,2').split(',')]
  steps = int(os.environ.get('ORACLE_STEPS', '128'))
  batch_size = int(os.environ.get('ORACLE_BATCH_SIZE', '8'))
  models = [
    ('base', os.environ.get('BASE_CKPT', str(ROOT / 'checkpoints/tinygsm/mdlm.ckpt')), False),
    ('step711', os.environ.get('STEP711_CKPT', ''), True),
  ]
  if any(not path or not Path(path).is_file() for _, path, _ in models):
    raise FileNotFoundError('Set BASE_CKPT and STEP711_CKPT to existing checkpoint files.')
  out_root = Path(os.environ.get(
    'ORACLE_OUT', '/root/autodl-tmp/s-flm/outputs/eval/20261022-oracle-rationale'))
  out_root.mkdir(parents=True, exist_ok=True)
  summary = {
    'validation_rows': len(rows),
    'subset_indices': indices,
    'sample_ids': [rows[i]['sample_id'] for i in indices],
    'subset_seed': 20261022,
    'steps': steps,
    'seeds': seeds,
    'weights': 'EMA',
    'condition_definitions': {
      'question_only': 'question + original separator; model generates the full Python program',
      'oracle_reasoning': 'question + cleaned GSM8K reference reasoning as text + original separator; model generates the full Python program',
    },
    'models': {},
  }
  all_records = {}
  for label, checkpoint, lora_enabled in models:
    config.lora.enabled = lora_enabled
    config.eval.checkpoint_path = checkpoint
    print(f'Loading {label}: {checkpoint}', flush=True)
    model = algo.MDLM.load_from_checkpoint(
      checkpoint, tokenizer=tokenizer, config=config,
      strict=config.eval.strict_loading, map_location='cpu')
    model.to(device)
    model._eval_mode()
    summary['models'][label] = {}
    for condition in ('question_only', 'oracle_reasoning'):
      for seed in seeds:
        records = _generate(
          model, rows, indices, tokenizer, str(config.data.separator),
          condition, steps, seed, batch_size, device, int(config.model.length))
        key = f'{label}/{condition}/steps{steps}/seed{seed}'
        all_records[key] = records
        summary['models'][label][f'{condition}/seed{seed}'] = _summary(records)
    del model
    torch.cuda.empty_cache()
  summary['elapsed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
  (out_root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
  (out_root / 'records.json').write_text(json.dumps(all_records, indent=2) + '\n')
  (out_root / 'COMPLETE').write_text(summary['elapsed_at'] + '\n')
  print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
  main()
