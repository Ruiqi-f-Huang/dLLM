"""Evaluate memorization of a fixed training subset without changing train RNG."""
import json
import random
import ast
import re
from pathlib import Path

import numpy as np
import torch

from callbacks.milestone_checkpoint import MilestoneCheckpoint
import sandbox_gsm8k
from trainer_base import TrainingContext


def _make_target_tensors(rows, tokenizer, length, separator):
    ids, prefixes, ends = [], [], []
    sep = tokenizer(separator, add_special_tokens=False).input_ids
    for row in rows:
        prefix = ([tokenizer.bos_token_id] + tokenizer(
            row['question'].strip(), add_special_tokens=False).input_ids + sep)
        target = tokenizer(row['code'].strip(), add_special_tokens=False).input_ids
        tokens = prefix + target + [tokenizer.eos_token_id]
        if len(tokens) > length:
            raise ValueError(f"Overlong target: {row['sample_id']}")
        prefixes.append(len(prefix))
        ends.append(len(tokens))
        ids.append(tokens + [tokenizer.pad_token_id] * (length - len(tokens)))
    return torch.tensor(ids), torch.tensor(prefixes), torch.tensor(ends)


def _parseable(text):
    fence = re.search(r'```(?:python)?\s*(.*?)```', text, flags=re.I | re.S)
    text = fence.group(1) if fence else text
    start = text.find('def ')
    try:
        ast.parse(text[start:].strip() if start >= 0 else text.strip())
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


@torch.no_grad()
def _generate(model, tokenizer, dataset):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    records = []
    for start in range(0, len(dataset), 8):
        rows = dataset[start:start + 8]
        lengths = torch.tensor([len(r['input_ids']) for r in rows], device=model.device)
        # Match main._pad_prefix_batch: unused slots are zeros; lengths gate them.
        prefixes = torch.zeros(len(rows), int(lengths.max()), dtype=torch.long, device=model.device)
        for i, row in enumerate(rows):
            prefixes[i, :int(lengths[i])] = torch.tensor(row['input_ids'], device=model.device)
        samples, _ = model.generate_samples(num_samples=len(rows), num_steps=128,
                                            prefix_tokens=prefixes, prefix_lengths=lengths)
        for i, row in enumerate(rows):
            token_ids = samples[i, int(lengths[i]):].cpu().tolist()
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            records.append({'token_ids': token_ids, 'full_response': text,
                            'full_strict_parseable': _parseable(text),
                            'full_correct': bool(sandbox_gsm8k.evaluate_samples(
                                text, row['response_ground_truth'], timeout_s=2))})
    return records


class OverfitProbe(MilestoneCheckpoint):
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        if int(trainer.global_step) == 0:
            self._evaluate(trainer, pl_module, 0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        should_evaluate = step in self.milestones and step not in self.saved
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if should_evaluate:
            result = self._evaluate(trainer, pl_module, step)
            if result['correct'] >= 14:
                trainer.should_stop = True

    @torch.no_grad()
    def _evaluate(self, trainer, model, step):
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
               torch.cuda.get_rng_state_all())
        was_training = model.training
        try:
            # No _eval_mode(): that method swaps in EMA; this probe needs raw.
            model.eval()
            rows = [json.loads(s) for s in Path(model.config.data.train_file)
                    .read_text().splitlines() if s.strip()]
            tokenizer = model.tokenizer
            ids, prefixes, ends = _make_target_tensors(
                rows, tokenizer, 512, str(model.config.data.separator))
            dataset = [{'input_ids': ids[i, :int(prefixes[i])].tolist(),
                        'response_ground_truth': r['reference_answer']}
                       for i, r in enumerate(rows)]
            torch.manual_seed(17)
            torch.cuda.manual_seed_all(17)
            total, count = 0.0, 0
            positions = torch.arange(512, device=model.device)[None, :]
            for start in range(0, len(rows), 8):
                x = ids[start:start + 8].to(model.device)
                prefix = prefixes[start:start + 8, None].to(model.device)
                end = ends[start:start + 8, None].to(model.device)
                noise_mask = positions >= prefix
                code_mask = noise_mask & (positions < end)
                losses, _ = model.nll(x, x, TrainingContext(),
                                      valid_tokens=noise_mask)
                total += float(losses[code_mask].sum())
                count += int(code_mask.sum())
            records = _generate(model, tokenizer, dataset)
            for row, record in zip(rows, records):
                record['sample_id'] = row['sample_id']
            result = {
                'step': step, 'weights': 'Base EMA initialization' if step == 0 else 'raw',
                'n': len(rows), 'seed': 1, 'nfe': 128,
                'correct': sum(r['full_correct'] for r in records),
                'parseable': sum(r['full_strict_parseable'] for r in records),
                'code_eos_nll': total / count,
                'nll_seed': 17, 'records': records,
            }
            out = self.dirpath.parent / 'probe'
            out.mkdir(exist_ok=True)
            (out / f'step-{step:06d}.json').write_text(json.dumps(result, indent=2))
            print('OVERFIT_PROBE ' + json.dumps(
                {k: v for k, v in result.items() if k != 'records'}), flush=True)
            return result
        finally:
            model.train(was_training)
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.set_rng_state(rng[2])
            torch.cuda.set_rng_state_all(rng[3])
