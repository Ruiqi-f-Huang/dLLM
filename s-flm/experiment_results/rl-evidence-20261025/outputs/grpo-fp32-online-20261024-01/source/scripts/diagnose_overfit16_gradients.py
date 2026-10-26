"""Partition LoRA gradients and test one code-only update on disposable models."""
import ast
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import hydra
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as entry
import algo
import dataloader
from scripts.eval_mdlm_denoising_gap import _make_target_tensors, _fixed_corruption


def tensors(rows, tokenizer, separator):
    ids, prefixes, ends = _make_target_tensors(rows, tokenizer, 512, separator)
    labels = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        code = row['code'].strip()
        doc = ast.parse(code).body[0].body[0]
        lines = code.splitlines(keepends=True)
        a = sum(map(len, lines[:doc.lineno - 1])) + len(lines[doc.lineno - 1].encode()[:doc.col_offset].decode())
        b = sum(map(len, lines[:doc.end_lineno - 1])) + len(lines[doc.end_lineno - 1].encode()[:doc.end_col_offset].decode())
        labels[i, int(prefixes[i]):int(ends[i])] = 2
        labels[i, int(ends[i]):] = 3
        offsets = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
        for j, (start, end) in enumerate(offsets):
            if start < b and end > a:
                labels[i, int(prefixes[i]) + j] = 1
    return ids, prefixes, ends, labels


def forward(model, batch, rate, rows):
    ids, prefix, ends, _ = batch
    x0, xt, masked = _fixed_corruption(ids, prefix, ends, rate, model.tokenizer.pad_token_id,
                                      model.mask_index, [r['sample_id'] for r in rows], 'cuda')
    t = torch.full((len(x0),), min(1., rate / (1. - float(model.config.noise.eps))), device='cuda')
    dalpha, alpha = model.noise(t)
    logp = model.forward(xt=xt, sigma=model._sigma_from_alphat(alpha[:, None]))
    ce = -logp.gather(-1, x0[:, :, None]).squeeze(-1)
    weight = -dalpha[:, None] / (1. - alpha[:, None])
    return ce, ce * weight, masked, logp.argmax(-1), x0, float(weight[0, 0])


def norm(x):
    return float(torch.linalg.vector_norm(x.double()))


def cosine(x, y):
    denominator = norm(x) * norm(y)
    return float(torch.dot(x.double(), y.double()) / denominator) if denominator else None


def frozen_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def measure(model, data, rows):
    code = data[3] == 2
    total, probability, correct = 0., 0., 0
    for start in range(0, len(rows), 4):
        batch = tuple(x[start:start + 4] for x in data)
        ce, _, _, prediction, x0, _ = forward(model, batch, 1., rows[start:start + 4])
        mask = code[start:start + 4].to('cuda')
        total += float(ce[mask].sum())
        probability += float((-ce[mask]).exp().sum())
        correct += int((prediction[mask] == x0[mask]).sum())
    count = int(code.sum())
    return {'code_ce': total / count, 'mean_gold_probability': probability / count,
            'top1_correct_tokens': correct, 'tokens': count, 'top1_accuracy': correct / count}


@hydra.main(version_base=None, config_path='../configs', config_name='sft-overfit16')
def run(config):
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    p = Path('/root/autodl-tmp/s-flm/outputs/overfit16-20261023-01')
    out = p / 'gradient-diagnostic'
    out.mkdir(exist_ok=True)
    rows = [json.loads(s) for s in Path(config.data.train_file).read_text().splitlines()]
    tokenizer = dataloader.get_tokenizer(config)
    data = tensors(rows, tokenizer, str(config.data.separator))
    denominator = int((data[3] > 0).sum())
    names = {1: 'docstring', 2: 'non_docstring_code_eos', 3: 'pad'}
    results = {'sample_ids': [r['sample_id'] for r in rows], 'supervised_tokens': denominator,
               'mode': 'eval, dropout disabled, LoRA only; fixed corruption; token-weighted MDLM contributions',
               'models': {}}
    for step in [0, 3000]:
        if step == 0:
            model = entry._initialize_finetune(algo.MDLM, config, tokenizer)
        else:
            model = algo.MDLM(config, tokenizer=tokenizer)
            ckpt = torch.load(p / 'checkpoints/step-003000.ckpt', map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['state_dict'], strict=True)
            del ckpt
        model.ema = None
        model.to('cuda').eval()
        parameters = [(n, v) for n, v in model.named_parameters() if v.requires_grad]
        assert all(n.endswith(('lora_A', 'lora_B')) for n, _ in parameters)
        params = [v for _, v in parameters]
        record = {'dtypes': dict(Counter(str(v.dtype) for v in params)), 'conditions': []}
        for rate in [.5, 1.]:
            gradients = {name: torch.zeros(sum(v.numel() for v in params), device='cuda') for name in names.values()}
            totals = {name: {'masked_tokens': 0, 'ce_sum': 0., 'objective_contribution': 0.} for name in names.values()}
            for start in range(0, len(rows), 4):
                batch = tuple(x[start:start + 4] for x in data)
                ce, weighted, masked, prediction, x0, coefficient = forward(model, batch, rate, rows[start:start + 4])
                labels = batch[3].to('cuda')
                for region, name in names.items():
                    mask = masked & (labels == region)
                    loss = weighted[mask].sum() / denominator
                    grads = torch.autograd.grad(loss, params, retain_graph=region != 3, allow_unused=True)
                    vector = torch.cat([(g if g is not None else torch.zeros_like(v)).detach().float().flatten()
                                        for g, v in zip(grads, params)])
                    gradients[name] += vector
                    totals[name]['masked_tokens'] += int(mask.sum())
                    totals[name]['ce_sum'] += float(ce[mask].sum().detach())
                    totals[name]['objective_contribution'] += float(loss.detach())
                del ce, weighted, loss, grads, vector
            total_gradient = sum(gradients.values())
            code_gradient = gradients['non_docstring_code_eos']
            for name, stats in totals.items():
                count = stats['masked_tokens']
                stats['masked_token_ce'] = stats['ce_sum'] / count
                stats['objective_gradient_norm'] = norm(gradients[name])
                stats['region_mean_ce_gradient_norm'] = norm(gradients[name]) * denominator / (count * coefficient)
                stats['cosine_with_code_gradient'] = cosine(gradients[name], code_gradient)
                stats['cosine_with_total_gradient'] = cosine(gradients[name], total_gradient)
            condition = {'mask_fraction': rate, 'time_weight': coefficient, 'regions': totals,
                         'total_gradient_norm': norm(total_gradient),
                         'total_code_gradient_cosine': cosine(total_gradient, code_gradient)}
            record['conditions'].append(condition)
            print(json.dumps({'step': step, **condition}), flush=True)
            del gradients, total_gradient, code_gradient
        before = measure(model, data, rows)
        repeat = measure(model, data, rows)
        assert before == repeat, 'No-update replay was not deterministic'
        frozen_before = frozen_hash(model)
        initial = [v.detach().clone() for v in params]
        optimizer = torch.optim.AdamW(params, lr=float(config.optim.lr),
                                     betas=(float(config.optim.beta1), float(config.optim.beta2)),
                                     eps=float(config.optim.eps), weight_decay=float(config.optim.weight_decay))
        optimizer.zero_grad(set_to_none=True)
        code_count = int((data[3] == 2).sum())
        for start in range(0, len(rows), 4):
            batch = tuple(x[start:start + 4] for x in data)
            ce, _, _, _, _, _ = forward(model, batch, 1., rows[start:start + 4])
            (ce[batch[3].to('cuda') == 2].sum() / code_count).backward()
        grad_by_group = {suffix: norm(torch.cat([v.grad.detach().float().flatten() for n, v in parameters if n.endswith(suffix)]))
                         for suffix in ['lora_A', 'lora_B']}
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float(config.trainer.gradient_clip_val)))
        optimizer.step()
        changes = [v.detach() - old for v, old in zip(params, initial)]
        update = {'before': before, 'after': measure(model, data, rows), 'fresh_adamw': True,
                  'lr': float(config.optim.lr), 'grad_norm_pre_clip': grad_norm,
                  'gradient_norm_by_group': grad_by_group,
                  'changed_parameter_elements': sum(int(torch.count_nonzero(v)) for v in changes),
                  'update_l2': norm(torch.cat([v.float().flatten() for v in changes])),
                  'finite_gradients': all(bool(torch.isfinite(v.grad).all()) for v in params),
                  'frozen_parameters_unchanged': frozen_before == frozen_hash(model)}
        assert update['finite_gradients'] and update['frozen_parameters_unchanged']
        record['one_code_only_update'] = update
        print(json.dumps({'step': step, 'one_code_only_update': update}), flush=True)
        results['models'][str(step)] = record
        del model, optimizer, params, parameters, initial, changes
        gc.collect()
        torch.cuda.empty_cache()
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    print('GRADIENT_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    run()
