"""Fixed-mask EMA diagnostic of existing LIFT pilot checkpoints; no updates."""
import argparse
import ast
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import tokenize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REGIONS = ['target_no_pad', 'non_docstring_code_eos', 'docstring_boundary',
           'eos', 'critical', 'variable', 'number', 'operator', 'pad']
OPERATORS = {'+', '-', '*', '/', '//', '%', '**', '<', '>', '<=', '>=', '==', '!='}


def critical_regions(code, offsets, pure_code):
    """BPE overlaps with local variable references, numeric literals, operators.

    Local names include their Store and Load occurrences. Builtin call heads
    and module names are excluded unless also assigned locally. Categories
    may overlap at BPE boundaries; critical is their union, never their sum.
    """
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def position(line, col, byte=False):
        if byte:
            col = len(lines[line - 1].encode('utf-8')[:col].decode('utf-8'))
        return starts[line - 1] + col

    bound = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    bound.update(n.arg for n in ast.walk(tree) if isinstance(n, ast.arg))
    spans = {'variable': [], 'number': [], 'operator': []}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in bound:
            spans['variable'].append((position(n.lineno, n.col_offset, True),
                                      position(n.end_lineno, n.end_col_offset, True)))
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        name = ('number' if token.type == tokenize.NUMBER else
                'operator' if token.type == tokenize.OP and token.string in OPERATORS else None)
        if name:
            spans[name].append((position(*token.start), position(*token.end)))
    labels = {name: [bool(keep and any(a < d and b > c for c, d in ranges))
                     for (a, b), keep in zip(offsets, pure_code)]
              for name, ranges in spans.items()}
    labels['critical'] = [any(labels[k][i] for k in spans) for i in range(len(offsets))]
    return labels


def fixed_uniforms(sample_id, seed, length):
    import torch
    digest = hashlib.sha256(f'{sample_id}|{seed}|lift-fixed-20261023-v1'.encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % (2**63 - 1))
    return torch.rand(length, generator=generator)


def prepare(rows, tokenizer, length, separator):
    import torch
    from lift_masking import code_region
    all_ids, all_regions, all_suffix, all_prefix = [], [], [], []
    sep = tokenizer(separator, add_special_tokens=False).input_ids
    for row in rows:
        code = row['code'].strip()
        encoded = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
        ids, pure, ok = code_region(code, tokenizer)
        assert ok and ids == encoded.input_ids, row['sample_id']
        labels = critical_regions(code, encoded.offset_mapping, pure)
        prefix = [tokenizer.bos_token_id] + tokenizer(row['question'].strip(), add_special_tokens=False).input_ids + sep
        start, end = len(prefix), len(prefix) + len(ids) + 1
        assert end <= length
        x0 = prefix + ids + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (length - end)
        region = torch.zeros(len(REGIONS), length, dtype=torch.bool)
        region[REGIONS.index('target_no_pad'), start:end] = True
        region[REGIONS.index('non_docstring_code_eos'), start:end] = torch.tensor(pure + [True])
        region[REGIONS.index('docstring_boundary'), start:end - 1] = ~torch.tensor(pure)
        region[REGIONS.index('eos'), end - 1] = True
        region[REGIONS.index('pad'), end:] = True
        for name, values in labels.items():
            region[REGIONS.index(name), start:end - 1] = torch.tensor(values)
        all_ids.append(x0); all_regions.append(region)
        all_suffix.append(torch.arange(length) >= start); all_prefix.append(start)
    return {'ids': torch.tensor(all_ids), 'regions': torch.stack(all_regions),
            'suffix': torch.stack(all_suffix), 'prefix': all_prefix,
            'sample_ids': [r['sample_id'] for r in rows]}


def conditions(data):
    import torch
    # Uniforms are shared across all q, giving nested masks within each seed.
    for rate in (.25, .5, .75, .9, 1.):
        for seed in ((17,) if rate == 1. else (17, 29)):
            uniforms = torch.stack([fixed_uniforms(s, seed, data['ids'].shape[1]) for s in data['sample_ids']])
            mask = (uniforms < rate) & data['suffix']
            yield rate, seed, mask


def main():
    import lightning as L
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import main as entry
    import algo
    import dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('pilot', type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    pilot = args.pilot.resolve(); out = pilot / 'fixed-mask-diagnostic'
    out.mkdir(exist_ok=True)
    cfg = OmegaConf.load(pilot / 'train-a/.hydra/config.yaml')
    tokenizer = dataloader.get_tokenizer(cfg)
    L.seed_everything(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    pool = [json.loads(s) for s in Path(cfg.data.train_file).read_text().splitlines() if s.strip()]
    selected = sorted(pool, key=lambda r: hashlib.sha256(('lift-fixed-train|' + r['sample_id']).encode()).hexdigest())[:256]
    valid = [json.loads(s) for s in Path(cfg.data.validation_file).read_text().splitlines() if s.strip()]
    splits = {'train256': prepare(selected, tokenizer, cfg.model.length, str(cfg.data.separator)),
              'validation632': prepare(valid, tokenizer, cfg.model.length, str(cfg.data.separator))}
    masks = {name: list(conditions(data)) for name, data in splits.items()}
    manifest = {'weights': 'EMA', 'no_training': True, 'regions': REGIONS,
                'mask_rates': [.25, .5, .75, .9, 1.], 'partial_mask_seeds': [17, 29],
                'full_mask_seeds': [17], 'time_conditioning': False,
                'precision': 'same BF16 backbone/FP32 log-probs as pilot evaluation',
                'scope': 'Prompt always visible; suffix (including PAD) corrupted. PAD scored only in its separate diagnostic region. CE has no diffusion-time coefficient.',
                'definitions': 'Variables are AST local Store/Load names; numbers are tokenize.NUMBER; operators are arithmetic/comparison OP tokens. BPE overlap labels; critical is union. Docstring-overlapping BPE excluded from code.',
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'data_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('train',cfg.data.train_file),('validation',cfg.data.validation_file)]},
                'splits': {}}
    for name, data in splits.items():
        manifest['splits'][name] = {'sample_ids': data['sample_ids'],
            'input_sha256': hashlib.sha256(data['ids'].numpy().tobytes()).hexdigest(),
            'region_sha256': hashlib.sha256(data['regions'].numpy().tobytes()).hexdigest(),
            'conditions': [{'rate': q, 'seed': s, 'mask_sha256': hashlib.sha256(m.numpy().tobytes()).hexdigest(),
                            'masked_tokens_by_region': {region:int((m & data['regions'][:,i]).sum()) for i,region in enumerate(REGIONS)}}
                           for q,s,m in masks[name]]}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    aggregates = []
    for label in ['base', 'a100', 'a200', 'b100', 'b200', 'c100', 'c200']:
        cfg.lora.enabled = label != 'base'
        if label == 'base':
            model = entry._initialize_finetune(algo.MDLM, cfg, tokenizer)
        else:
            cfg.eval.checkpoint_path = str(pilot / f'train-{label[0]}/checkpoints/step-{int(label[1:]):06d}.ckpt')
            model = entry._load_from_checkpoint(algo.MDLM, cfg, tokenizer)
        assert not model.time_conditioning
        model.to('cuda'); model._eval_mode(); model.eval()
        for split, data in splits.items():
            for rate, seed, mask in masks[split]:
                stats = np.zeros((len(data['ids']), len(REGIONS), 3), dtype=np.float64)
                with torch.no_grad():
                    for start in range(0, len(data['ids']), args.batch_size):
                        stop = start + args.batch_size
                        x0 = data['ids'][start:stop].cuda()
                        masked = mask[start:stop].cuda()
                        xt = x0.masked_fill(masked, model.mask_index)
                        t = torch.full((len(x0),), min(1., rate / (1 - float(cfg.noise.eps))), device='cuda')
                        _, alpha = model.noise(t)
                        logp = model.forward(xt=xt, sigma=model._sigma_from_alphat(alpha[:,None]))
                        ce = -logp.gather(-1, x0[...,None]).squeeze(-1)
                        correct = logp.argmax(-1) == x0
                        scored = data['regions'][start:stop].cuda() & masked[:,None,:]
                        packed = torch.stack([(ce[:,None,:] * scored).sum(-1),
                                              (correct[:,None,:] & scored).sum(-1),
                                              scored.sum(-1)], dim=-1)
                        stats[start:start + len(x0)] = packed.cpu().numpy()
                        del logp, ce, correct, packed
                filename = f'{label}-{split}-q{rate:g}-seed{seed}.npz'
                np.savez_compressed(out / filename, stats=stats)
                totals = stats.sum(0)
                for i, region in enumerate(REGIONS):
                    ce_sum, n_correct, n_tokens = totals[i]
                    aggregates.append({'model':label,'split':split,'rate':rate,'seed':seed,
                        'region':region,'ce_sum':float(ce_sum),'correct':int(n_correct),'tokens':int(n_tokens),
                        'ce':float(ce_sum / n_tokens) if n_tokens else None,
                        'accuracy':float(n_correct / n_tokens) if n_tokens else None})
                key = REGIONS.index('critical')
                print(json.dumps({'model':label,'split':split,'rate':rate,'seed':seed,
                                  'critical_ce':float(totals[key,0]/totals[key,2]),
                                  'critical_top1':float(totals[key,1]/totals[key,2])}),flush=True)
                (out / 'aggregate.json').write_text(json.dumps(aggregates,indent=2)+'\n')
        del model
        gc.collect(); torch.cuda.empty_cache()
    (out / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
