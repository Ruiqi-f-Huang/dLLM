"""Compare fixed partial/full-MASK prediction using existing overfit weights."""
import ast
import gc
import json
import sys
from pathlib import Path

import hydra
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as entry
import algo
import dataloader
from scripts.eval_mdlm_denoising_gap import _make_target_tensors, _fixed_corruption


@hydra.main(version_base=None, config_path='../configs', config_name='sft-overfit16')
@torch.no_grad()
def run(config):
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    p = Path('/root/autodl-tmp/s-flm/outputs/overfit16-20261023-01')
    rows = [json.loads(s) for s in Path(config.data.train_file).read_text().splitlines()]
    tokenizer = dataloader.get_tokenizer(config)
    ids, prefixes, ends = _make_target_tensors(rows, tokenizer, 512, str(config.data.separator))
    regions = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        code = row['code'].strip()
        doc = ast.parse(code).body[0].body[0]
        assert isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str)
        lines = code.splitlines(keepends=True)
        a = sum(map(len, lines[:doc.lineno - 1])) + len(
            lines[doc.lineno - 1].encode()[:doc.col_offset].decode())
        b = sum(map(len, lines[:doc.end_lineno - 1])) + len(
            lines[doc.end_lineno - 1].encode()[:doc.end_col_offset].decode())
        tokens = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
        assert tokens.input_ids == ids[i, int(prefixes[i]):int(ends[i]) - 1].tolist()
        regions[i, int(prefixes[i]):int(ends[i])] = 2
        for j, (start, end) in enumerate(tokens.offset_mapping):
            if start < b and end > a:
                regions[i, int(prefixes[i]) + j] = 1
    result = {'rates': [0.5, 0.9, 1.0], 'sample_ids': [r['sample_id'] for r in rows],
              'region_definition': 'docstring vs non-docstring code including EOS; PAD never scored',
              'models': {}}
    for step in [0, 1000, 3000]:
        if step == 0:
            model = entry._initialize_finetune(algo.MDLM, config, tokenizer)
        else:
            model = algo.MDLM(config, tokenizer=tokenizer)
            ckpt = torch.load(p / f'checkpoints/step-{step:06d}.ckpt', map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['state_dict'], strict=True)
            del ckpt
        model.ema = None
        model.to('cuda').eval()
        stats = []
        for rate in result['rates']:
            totals = {name: {'ce_sum': 0., 'correct': 0, 'tokens': 0}
                      for name in ['code_eos', 'docstring', 'non_docstring_code_eos']}
            per_question = []
            for start in range(0, len(rows), 8):
                stop = start + 8
                x0, xt, masked = _fixed_corruption(
                    ids[start:stop], prefixes[start:stop], ends[start:stop], rate,
                    tokenizer.pad_token_id, model.mask_index,
                    result['sample_ids'][start:stop], 'cuda')
                t = torch.full((len(x0),), min(1., rate / (1. - float(config.noise.eps))), device='cuda')
                _, alpha = model.noise(t)
                logp = model.forward(xt=xt, sigma=model._sigma_from_alphat(alpha[:, None]))
                labels = regions[start:stop].to('cuda')
                per = [{'sample_id': r['sample_id']} for r in rows[start:stop]]
                for name, target_region in [('code_eos', labels > 0), ('docstring', labels == 1),
                                             ('non_docstring_code_eos', labels == 2)]:
                    mask = masked & target_region
                    for i in range(len(x0)):
                        scored = logp[i, mask[i]]
                        target = x0[i, mask[i]]
                        count = len(target)
                        ce = float(-scored.gather(1, target[:, None]).sum())
                        correct = int((scored.argmax(-1) == target).sum())
                        per[i][name] = {'ce_sum': ce, 'correct': correct, 'tokens': count}
                        totals[name]['ce_sum'] += ce
                        totals[name]['correct'] += correct
                        totals[name]['tokens'] += count
                per_question.extend(per)
                del logp
            aggregate = {name: {'ce': v['ce_sum'] / v['tokens'],
                                'accuracy': v['correct'] / v['tokens'], 'tokens': v['tokens']}
                         for name, v in totals.items()}
            stats.append({'rate': rate, 'aggregate': aggregate, 'per_question': per_question})
            print(json.dumps({'step': step, 'rate': rate, **aggregate}), flush=True)
        result['models'][str(step)] = {'weights': 'Base EMA' if step == 0 else 'raw', 'mask_tests': stats}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    out = p / 'mask-diagnostic'
    out.mkdir(exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    print('DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    run()
