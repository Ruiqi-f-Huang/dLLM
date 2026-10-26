"""Generate code bodies from question + fixed gold function header/docstring."""
import ast
import gc
import json
import random
import re
import sys
from pathlib import Path

import hydra
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as entry
import algo
import dataloader
import sandbox_gsm8k
from callbacks.overfit_probe import _parseable
from scripts.train_overfit16_fullmask_codeonly import fp32_backbone


def char_offset(lines, line, byte_column):
    return sum(map(len, lines[:line - 1])) + len(lines[line - 1].encode()[:byte_column].decode())


def fixed_prefix(row, tokenizer, separator, style='reference'):
    code = row['code'].strip()
    function = ast.parse(code).body[0]
    doc = function.body[0]
    assert isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant)
    assert isinstance(doc.value.value, str)
    assert re.sub(r'\s+', ' ', doc.value.value).strip() == re.sub(r'\s+', ' ', row['question']).strip()
    lines = code.splitlines(keepends=True)
    doc_end = char_offset(lines, doc.end_lineno, doc.end_col_offset)
    body_start = char_offset(lines, function.body[1].lineno, function.body[1].col_offset)
    encoded = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
    k = 0
    for a, b in encoded.offset_mapping:
        if b > body_start:
            break
        k += 1
    end = encoded.offset_mapping[k - 1][1]
    assert end >= doc_end and not code[end:body_start].strip()
    prefix_text = tokenizer.decode(encoded.input_ids[:k], skip_special_tokens=False)
    assert prefix_text == code[:end]
    parsed = ast.parse(prefix_text)
    assert len(parsed.body) == 1 and len(parsed.body[0].body) == 1
    assert isinstance(parsed.body[0].body[0], ast.Expr)
    question_ids = ([tokenizer.bos_token_id] + tokenizer(
        row['question'].strip(), add_special_tokens=False).input_ids
        + tokenizer(separator, add_special_tokens=False).input_ids)
    code_ids = encoded.input_ids[:k]
    if style == 'multiline':
        header_end = char_offset(lines, doc.lineno, doc.col_offset)
        prefix_text = code[:header_end] + '\"\"\"\n    ' + row['question'].strip() + '\n    \"\"\"\n'
        parsed = ast.parse(prefix_text)
        assert len(parsed.body[0].body) == 1
        assert re.sub(r'\s+', ' ', ast.get_docstring(parsed.body[0])).strip() == re.sub(r'\s+', ' ', row['question']).strip()
        code_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        assert tokenizer.decode(code_ids, skip_special_tokens=False) == prefix_text
        k = len(code_ids)
    else:
        assert style == 'reference'
    prefix_ids = question_ids + code_ids
    assert len(prefix_ids) < 512 and tokenizer.eos_token_id not in encoded.input_ids[:k]
    return {'sample_id': row['sample_id'], 'provided_code_prefix': prefix_text,
            'provided_code_tokens': k, 'question_prefix_tokens': len(question_ids),
            'prefix_ids': prefix_ids, 'generation_slots': 512 - len(prefix_ids),
            'body_start_char': body_start, 'provided_end_char': end if style == 'reference' else None,
            'style': style}


def strict_parseable(text):
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError):
        return False


@torch.no_grad()
def generate(model, rows, prefixes, tokenizer):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    records = []
    for start in range(0, len(rows), 8):
        batch = prefixes[start:start + 8]
        lengths = torch.tensor([len(x['prefix_ids']) for x in batch], device='cuda')
        fixed = torch.zeros(len(batch), int(lengths.max()), dtype=torch.long, device='cuda')
        for i, item in enumerate(batch):
            fixed[i, :int(lengths[i])] = torch.tensor(item['prefix_ids'], device='cuda')
        samples, _ = model.generate_samples(num_samples=len(batch), num_steps=128,
                                           prefix_tokens=fixed, prefix_lengths=lengths)
        for i, (row, item) in enumerate(zip(rows[start:start + 8], batch)):
            n = int(lengths[i])
            assert torch.equal(samples[i, :n], fixed[i, :n]), 'Sampler changed fixed prefix'
            target_ids = samples[i, item['question_prefix_tokens']:].cpu().tolist()
            suffix_ids = samples[i, n:].cpu().tolist()
            eos = target_ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in target_ids else None
            program = tokenizer.decode(target_ids, skip_special_tokens=True)
            trimmed = tokenizer.decode(target_ids[:eos] if eos is not None else target_ids, skip_special_tokens=True)
            correct = bool(sandbox_gsm8k.evaluate_samples(program, row['reference_answer'], timeout_s=2))
            trimmed_correct = bool(sandbox_gsm8k.evaluate_samples(trimmed, row['reference_answer'], timeout_s=2))
            records.append({'sample_id': row['sample_id'], 'provided_code_prefix': item['provided_code_prefix'],
                            'target_token_ids': target_ids, 'generated_suffix_ids': suffix_ids,
                            'program': program, 'eos_program': trimmed, 'eos_target_position': eos,
                            'correct': correct, 'parseable': _parseable(program),
                            'strict_parseable': strict_parseable(program),
                            'strict_execution_correct': strict_parseable(program) and correct,
                            'eos_correct': trimmed_correct, 'eos_parseable': _parseable(trimmed)})
    return records


@hydra.main(version_base=None, config_path='../configs', config_name='sft-overfit16-fullmask-codeonly')
@torch.no_grad()
def run(config):
    import os
    import wandb
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    p = Path('/root/autodl-tmp/s-flm/outputs/overfit16-fullmask-codeonly-20261023-01')
    style = config.get('fixed_prefix_style', 'reference')
    suffix = '-multiline' if style == 'multiline' else ''
    out = p / ('fixed-prefix-diagnostic' + suffix)
    out.mkdir(exist_ok=True)
    rows = [json.loads(s) for s in Path(config.data.train_file).read_text().splitlines()]
    tokenizer = dataloader.get_tokenizer(config)
    prefixes = [fixed_prefix(row, tokenizer, str(config.data.separator), style) for row in rows]
    (out / 'prefix_manifest.json').write_text(json.dumps(prefixes, indent=2))
    summary = []
    for step in [0, 1000, 3000]:
        if step == 0:
            model = entry._initialize_finetune(algo.MDLM, config, tokenizer)
        else:
            model = algo.MDLM(config, tokenizer=tokenizer)
            c = torch.load(p / f'resume-to3000/checkpoints/step-{step:06d}.ckpt', map_location='cpu', weights_only=False)
            model.load_state_dict(c['state_dict'], strict=True)
            del c
        model.ema = None
        model.to('cuda').eval()
        fp32_backbone(model)
        records = generate(model, rows, prefixes, tokenizer)
        baseline_file = (p if step == 0 else p / 'resume-to3000') / f'probe/step-{step:06d}.json'
        baseline = json.loads(baseline_file.read_text())['records']
        assert [x['sample_id'] for x in records] == [x['sample_id'] for x in baseline]
        counts = {key: sum(x[key] for x in records) for key in
                  ['correct', 'parseable', 'strict_parseable', 'strict_execution_correct', 'eos_correct', 'eos_parseable']}
        baseline_eos_correct = 0
        for row, x in zip(rows, baseline):
            ids = x['token_ids']; end = ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in ids else None
            text = tokenizer.decode(ids[:end] if end is not None else ids, skip_special_tokens=True)
            baseline_eos_correct += bool(sandbox_gsm8k.evaluate_samples(text, row['reference_answer'], timeout_s=2))
        result = {'step': step, 'n': len(rows), 'precision': 'FP32, TF32 off', 'seed': 1, 'nfe': 128,
                  'prefix_style': style,
                  'fixed_prefix': counts, 'question_only': {'correct': sum(x['full_correct'] for x in baseline),
                  'parseable': sum(x['full_strict_parseable'] for x in baseline), 'eos_correct': baseline_eos_correct},
                  'wrong_to_right': [x['sample_id'] for x,y in zip(records,baseline) if x['correct'] and not y['full_correct']],
                  'right_to_wrong': [x['sample_id'] for x,y in zip(records,baseline) if not x['correct'] and y['full_correct']]}
        summary.append(result)
        (out / f'step-{step:06d}.json').write_text(json.dumps({'summary': result, 'records': records}, indent=2))
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('FIXED_PREFIX ' + json.dumps(result), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    os.environ.pop('WANDB_MODE', None)
    wb = wandb.init(entity='ruiqi-f-huang-nus', project='s-flm', id='x4989caj', resume='must',
                    mode='online', dir=str(out), settings=wandb.Settings(init_timeout=60))
    table = wandb.Table(columns=['step', 'question_only_correct', 'fixed_prefix_correct',
                                'question_only_parseable', 'fixed_prefix_parseable',
                                'question_only_eos_correct', 'fixed_prefix_eos_correct'])
    for r in summary:
        table.add_data(r['step'],r['question_only']['correct'],r['fixed_prefix']['correct'],
                       r['question_only']['parseable'],r['fixed_prefix']['parseable'],
                       r['question_only']['eos_correct'],r['fixed_prefix']['eos_correct'])
    wb.summary['fixed_prefix_generation_table' + suffix.replace('-', '_')] = table
    wb.summary['fixed_prefix_note' + suffix.replace('-', '_')] = 'No training. Header and question-only docstring, no computational statements. FP32, seed1, 128NFE, length512. Style=' + style
    artifact = wandb.Artifact('overfit16-codeonly-fixed-prefix' + suffix, type='diagnostic')
    for f in out.glob('*.json'):
        artifact.add_file(str(f), name=f.name)
    artifact.add_file(str(Path(__file__)))
    wb.log_artifact(artifact)
    wb.finish()
    (out / 'COMPLETE').write_text('three models evaluated without training\n')
    print('FIXED_PREFIX_COMPLETE', flush=True)


if __name__ == '__main__':
    run()
