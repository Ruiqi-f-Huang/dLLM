"""Compare correct versus unrelated, exactly length-matched question prefixes."""
import gc
import json
import re
import sys
from pathlib import Path

import hydra
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as entry
import algo
import dataloader
from scripts.diagnose_overfit16_gradients import tensors
from scripts.eval_mdlm_denoising_gap import _fixed_corruption


def words(text):
    stop = set('a an the is are was were of to in on at for and or if how many much does do did has have had there what with from it its that this by as than more less each per total'.split())
    return set(re.findall(r'[a-z]+', text.lower())) - stop


def pair_questions(rows, pool, tokenizer):
    lengths = tokenizer([r['question'].strip() for r in pool], add_special_tokens=False).input_ids
    excluded = {r['sample_id'] for r in rows}
    used = set()
    pairs = []
    for row in rows:
        original = tokenizer(row['question'].strip(), add_special_tokens=False).input_ids
        w = words(row['question'])
        candidates = []
        for candidate, tokens in zip(pool, lengths):
            if candidate['sample_id'] in excluded | used or len(tokens) != len(original):
                continue
            other = words(candidate['question'])
            similarity = len(w & other) / max(len(w | other), 1)
            candidates.append((similarity, candidate['sample_id'], candidate, tokens))
        if not candidates:
            raise ValueError(f"No exactly length-matched unrelated question for {row['sample_id']}")
        score, _, selected, tokens = min(candidates, key=lambda x: (x[0], x[1]))
        assert score < .15, 'No sufficiently unrelated candidate'
        used.add(selected['sample_id'])
        pairs.append({'sample_id': row['sample_id'], 'correct_question': row['question'],
                      'wrong_question_id': selected['sample_id'], 'wrong_question': selected['question'],
                      'question_tokens': len(original), 'wrong_tokens': tokens, 'word_jaccard': score})
    return pairs


@hydra.main(version_base=None, config_path='../configs', config_name='sft-overfit16')
@torch.no_grad()
def run(config):
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    p = Path('/root/autodl-tmp/s-flm/outputs/overfit16-20261023-01')
    out = p / 'question-conditioning'
    out.mkdir(exist_ok=True)
    rows = [json.loads(s) for s in Path(config.data.train_file).read_text().splitlines()]
    pool = [json.loads(s) for s in (ROOT / 'data/gsm8k_pot/sft_train.jsonl').read_text().splitlines()]
    tokenizer = dataloader.get_tokenizer(config)
    pairs = pair_questions(rows, pool, tokenizer)
    (out / 'pairs.json').write_text(json.dumps(pairs, indent=2))
    ids, prefixes, ends, labels = tensors(rows, tokenizer, str(config.data.separator))
    result = {'pairs': pairs, 'pair_source': 'original SFT train, excluding the 16 overfit questions',
              'length_control': 'exact tokenizer question length match; all target, PAD and MASK positions identical',
              'models': {}}
    cases = [('half_mask_original_docstring', .5, False),
             ('half_mask_docstring_hidden', .5, True), ('full_mask', 1., False)]
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
        condition_results = []
        for name, rate, hide_docstring in cases:
            records = []
            for start in range(0, len(rows), 8):
                stop = start + 8
                x0, xt, masked = _fixed_corruption(
                    ids[start:stop], prefixes[start:stop], ends[start:stop], rate,
                    tokenizer.pad_token_id, model.mask_index, [r['sample_id'] for r in rows[start:stop]], 'cuda')
                regions = labels[start:stop].to('cuda')
                if hide_docstring:
                    masked |= regions == 1
                    xt[regions == 1] = model.mask_index
                wrong = xt.clone()
                for i, pair in enumerate(pairs[start:stop]):
                    tokens = torch.tensor(pair['wrong_tokens'], device='cuda')
                    assert len(tokens) == pair['question_tokens']
                    wrong[i, 1:1 + len(tokens)] = tokens
                    assert torch.equal(wrong[i, int(prefixes[start + i]):], xt[i, int(prefixes[start + i]):])
                t = torch.full((len(x0),), min(1., rate / (1. - float(config.noise.eps))), device='cuda')
                _, alpha = model.noise(t)
                sigma = model._sigma_from_alphat(alpha[:, None])
                output = {}
                for condition, inputs in [('correct', xt), ('wrong', wrong)]:
                    logp = model.forward(xt=inputs, sigma=sigma)
                    ce = -logp.gather(-1, x0[:, :, None]).squeeze(-1)
                    prediction = logp.argmax(-1)
                    output[condition] = (ce.cpu(), prediction.cpu())
                    del logp
                for i, row in enumerate(rows[start:stop]):
                    rec = {'sample_id': row['sample_id']}
                    for region, number in [('code_eos_without_docstring', 2), ('docstring', 1)]:
                        mask = (masked[i] & (regions[i] == number)).cpu()
                        count = int(mask.sum())
                        data = {'tokens': count}
                        for condition, (ce, prediction) in output.items():
                            data[condition] = {'ce_sum': float(ce[i, mask].sum()),
                                               'correct_tokens': int((prediction[i, mask] == x0[i].cpu()[mask]).sum())}
                        data['changed_predictions'] = int((output['correct'][1][i, mask] != output['wrong'][1][i, mask]).sum())
                        rec[region] = data
                    records.append(rec)
            aggregate = {}
            for region in ['code_eos_without_docstring', 'docstring']:
                n = sum(r[region]['tokens'] for r in records)
                d = {'tokens': n}
                for condition in ['correct', 'wrong']:
                    d[condition] = {'ce': sum(r[region][condition]['ce_sum'] for r in records) / n,
                                    'accuracy': sum(r[region][condition]['correct_tokens'] for r in records) / n}
                d['wrong_minus_correct_ce'] = d['wrong']['ce'] - d['correct']['ce']
                d['correct_minus_wrong_accuracy'] = d['correct']['accuracy'] - d['wrong']['accuracy']
                d['prediction_change_rate'] = sum(r[region]['changed_predictions'] for r in records) / n
                d['questions_correct_ce_lower'] = sum(r[region]['correct']['ce_sum'] < r[region]['wrong']['ce_sum'] for r in records)
                aggregate[region] = d
            condition_results.append({'case': name, 'mask_fraction': rate, 'aggregate': aggregate, 'records': records})
            print(json.dumps({'step': step, 'case': name, **aggregate}), flush=True)
        result['models'][str(step)] = condition_results
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    print('QUESTION_CONDITIONING_COMPLETE', flush=True)


if __name__ == '__main__':
    run()
