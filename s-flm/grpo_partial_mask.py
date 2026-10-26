"""Paired teacher-forced Partial-MASK plans and question-level summaries."""
from collections import defaultdict
import math
import random
import torch
from grpo_overfit import stable_seed


def make_mask(eligible, probability, sample_id, reward, replicate):
    if eligible.dtype != torch.bool or eligible.ndim != 1 or not 0 < probability <= 1:
        raise ValueError('Expected a 1-D bool region and rate in (0,1].')
    positions=eligible.nonzero().flatten()
    if not len(positions):raise ValueError('Empty answer region.')
    generator=torch.Generator(device='cpu').manual_seed(
        stable_seed('partial-mask',sample_id,int(reward),replicate))
    # The permutation does not depend on rate, yielding nested masks.
    count=min(len(positions),max(1,int(math.floor(len(positions)*probability+.5))))
    chosen=positions[torch.randperm(len(positions),generator=generator)[:count].to(positions.device)]
    mask=torch.zeros_like(eligible);mask[chosen]=True
    return mask


def perturb_completion(tokens, prefix_length, masked, mask_id):
    if masked[:prefix_length].any():raise ValueError('Question prefix must stay visible.')
    noisy=tokens.clone()
    # Non-answer suffix, including PAD and first-EOS tail, always stays masked.
    noisy[prefix_length:]=mask_id
    # Caller supplies only eligible answer positions for restoration separately.
    return noisy


def noisy_input(tokens, prefix_length, eligible, masked, mask_id):
    if (masked & ~eligible).any() or eligible[:prefix_length].any():
        raise ValueError('Mask/answer region crosses prompt boundary.')
    noisy=perturb_completion(tokens,prefix_length,masked,mask_id)
    restore=eligible & ~masked
    noisy[restore]=tokens[restore]
    return noisy


def question_scores(records):
    grouped=defaultdict(list)
    for record in records:
        if record['tokens']:
            key=(record['model'],record['rate'],record['reward'],record['region'],record['sample_id'])
            grouped[key].append(record)
    result=[]
    for (model,rate,reward,region,sid),rows in sorted(grouped.items()):
        logp=sum(x['mean_logp'] for x in rows)/len(rows)
        result.append(dict(model=model,rate=rate,reward=reward,region=region,sample_id=sid,
            mean_logp=logp,ce=-logp,top1=sum(x['top1'] for x in rows)/len(rows),
            mask_replicates=len(rows),scored_tokens=sum(x['tokens'] for x in rows)))
    return result


def paired_summary(base,current,draws=2000):
    a={x['sample_id']:x for x in base};b={x['sample_id']:x for x in current}
    if a.keys()!=b.keys():raise ValueError('Paired question coverage differs.')
    if not a:return dict(questions=0)
    ids=sorted(a)
    rng=random.Random(271828)
    result=dict(questions=len(ids),base_ce=sum(a[k]['ce'] for k in ids)/len(ids),
                current_ce=sum(b[k]['ce'] for k in ids)/len(ids))
    for metric in ['mean_logp','top1']:
        delta=[b[k][metric]-a[k][metric] for k in ids]
        bootstrap=sorted(sum(rng.choices(delta,k=len(ids)))/len(ids) for _ in range(draws))
        result[metric+'_delta']=sum(delta)/len(ids)
        result[metric+'_ci95']=[bootstrap[math.floor(.025*(draws-1))],bootstrap[math.ceil(.975*(draws-1))]]
        result[metric+'_improved_questions']=sum(x>0 for x in delta)
    result['ce_delta']=-result['mean_logp_delta']
    return result
