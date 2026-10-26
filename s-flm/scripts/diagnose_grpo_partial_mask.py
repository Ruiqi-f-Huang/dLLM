#!/usr/bin/env python3
"""Read-only, paired Partial-MASK scoring of existing online RL checkpoints."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True)
    p.add_argument('--online-run',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--replicates',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--rates',type=float,nargs='+',default=[1.,.75,.25])
    p.add_argument('--models',nargs='+',choices=['base','lr3e-06-step250','lr3e-05-step250'],
                   default=['base','lr3e-05-step250'])
    args=p.parse_args()
    if args.replicates<1 or args.batch_size<1 or len(set(args.rates))!=len(args.rates) or any(not 0<q<=1 for q in args.rates):
        p.error('Invalid mask rates or batch/replicate count.')
    if args.models[0] != 'base' or len(set(args.models)) != len(args.models) or len(args.models)<2:
        p.error('Base must be first, with at least one distinct RL model.')
    import torch
    from grpo_fixed32 import read_jsonl,load_model
    from grpo_monitor import code_regions
    from diffu_grpo import completion_mask,proxy_logps
    from grpo_partial_mask import make_mask,noisy_input,question_scores,paired_summary
    online=Path(args.online_run).resolve()
    if not (online/'COMPLETE').exists():raise ValueError('Online run is incomplete.')
    run=Path(args.output).resolve();run.mkdir(parents=True,exist_ok=False)
    (run/'run.json').write_text(json.dumps(vars(args),indent=2))
    try:
        questions=read_jsonl(online/'questions.jsonl');pairs=read_jsonl(online/'pairs.jsonl')
        if len(questions)!=32 or len(pairs)!=64:raise ValueError('Expected frozen 32-question, 64-program set.')
        for src in [online/'questions.jsonl',online/'pairs.jsonl',Path(__file__),ROOT/'grpo_partial_mask.py']:
            shutil.copy2(src,run/src.name)
        checkpoints={'base':None,
            'lr3e-06-step250':str(online/'lr3e-06/train/step-000250.ckpt'),
            'lr3e-05-step250':str(online/'lr3e-05/train/step-000250.ckpt')}
        checkpoints={name:checkpoints[name] for name in args.models}
        for path in checkpoints.values():
            if path and not Path(path).is_file():raise FileNotFoundError(path)
        (run/'checkpoints.json').write_text(json.dumps(checkpoints,indent=2))
        plan=None;all_records=[];fullmask_scores={}
        for label,checkpoint in checkpoints.items():
            (run/'STAGE').write_text(label+'\n');print(label,flush=True)
            model,tokenizer=load_model(args.base,checkpoint)
            model.config.backbone_compute_dtype='float32'
            model.requires_grad_(False);model.eval()
            if model.config.algo.time_conditioning:raise ValueError('Protocol assumes the current time-unconditioned Base.')
            if plan is None:
                by_id={q['sample_id']:q for q in questions};seen=set();items=[];metadata=[]
                separator=tokenizer(str(model.config.data.separator),add_special_tokens=False).input_ids
                for record in pairs:
                    sid=record['sample_id'];reward=record['reward'];key=(sid,reward)
                    if reward not in (0,1) or key in seen or sid not in by_id:raise ValueError('Invalid fixed pair.')
                    seen.add(key)
                    prefix=[tokenizer.bos_token_id]+tokenizer(by_id[sid]['question'].strip(),add_special_tokens=False).input_ids+separator
                    tokens=torch.tensor(record['tokens'],dtype=torch.long)
                    if len(tokens)!=model.num_tokens or tokens[:len(prefix)].tolist()!=prefix:raise ValueError('Prefix/token mismatch.')
                    eligible=completion_mask(tokens[None],torch.tensor([len(prefix)]),tokenizer.eos_token_id,
                                             tokenizer.pad_token_id,model.mask_index)[0]
                    suffix=tokens[len(prefix):].tolist()
                    eos=suffix.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in suffix else None
                    suffix=suffix[:eos] if eos is not None else suffix
                    text=tokenizer.decode(suffix,skip_special_tokens=True)
                    code=torch.zeros_like(eligible);doc=torch.zeros_like(eligible);available=False
                    chars=code_regions(text)
                    try:
                        encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
                        if chars is not None and list(encoded['input_ids'])==suffix:
                            available=True
                            for j,(a,b) in enumerate(encoded['offset_mapping']):
                                if b<=a:continue
                                flags=chars[a:b]
                                if all(flags):code[len(prefix)+j]=True
                                elif not any(flags):doc[len(prefix)+j]=True
                            if eos is not None:code[len(prefix)+eos]=True
                    except (ValueError,TypeError,NotImplementedError,KeyError):pass
                    regions={'all_answer':eligible,'code_eos':code & eligible,'docstring_comment':doc & eligible}
                    metadata.append(dict(sample_id=sid,reward=reward,prefix_length=len(prefix),
                        eligible_tokens=int(eligible.sum()),code_region_available=available,
                        code_eos_tokens=int(regions['code_eos'].sum()),docstring_comment_tokens=int(regions['docstring_comment'].sum())))
                    for rate in args.rates:
                        for replicate in range(1 if rate==1 else args.replicates):
                            masked=make_mask(eligible,rate,sid,reward,replicate)
                            items.append(dict(sample_id=sid,reward=reward,rate=rate,replicate=replicate,
                                tokens=tokens,prefix_length=len(prefix),eligible=eligible,regions=regions,
                                masked=masked,noisy=noisy_input(tokens,len(prefix),eligible,masked,model.mask_index)))
                if len(seen)!=64:raise ValueError('Pair coverage is incomplete.')
                plan=items
                (run/'program_metadata.json').write_text(json.dumps(metadata,indent=2))
                torch.save([{k:v for k,v in x.items() if k!='noisy'} for x in plan],run/'mask_plan.pt')
                (run/'mask_plan_sha256.txt').write_text(hashlib.sha256((run/'mask_plan.pt').read_bytes()).hexdigest()+'\n')
            outputs=[];fullmask_scores[label]={}
            with (run/f'{label}.jsonl').open('x') as handle,torch.inference_mode():
                for start in range(0,len(plan),args.batch_size):
                    batch=plan[start:start+args.batch_size]
                    xt=torch.stack([x['noisy'] for x in batch]).to(model.device)
                    targets=torch.stack([x['tokens'] for x in batch]).to(model.device)
                    # Time does not condition this model; retain schedule-consistent values where possible.
                    t=torch.tensor([min(1.,x['rate']/max(1e-8,1-float(model.config.noise.eps))) for x in batch],device=model.device)[:,None]
                    _,alpha=model.noise(t);sigma=model._sigma_from_alphat(alpha)
                    if start==0:
                        raw=model.backbone(None,xt,model._process_sigma(sigma),None)
                        if raw.dtype!=torch.float32:raise ValueError('FP32 forward not active.')
                        del raw
                    prediction=model.forward(xt=xt,sigma=sigma)
                    target_logps=prediction.gather(-1,targets[:,:,None]).squeeze(-1)
                    top1=prediction.argmax(-1).eq(targets)
                    for j,item in enumerate(batch):
                        if item['rate']==1:
                            fullmask_scores[label][(item['sample_id'],item['reward'])]=target_logps[j][item['masked'].to(model.device)].mean().item()
                        for region,positions in item['regions'].items():
                            selected=(item['masked'] & positions).to(model.device);n=int(selected.sum())
                            logp=target_logps[j][selected]
                            if n and not torch.isfinite(logp).all():raise FloatingPointError('Invalid target log probabilities.')
                            row=dict(model=label,sample_id=item['sample_id'],reward=item['reward'],
                                rate=item['rate'],replicate=item['replicate'],region=region,tokens=n,
                                answer_masked_tokens=int(item['masked'].sum()),
                                realized_rate=float(item['masked'].sum()/item['eligible'].sum()),
                                mean_logp=logp.mean().item() if n else None,top1=top1[j][selected].float().mean().item() if n else None)
                            handle.write(json.dumps(row)+'\n');outputs.append(row)
                    handle.flush()
                    if start % (args.batch_size*16)==0:print(json.dumps(dict(model=label,scored=min(start+args.batch_size,len(plan)),total=len(plan))),flush=True)
                    del prediction,target_logps,top1
                # Check q=1 against the exact existing full-MASK proxy on identical fixed samples.
                differences=[]
                for record in pairs:
                    sid=record['sample_id'];reward=record['reward']
                    item=next(x for x in plan if x['sample_id']==sid and x['reward']==reward)
                    tokens=item['tokens'][None].to(model.device)
                    lengths=torch.tensor([item['prefix_length']],device=model.device)
                    old=proxy_logps(model,tokens,lengths,0.,42)[0][item['eligible'].to(model.device)].mean().item()
                    if 1. in args.rates:differences.append(abs(old-fullmask_scores[label][(sid,reward)]))
                tolerance=2e-5
                max_difference=max(differences,default=0.)
                if max_difference>tolerance:raise ValueError(f'Full-MASK proxy reproduction failed: {max_difference}')
                (run/f'{label}-q1-check.json').write_text(json.dumps(dict(max_difference=max_difference,tolerance=tolerance)))
            all_records.extend(outputs)
            del model;gc.collect();torch.cuda.empty_cache()
        scores=question_scores(all_records)
        (run/'per_question.json').write_text(json.dumps(scores,indent=2))
        comparisons=[]
        for label in list(checkpoints)[1:]:
            for rate in args.rates:
                for reward in [1,0]:
                    for region in ['all_answer','code_eos','docstring_comment']:
                        select=lambda model:[x for x in scores if x['model']==model and x['rate']==rate and x['reward']==reward and x['region']==region]
                        comparisons.append(dict(model=label,rate=rate,reward=reward,region=region,
                                                **paired_summary(select('base'),select(label))))
        result=dict(comparisons=comparisons,protocol=dict(rates=args.rates,partial_mask_replicates=args.replicates,
            full_mask_replicates=1,precision='float32',answer_definition='prefix excluded, first EOS included, PAD/MASK excluded',
            context='Only unmasked eligible target tokens restored. All other suffix tokens always MASK.',
            uncertainty='Question-cluster paired bootstrap after averaging mask replicates.'),
            limitations=['Teacher-forced visible target tokens, not actual sampler trajectories.',
                'Failed program means reconstruction of that failed program, not a correct gold answer.',
                'Code regions unavailable on invalid source or token alignment mismatch; coverage reported.',
                'No ELBO weighting; only masked tokens scored. No model updates.',
                'Same selected 32 training questions; not generalization or a causal proof of objective mismatch.'])
        (run/'comparison.json').write_text(json.dumps(result,indent=2))
        (run/'STAGE').write_text('complete\n');(run/'COMPLETE').touch()
    except BaseException:
        (run/'FAILED').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
