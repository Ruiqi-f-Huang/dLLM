#!/usr/bin/env python3
"""Replay fixed Base ancestral states; two models, one sampling temperature."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True)
    p.add_argument('--online-run',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--trajectory-run',help='Reuse saved Base trajectories instead of sampling.')
    p.add_argument('--candidate-checkpoint',help='Alternative RL checkpoint to score.')
    p.add_argument('--trajectories',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    if args.trajectories<1 or args.batch_size<1:p.error('Positive batch and trajectory count required.')
    import torch
    from grpo_fixed32 import read_jsonl,load_model
    from grpo_overfit import stable_seed
    from train_mdlm_grpo import execution_reward
    from samplers import AncestralSampler,AncestralContext
    from diffu_grpo import completion_mask
    from grpo_partial_mask import paired_summary
    from grpo_trajectory import transition_masks,kernel_log_probability
    online=Path(args.online_run).resolve()
    checkpoint=Path(args.candidate_checkpoint) if args.candidate_checkpoint else online/'lr3e-05/train/step-000250.ckpt'
    if not (online/'COMPLETE').exists() or not checkpoint.is_file():raise ValueError('Source experiment incomplete.')
    run=Path(args.output).resolve();run.mkdir(parents=True,exist_ok=False)
    (run/'run.json').write_text(json.dumps(vars(args),indent=2))
    try:
        rows=read_jsonl(online/'questions.jsonl')
        if len(rows)!=32 or len({x['sample_id'] for x in rows})!=32:raise ValueError('Expected original fixed 32 questions.')
        (run/'questions.jsonl').write_text((online/'questions.jsonl').read_text())
        (run/'runner.py').write_text(Path(__file__).read_text())
        (run/'grpo_trajectory.py').write_text((ROOT/'grpo_trajectory.py').read_text())
        if args.trajectory_run:
            source=Path(args.trajectory_run)
            if not (source/'COMPLETE').exists():raise ValueError('Trajectory source incomplete.')
            if (source/'questions.jsonl').read_bytes() != (run/'questions.jsonl').read_bytes():
                raise ValueError('Trajectory question identity mismatch.')
            trajectories=torch.load(source/'base_trajectories.pt',map_location='cpu',weights_only=False)
            trajectory_metadata=json.loads((source/'trajectory_metadata.json').read_text())
            for name in ['base_trajectories.pt','trajectory_metadata.json','trajectory_sha256.txt']:
                __import__('shutil').copy2(source/name,run/name)
        else:
            (run/'STAGE').write_text('collect-base-trajectories\n')
            model,tokenizer=load_model(args.base,None)
            model.config.backbone_compute_dtype='float32';model.requires_grad_(False);model.eval()
            sampler=model.sampler
            if not isinstance(sampler,AncestralSampler) or sampler.noise_removal!='ancestral' or sampler.steps_policy!='full':
                raise ValueError('Need the original full ancestral sampler.')
            sampler.temperature=.3
            sep=tokenizer(str(model.config.data.separator),add_special_tokens=False).input_ids
            trajectories=[];trajectory_metadata=[]
            with torch.inference_mode():
                for row in rows:
                    seed=stable_seed('actual-base-trajectory',20261024,row['sample_id'])
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    prefix=[tokenizer.bos_token_id]+tokenizer(row['question'].strip(),add_special_tokens=False).input_ids+sep
                    ids=torch.tensor([prefix],device=model.device).expand(args.trajectories,-1)
                    lengths=torch.full((args.trajectories,),len(prefix),device=model.device,dtype=torch.long)
                    state=sampler.init_state(model,args.trajectories,num_steps=32,eps=1e-5,
                                             prefix_tokens=ids,prefix_lengths=lengths)
                    saved=[]
                    while not state.done:
                        step=state.step_idx;t=state.timesteps[step]*state.ones
                        _,at=model.noise(t)
                        if step==31:ass=torch.ones_like(at)
                        else:_,ass=model.noise(state.timesteps[step+1]*state.ones)
                        h=((ass-at)/(1-at)).flatten()[0].item()
                        before=state.xt.detach().cpu().clone()
                        sampler.step(model,state)
                        saved.append(dict(step=step,t=float(t.flatten()[0]),denoise_probability=h,
                                          before=before,after=state.xt.detach().cpu().clone()))
                    final=state.xt.detach().cpu().clone()
                    for j in range(args.trajectories):
                        suffix=final[j,len(prefix):].tolist()
                        if tokenizer.eos_token_id in suffix:suffix=suffix[:suffix.index(tokenizer.eos_token_id)]
                        text=tokenizer.decode(suffix,skip_special_tokens=True)
                        reward=execution_reward(text,row['answer'],2.)
                        metadata=dict(sample_id=row['sample_id'],trajectory=j,seed=seed,reward=reward,
                                      prefix_length=len(prefix),response=text)
                        valid=completion_mask(final[j:j+1],torch.tensor([len(prefix)]),tokenizer.eos_token_id,
                                              tokenizer.pad_token_id,model.mask_index)[0]
                        transitions=[]
                        for x in saved:
                            before=x['before'][j];after=x['after'][j]
                            transition_masks(before,after,len(prefix),model.mask_index)
                            if not torch.equal(before[before!=model.mask_index],final[j][before!=model.mask_index]):
                                raise ValueError('Revealed tokens disagree with final Base output.')
                            transitions.append(dict(step=x['step'],t=x['t'],denoise_probability=x['denoise_probability'],
                                                    before=before,after=after))
                        trajectories.append(dict(**metadata,final_tokens=final[j],valid_completion=valid,states=transitions))
                        trajectory_metadata.append(metadata)
                    print(json.dumps(dict(collected_questions=len(trajectories)//args.trajectories,total_questions=32)),flush=True)
            torch.save(trajectories,run/'base_trajectories.pt')
            (run/'trajectory_metadata.json').write_text(json.dumps(trajectory_metadata,ensure_ascii=False,indent=2))
            (run/'trajectory_sha256.txt').write_text(hashlib.sha256((run/'base_trajectories.pt').read_bytes()).hexdigest()+'\n')
            del model;gc.collect();torch.cuda.empty_cache()
        chosen_steps=[0,8,16,24,31]
        plan=[(trajectory,trajectory['states'][step]) for trajectory in trajectories for step in chosen_steps]
        all_scores=[]
        for label,path in [('base',None),('rl250',str(checkpoint))]:
            (run/'STAGE').write_text('score-'+label+'\n')
            model,tokenizer=load_model(args.base,path)
            model.config.backbone_compute_dtype='float32';model.requires_grad_(False);model.eval()
            with (run/f'{label}.jsonl').open('x') as handle,torch.inference_mode():
                for start in range(0,len(plan),args.batch_size):
                    batch=plan[start:start+args.batch_size]
                    xt=torch.stack([state['before'] for trajectory,state in batch]).to(model.device)
                    final=torch.stack([trajectory['final_tokens'] for trajectory,state in batch]).to(model.device)
                    times=torch.tensor([state['t'] for trajectory,state in batch],device=model.device)[:,None]
                    _,alpha=model.noise(times);sigma=model._sigma_from_alphat(alpha)
                    if start==0:
                        raw=model.backbone(None,xt,model._process_sigma(sigma),None)
                        if raw.dtype!=torch.float32:raise ValueError('FP32 not active.')
                        del raw
                    # Two scoring temperatures are diagnostics on the same states, not new sampling groups.
                    for temperature in [1.,.3]:
                        prediction=model.forward(xt=xt,sigma=sigma,
                            context=AncestralContext(temperature=temperature))
                        target_logp=prediction.gather(-1,final[:,:,None]).squeeze(-1)
                        top1=prediction.argmax(-1).eq(final)
                        for j,(trajectory,state) in enumerate(batch):
                            masked,revealed,remained=transition_masks(state['before'],state['after'],
                                trajectory['prefix_length'],model.mask_index)
                            regions={'remaining_completion':masked & trajectory['valid_completion'],
                                     'revealed_completion':revealed & trajectory['valid_completion'],
                                     'revealed_all_suffix':revealed}
                            for region,selected in regions.items():
                                selected=selected.to(model.device);n=int(selected.sum())
                                values=target_logp[j][selected]
                                if n and not torch.isfinite(values).all():raise FloatingPointError('Non-finite target score.')
                                record=dict(model=label,sample_id=trajectory['sample_id'],trajectory=trajectory['trajectory'],
                                    reward=trajectory['reward'],step=state['step'],t=state['t'],region=region,
                                    score_temperature=temperature,tokens=n,
                                    mean_logp=values.mean().item() if n else None,
                                    top1=top1[j][selected].float().mean().item() if n else None,
                                    remaining_mask_fraction=float(masked.sum()/(len(masked)-trajectory['prefix_length'])))
                                handle.write(json.dumps(record)+'\n');all_scores.append(record)
                            if temperature==.3:
                                # All observed new tokens equal their final Base tokens by absorption.
                                exact=kernel_log_probability(target_logp[j],revealed.to(model.device),
                                    remained.to(model.device),state['denoise_probability'])
                                handle.write(json.dumps(dict(model=label,sample_id=trajectory['sample_id'],
                                    trajectory=trajectory['trajectory'],step=state['step'],reward=trajectory['reward'],
                                    transition_log_probability=exact,masked_positions=int(masked.sum()),
                                    newly_revealed=int(revealed.sum()),remained_masked=int(remained.sum())))+'\n')
                        del prediction,target_logp,top1
                    handle.flush()
                    if start % (args.batch_size*8)==0:print(json.dumps(dict(model=label,scored=min(start+args.batch_size,len(plan)),total=len(plan))),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
        # Average trajectories within question/reward/state before uncertainty calculations.
        groups=defaultdict(list)
        for x in all_scores:
            if x['tokens']:groups[(x['model'],x['sample_id'],x['reward'],x['step'],x['region'],x['score_temperature'])].append(x)
        macro=[]
        for (model,sid,reward,step,region,temp),values in groups.items():
            macro.append(dict(model=model,sample_id=sid,reward=reward,step=step,region=region,temperature=temp,
                trajectories=len(values),mean_logp=sum(x['mean_logp'] for x in values)/len(values),
                ce=-sum(x['mean_logp'] for x in values)/len(values),top1=sum(x['top1'] for x in values)/len(values)))
        (run/'per_question.json').write_text(json.dumps(macro,indent=2))
        comparisons=[]
        for reward in [1.,0.]:
            for step in chosen_steps:
                for region in ['remaining_completion','revealed_completion','revealed_all_suffix']:
                    for temp in [1.,.3]:
                        select=lambda model:[x for x in macro if x['model']==model and x['reward']==reward and x['step']==step and x['region']==region and x['temperature']==temp]
                        comparisons.append(dict(reward=reward,step=step,region=region,temperature=temp,
                                                **paired_summary(select('base'),select('rl250'))))
        (run/'comparison.json').write_text(json.dumps(dict(comparisons=comparisons,
            trajectory_count=len(trajectories),successful_trajectories=sum(x['reward'] for x in trajectories),
            protocol=dict(n_questions=32,trajectories_per_question=args.trajectories,rollout_temperature=.3,
                          nfe=32,scored_steps=chosen_steps,precision='float32',models=['base','rl250']),
            limitations=['Fixed Base state distribution; not RL-policy trajectories or a causal success evaluation.',
                'Targets are Base realized/future tokens; failed trajectories are not correct gold programs.',
                'Completion boundary uses final Base EOS retrospectively; full suffix transition score also saved.',
                'Temperature .3 conditional transition probabilities use the actual absorbing kernel; no claim of final-program likelihood.',
                'Success/failure strata may cover different questions; paired model comparisons are within each stratum.',
                'Only two Base trajectories per question; bootstrap clusters trajectories by question.']),indent=2))
        (run/'STAGE').write_text('complete\n');(run/'COMPLETE').touch()
    except BaseException:
        (run/'FAILED').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
