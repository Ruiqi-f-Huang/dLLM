#!/usr/bin/env python3
"""Expanded fixed32 positive-only partial75 fit; failure programs evaluation-only."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--precision', choices=['bfloat16', 'float32'], required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--microbatch', type=int, default=2)
    args = parser.parse_args()
    if min(args.steps, args.count, args.microbatch) < 1 or args.lr <= 0:
        parser.error('Positive counts and learning rate required.')
    import torch
    from grpo_fixed32 import read_jsonl, load_model
    from grpo_overfit import stable_seed
    from grpo_controlled import signed_loss, flatten_gradients, gradient_geometry
    from diffu_grpo import completion_mask, partial_completion_mask
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'run.json').write_text(json.dumps(vars(args), indent=2))
    try:
        rows = read_jsonl(args.questions)
        rows.sort(key=lambda r: (stable_seed('controlled-pair-subset',42,r['sample_id']), r['sample_id']))
        rows = rows[:args.count]
        if len(rows) != args.count or len({r['sample_id'] for r in rows}) != args.count:
            raise ValueError('Not enough unique questions.')
        pair_records = read_jsonl(args.pairs)
        model, tokenizer = load_model(args.base, None)
        model.config.backbone_compute_dtype = args.precision
        # No outer mixed-precision trainer is used. float32 disables DiT autocast.
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('FP32 master parameters required for controlled comparison.')
        separator = tokenizer(str(model.config.data.separator), add_special_tokens=False).input_ids
        tokens, lengths, rewards, metadata = [], [], [], []
        for row in rows:
            prefix = [tokenizer.bos_token_id] + tokenizer(row['question'].strip(),add_special_tokens=False).input_ids + separator
            matches = [r for r in pair_records if r['sample_id']==row['sample_id']]
            if len(matches) != 2 or {r['reward'] for r in matches} != {0,1}:
                raise ValueError('Each question requires exactly one success and one failure.')
            for record in sorted(matches,key=lambda r:-r['reward']):
                if record['tokens'][:len(prefix)] != prefix or len(record['tokens']) != model.num_tokens:
                    raise ValueError('Diagnostic token/prefix mismatch.')
                tokens.append(record['tokens']); lengths.append(len(prefix)); rewards.append(record['reward'])
                metadata.append(dict(sample_id=row['sample_id'],reward=record['reward']))
        tokens = torch.tensor(tokens,device=model.device)
        lengths = torch.tensor(lengths,device=model.device)
        advantages = torch.tensor(rewards,device=model.device,dtype=torch.float32)*.5
        # Retain the signed-run scaling and denominator; failures contribute zero.
        # Same 16 programs/masks remain available for paired evaluation.
        valid = completion_mask(tokens,lengths,tokenizer.eos_token_id,tokenizer.pad_token_id,model.mask_index)
        if not valid.any(1).all(): raise ValueError('Empty diagnostic completion.')
        (output/'fixed_batch.json').write_text(json.dumps(dict(records=metadata,
            valid_lengths=valid.sum(1).tolist(),
            source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.questions,args.pairs]}),indent=2))
        train_valid = partial_completion_mask(valid, .75, 2026102401)
        heldout_valid = [partial_completion_mask(valid, .75, 2026102402+i) for i in range(4)]
        def make_input(mask):
            xt=tokens.clone()
            suffix=torch.arange(tokens.shape[1],device=model.device)[None,:]>=lengths[:,None]
            xt[suffix]=model.mask_index
            xt[valid & ~mask]=tokens[valid & ~mask]
            return xt
        train_input=make_input(train_valid)
        heldout_inputs=[make_input(mask) for mask in heldout_valid]
        torch.save(dict(tokens=tokens.cpu(),lengths=lengths.cpu(),train_mask=train_valid.cpu(),
            heldout_masks=[x.cpu() for x in heldout_valid],metadata=metadata),output/'mask_plan.pt')
        (output/'runner.py').write_text(Path(__file__).read_text())
        def plan_logps(sl, heldout=None):
            xt=train_input[sl] if heldout is None else heldout_inputs[heldout][sl]
            _,alpha=model.noise(torch.ones(len(xt),1,device=model.device))
            logps=model.forward(xt=xt,sigma=model._sigma_from_alphat(alpha))
            return logps.gather(-1,tokens[sl,:,None]).squeeze(-1)
        valid=train_valid
        parameters = [p for p in model.parameters() if p.requires_grad]
        initial = [p.detach().clone() for p in parameters]
        frozen = {n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                  for n,p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW(parameters,lr=args.lr,weight_decay=0)
        reference = []
        with torch.no_grad():
            for start in range(0,len(tokens),args.microbatch):
                sl=slice(start,start+args.microbatch)
                reference.append(plan_logps(sl))
        reference = torch.cat(reference)
        count = len(tokens)

        def gradients(kind):
            optimizer.zero_grad(set_to_none=True)
            total=0.
            for start in range(0,count,args.microbatch):
                sl=slice(start,start+args.microbatch)
                logps=plan_logps(sl)
                if kind in ('program','token'):
                    loss=signed_loss(logps,valid[sl],advantages[sl],kind)
                    weight=(len(tokens[sl])/count if kind=='program' else
                            valid[sl].sum().item()/valid.sum().item())
                else:
                    delta=reference[sl][valid[sl]]-logps[valid[sl]]
                    loss=.04*(delta.exp()-delta-1).mean()
                    weight=valid[sl].sum().item()/valid.sum().item()
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss.')
                (loss*weight).backward()
                total+=loss.item()*weight
            return total, flatten_gradients(parameters)

        @torch.no_grad()
        def scores(step, heldout=None):
            valid=train_valid if heldout is None else heldout_valid[heldout]
            logps=[]
            for start in range(0,count,args.microbatch):
                sl=slice(start,start+args.microbatch)
                logps.append(plan_logps(sl,heldout))
            logps=torch.cat(logps)
            means=torch.where(valid,logps,0).sum(1)/valid.sum(1)
            if not torch.isfinite(means).all():raise FloatingPointError('Non-finite scores.')
            # Check output dtype on a small direct call to verify precision, not just cast scores.
            xt=tokens[:1].clone()
            positions=torch.arange(xt.shape[1],device=model.device)[None,:]
            xt[positions>=lengths[:1,None]]=model.mask_index
            t=torch.ones(1,1,device=model.device)
            _,alpha=model.noise(t)
            sigma=model._process_sigma(model._sigma_from_alphat(alpha))
            raw=model.backbone(None,xt,sigma,None)
            expected=torch.float32 if args.precision=='float32' else torch.bfloat16
            if raw.dtype != expected:raise ValueError(f'Wrong raw output dtype: {raw.dtype}')
            update=math.sqrt(sum((p-s).float().square().sum().item() for p,s in zip(parameters,initial)))
            record=dict(step=step,program_loss=signed_loss(logps,valid,advantages,'program').item(),
                token_loss=signed_loss(logps,valid,advantages,'token').item(),
                success_mean_logp=means[::2].mean().item(),failure_mean_logp=means[1::2].mean().item(),
                mean_gap=(means[::2]-means[1::2]).mean().item(),parameter_delta_norm=update,
                raw_output_dtype=str(raw.dtype),per_program=[dict(m,mean_logp=v) for m,v in zip(metadata,means.tolist())])
            return record

        geometry=[]
        measurements=[]
        with (output/'scores.jsonl').open('x') as score_file, (output/'updates.jsonl').open('x') as update_file:
            def save_score(step):
                r=scores(step);r['heldout']=[scores(step,i) for i in range(4)];measurements.append(r)
                score_file.write(json.dumps(r)+'\n');score_file.flush()
                print(json.dumps({k:v for k,v in r.items() if k not in ('per_program','heldout')}),flush=True)
            save_score(0)
            for step in range(1,args.steps+1):
                if step in (1,2,11,args.steps):
                    _,pg=gradients('program');_,tg=gradients('token');_,kg=gradients('reference')
                    geometry.append(dict(before_update=step,
                        program_vs_reference=gradient_geometry(pg,kg),
                        token_vs_reference=gradient_geometry(tg,kg),
                        program_vs_token=gradient_geometry(pg,tg)))
                    (output/'gradient_geometry.json').write_text(json.dumps(geometry,indent=2))
                loss,grad=gradients('program')
                if not torch.isfinite(grad).all():raise FloatingPointError('Non-finite gradient.')
                norm=torch.nn.utils.clip_grad_norm_(parameters,.2,error_if_nonfinite=True)
                optimizer.step()
                update_file.write(json.dumps(dict(step=step,loss_before_update=loss,gradient_norm=float(norm)))+'\n')
                update_file.flush()
                if step in (1,2,10,50,100,args.steps):save_score(step)
                if step in (50,100,args.steps):
                    torch.save(dict(adapters={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},args=vars(args),step=step),output/f'adapters-step{step:06d}.pt')
        changed=[n for n,p in model.named_parameters() if not p.requires_grad and
            hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()!=frozen[n]]
        if changed:raise ValueError(f'Frozen parameters changed: {changed}')
        torch.save(dict(adapters={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},optimizer=optimizer.state_dict(),args=vars(args)),output/'final_adapters.pt')
        (output/'summary.json').write_text(json.dumps(dict(
            objective='positive-only fixed 75% mask; success A=.5, failure A=0; original denominator; no KL/clipping; diagnostic only',
            frozen_changed=changed,initial=measurements[0],final=measurements[-1],
            gap_delta=measurements[-1]['mean_gap']-measurements[0]['mean_gap'],
            program_loss_delta=measurements[-1]['program_loss']-measurements[0]['program_loss']),indent=2))
        (output/'COMPLETE').touch()
    except BaseException:
        import traceback
        (output/'FAILED').write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()
