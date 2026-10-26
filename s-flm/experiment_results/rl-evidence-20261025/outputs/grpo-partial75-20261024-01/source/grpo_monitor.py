"""Detached observations and explicit gradient decomposition for online GRPO."""
import ast
import io
import tokenize
import torch
from diffu_grpo import clipped_loss, proxy_logps
from grpo_controlled import flatten_gradients, gradient_geometry


def ratio_metrics(current, old, reference, advantages, valid, epsilon, beta):
    with torch.no_grad():
        ratio = (current[valid]-old[valid]).exp()
        adv = advantages[:, None].expand_as(valid)[valid]
        surrogate = torch.minimum(ratio*adv, ratio.clamp(1-epsilon,1+epsilon)*adv)
        delta = reference[valid]-current[valid]
        kl = delta.exp()-delta-1
        result = {}
        for name, mask in [('all',torch.ones_like(adv,dtype=torch.bool)),
                           ('positive',adv>0),('negative',adv<0),('zero',adv==0)]:
            x=ratio[mask]
            result[name]=dict(tokens=x.numel(),policy_loss_contribution=(-surrogate[mask].sum()/len(ratio)).item(),
                             kl_contribution=(beta*kl[mask].sum()/len(ratio)).item())
            if x.numel():
                result[name].update(ratio_quantiles=torch.quantile(x.float(),
                    torch.tensor([.05,.5,.95,.99],device=x.device)).tolist(),
                    below_clip=(x<1-epsilon).float().mean().item(),
                    above_clip=(x>1+epsilon).float().mean().item())
        return result


def gradient_parts(model, parameters, tokens, lengths, old, reference,
                   advantages, valid, seed, args):
    vectors=[]
    values=[]
    for mode in ('policy','reference'):
        model.zero_grad(set_to_none=True)
        total=0.
        for start in range(0,len(tokens),args.microbatch):
            sl=slice(start,start+args.microbatch)
            weight=valid[sl].sum().item()/valid.sum().item()
            if not weight:continue
            current=proxy_logps(model,tokens[sl],lengths[sl],args.prompt_mask,seed,getattr(args,'completion_mask',1.))
            a=advantages[sl] if mode=='policy' else torch.zeros_like(advantages[sl])
            beta=0. if mode=='policy' else args.beta
            loss,_=clipped_loss(current,old[sl],reference[sl],a,valid[sl],args.epsilon,beta)
            (loss*weight).backward();total+=loss.item()*weight
        vectors.append(flatten_gradients(parameters));values.append(total)
    result=gradient_geometry(*vectors)
    result.update(policy_loss=values[0],kl_penalty=values[1],
                  combined_gradient_norm=(vectors[0]+vectors[1]).norm().item())
    model.zero_grad(set_to_none=True)
    return result,vectors[0]


def parameter_step(parameters, before, gradient_norm, policy_gradient=None):
    with torch.no_grad():
        update=torch.cat([(p-b).float().flatten() for p,b in zip(parameters,before)])
        norm=update.norm().item()
        previous=sum(b.float().square().sum().item() for b in before)**.5
        result=dict(update_norm=norm,parameter_norm_before=previous,
                    update_relative_norm=norm/previous if previous else None,
                    gradient_clip_scale=min(1.,.2/(float(gradient_norm)+1e-6)))
        if policy_gradient is not None and norm and policy_gradient.norm().item():
            result['update_cosine_negative_policy_gradient']=torch.dot(update,-policy_gradient).item()/(norm*policy_gradient.norm().item())
        return result


def code_regions(text):
    """Character regions; invalid source is explicitly unavailable."""
    try:
        tree=ast.parse(text)
    except (SyntaxError,ValueError):return None
    lines=text.splitlines(keepends=True)
    offsets=[0]
    for line in lines:offsets.append(offsets[-1]+len(line))
    result=[True]*len(text)
    # AST columns use UTF-8 bytes, so convert to character indices.
    def index(line,col):
        return offsets[line-1]+len(lines[line-1].encode()[:col].decode())
    for node in ast.walk(tree):
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body:
            first=node.body[0]
            if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str):
                for i in range(index(first.lineno,first.col_offset),index(first.end_lineno,first.end_col_offset)):
                    result[i]=False
    try:
        for item in tokenize.generate_tokens(io.StringIO(text).readline):
            if item.type==tokenize.COMMENT:
                a=offsets[item.start[0]-1]+item.start[1];b=offsets[item.end[0]-1]+item.end[1]
                for i in range(a,b):result[i]=False
    except tokenize.TokenError:return None
    return result


def ast_signature(text):
    try:
        tree=ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.body:
                x=node.body[0]
                if isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str):
                    node.body=node.body[1:]
        return ast.dump(tree,include_attributes=False)
    except (SyntaxError,ValueError):return None
