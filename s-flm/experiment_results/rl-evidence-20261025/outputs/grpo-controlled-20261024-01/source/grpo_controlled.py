"""Fixed-rollout signed proxy objective and gradient diagnostics."""
import torch


def signed_loss(logps, valid, advantages, reduction):
    # No clipping, reference penalty or refresh of the fixed advantages.
    selected = torch.where(valid, logps, 0.)
    counts = valid.sum(1)
    if (counts == 0).any():
        raise ValueError('Empty completion in fixed diagnostic batch.')
    if reduction == 'program':
        return -(advantages * selected.sum(1) / counts).mean()
    if reduction == 'token':
        return -(advantages[:, None] * selected).sum() / counts.sum()
    raise ValueError(reduction)


def flatten_gradients(parameters):
    return torch.cat([(p.grad.detach().float().flatten() if p.grad is not None
                       else torch.zeros(p.numel(), device=p.device)) for p in parameters])


def gradient_geometry(policy, reference):
    a, b = policy.norm().item(), reference.norm().item()
    cosine = (torch.dot(policy, reference).item()/(a*b)) if a and b else None
    return dict(policy_gradient_norm=a, reference_gradient_norm=b,
                reference_to_policy_norm=b/a if a else None, cosine=cosine)
